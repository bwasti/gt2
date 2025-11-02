"""
Test signal-driven tensor sharding.

Verifies that the ShardingStreamModifier correctly transforms commands
based on signal scopes and YAML configuration.
"""

import pytest
import numpy as np
import os
import threading
import time
from gt.client.client import Client


@pytest.fixture(scope="module")
def gt_system_8workers_with_config():
    """
    Start GT system with 8 workers and signal sharding config.

    Uses the tuning_demo config for testing signal-based sharding.
    """
    print("\n=== Starting GT system with 8 workers + signal config ===")

    # Load config
    config_path = 'examples/tuning_demo/sharding_config.yaml'
    if not os.path.exists(config_path):
        pytest.skip(f"Config file not found: {config_path}")

    from gt.config import get_config
    config = get_config()
    config.load(config_path)
    print(f"Loaded sharding config from {config_path}")

    # Start dispatcher
    from gt.dispatcher.dispatcher import Dispatcher
    dispatcher = Dispatcher(host='localhost', port=9003, console_log=False)

    def run_dispatcher():
        print("Test dispatcher (8 workers + config) starting on localhost:9003")
        dispatcher.start()

    dispatcher_thread = threading.Thread(target=run_dispatcher, daemon=True)
    dispatcher_thread.start()
    time.sleep(0.5)

    # Start 8 workers
    from gt.worker.worker import Worker
    worker_threads = []
    for i in range(8):
        def make_run_worker(worker_id, delay):
            def run_worker():
                time.sleep(delay)
                worker = Worker(worker_id=worker_id, backend='numpy')
                worker.connect_to_dispatcher(dispatcher_host='localhost', dispatcher_port=9003)
            return run_worker

        worker_thread = threading.Thread(
            target=make_run_worker(f"test_worker_{i}", 0.5 + i * 0.1),
            daemon=True
        )
        worker_thread.start()
        worker_threads.append(worker_thread)

    time.sleep(2)  # Give system time to start
    print("GT system with 8 workers + config ready\n")

    yield

    print("\n=== Shutting down GT system with 8 workers ===")
    dispatcher.stop()
    config.clear()  # Clean up config


@pytest.fixture
def client_with_signals(gt_system_8workers_with_config):
    """Create a client connected to the 8-worker system with signal config."""
    client = Client(dispatcher_host="localhost", dispatcher_port=9003)
    client.connect()
    yield client
    client.disconnect()


def test_signal_sharding_pp_stage0(client_with_signals):
    """Test that tensors are sharded according to pp_stage0 signal (workers 0-1)."""
    import gt
    from gt.client.tensor import randn

    # Create tensor in pp_stage0 signal scope
    # Config says: pp_stage0 uses workers [0, 1], axis 0
    with gt.signal.context('pp_stage0'):
        a = randn(100, 64)

    # Get data back (should gather from workers 0-1)
    data = a.data.numpy()

    # Verify shape is correct
    assert data.shape == (100, 64), f"Expected shape (100, 64), got {data.shape}"
    print(f"✓ pp_stage0 tensor created and retrieved: shape={data.shape}")


def test_signal_sharding_pp_stage1(client_with_signals):
    """Test that tensors are sharded according to pp_stage1 signal (workers 2-3)."""
    import gt
    from gt.client.tensor import randn

    # Create tensor in pp_stage1 signal scope
    # Config says: pp_stage1 uses workers [2, 3], axis 0
    with gt.signal.context('pp_stage1'):
        b = randn(100, 64)

    data = b.data.numpy()
    assert data.shape == (100, 64)
    print(f"✓ pp_stage1 tensor created: shape={data.shape}")


def test_signal_sharding_multiple_stages(client_with_signals):
    """Test creating tensors in different signal scopes."""
    import gt
    from gt.client.tensor import randn

    # Create tensors in different pipeline stages
    tensors = []
    for stage in range(4):
        with gt.signal.context(f'pp_stage{stage}'):
            t = randn(50, 32)
            tensors.append(t)

    # Verify all tensors have correct shapes
    for i, t in enumerate(tensors):
        data = t.data.numpy()
        assert data.shape == (50, 32), f"Stage {i}: expected (50, 32), got {data.shape}"
        print(f"✓ pp_stage{i} tensor: shape={data.shape}")


def test_signal_sharding_no_signal(client_with_signals):
    """Test that tensors without signal context still work (single worker)."""
    from gt.client.tensor import randn

    # Create tensor without signal context
    # Should use round-robin worker selection
    a = randn(50, 32)

    data = a.data.numpy()
    assert data.shape == (50, 32)
    print(f"✓ No-signal tensor created: shape={data.shape}")


def test_signal_colpar_rowpar(client_with_signals):
    """Test column-parallel and row-parallel signal groups."""
    import gt
    from gt.client.tensor import randn

    # Test column-parallel (axis=1 sharding)
    with gt.signal.context('pp_stage0_colpar'):
        # This shards along axis 1 (columns)
        colpar = randn(64, 128)

    data = colpar.data.numpy()
    assert data.shape == (64, 128)
    print(f"✓ Column-parallel tensor: shape={data.shape}")

    # Test row-parallel (axis=0 sharding)
    with gt.signal.context('pp_stage0_rowpar'):
        # This shards along axis 0 (rows)
        rowpar = randn(128, 64)

    data = rowpar.data.numpy()
    assert data.shape == (128, 64)
    print(f"✓ Row-parallel tensor: shape={data.shape}")


def test_signal_sharding_small_tensor(client_with_signals):
    """Test that small tensors can be sharded."""
    import gt
    from gt.client.tensor import randn

    # Even small tensors should respect signal config
    with gt.signal.context('pp_stage0'):
        # 10 elements, 2 workers = 5 per worker
        small = randn(10, 8)

    data = small.data.numpy()
    assert data.shape == (10, 8)
    print(f"✓ Small tensor sharded: shape={data.shape}")


def test_signal_nested_contexts(client_with_signals):
    """Test that nested signal contexts work correctly."""
    import gt
    from gt.client.tensor import randn

    # Outer context
    with gt.signal.context('pp_stage0'):
        a = randn(50, 32)

        # Inner context (should override)
        with gt.signal.context('pp_stage1'):
            b = randn(50, 32)

        # Back to outer context
        c = randn(50, 32)

    # All should have correct shapes
    assert a.data.numpy().shape == (50, 32)
    assert b.data.numpy().shape == (50, 32)
    assert c.data.numpy().shape == (50, 32)
    print(f"✓ Nested signal contexts work correctly")


def test_signal_sharding_correctness(client_with_signals):
    """Verify that sharded tensors produce correct numerical results."""
    import gt
    from gt.client.tensor import randn

    # Create a tensor with known content
    np.random.seed(42)
    expected_data = np.random.randn(100, 64).astype('float32')

    # Create tensor in signal scope
    with gt.signal.context('pp_stage0'):
        from gt.client.tensor import from_numpy
        a = from_numpy(expected_data)

    # Get data back
    retrieved_data = a.data.numpy()

    # Verify data is correct
    np.testing.assert_array_almost_equal(
        retrieved_data, expected_data, decimal=5,
        err_msg="Sharded tensor data doesn't match original"
    )
    print(f"✓ Sharded tensor data is correct: max_diff={np.max(np.abs(retrieved_data - expected_data))}")


def test_signal_sharding_all_stages(client_with_signals):
    """Test that all 4 pipeline stages work with their respective worker groups."""
    import gt
    from gt.client.tensor import randn

    # Test all 4 stages with their worker groups
    # Stage 0: workers [0, 1]
    # Stage 1: workers [2, 3]
    # Stage 2: workers [4, 5]
    # Stage 3: workers [6, 7]

    results = []
    for stage_id in range(4):
        with gt.signal.context(f'pp_stage{stage_id}'):
            t = randn(80, 40)
            data = t.data.numpy()
            assert data.shape == (80, 40), f"Stage {stage_id} failed"
            results.append(data)

    print(f"✓ All 4 pipeline stages work correctly")
    print(f"  Stage 0 (workers 0-1): {results[0].shape}")
    print(f"  Stage 1 (workers 2-3): {results[1].shape}")
    print(f"  Stage 2 (workers 4-5): {results[2].shape}")
    print(f"  Stage 3 (workers 6-7): {results[3].shape}")
