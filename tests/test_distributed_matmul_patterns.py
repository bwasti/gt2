"""
Test all distributed matrix multiplication patterns.

Validates three key distributed matmul strategies:

1. EMBARRASSINGLY PARALLEL (Row-parallel A):
   - A is sharded by rows (axis 0)
   - B is replicated on all workers
   - Each worker computes: A_shard @ B → C_shard
   - Result C is sharded by rows
   - No communication during computation!

   Example: (128, 64) sharded @ (64, 32) replicated
   - Worker 0: (32, 64) @ (64, 32) → (32, 32)
   - Worker 1: (32, 64) @ (64, 32) → (32, 32)
   - Worker 2: (32, 64) @ (64, 32) → (32, 32)
   - Worker 3: (32, 64) @ (64, 32) → (32, 32)
   - Final: (128, 32) sharded

2. ALL-GATHER (Both sharded):
   - A is sharded by rows (axis 0)
   - B is sharded by rows (axis 0)
   - All-gather B to get full B on all workers
   - Each worker computes: A_shard @ B_full → C_shard
   - Result C is sharded by rows
   - Communication: All-gather B before computation

   Example: (128, 64) sharded @ (64, 32) sharded
   - All-gather B: each worker gets full (64, 32)
   - Worker 0: (32, 64) @ (64, 32) → (32, 32)
   - Worker 1: (32, 64) @ (64, 32) → (32, 32)
   - Worker 2: (32, 64) @ (64, 32) → (32, 32)
   - Worker 3: (32, 64) @ (64, 32) → (32, 32)
   - Final: (128, 32) sharded

3. ALL-REDUCE (Column-parallel B):
   - A is replicated on all workers
   - B is sharded by columns (axis 1)
   - Each worker computes: A @ B_shard → partial_C
   - All-reduce (sum) across workers to get final C
   - Result C is replicated on all workers
   - Communication: All-reduce C after computation

   Example: (128, 64) replicated @ (64, 32) column-sharded
   - Worker 0: (128, 64) @ (64, 8) → (128, 8) partial
   - Worker 1: (128, 64) @ (64, 8) → (128, 8) partial
   - Worker 2: (128, 64) @ (64, 8) → (128, 8) partial
   - Worker 3: (128, 64) @ (64, 8) → (128, 8) partial
   - All-reduce: sum partials → (128, 32) final

   NOTE: This pattern is not yet implemented in GT.
   It requires column sharding support.
"""

import pytest
import numpy as np
from gt.client.tensor import randn, from_numpy


def test_embarrassingly_parallel_matmul(client_4workers):
    """
    Test Pattern 1: EMBARRASSINGLY PARALLEL

    A is sharded by rows, B is replicated.
    This is the most efficient pattern - no communication during matmul!
    """
    print("\n" + "="*70)
    print("PATTERN 1: EMBARRASSINGLY PARALLEL (Row-parallel A)")
    print("="*70)

    # Create sharded A: (128, 64) → 4 workers × (32, 64)
    a = randn(128, 64)
    print(f"Created A: shape=(128, 64), sharded by rows")
    print(f"  Worker 0-3: each has (32, 64) shard")

    # Create replicated B: (64, 32) on single worker
    b_data = np.random.randn(64, 32).astype('float32')
    b = from_numpy(b_data)
    print(f"Created B: shape=(64, 32), replicated to all workers")

    # Matmul: C = A @ B
    print(f"\nComputing C = A @ B (embarrassingly parallel)...")
    print(f"  Worker 0: (32, 64) @ (64, 32) → (32, 32)")
    print(f"  Worker 1: (32, 64) @ (64, 32) → (32, 32)")
    print(f"  Worker 2: (32, 64) @ (64, 32) → (32, 32)")
    print(f"  Worker 3: (32, 64) @ (64, 32) → (32, 32)")
    print(f"  Communication: B broadcast (one-time), no reduction needed")

    c = a @ b
    c_data = c.data.numpy()

    # Verify correctness
    a_data = a.data.numpy()
    expected = a_data @ b_data

    assert c_data.shape == (128, 32), f"Expected (128, 32), got {c_data.shape}"
    np.testing.assert_array_almost_equal(c_data, expected, decimal=4,
        err_msg="Embarrassingly parallel matmul failed")

    max_diff = np.max(np.abs(c_data - expected))
    print(f"\n✓ PASSED: Result shape {c_data.shape}, max error {max_diff:.2e}")
    print("="*70)


def test_all_gather_matmul(client_4workers):
    """
    Test Pattern 2: ALL-GATHER

    Both A and B are sharded by rows.
    Requires all-gather of B before computation.
    """
    print("\n" + "="*70)
    print("PATTERN 2: ALL-GATHER (Both sharded by rows)")
    print("="*70)

    # Create sharded A: (128, 64) → 4 workers × (32, 64)
    a = randn(128, 64)
    print(f"Created A: shape=(128, 64), sharded by rows")
    print(f"  Worker 0-3: each has (32, 64) shard")

    # Create sharded B: (64, 32) → 4 workers × (16, 32)
    b = randn(64, 32)
    print(f"Created B: shape=(64, 32), sharded by rows")
    print(f"  Worker 0-3: each has (16, 32) shard")

    # Matmul: C = A @ B
    print(f"\nComputing C = A @ B (all-gather strategy)...")
    print(f"  Step 1: All-gather B")
    print(f"    - Collect (16, 32) shards from all 4 workers")
    print(f"    - Concatenate → full (64, 32) matrix")
    print(f"    - Broadcast full B to all workers")
    print(f"  Step 2: Local matmul on each worker")
    print(f"    Worker 0: (32, 64) @ (64, 32) → (32, 32)")
    print(f"    Worker 1: (32, 64) @ (64, 32) → (32, 32)")
    print(f"    Worker 2: (32, 64) @ (64, 32) → (32, 32)")
    print(f"    Worker 3: (32, 64) @ (64, 32) → (32, 32)")
    print(f"  Communication: All-gather B (size: 64×32 = 2KB for float32)")

    c = a @ b
    c_data = c.data.numpy()

    # Verify correctness
    a_data = a.data.numpy()
    b_data = b.data.numpy()
    expected = a_data @ b_data

    assert c_data.shape == (128, 32), f"Expected (128, 32), got {c_data.shape}"
    np.testing.assert_array_almost_equal(c_data, expected, decimal=4,
        err_msg="All-gather matmul failed")

    max_diff = np.max(np.abs(c_data - expected))
    print(f"\n✓ PASSED: Result shape {c_data.shape}, max error {max_diff:.2e}")
    print("="*70)


def test_large_all_gather_matmul(client_4workers):
    """
    Test all-gather pattern with larger matrices.

    This validates the pattern scales to more realistic sizes.
    """
    print("\n" + "="*70)
    print("PATTERN 2 (LARGE): ALL-GATHER with 512×256 matrices")
    print("="*70)

    # Large sharded matrices
    a = randn(512, 256)  # Each worker: (128, 256)
    b = randn(256, 128)  # Each worker: (64, 128)

    print(f"Created A: shape=(512, 256), sharded → 4 × (128, 256)")
    print(f"Created B: shape=(256, 128), sharded → 4 × (64, 128)")

    print(f"\nComputing C = A @ B...")
    print(f"  All-gather B: 4 × (64, 128) → (256, 128) [32 KB]")
    print(f"  Local matmul: 4 × [(128, 256) @ (256, 128) → (128, 128)]")

    c = a @ b
    c_data = c.data.numpy()

    # Verify
    a_data = a.data.numpy()
    b_data = b.data.numpy()
    expected = a_data @ b_data

    assert c_data.shape == (512, 128), f"Expected (512, 128), got {c_data.shape}"
    np.testing.assert_array_almost_equal(c_data, expected, decimal=3,
        err_msg="Large all-gather matmul failed")

    max_diff = np.max(np.abs(c_data - expected))
    print(f"\n✓ PASSED: Result shape {c_data.shape}, max error {max_diff:.2e}")
    print("="*70)


def test_chained_distributed_matmul(client_4workers):
    """
    Test chaining multiple distributed matmuls.

    This validates that sharded results can be used in subsequent operations.
    D = (A @ B) @ C with all matrices sharded.
    """
    print("\n" + "="*70)
    print("CHAINED MATMUL: (A @ B) @ C with sharding")
    print("="*70)

    # Create sharded matrices
    a = randn(64, 32)  # (64, 32) → 4 × (16, 32)
    b = randn(32, 16)  # (32, 16) → 4 × (8, 16)
    c = randn(16, 8)   # (16, 8) → 4 × (4, 8)

    print(f"Created A: (64, 32) sharded → 4 × (16, 32)")
    print(f"Created B: (32, 16) sharded → 4 × (8, 16)")
    print(f"Created C: (16, 8) sharded → 4 × (4, 8)")

    # First matmul: temp = A @ B
    print(f"\nStep 1: temp = A @ B")
    print(f"  Pattern: ALL-GATHER (both sharded)")
    print(f"  Result: (64, 16) sharded → 4 × (16, 16)")
    temp = a @ b
    temp_data = temp.data.numpy()
    assert temp_data.shape == (64, 16)

    # Second matmul: d = temp @ C
    print(f"\nStep 2: d = temp @ C")
    print(f"  Pattern: ALL-GATHER (both sharded)")
    print(f"  Result: (64, 8) sharded → 4 × (16, 8)")
    d = temp @ c
    d_data = d.data.numpy()

    # Verify
    a_data = a.data.numpy()
    b_data = b.data.numpy()
    c_data = c.data.numpy()
    expected = (a_data @ b_data) @ c_data

    assert d_data.shape == (64, 8), f"Expected (64, 8), got {d_data.shape}"
    np.testing.assert_array_almost_equal(d_data, expected, decimal=4,
        err_msg="Chained matmul failed")

    max_diff = np.max(np.abs(d_data - expected))
    print(f"\n✓ PASSED: Final shape {d_data.shape}, max error {max_diff:.2e}")
    print("="*70)


@pytest.mark.skip(reason="Column sharding (all-reduce pattern) not yet implemented")
def test_all_reduce_matmul(client_4workers):
    """
    Test Pattern 3: ALL-REDUCE (Column-parallel B)

    A is replicated, B is sharded by columns.
    Requires all-reduce to combine partial results.

    NOTE: This pattern requires column sharding support, which is not
    yet implemented in GT. Keeping this test as documentation of the
    desired behavior for future implementation.
    """
    print("\n" + "="*70)
    print("PATTERN 3: ALL-REDUCE (Column-parallel B)")
    print("="*70)

    # This would require:
    # 1. Column sharding support in randn() or explicit API
    # 2. Detection of column-sharded matmul in dispatcher
    # 3. All-reduce implementation for partial result aggregation

    # Example usage (future):
    # a = randn(128, 64)  # Replicated
    # b = randn(64, 32, shard_axis=1)  # Column-sharded: 4 × (64, 8)
    # c = a @ b  # Each worker computes (128, 64) @ (64, 8) → (128, 8)
    #            # All-reduce: sum 4 × (128, 8) → (128, 32)

    print("Column sharding not yet implemented.")
    print("This pattern would enable:")
    print("  - Model parallelism (different parts of weight matrix on different GPUs)")
    print("  - Memory-efficient inference for large models")
    print("  - Load balancing when reduction dimension is large")
    print("="*70)

    pytest.skip("Column sharding not implemented")


def test_mixed_patterns_in_workflow(client_4workers):
    """
    Test real-world scenario: mix of parallel and all-gather patterns.

    Simulates a mini neural network forward pass with sharded activations.
    Uses row-parallel strategy where activations are sharded, weights are replicated.
    """
    print("\n" + "="*70)
    print("MIXED PATTERNS: Neural network with sharded activations")
    print("="*70)

    # Input: large batch, sharded across workers
    x = randn(128, 64)
    print(f"Input x: shape=(128, 64), sharded → 4 × (32, 64)")
    print(f"  Each worker has a different batch of samples")

    # Weight matrix: replicated (small enough to fit on each worker)
    W_data = np.random.randn(64, 32).astype('float32')
    W = from_numpy(W_data)
    print(f"Weight W: shape=(64, 32), replicated on all workers")

    # First layer: h = x @ W
    print(f"\nLayer 1: h = x @ W")
    print(f"  Pattern: EMBARRASSINGLY PARALLEL (only x sharded)")
    print(f"  Worker 0: (32, 64) @ (64, 32) → (32, 32)")
    print(f"  Worker 1: (32, 64) @ (64, 32) → (32, 32)")
    print(f"  Worker 2: (32, 64) @ (64, 32) → (32, 32)")
    print(f"  Worker 3: (32, 64) @ (64, 32) → (32, 32)")
    print(f"  Result h: (128, 32) sharded → 4 × (32, 32)")

    h = x @ W
    h_data = h.data.numpy()

    # Verify
    x_data = x.data.numpy()
    expected = x_data @ W_data

    assert h_data.shape == (128, 32), f"Expected (128, 32), got {h_data.shape}"
    np.testing.assert_array_almost_equal(h_data, expected, decimal=4,
        err_msg="Mixed pattern layer 1 failed")

    max_diff = np.max(np.abs(h_data - expected))
    print(f"✓ Layer 1 output: shape {h_data.shape}, max error {max_diff:.2e}")

    # Second weight matrix: also replicated
    W2_data = np.random.randn(32, 16).astype('float32')
    W2 = from_numpy(W2_data)
    print(f"\nWeight W2: shape=(32, 16), replicated on all workers")

    # Second layer: output = h @ W2
    print(f"\nLayer 2: output = h @ W2")
    print(f"  Pattern: EMBARRASSINGLY PARALLEL (only h sharded)")
    print(f"  Same as Layer 1 - each worker processes its batch independently")
    print(f"  Result output: (128, 16) sharded → 4 × (32, 16)")

    output = h @ W2
    output_data = output.data.numpy()

    # Verify
    expected_output = h_data @ W2_data

    assert output_data.shape == (128, 16), f"Expected (128, 16), got {output_data.shape}"
    np.testing.assert_array_almost_equal(output_data, expected_output, decimal=4,
        err_msg="Mixed pattern layer 2 failed")

    max_diff = np.max(np.abs(output_data - expected_output))
    print(f"✓ Layer 2 output: shape {output_data.shape}, max error {max_diff:.2e}")
    print("\n✓ PASSED: Complete forward pass with data parallelism")
    print("  This is data-parallel training: each worker processes different samples")
    print("="*70)
