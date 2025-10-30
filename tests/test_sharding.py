"""
Test tensor sharding across multiple workers.

Verifies that the dispatcher can shard tensors and perform distributed operations.
"""

import pytest
import numpy as np
from gt.client.tensor import from_numpy


def test_sharded_randn_4_workers(client_4workers):
    """Test that randn is sharded across 4 workers."""
    from gt.client.tensor import randn

    # Create a tensor with shape (128, 128)
    # With 4 workers, this should be sharded as (32, 128) on each worker
    a = randn(128, 128)

    # Get the data back - should automatically gather from all workers
    data = a.data.numpy()

    # Verify shape is correct
    assert data.shape == (128, 128), f"Expected shape (128, 128), got {data.shape}"
    print(f"✓ Sharded randn created: shape={data.shape}")


def test_sharded_matmul_4_workers(client_4workers):
    """Test distributed matmul with sharding across 4 workers."""
    from gt.client.tensor import randn, from_numpy

    # Create sharded matrix A (128, 64) - will be sharded as (32, 64) on each worker
    a = randn(128, 64)

    # Create matrix B (64, 32) - small enough it won't be sharded
    # To ensure B isn't sharded, we'll use from_numpy which bypasses sharding logic
    b_data = np.random.randn(64, 32).astype('float32')
    b = from_numpy(b_data)

    # Perform distributed matmul: C = A @ B
    # Each worker computes A_shard @ B locally
    # Result C should be sharded (32, 32) on each worker, total (128, 32)
    c = a @ b

    # Get result data
    c_data = c.data.numpy()

    # Verify shape
    assert c_data.shape == (128, 32), f"Expected shape (128, 32), got {c_data.shape}"
    print(f"✓ Distributed matmul completed: result shape={c_data.shape}")

    # Verify correctness by computing same thing in numpy
    a_data = a.data.numpy()
    b_data = b.data.numpy()
    expected = a_data @ b_data

    np.testing.assert_array_almost_equal(c_data, expected, decimal=4,
        err_msg="Distributed matmul result doesn't match numpy computation")
    print(f"✓ Distributed matmul result matches numpy: max_diff={np.max(np.abs(c_data - expected))}")


def test_sharded_matmul_multiple_ops(client_4workers):
    """Test multiple distributed operations in sequence."""
    from gt.client.tensor import randn

    # Create sharded tensors
    a = randn(64, 32)  # Sharded to (16, 32) per worker with 4 workers
    b = randn(32, 16)
    c = randn(16, 8)

    # Chain matmuls: D = (A @ B) @ C
    temp = a @ b  # (64, 16) sharded
    d = temp @ c  # (64, 8) sharded

    # Get result
    d_data = d.data.numpy()

    # Verify shape
    assert d_data.shape == (64, 8), f"Expected shape (64, 8), got {d_data.shape}"
    print(f"✓ Chained distributed matmuls: final shape={d_data.shape}")

    # Verify correctness
    a_data = a.data.numpy()
    b_data = b.data.numpy()
    c_data = c.data.numpy()
    expected = (a_data @ b_data) @ c_data

    np.testing.assert_array_almost_equal(d_data, expected, decimal=4,
        err_msg="Chained matmul doesn't match numpy")
    print(f"✓ Chained matmul result matches numpy: max_diff={np.max(np.abs(d_data - expected))}")


def test_sharding_with_large_tensors(client_4workers):
    """Test sharding with larger tensors to see actual distribution benefits."""
    from gt.client.tensor import randn

    # Large sharded matrix
    a = randn(512, 256)  # Each worker gets (128, 256) with 4 workers
    b = randn(256, 128)

    # Distributed matmul
    c = a @ b

    # Verify
    c_data = c.data.numpy()
    assert c_data.shape == (512, 128)

    a_data = a.data.numpy()
    b_data = b.data.numpy()
    expected = a_data @ b_data

    np.testing.assert_array_almost_equal(c_data, expected, decimal=3)
    print(f"✓ Large sharded matmul (512x256 @ 256x128): verified")
