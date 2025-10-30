"""
Tests for distributed reduction operations (sum, mean).

These operations require all-reduce since each worker has only part of the data.
"""

import numpy as np
from gt.client.tensor import randn


def test_distributed_sum_4_workers(client_4workers):
    """Test distributed sum with 4 workers."""
    # Create sharded tensor
    a = randn(128, 64)  # Sharded as (32, 64) per worker

    # Compute sum
    result = a.sum()
    result_value = result.data.numpy()

    # Verify: compare with numpy
    a_full = a.data.numpy()
    expected = np.sum(a_full)

    # Should match within floating point precision
    np.testing.assert_allclose(result_value, expected, rtol=1e-5)
    print(f"✓ Distributed sum: result={result_value:.6f}, expected={expected:.6f}")


def test_distributed_mean_4_workers(client_4workers):
    """Test distributed mean with 4 workers."""
    # Create sharded tensor
    a = randn(128, 64)  # Sharded as (32, 64) per worker

    # Compute mean
    result = a.mean()
    result_value = result.data.numpy()

    # Verify: compare with numpy
    a_full = a.data.numpy()
    expected = np.mean(a_full)

    # Should match within floating point precision
    np.testing.assert_allclose(result_value, expected, rtol=1e-5)
    print(f"✓ Distributed mean: result={result_value:.6f}, expected={expected:.6f}")


def test_sum_after_matmul_4_workers(client_4workers):
    """Test sum on result of distributed matmul."""
    # Create sharded matrices
    a = randn(128, 64)  # Sharded
    b_data = np.random.randn(64, 32).astype('float32')

    from gt.client.tensor import from_numpy
    b = from_numpy(b_data)  # Not sharded

    # Distributed matmul
    c = a @ b  # Result is sharded: (32, 32) per worker

    # Sum the result (should use distributed reduction)
    total = c.sum()
    result_value = total.data.numpy()

    # Verify
    a_full = a.data.numpy()
    c_expected = a_full @ b_data
    expected_sum = np.sum(c_expected)

    np.testing.assert_allclose(result_value, expected_sum, rtol=1e-4)
    print(f"✓ Sum after matmul: result={result_value:.6f}, expected={expected_sum:.6f}")


def test_mean_after_matmul_4_workers(client_4workers):
    """Test mean on result of distributed matmul."""
    # Create sharded matrices
    a = randn(128, 64)  # Sharded
    b_data = np.random.randn(64, 32).astype('float32')

    from gt.client.tensor import from_numpy
    b = from_numpy(b_data)  # Not sharded

    # Distributed matmul
    c = a @ b  # Result is sharded: (32, 32) per worker

    # Mean of the result
    avg = c.mean()
    result_value = avg.data.numpy()

    # Verify
    a_full = a.data.numpy()
    c_expected = a_full @ b_data
    expected_mean = np.mean(c_expected)

    np.testing.assert_allclose(result_value, expected_mean, rtol=1e-4)
    print(f"✓ Mean after matmul: result={result_value:.6f}, expected={expected_mean:.6f}")
