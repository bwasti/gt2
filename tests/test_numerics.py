"""
Numeric correctness tests for GT.

Tests that operations produce correct results compared to numpy.

Keep this SIMPLE and READABLE.
"""

import pytest
import numpy as np
from gt.client.tensor import randn, from_numpy, zeros


def test_zeros(client):
    """Test that zeros creates correct tensor."""
    t = zeros(3, 4)
    data = t.data

    expected = np.zeros((3, 4), dtype='float32')
    np.testing.assert_array_equal(data, expected)


def test_from_numpy(client):
    """Test creating tensor from numpy array."""
    np_array = np.array([[1, 2], [3, 4]], dtype=np.float32)
    t = from_numpy(np_array)
    data = t.data

    np.testing.assert_array_equal(data, np_array)


def test_addition(client):
    """Test element-wise addition."""
    a_np = np.array([[1, 2], [3, 4]], dtype=np.float32)
    b_np = np.array([[5, 6], [7, 8]], dtype=np.float32)

    a = from_numpy(a_np)
    b = from_numpy(b_np)
    c = a + b

    expected = a_np + b_np
    np.testing.assert_array_almost_equal(c.data, expected)


def test_subtraction(client):
    """Test element-wise subtraction."""
    a_np = np.array([[10, 20], [30, 40]], dtype=np.float32)
    b_np = np.array([[1, 2], [3, 4]], dtype=np.float32)

    a = from_numpy(a_np)
    b = from_numpy(b_np)
    c = a - b

    expected = a_np - b_np
    np.testing.assert_array_almost_equal(c.data, expected)


def test_multiplication(client):
    """Test element-wise multiplication."""
    a_np = np.array([[1, 2], [3, 4]], dtype=np.float32)
    b_np = np.array([[2, 3], [4, 5]], dtype=np.float32)

    a = from_numpy(a_np)
    b = from_numpy(b_np)
    c = a * b

    expected = a_np * b_np
    np.testing.assert_array_almost_equal(c.data, expected)


def test_division(client):
    """Test element-wise division."""
    a_np = np.array([[10, 20], [30, 40]], dtype=np.float32)
    b_np = np.array([[2, 4], [5, 8]], dtype=np.float32)

    a = from_numpy(a_np)
    b = from_numpy(b_np)
    c = a / b

    expected = a_np / b_np
    np.testing.assert_array_almost_equal(c.data, expected)


def test_matmul(client):
    """Test matrix multiplication."""
    a_np = np.array([[1, 2], [3, 4]], dtype=np.float32)
    b_np = np.array([[5, 6], [7, 8]], dtype=np.float32)

    a = from_numpy(a_np)
    b = from_numpy(b_np)
    c = a @ b

    expected = a_np @ b_np
    np.testing.assert_array_almost_equal(c.data, expected)


def test_matmul_non_square(client):
    """Test matrix multiplication with non-square matrices."""
    a_np = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)  # 2x3
    b_np = np.array([[7, 8], [9, 10], [11, 12]], dtype=np.float32)  # 3x2

    a = from_numpy(a_np)
    b = from_numpy(b_np)
    c = a @ b

    expected = a_np @ b_np
    np.testing.assert_array_almost_equal(c.data, expected)


def test_exp(client):
    """Test exponential function."""
    a_np = np.array([[0, 1], [2, 3]], dtype=np.float32)

    a = from_numpy(a_np)
    b = a.exp()

    expected = np.exp(a_np)
    np.testing.assert_array_almost_equal(b.data, expected)


def test_log(client):
    """Test natural logarithm."""
    a_np = np.array([[1, 2], [3, 4]], dtype=np.float32)

    a = from_numpy(a_np)
    b = a.log()

    expected = np.log(a_np)
    np.testing.assert_array_almost_equal(b.data, expected)


def test_sum(client):
    """Test sum reduction."""
    a_np = np.array([[1, 2], [3, 4]], dtype=np.float32)

    a = from_numpy(a_np)
    b = a.sum()

    expected = np.sum(a_np)
    np.testing.assert_almost_equal(b.data, expected)


def test_mean(client):
    """Test mean reduction."""
    a_np = np.array([[1, 2], [3, 4]], dtype=np.float32)

    a = from_numpy(a_np)
    b = a.mean()

    expected = np.mean(a_np)
    np.testing.assert_almost_equal(b.data, expected)


def test_chained_operations(client):
    """Test chaining multiple operations."""
    a_np = np.array([[1, 2], [3, 4]], dtype=np.float32)
    b_np = np.array([[5, 6], [7, 8]], dtype=np.float32)

    a = from_numpy(a_np)
    b = from_numpy(b_np)

    # (a + b) * a
    c = (a + b) * a

    expected = (a_np + b_np) * a_np
    np.testing.assert_array_almost_equal(c.data, expected)


def test_complex_expression(client):
    """Test complex expression: (a @ b) + (c * d)."""
    a_np = np.array([[1, 2], [3, 4]], dtype=np.float32)
    b_np = np.array([[5, 6], [7, 8]], dtype=np.float32)
    c_np = np.array([[2, 2], [2, 2]], dtype=np.float32)
    d_np = np.array([[3, 3], [3, 3]], dtype=np.float32)

    a = from_numpy(a_np)
    b = from_numpy(b_np)
    c = from_numpy(c_np)
    d = from_numpy(d_np)

    result = (a @ b) + (c * d)

    expected = (a_np @ b_np) + (c_np * d_np)
    np.testing.assert_array_almost_equal(result.data, expected)


def test_randn_shape(client):
    """Test that randn creates tensor with correct shape."""
    t = randn(3, 4, 5)
    data = t.data

    assert data.shape == (3, 4, 5)
    assert data.dtype == np.float32


def test_randn_statistics(client):
    """Test that randn produces values with approximately correct statistics."""
    # Large enough sample to test distribution
    t = randn(1000, 1000)
    data = t.data

    # Mean should be close to 0
    mean = np.mean(data)
    assert abs(mean) < 0.05, f"Mean {mean} too far from 0"

    # Std should be close to 1
    std = np.std(data)
    assert abs(std - 1.0) < 0.05, f"Std {std} too far from 1"
