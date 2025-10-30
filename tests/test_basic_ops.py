"""
Test basic GT operations.

Tests tensor creation, arithmetic, and data retrieval.
"""

import gt
import numpy as np
import pytest


def test_randn():
    """Test random tensor creation."""
    a = gt.randn(3, 3)
    assert a.shape == (3, 3)
    assert a.dtype == "float32"

    # Should be able to get data
    data = a.data.numpy()
    assert data.shape == (3, 3)
    assert data.dtype == np.float32


def test_zeros():
    """Test zeros tensor creation."""
    a = gt.zeros(2, 4)
    assert a.shape == (2, 4)

    data = a.data.numpy()
    np.testing.assert_array_equal(data, np.zeros((2, 4), dtype=np.float32))


def test_from_numpy():
    """Test creating tensor from numpy array."""
    np_array = np.array([[1, 2], [3, 4]], dtype=np.float32)
    a = gt.from_numpy(np_array)

    assert a.shape == (2, 2)
    data = a.data.numpy()
    np.testing.assert_array_equal(data, np_array)


def test_addition():
    """Test tensor addition."""
    a = gt.from_numpy(np.array([[1, 2], [3, 4]], dtype='float32'))
    b = gt.from_numpy(np.array([[5, 6], [7, 8]], dtype='float32'))

    c = a + b
    assert c.shape == (2, 2)

    data = c.data.numpy()
    expected = np.array([[6, 8], [10, 12]], dtype='float32')
    np.testing.assert_array_equal(data, expected)


def test_multiplication():
    """Test element-wise multiplication."""
    a = gt.from_numpy(np.array([[2, 3], [4, 5]], dtype='float32'))
    b = gt.from_numpy(np.array([[1, 2], [3, 4]], dtype='float32'))

    c = a * b
    data = c.data.numpy()
    expected = np.array([[2, 6], [12, 20]], dtype='float32')
    np.testing.assert_array_equal(data, expected)


def test_matmul():
    """Test matrix multiplication."""
    a = gt.from_numpy(np.array([[1, 2], [3, 4]], dtype='float32'))
    b = gt.from_numpy(np.array([[5, 6], [7, 8]], dtype='float32'))

    c = a @ b
    assert c.shape == (2, 2)

    data = c.data.numpy()
    expected = np.array([[19, 22], [43, 50]], dtype='float32')
    np.testing.assert_array_equal(data, expected)


def test_transpose():
    """Test transpose operation."""
    a = gt.from_numpy(np.array([[1, 2, 3], [4, 5, 6]], dtype='float32'))

    b = a.T
    assert b.shape == (3, 2)

    data = b.data.numpy()
    expected = np.array([[1, 4], [2, 5], [3, 6]], dtype='float32')
    np.testing.assert_array_equal(data, expected)


def test_mean():
    """Test mean operation."""
    a = gt.from_numpy(np.array([[1, 2], [3, 4]], dtype='float32'))

    b = a.mean()
    assert b.shape == ()

    result = b.item()
    expected = 2.5
    assert abs(result - expected) < 1e-5


def test_subtraction():
    """Test tensor subtraction."""
    a = gt.from_numpy(np.array([[10, 20], [30, 40]], dtype='float32'))
    b = gt.from_numpy(np.array([[1, 2], [3, 4]], dtype='float32'))

    c = a - b
    data = c.data.numpy()
    expected = np.array([[9, 18], [27, 36]], dtype='float32')
    np.testing.assert_array_equal(data, expected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
