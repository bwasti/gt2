"""Test axis-aware sum operations."""

import numpy as np
import pytest
import gt


def test_sum_axis_0():
    """Test sum along axis 0."""
    a = gt.from_numpy(np.array([[1, 2, 3], [4, 5, 6]], dtype='float32'))
    result = a.sum(axis=0, keepdims=False)
    result_data = result.data.numpy()
    expected = np.array([5, 7, 9], dtype='float32')

    assert result.shape == (3,), f"Shape mismatch: {result.shape} != (3,)"
    assert np.allclose(result_data, expected), f"Result mismatch: {result_data} != {expected}"


def test_sum_axis_0_keepdims():
    """Test sum along axis 0 with keepdims."""
    a = gt.from_numpy(np.array([[1, 2, 3], [4, 5, 6]], dtype='float32'))
    result = a.sum(axis=0, keepdims=True)
    result_data = result.data.numpy()
    expected = np.array([[5, 7, 9]], dtype='float32')

    assert result.shape == (1, 3), f"Shape mismatch: {result.shape} != (1, 3)"
    assert np.allclose(result_data, expected), f"Result mismatch: {result_data} != {expected}"


def test_sum_axis_1():
    """Test sum along axis 1."""
    a = gt.from_numpy(np.array([[1, 2, 3], [4, 5, 6]], dtype='float32'))
    result = a.sum(axis=1, keepdims=False)
    result_data = result.data.numpy()
    expected = np.array([6, 15], dtype='float32')

    assert result.shape == (2,), f"Shape mismatch: {result.shape} != (2,)"
    assert np.allclose(result_data, expected), f"Result mismatch: {result_data} != {expected}"


def test_sum_full_reduction():
    """Test full reduction (sum all elements)."""
    a = gt.from_numpy(np.array([[1, 2, 3], [4, 5, 6]], dtype='float32'))
    result = a.sum()
    result_data = result.data.numpy()
    expected = 21.0

    assert result.shape == (), f"Shape mismatch: {result.shape} != ()"
    assert np.allclose(result_data, expected), f"Result mismatch: {result_data} != {expected}"
