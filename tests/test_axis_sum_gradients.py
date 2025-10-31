"""Test axis-aware sum gradients (remote execution)."""

import numpy as np
import pytest
import gt


def test_full_reduction_gradient():
    """Test gradient of full reduction."""
    x = gt.from_numpy(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype='float32'), requires_grad=True)
    y = x.sum()
    y.backward()
    grad = x.grad.data.numpy()
    expected = np.ones((2, 3), dtype='float32')

    assert grad.shape == expected.shape, f"Shape mismatch: {grad.shape} vs {expected.shape}"
    assert np.allclose(grad, expected), f"Gradient mismatch: {grad} vs {expected}"


def test_axis_0_reduction_gradient():
    """Test gradient of axis=0 reduction."""
    x = gt.from_numpy(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype='float32'), requires_grad=True)
    y = x.sum(axis=0, keepdims=False)
    loss = y.sum()  # Sum to scalar to get gradient
    loss.backward()
    grad = x.grad.data.numpy()
    expected = np.ones((2, 3), dtype='float32')

    assert grad.shape == expected.shape, f"Shape mismatch: {grad.shape} vs {expected.shape}"
    assert np.allclose(grad, expected), f"Gradient mismatch: {grad} vs {expected}"


def test_axis_1_reduction_gradient():
    """Test gradient of axis=1 reduction."""
    x = gt.from_numpy(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype='float32'), requires_grad=True)
    y = x.sum(axis=1, keepdims=False)
    loss = y.sum()  # Sum to scalar to get gradient
    loss.backward()
    grad = x.grad.data.numpy()
    expected = np.ones((2, 3), dtype='float32')

    assert grad.shape == expected.shape, f"Shape mismatch: {grad.shape} vs {expected.shape}"
    assert np.allclose(grad, expected), f"Gradient mismatch: {grad} vs {expected}"
