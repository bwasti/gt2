"""
Test for Bug 1: sqrt() backward pass shape mismatch with keepdims.

This reproduces the LayerNorm pattern that was failing.
"""
import pytest
import numpy as np
import gt


def test_sqrt_backward_with_keepdims(client):
    """
    Test sqrt() backward when result has keepdims=True.

    This reproduces the LayerNorm pattern:
    mean = x.mean(axis=-1, keepdims=True)
    variance = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    std = (variance + eps).sqrt()
    x_norm = (x - mean) / std
    """
    x = gt.randn(2, 4, 10, requires_grad=True)

    # LayerNorm pattern
    mean = x.mean(axis=-1, keepdims=True)  # Shape: (2, 4, 1)
    variance = ((x - mean) ** 2).mean(axis=-1, keepdims=True)  # Shape: (2, 4, 1)
    eps = gt.from_numpy(np.array(1e-5, dtype='float32'))
    std = (variance + eps).sqrt()  # Shape: (2, 4, 1)
    x_norm = (x - mean) / std  # Shape: (2, 4, 10) - broadcasting happens here

    # This should work without shape mismatch
    loss = (x_norm ** 2).mean()
    loss.backward()

    # Check gradient exists and has correct shape
    assert x.grad is not None
    assert x.grad.data.numpy().shape == (2, 4, 10)


def test_sqrt_backward_simple(client):
    """Test simple sqrt() backward pass."""
    x = gt.from_numpy(np.array([4.0, 9.0, 16.0], dtype='float32'), requires_grad=True)

    y = x.sqrt()  # sqrt([4, 9, 16]) = [2, 3, 4]
    loss = y.sum()  # sum = 9
    loss.backward()

    # Gradient of sqrt(x) is 1/(2*sqrt(x))
    # For x = [4, 9, 16], sqrt(x) = [2, 3, 4]
    # Gradient = [1/4, 1/6, 1/8] = [0.25, 0.1667, 0.125]
    expected_grad = np.array([0.25, 1.0/6.0, 0.125], dtype='float32')

    assert x.grad is not None
    assert np.allclose(x.grad.data.numpy(), expected_grad, atol=1e-4)


def test_broadcasting_gradient_reduction(client):
    """Test that gradients are correctly reduced when broadcasting occurs."""
    # Create a tensor with shape (3, 1) - will be broadcasted
    a = gt.from_numpy(np.array([[1.0], [2.0], [3.0]], dtype='float32'), requires_grad=True)

    # Create a tensor with shape (3, 4)
    b = gt.from_numpy(np.ones((3, 4), dtype='float32'), requires_grad=True)

    # Broadcasting: (3, 1) * (3, 4) -> (3, 4)
    c = a * b

    # Sum to scalar
    loss = c.sum()
    loss.backward()

    # Gradient for 'a' should be summed over axis 1 and keep dims
    # c = a * b, so dc/da = b
    # Since b is all ones and has shape (3, 4), the gradient is the sum over axis 1
    # Expected: [[4], [4], [4]]
    assert a.grad is not None
    assert a.grad.data.numpy().shape == (3, 1)  # Must keep the singleton dimension!
    assert np.allclose(a.grad.data.numpy(), np.array([[4.0], [4.0], [4.0]], dtype='float32'))

    # Gradient for 'b' should match its shape
    # dc/db = a (broadcasted to (3, 4))
    # Expected: [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]
    assert b.grad is not None
    assert b.grad.data.numpy().shape == (3, 4)
    expected_b_grad = np.array([[1.0, 1.0, 1.0, 1.0],
                                 [2.0, 2.0, 2.0, 2.0],
                                 [3.0, 3.0, 3.0, 3.0]], dtype='float32')
    assert np.allclose(b.grad.data.numpy(), expected_b_grad)


def test_division_with_keepdims_broadcasting(client):
    """Test division when denominator has keepdims=True."""
    x = gt.randn(2, 3, 4, requires_grad=True)

    # Create divisor with keepdims
    divisor = gt.ones(2, 3, 1, requires_grad=True) * 2.0  # Shape: (2, 3, 1)

    # Division with broadcasting: (2, 3, 4) / (2, 3, 1) -> (2, 3, 4)
    result = x / divisor

    loss = result.sum()
    loss.backward()

    # Check shapes are preserved
    assert x.grad is not None
    assert x.grad.data.numpy().shape == (2, 3, 4)

    assert divisor.grad is not None
    assert divisor.grad.data.numpy().shape == (2, 3, 1)  # Must keep singleton dimension!
