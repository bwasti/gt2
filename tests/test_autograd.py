"""
Test autograd with PyTorch-like API.

Tests: requires_grad, backward(), .grad
"""

import pytest
import numpy as np
from gt.client.tensor import from_numpy


def test_simple_add(client):
    """Test: (a + b).sum().backward()"""
    # Create tensors with requires_grad
    a = from_numpy(np.array([1.0, 2.0, 3.0], dtype='float32'), requires_grad=True)
    b = from_numpy(np.array([4.0, 5.0, 6.0], dtype='float32'), requires_grad=True)

    # Forward pass
    c = a + b
    loss = c.sum()

    # Backward pass
    loss.backward()

    # Get gradients
    grad_a = a.grad.data.numpy()
    grad_b = b.grad.data.numpy()

    # Check gradients: d/da sum(a + b) = [1, 1, 1]
    expected = np.ones(3, dtype='float32')
    np.testing.assert_array_almost_equal(grad_a, expected, decimal=5)
    np.testing.assert_array_almost_equal(grad_b, expected, decimal=5)


def test_multiplication(client):
    """Test: (a * b).sum().backward()"""
    a_val = np.array([2.0, 3.0, 4.0], dtype='float32')
    b_val = np.array([5.0, 6.0, 7.0], dtype='float32')

    a = from_numpy(a_val, requires_grad=True)
    b = from_numpy(b_val, requires_grad=True)

    # Forward
    c = a * b
    loss = c.sum()

    # Backward
    loss.backward()

    grad_a = a.grad.data.numpy()
    grad_b = b.grad.data.numpy()

    # d/da sum(a * b) = b, d/db sum(a * b) = a
    np.testing.assert_array_almost_equal(grad_a, b_val, decimal=5)
    np.testing.assert_array_almost_equal(grad_b, a_val, decimal=5)


def test_complex_expression(client):
    """Test: ((a + b) * a).sum().backward()"""
    a_val = np.array([1.0, 2.0, 3.0], dtype='float32')
    b_val = np.array([4.0, 5.0, 6.0], dtype='float32')

    a = from_numpy(a_val, requires_grad=True)
    b = from_numpy(b_val, requires_grad=True)

    # Forward: loss = sum((a + b) * a) = sum(a² + ab)
    c = a + b
    d = c * a
    loss = d.sum()

    # Backward
    loss.backward()

    grad_a = a.grad.data.numpy()
    grad_b = b.grad.data.numpy()

    # Analytical: d/da sum((a+b)*a) = 2a + b, d/db = a
    expected_grad_a = 2 * a_val + b_val
    expected_grad_b = a_val

    np.testing.assert_array_almost_equal(grad_a, expected_grad_a, decimal=4)
    np.testing.assert_array_almost_equal(grad_b, expected_grad_b, decimal=4)


def test_division(client):
    """Test: (a / b).sum().backward()"""
    a_val = np.array([10.0, 20.0, 30.0], dtype='float32')
    b_val = np.array([2.0, 4.0, 5.0], dtype='float32')

    a = from_numpy(a_val, requires_grad=True)
    b = from_numpy(b_val, requires_grad=True)

    c = a / b
    loss = c.sum()
    loss.backward()

    grad_a = a.grad.data.numpy()
    grad_b = b.grad.data.numpy()

    # d/da sum(a/b) = 1/b
    # d/db sum(a/b) = -a/b²
    expected_grad_a = 1.0 / b_val
    expected_grad_b = -a_val / (b_val ** 2)

    np.testing.assert_array_almost_equal(grad_a, expected_grad_a, decimal=4)
    np.testing.assert_array_almost_equal(grad_b, expected_grad_b, decimal=4)


def test_exp(client):
    """Test: exp(a).sum().backward()"""
    a_val = np.array([0.0, 1.0, 2.0], dtype='float32')
    a = from_numpy(a_val, requires_grad=True)

    b = a.exp()
    loss = b.sum()
    loss.backward()

    grad_a = a.grad.data.numpy()

    # d/da sum(exp(a)) = exp(a)
    expected = np.exp(a_val)
    np.testing.assert_array_almost_equal(grad_a, expected, decimal=4)


def test_log(client):
    """Test: log(a).sum().backward()"""
    a_val = np.array([1.0, 2.0, 3.0], dtype='float32')
    a = from_numpy(a_val, requires_grad=True)

    b = a.log()
    loss = b.sum()
    loss.backward()

    grad_a = a.grad.data.numpy()

    # d/da sum(log(a)) = 1/a
    expected = 1.0 / a_val
    np.testing.assert_array_almost_equal(grad_a, expected, decimal=4)
