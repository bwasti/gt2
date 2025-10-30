"""
Test PyTorch-like operations.
"""

import pytest
import numpy as np
from gt.client.tensor import from_numpy


def test_relu(client):
    """Test ReLU activation"""
    x_val = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype='float32')
    x = from_numpy(x_val)

    y = x.relu()
    result = y.data.numpy()

    expected = np.maximum(0, x_val)
    np.testing.assert_array_almost_equal(result, expected, decimal=5)


def test_sigmoid(client):
    """Test sigmoid activation"""
    x_val = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype='float32')
    x = from_numpy(x_val)

    y = x.sigmoid()
    result = y.data.numpy()

    expected = 1.0 / (1.0 + np.exp(-x_val))
    np.testing.assert_array_almost_equal(result, expected, decimal=5)


def test_tanh(client):
    """Test tanh activation"""
    x_val = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype='float32')
    x = from_numpy(x_val)

    y = x.tanh()
    result = y.data.numpy()

    expected = np.tanh(x_val)
    np.testing.assert_array_almost_equal(result, expected, decimal=5)


def test_relu_gradient(client):
    """Test ReLU gradient"""
    x_val = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype='float32')
    x = from_numpy(x_val, requires_grad=True)

    y = x.relu()
    loss = y.sum()
    loss.backward()

    grad = x.grad.data.numpy()

    # d/dx relu(x) = 1 if x > 0 else 0
    expected = (x_val > 0).astype('float32')
    np.testing.assert_array_almost_equal(grad, expected, decimal=5)


def test_sigmoid_gradient(client):
    """Test sigmoid gradient"""
    x_val = np.array([0.0, 1.0, 2.0], dtype='float32')
    x = from_numpy(x_val, requires_grad=True)

    y = x.sigmoid()
    loss = y.sum()
    loss.backward()

    grad = x.grad.data.numpy()

    # d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
    sig = 1.0 / (1.0 + np.exp(-x_val))
    expected = sig * (1.0 - sig)
    np.testing.assert_array_almost_equal(grad, expected, decimal=4)


def test_tanh_gradient(client):
    """Test tanh gradient"""
    x_val = np.array([0.0, 1.0, 2.0], dtype='float32')
    x = from_numpy(x_val, requires_grad=True)

    y = x.tanh()
    loss = y.sum()
    loss.backward()

    grad = x.grad.data.numpy()

    # d/dx tanh(x) = 1 - tanh²(x)
    tanh_val = np.tanh(x_val)
    expected = 1.0 - tanh_val ** 2
    np.testing.assert_array_almost_equal(grad, expected, decimal=4)
