"""
Test neural network modules and loss functions.
"""

import pytest
import numpy as np
import gt
from gt.client import nn


def test_linear_forward(client):
    """Test Linear layer forward pass"""
    # Create simple linear layer: 3 -> 2
    layer = nn.Linear(3, 2, bias=False)

    # Set known weights
    w = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype='float32')
    layer.weight = gt.tensor(w, requires_grad=True)
    layer._parameters = [layer.weight]

    # Input: [1, 2, 3]
    x = gt.tensor([1.0, 2.0, 3.0])

    # Forward: [1,2,3] @ [[1,2],[3,4],[5,6]] = [1*1+2*3+3*5, 1*2+2*4+3*6] = [22, 28]
    y = layer(x)
    result = y.data.numpy()

    expected = np.array([22.0, 28.0], dtype='float32')
    np.testing.assert_array_almost_equal(result, expected, decimal=5)


def test_linear_with_bias(client):
    """Test Linear layer with bias"""
    layer = nn.Linear(2, 3, bias=True)

    # Set known weights and bias
    w = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype='float32')
    b = np.array([0.1, 0.2, 0.3], dtype='float32')

    layer.weight = gt.tensor(w, requires_grad=True)
    layer.bias = gt.tensor(b, requires_grad=True)
    layer._parameters = [layer.weight, layer.bias]

    x = gt.tensor([1.0, 2.0])

    y = layer(x)
    result = y.data.numpy()

    # [1,2] @ [[1,2,3],[4,5,6]] + [0.1,0.2,0.3] = [1+8, 2+10, 3+12] + [0.1,0.2,0.3] = [9.1, 12.2, 15.3]
    expected = np.array([9.1, 12.2, 15.3], dtype='float32')
    np.testing.assert_array_almost_equal(result, expected, decimal=4)


def test_mse_loss(client):
    """Test MSE loss"""
    pred = gt.tensor([1.0, 2.0, 3.0])
    target = gt.tensor([1.5, 2.5, 3.5])

    loss = nn.mse_loss(pred, target)
    result = loss.data.numpy()

    # MSE = mean((pred - target)²) = mean([0.25, 0.25, 0.25]) = 0.25
    expected = 0.25
    np.testing.assert_almost_equal(result, expected, decimal=5)


def test_mse_loss_gradient(client):
    """Test MSE loss gradient"""
    pred = gt.tensor([1.0, 2.0, 3.0], requires_grad=True)
    target = gt.tensor([1.5, 2.5, 3.5])

    loss = nn.mse_loss(pred, target)
    loss.backward()

    grad = pred.grad.data.numpy()

    # d/dpred MSE = 2 * (pred - target) / n
    # For pred=[1,2,3], target=[1.5,2.5,3.5]: diff=[-0.5,-0.5,-0.5]
    # grad = 2 * [-0.5,-0.5,-0.5] / 3 = [-0.333, -0.333, -0.333]
    expected = 2.0 * (np.array([1.0, 2.0, 3.0]) - np.array([1.5, 2.5, 3.5])) / 3.0
    np.testing.assert_array_almost_equal(grad, expected, decimal=3)


def test_binary_cross_entropy(client):
    """Test binary cross entropy loss"""
    # Predicted probabilities (after sigmoid)
    pred = gt.tensor([0.7, 0.3, 0.9])
    target = gt.tensor([1.0, 0.0, 1.0])

    loss = nn.binary_cross_entropy(pred, target)
    result = loss.data.numpy()

    # BCE = -mean(target*log(pred) + (1-target)*log(1-pred))
    # For pred=[0.7,0.3,0.9], target=[1,0,1]:
    #   = -mean([log(0.7), log(0.7), log(0.9)])
    #   = -mean([-0.357, -0.357, -0.105]) ≈ 0.273
    expected = -(np.log(0.7) + np.log(0.7) + np.log(0.9)) / 3.0
    np.testing.assert_almost_equal(result, expected, decimal=3)


def test_module_parameters(client):
    """Test Module.parameters()"""
    layer = nn.Linear(2, 3, bias=True)

    params = layer.parameters()
    assert len(params) == 2  # weight and bias

    # Check that parameters have requires_grad
    assert all(p.requires_grad for p in params)


def test_zero_grad(client):
    """Test Module.zero_grad()"""
    layer = nn.Linear(2, 3, bias=False)

    x = gt.tensor([1.0, 2.0])
    y = layer(x)
    loss = y.sum()
    loss.backward()

    # Check gradient exists
    assert layer.weight.grad is not None

    # Zero grad
    layer.zero_grad()

    # Check gradient is None
    assert layer.weight.grad is None
