"""
Test new tensor operations added for PyTorch compatibility.

Tests for features requested in plm/MISSING.md:
- gt.ones()
- Tensor.mean(axis=..., keepdims=...)
- Tensor.max(axis=..., keepdims=...)
"""

import numpy as np
import pytest


def test_ones(client):
    """Test gt.ones() tensor creation."""
    import gt

    # Basic ones creation
    x = gt.ones(5, 10)
    data = x.data.numpy()

    assert data.shape == (5, 10)
    assert np.all(data == 1.0)

    # Test with different dtype
    y = gt.ones(3, 3, dtype='float32')
    y_data = y.data.numpy()
    assert y_data.dtype == np.float32
    assert np.all(y_data == 1.0)


def test_ones_with_grad(client):
    """Test gt.ones() with gradient tracking."""
    import gt

    # Test with requires_grad
    y = gt.ones(3, 3, requires_grad=True)
    z = (y * 2).sum()
    z.backward()

    grad_data = y.grad.data.numpy()
    assert grad_data.shape == (3, 3)
    assert np.all(grad_data == 2.0)


def test_mean_with_axis(client):
    """Test Tensor.mean() with axis parameter."""
    import gt

    x = gt.from_numpy(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))

    # Mean along axis 0
    m0 = x.mean(axis=0)
    expected_0 = np.array([2.5, 3.5, 4.5])
    assert np.allclose(m0.data.numpy(), expected_0)

    # Mean along axis 1
    m1 = x.mean(axis=1)
    expected_1 = np.array([2.0, 5.0])
    assert np.allclose(m1.data.numpy(), expected_1)

    # Global mean
    m_all = x.mean()
    assert np.allclose(m_all.data.numpy(), 3.5)


def test_mean_with_keepdims(client):
    """Test Tensor.mean() with keepdims parameter."""
    import gt

    x = gt.from_numpy(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))

    # Mean with keepdims=True
    m0_kd = x.mean(axis=0, keepdims=True)
    assert m0_kd.data.numpy().shape == (1, 3)

    m1_kd = x.mean(axis=1, keepdims=True)
    assert m1_kd.data.numpy().shape == (2, 1)


def test_max_with_axis(client):
    """Test Tensor.max() with axis parameter."""
    import gt

    x = gt.from_numpy(np.array([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]]))

    # Max along axis 0
    m0 = x.max(axis=0)
    expected_0 = np.array([4.0, 5.0, 6.0])
    assert np.allclose(m0.data.numpy(), expected_0)

    # Max along axis 1
    m1 = x.max(axis=1)
    expected_1 = np.array([5.0, 6.0])
    assert np.allclose(m1.data.numpy(), expected_1)

    # Global max
    m_all = x.max()
    assert np.allclose(m_all.data.numpy(), 6.0)


def test_max_with_keepdims(client):
    """Test Tensor.max() with keepdims parameter."""
    import gt

    x = gt.from_numpy(np.array([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]]))

    # Max with keepdims=True
    m0_kd = x.max(axis=0, keepdims=True)
    assert m0_kd.data.numpy().shape == (1, 3)

    m1_kd = x.max(axis=1, keepdims=True)
    assert m1_kd.data.numpy().shape == (2, 1)


def test_stable_softmax_pattern(client):
    """Test the stable softmax pattern: x - x.max(axis=-1, keepdims=True)"""
    import gt

    # This is a critical use case from MISSING.md
    logits = gt.from_numpy(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    max_logits = logits.max(axis=-1, keepdims=True)
    normalized = logits - max_logits

    result = normalized.data.numpy()

    # Check shape is preserved
    assert result.shape == (2, 3)

    # Check values are correct (should be shifted by max)
    expected = np.array([[-2.0, -1.0, 0.0], [-2.0, -1.0, 0.0]])
    assert np.allclose(result, expected)


def test_existing_features_still_work(client):
    """Verify that existing features (sqrt, reshape, etc.) still work."""
    import gt

    # sqrt()
    x = gt.from_numpy(np.array([4.0, 9.0, 16.0]))
    y = x.sqrt()
    assert np.allclose(y.data.numpy(), [2.0, 3.0, 4.0])

    # reshape()
    a = gt.randn(2, 3, 4)
    b = a.reshape(6, 4)
    assert b.data.numpy().shape == (6, 4)

    # unsqueeze()
    c = gt.randn(10, 20)
    d = c.unsqueeze(0)
    assert d.data.numpy().shape == (1, 10, 20)

    # squeeze()
    e = gt.randn(1, 10, 1, 20)
    f = e.squeeze()
    assert f.data.numpy().shape == (10, 20)
