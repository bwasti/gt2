"""
Quick protocol sanity tests.

Run this before/after protocol changes to ensure basic correctness.
Should complete in < 5 seconds.

Usage:
    pytest tests/test_protocol_quick.py -v
"""

import pytest
import numpy as np
import gt


def test_create_and_fetch(client):
    """Test CreateTensor and GetData protocol."""
    a = gt.from_numpy(np.array([[1, 2], [3, 4]], dtype='float32'))
    data = a.data.numpy()
    np.testing.assert_array_equal(data, [[1, 2], [3, 4]])


def test_binary_op(client):
    """Test BinaryOp protocol (add, mul, matmul)."""
    a = gt.from_numpy(np.array([[1, 2], [3, 4]], dtype='float32'))
    b = gt.from_numpy(np.array([[5, 6], [7, 8]], dtype='float32'))

    # Add
    c = a + b
    np.testing.assert_array_equal(c.data.numpy(), [[6, 8], [10, 12]])

    # Mul
    d = a * b
    np.testing.assert_array_equal(d.data.numpy(), [[5, 12], [21, 32]])

    # Matmul
    e = a @ b
    np.testing.assert_array_equal(e.data.numpy(), [[19, 22], [43, 50]])


def test_unary_op(client):
    """Test UnaryOp protocol (sum, mean, exp, relu)."""
    a = gt.from_numpy(np.array([[1, 2], [3, 4]], dtype='float32'))

    # Sum
    s = a.sum()
    assert s.data.numpy() == 10.0

    # Mean
    m = a.mean()
    assert m.data.numpy() == 2.5

    # ReLU
    b = gt.from_numpy(np.array([[-1, 2], [-3, 4]], dtype='float32'))
    r = b.relu()
    np.testing.assert_array_equal(r.data.numpy(), [[0, 2], [0, 4]])


def test_free_tensor(client):
    """Test FreeTensor protocol (garbage collection)."""
    a = gt.randn(100, 100)
    tensor_id = a.id

    # Tensor should exist
    data = a.data.numpy()
    assert data.shape == (100, 100)

    # Delete reference - should trigger FreeTensor
    del a

    # Create new tensor to verify system still works
    b = gt.randn(10, 10)
    assert b.data.numpy().shape == (10, 10)


def test_chained_operations(client):
    """Test multiple operations in sequence."""
    a = gt.from_numpy(np.array([[1, 2], [3, 4]], dtype='float32'))
    b = gt.from_numpy(np.array([[5, 6], [7, 8]], dtype='float32'))

    # Chain: (a @ b) + a
    result = (a @ b) + a

    expected = np.array([[20, 24], [46, 54]], dtype='float32')
    np.testing.assert_array_equal(result.data.numpy(), expected)


def test_backward_protocol(client):
    """Test backward pass (generates many operations)."""
    a = gt.randn(4, 4, requires_grad=True)
    b = gt.randn(4, 4, requires_grad=True)

    # Forward
    c = a @ b
    loss = c.sum()

    # Backward
    loss.backward()

    # Check gradients exist
    assert a.grad is not None
    assert b.grad is not None
    assert a.grad.data.numpy().shape == (4, 4)
    assert b.grad.data.numpy().shape == (4, 4)


def test_randn_protocol(client):
    """Test UnaryOp protocol for tensor creation."""
    a = gt.randn(10, 20)

    assert a.shape == (10, 20)
    data = a.data.numpy()
    assert data.shape == (10, 20)
    assert data.dtype == np.float32


def test_zeros_protocol(client):
    """Test zeros creation."""
    a = gt.zeros(5, 5)

    data = a.data.numpy()
    np.testing.assert_array_equal(data, np.zeros((5, 5)))


def test_reshape_protocol(client):
    """Test ReshapeOp protocol."""
    a = gt.from_numpy(np.array([[1, 2, 3, 4]], dtype='float32'))
    b = a.reshape(2, 2)

    expected = np.array([[1, 2], [3, 4]], dtype='float32')
    np.testing.assert_array_equal(b.data.numpy(), expected)


def test_protocol_stress(client):
    """Stress test: many operations in quick succession."""
    tensors = []

    # Create many tensors
    for i in range(20):
        t = gt.randn(10, 10)
        tensors.append(t)

    # Do operations on them
    result = tensors[0]
    for i in range(1, 10):
        result = result + tensors[i]

    # Verify result
    data = result.data.numpy()
    assert data.shape == (10, 10)

    # Cleanup - test FreeTensor protocol
    del tensors
    del result


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
