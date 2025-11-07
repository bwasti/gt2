"""
Tests for tensor manipulation operations: cat, split, chunk, transpose, permute.
"""
import pytest
import numpy as np
import gt


def test_cat_axis0(client):
    """Test concatenating tensors along axis 0."""
    a = gt.from_numpy(np.array([[1, 2], [3, 4]], dtype='float32'))
    b = gt.from_numpy(np.array([[5, 6]], dtype='float32'))

    c = gt.cat([a, b], axis=0)
    result = c.data.numpy()

    expected = np.array([[1, 2], [3, 4], [5, 6]], dtype='float32')
    assert result.shape == (3, 2)
    assert np.allclose(result, expected)


def test_cat_axis1(client):
    """Test concatenating tensors along axis 1."""
    a = gt.from_numpy(np.array([[1, 2], [3, 4]], dtype='float32'))
    b = gt.from_numpy(np.array([[5], [6]], dtype='float32'))

    c = gt.cat([a, b], axis=1)
    result = c.data.numpy()

    expected = np.array([[1, 2, 5], [3, 4, 6]], dtype='float32')
    assert result.shape == (2, 3)
    assert np.allclose(result, expected)


def test_cat_multiple_tensors(client):
    """Test concatenating more than 2 tensors."""
    a = gt.from_numpy(np.array([[1]], dtype='float32'))
    b = gt.from_numpy(np.array([[2]], dtype='float32'))
    c = gt.from_numpy(np.array([[3]], dtype='float32'))

    result_tensor = gt.cat([a, b, c], axis=0)
    result = result_tensor.data.numpy()

    expected = np.array([[1], [2], [3]], dtype='float32')
    assert result.shape == (3, 1)
    assert np.allclose(result, expected)


def test_cat_with_gradient(client):
    """Test that cat supports gradients properly."""
    a = gt.from_numpy(np.array([[1, 2]], dtype='float32'), requires_grad=True)
    b = gt.from_numpy(np.array([[3, 4]], dtype='float32'), requires_grad=True)

    c = gt.cat([a, b], axis=0)  # Shape: (2, 2)
    loss = c.sum()
    loss.backward()

    # Gradient should split back: each input gets gradient of 1s with original shape
    assert a.grad is not None
    assert b.grad is not None
    assert np.allclose(a.grad.data.numpy(), np.ones((1, 2), dtype='float32'))
    assert np.allclose(b.grad.data.numpy(), np.ones((1, 2), dtype='float32'))


def test_split_basic(client):
    """Test splitting a tensor into chunks."""
    x = gt.from_numpy(np.arange(12).reshape(6, 2).astype('float32'))

    chunks = x.split(split_size=2, dim=0)  # Split into chunks of size 2 along axis 0

    assert len(chunks) == 3

    # Check each chunk
    assert chunks[0].data.numpy().shape == (2, 2)
    assert chunks[1].data.numpy().shape == (2, 2)
    assert chunks[2].data.numpy().shape == (2, 2)

    assert np.allclose(chunks[0].data.numpy(), np.array([[0, 1], [2, 3]], dtype='float32'))
    assert np.allclose(chunks[1].data.numpy(), np.array([[4, 5], [6, 7]], dtype='float32'))
    assert np.allclose(chunks[2].data.numpy(), np.array([[8, 9], [10, 11]], dtype='float32'))


def test_split_uneven(client):
    """Test splitting when size doesn't divide evenly."""
    x = gt.from_numpy(np.arange(10).reshape(5, 2).astype('float32'))

    chunks = x.split(split_size=2, dim=0)  # 5 doesn't divide evenly by 2

    assert len(chunks) == 3  # Ceiling division: (5 + 2 - 1) // 2 = 3
    assert chunks[0].data.numpy().shape == (2, 2)
    assert chunks[1].data.numpy().shape == (2, 2)
    assert chunks[2].data.numpy().shape == (1, 2)  # Last chunk is smaller


def test_chunk_basic(client):
    """Test chunking a tensor into N pieces."""
    x = gt.from_numpy(np.arange(12).reshape(6, 2).astype('float32'))

    chunks = x.chunk(chunks=3, dim=0)  # Split into 3 chunks along axis 0

    assert len(chunks) == 3
    assert all(c.data.numpy().shape == (2, 2) for c in chunks)


def test_chunk_uneven(client):
    """Test chunking when size doesn't divide evenly."""
    x = gt.from_numpy(np.arange(10).reshape(5, 2).astype('float32'))

    chunks = x.chunk(chunks=2, dim=0)  # 5 doesn't divide evenly by 2

    assert len(chunks) == 2
    assert chunks[0].data.numpy().shape == (3, 2)  # Ceiling: (5 + 2 - 1) // 2 = 3
    assert chunks[1].data.numpy().shape == (2, 2)


def test_transpose_specific_dims(client):
    """Test transpose with specific dimensions."""
    x = gt.from_numpy(np.arange(24).reshape(2, 3, 4).astype('float32'))

    # Transpose dims 0 and 1
    y = x.transpose(0, 1)
    assert y.data.numpy().shape == (3, 2, 4)

    # Transpose dims 1 and 2
    z = x.transpose(1, 2)
    assert z.data.numpy().shape == (2, 4, 3)


def test_transpose_default(client):
    """Test .T property (transpose last two dimensions)."""
    x = gt.from_numpy(np.arange(24).reshape(2, 3, 4).astype('float32'))

    y = x.T
    assert y.data.numpy().shape == (2, 4, 3)  # Last two dims swapped

    # Verify data is correctly transposed
    x_np = x.data.numpy()
    y_np = y.data.numpy()
    assert np.allclose(y_np, np.transpose(x_np, (0, 2, 1)))


def test_permute_basic(client):
    """Test permute with various dimension orderings."""
    x = gt.from_numpy(np.arange(24).reshape(2, 3, 4).astype('float32'))

    # Permute to (4, 2, 3) - dims (2, 0, 1)
    y = x.permute(2, 0, 1)
    assert y.data.numpy().shape == (4, 2, 3)

    # Verify data is correctly permuted
    x_np = x.data.numpy()
    y_np = y.data.numpy()
    assert np.allclose(y_np, np.transpose(x_np, (2, 0, 1)))


def test_permute_with_tuple(client):
    """Test permute accepts tuple of dimensions."""
    x = gt.from_numpy(np.arange(24).reshape(2, 3, 4).astype('float32'))

    # Pass dimensions as tuple
    y = x.permute((1, 2, 0))
    assert y.data.numpy().shape == (3, 4, 2)

    x_np = x.data.numpy()
    y_np = y.data.numpy()
    assert np.allclose(y_np, np.transpose(x_np, (1, 2, 0)))


def test_permute_reverse_dims(client):
    """Test permute to reverse all dimensions."""
    x = gt.from_numpy(np.arange(60).reshape(3, 4, 5).astype('float32'))

    # Reverse all dimensions: (3, 4, 5) -> (5, 4, 3)
    y = x.permute(2, 1, 0)
    assert y.data.numpy().shape == (5, 4, 3)

    x_np = x.data.numpy()
    y_np = y.data.numpy()
    assert np.allclose(y_np, np.transpose(x_np, (2, 1, 0)))


def test_transpose_gradient(client):
    """Test transpose propagates gradients correctly."""
    x = gt.from_numpy(np.array([[1, 2, 3], [4, 5, 6]], dtype='float32'), requires_grad=True)

    y = x.transpose(0, 1)  # (2, 3) -> (3, 2)
    loss = y.sum()
    loss.backward()

    # Gradient should be transposed back to match x
    assert x.grad is not None
    assert x.grad.data.numpy().shape == (2, 3)
    assert np.allclose(x.grad.data.numpy(), np.ones((2, 3), dtype='float32'))


def test_permute_gradient(client):
    """Test permute propagates gradients correctly."""
    x = gt.from_numpy(np.arange(24).reshape(2, 3, 4).astype('float32'), requires_grad=True)

    y = x.permute(2, 0, 1)  # (2, 3, 4) -> (4, 2, 3)
    loss = y.sum()
    loss.backward()

    # Gradient should be permuted back to match x
    assert x.grad is not None
    assert x.grad.data.numpy().shape == (2, 3, 4)
    assert np.allclose(x.grad.data.numpy(), np.ones((2, 3, 4), dtype='float32'))


def test_split_chunk_composition(client):
    """Test that split/chunk results can be recombined with cat."""
    x = gt.from_numpy(np.arange(12).reshape(6, 2).astype('float32'))

    # Split into chunks
    chunks = x.split(split_size=2, dim=0)

    # Recombine with cat
    y = gt.cat(chunks, axis=0)

    # Should recover original tensor
    assert np.allclose(y.data.numpy(), x.data.numpy())


def test_complex_manipulation_chain(client):
    """Test chaining multiple tensor manipulation operations."""
    # Create a tensor and perform multiple operations
    x = gt.from_numpy(np.arange(24).reshape(2, 3, 4).astype('float32'))

    # Permute dims
    y = x.permute(1, 0, 2)  # (3, 2, 4)

    # Split along axis 0
    chunks = y.split(split_size=1, dim=0)
    assert len(chunks) == 3

    # Concatenate back
    z = gt.cat(chunks, axis=0)

    # Should match y
    assert np.allclose(z.data.numpy(), y.data.numpy())
