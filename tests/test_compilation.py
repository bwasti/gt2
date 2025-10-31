"""
Test compilation with batching and varying operation patterns.

This test exercises the compilation cache with different operation sequences
to expose issues with tensor dependency tracking.
"""

import numpy as np
import pytest
import gt


def test_compilation_simple_chain():
    """Test simple operation chain with compilation."""
    # Create inputs
    x = gt.from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]], dtype='float32'), requires_grad=True)
    w = gt.from_numpy(np.array([[0.5, 0.5], [0.5, 0.5]], dtype='float32'), requires_grad=True)

    # Forward pass with multiple operations
    h1 = x @ w
    h2 = h1.relu()
    h3 = h2 * gt.from_numpy(np.array(2.0, dtype='float32'))
    loss = h3.sum()

    # Backward pass
    loss.backward()

    # Check gradients exist
    assert x.grad is not None
    assert w.grad is not None

    # Check forward result
    loss_val = loss.data.numpy()
    assert loss_val.shape == ()
    assert loss_val > 0


def test_compilation_varying_patterns():
    """Test compilation with varying operation patterns."""
    results = []

    for i in range(5):
        x = gt.from_numpy(np.random.randn(4, 4).astype('float32'), requires_grad=True)
        w = gt.from_numpy(np.random.randn(4, 4).astype('float32'), requires_grad=True)

        # Vary the operation pattern each iteration
        if i % 3 == 0:
            # Pattern 1: matmul -> relu -> sum
            h = x @ w
            h = h.relu()
            loss = h.sum()
        elif i % 3 == 1:
            # Pattern 2: add -> mul -> mean
            h = x + w
            h = h * gt.from_numpy(np.array(0.5, dtype='float32'))
            loss = h.mean()
        else:
            # Pattern 3: matmul -> add -> sigmoid -> sum
            h = x @ w
            h = h + gt.from_numpy(np.array(1.0, dtype='float32'))
            h = h.sigmoid()
            loss = h.sum()

        loss.backward()

        # Store results
        results.append({
            'loss': loss.data.numpy().item(),
            'x_grad_sum': x.grad.data.numpy().sum(),
            'w_grad_sum': w.grad.data.numpy().sum()
        })

    # Verify all iterations completed
    assert len(results) == 5
    for r in results:
        assert not np.isnan(r['loss'])
        assert not np.isnan(r['x_grad_sum'])
        assert not np.isnan(r['w_grad_sum'])


def test_compilation_with_scalar_operations():
    """Test compilation with operations that produce scalar intermediate tensors."""
    x = gt.from_numpy(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype='float32'), requires_grad=True)

    # Operations that produce scalars
    s1 = x.sum()  # Scalar
    s2 = s1 * gt.from_numpy(np.array(2.0, dtype='float32'))  # Scalar * scalar
    s3 = s2 + gt.from_numpy(np.array(1.0, dtype='float32'))  # Scalar + scalar

    # Backward through scalar chain
    s3.backward()

    # Check gradient
    grad = x.grad.data.numpy()
    expected = np.full((2, 3), 2.0, dtype='float32')  # All 2.0 due to * 2.0
    assert grad.shape == (2, 3)
    assert np.allclose(grad, expected)


def test_compilation_with_branching():
    """Test compilation with branching computation graphs."""
    x = gt.from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]], dtype='float32'), requires_grad=True)

    # Create branching computation
    branch1 = x * gt.from_numpy(np.array(2.0, dtype='float32'))
    branch2 = x * gt.from_numpy(np.array(3.0, dtype='float32'))

    # Combine branches
    combined = branch1 + branch2
    loss = combined.sum()

    loss.backward()

    # Check gradient (should be 2.0 + 3.0 = 5.0 for each element)
    grad = x.grad.data.numpy()
    expected = np.full((2, 2), 5.0, dtype='float32')
    assert np.allclose(grad, expected)


def test_compilation_with_reshape_operations():
    """Test compilation with reshape operations in the graph."""
    x = gt.from_numpy(np.array([[1.0, 2.0, 3.0, 4.0]], dtype='float32'), requires_grad=True)

    # Reshape operations
    reshaped = x.reshape(2, 2)
    h = reshaped * gt.from_numpy(np.array(2.0, dtype='float32'))
    loss = h.sum()

    loss.backward()

    # Check gradient
    grad = x.grad.data.numpy()
    expected = np.full((1, 4), 2.0, dtype='float32')
    assert grad.shape == (1, 4)
    assert np.allclose(grad, expected)


def test_compilation_cache_with_same_structure():
    """Test that compilation cache works when structure is the same."""
    losses = []

    for i in range(3):
        # Same structure, different data
        x = gt.from_numpy(np.random.randn(3, 3).astype('float32'), requires_grad=True)
        w = gt.from_numpy(np.random.randn(3, 3).astype('float32'), requires_grad=True)

        # Same operation sequence
        h = x @ w
        h = h.relu()
        loss = h.sum()
        loss.backward()

        losses.append(loss.data.numpy().item())

    # All should complete successfully
    assert len(losses) == 3
    assert all(not np.isnan(l) for l in losses)


def test_compilation_with_axis_reduction():
    """Test compilation with axis-aware reductions."""
    x = gt.from_numpy(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype='float32'), requires_grad=True)

    # Axis reduction
    reduced = x.sum(axis=1, keepdims=False)  # Shape (2,)
    loss = reduced.sum()  # Scalar

    loss.backward()

    # Check gradient (should be all 1.0)
    grad = x.grad.data.numpy()
    expected = np.ones((2, 3), dtype='float32')
    assert np.allclose(grad, expected)
