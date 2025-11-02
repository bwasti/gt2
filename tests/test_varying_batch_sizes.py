"""
Test varying batch sizes in autograd operations.

Regression test for batch size bugs in matmul + transpose patterns
(e.g., attention mechanisms in transformers).
"""

import pytest
import numpy as np
from gt.client.tensor import from_numpy


def test_matmul_backward_varying_batch_sizes(client):
    """Test matmul+backward with different batch sizes in sequence."""
    # Create weight matrix (2D) - like attention projection weights
    weight_data = np.ones((64, 64), dtype='float32')
    weight = from_numpy(weight_data)
    weight = weight.requires_grad_(True)

    batch_sizes = [8, 4, 8, 2, 16, 4]

    for i, batch_size in enumerate(batch_sizes):
        # Input: (batch, seq, hidden) - mimics attention input
        x_data = np.random.randn(batch_size, 128, 64).astype('float32')
        x = from_numpy(x_data)

        # Forward: matmul
        # (batch, 128, 64) @ (64, 64) -> (batch, 128, 64)
        y = x @ weight

        # Verify shape
        assert y.shape == (batch_size, 128, 64), \
            f"Batch {i} (size {batch_size}): Wrong output shape {y.shape}"

        # Loss
        loss = y.sum()

        # Backward - this calls grad_left = grad_output @ right.T
        loss.backward()

        # Verify gradient shape
        assert weight.grad is not None, \
            f"Batch {i} (size {batch_size}): No gradient computed"
        grad_data = weight.grad.data.numpy()
        assert grad_data.shape == (64, 64), \
            f"Batch {i} (size {batch_size}): Wrong gradient shape {grad_data.shape}"

        # Optimizer step
        weight -= 0.001 * weight.grad
        weight.grad.zero_()


def test_3d_matmul_varying_batch_sizes(client):
    """Test 3D @ 2D matmul with varying batch sizes."""
    # Weight matrix (2D)
    W = from_numpy(np.random.randn(32, 16).astype('float32'))

    batch_sizes = [1, 4, 8, 2, 16]

    for batch_size in batch_sizes:
        # Input: (batch, seq, hidden)
        x_data = np.random.randn(batch_size, 10, 32).astype('float32')
        x = from_numpy(x_data)

        # Matmul: (batch, 10, 32) @ (32, 16) -> (batch, 10, 16)
        y = x @ W

        # Verify shape
        assert y.shape == (batch_size, 10, 16), \
            f"Batch size {batch_size}: Wrong output shape {y.shape}"

        # Verify values are correct
        y_data = y.data.numpy()
        W_data = W.data.numpy()
        expected = x_data @ W_data

        np.testing.assert_allclose(y_data, expected, rtol=1e-5,
            err_msg=f"Batch size {batch_size}: Result doesn't match numpy")


def test_matmul_transpose_varying_batch_sizes(client):
    """Test matmul with transpose operations across different batch sizes."""
    # This pattern is common in attention mechanisms
    batch_sizes = [1, 2, 4, 8]

    for batch_size in batch_sizes:
        # Query, Key, Value matrices
        Q_data = np.random.randn(batch_size, 8, 64).astype('float32')
        K_data = np.random.randn(batch_size, 8, 64).astype('float32')

        Q = from_numpy(Q_data)
        K = from_numpy(K_data)

        # Attention scores: Q @ K.T
        # Note: We need to transpose the last two dims of K
        # For now, test with 2D case
        if batch_size == 1:
            Q_2d = Q[0]  # (8, 64)
            K_2d = K[0]  # (8, 64)

            # Q @ K.T: (8, 64) @ (64, 8) -> (8, 8)
            scores = Q_2d @ K_2d.T

            assert scores.shape == (8, 8), \
                f"Batch size {batch_size}: Wrong scores shape {scores.shape}"

            # Verify correctness
            scores_data = scores.data.numpy()
            expected = Q_data[0] @ K_data[0].T

            np.testing.assert_allclose(scores_data, expected, rtol=1e-5,
                err_msg=f"Batch size {batch_size}: Scores don't match numpy")


def test_gradient_accumulation_varying_batches(client):
    """Test gradient accumulation with varying batch sizes."""
    # Simulate gradient accumulation across different batch sizes
    W = from_numpy(np.ones((10, 10), dtype='float32'))
    W = W.requires_grad_(True)

    batch_sizes = [4, 8, 4, 2]
    accumulated_grad = None

    for batch_size in batch_sizes:
        x_data = np.random.randn(batch_size, 10).astype('float32')
        x = from_numpy(x_data)

        # Forward
        y = x @ W
        loss = y.sum()

        # Backward
        loss.backward()

        # Store gradient
        if accumulated_grad is None:
            accumulated_grad = W.grad.data.numpy().copy()
        else:
            accumulated_grad += W.grad.data.numpy()

        # Don't zero gradient yet - accumulate across batches

    # Verify we accumulated something
    assert accumulated_grad is not None
    assert not np.allclose(accumulated_grad, 0), "Accumulated gradient is zero"
