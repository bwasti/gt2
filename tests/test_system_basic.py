"""
Test basic system functionality with multiple iterations.

Verifies that the system can run multiple operations in sequence
and exit cleanly without hanging.
"""

import pytest


def test_multiple_iterations(client):
    """Test that system can run multiple iterations of operations."""
    import gt

    # Run multiple iterations
    num_iterations = 10

    for i in range(num_iterations):
        a = gt.randn(256, 256)
        b = gt.randn(256, 256)
        c = a @ b
        d = c.sum()
        result = d.data.numpy()

        # Verify result is a scalar
        assert result.shape == (), f"Expected scalar, got shape {result.shape}"

    # Test passed - system ran all iterations without hanging


def test_basic_matmul_chain(client):
    """Test chained matmul operations."""
    import gt

    # Create a chain of matmuls
    a = gt.randn(128, 64)
    b = gt.randn(64, 32)
    c = gt.randn(32, 16)

    # Chain them
    x = a @ b  # 128 x 32
    y = x @ c  # 128 x 16

    result = y.data.numpy()
    assert result.shape == (128, 16), f"Expected (128, 16), got {result.shape}"
