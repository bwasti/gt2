"""
Test in-place operations (functional ID swapping approach).

These tests verify that in-place operations like -= work correctly
by swapping tensor IDs on the client side while keeping worker operations functional.
"""

import pytest
import numpy as np
import gt


def test_isub_basic(client):
    """Test basic in-place subtraction."""
    a = gt.tensor([5.0, 10.0], dtype='float32')
    b = gt.tensor([2.0, 3.0], dtype='float32')

    # Store original ID to verify it changes
    original_id = a.id

    # In-place subtraction (returns self)
    a -= b

    # Note: In Python, `a -= b` is equivalent to `a = a.__isub__(b)`
    # So the same object is returned but we can't capture it in assignment

    # Check that ID changed (functional approach)
    assert a.id != original_id, "ID should change after in-place op"

    # Check result is correct
    expected = np.array([3.0, 7.0], dtype='float32')
    np.testing.assert_allclose(a.data.numpy(), expected)


def test_isub_scalar(client):
    """Test in-place subtraction with scalar."""
    a = gt.tensor([10.0, 20.0], dtype='float32')

    a -= 5.0

    expected = np.array([5.0, 15.0], dtype='float32')
    np.testing.assert_allclose(a.data.numpy(), expected)


def test_isub_updates_parameter(client):
    """Test that -= actually updates a parameter (key for gradient descent)."""
    # This simulates param -= lr * grad
    param = gt.tensor([1.0, 2.0, 3.0], dtype='float32')
    grad = gt.tensor([0.1, 0.2, 0.3], dtype='float32')
    lr = 0.1

    print(f"Initial param: {param.data.numpy()}")
    print(f"Grad: {grad.data.numpy()}")

    # Update: param -= lr * grad
    param -= lr * grad

    print(f"After update: {param.data.numpy()}")

    expected = np.array([0.99, 1.98, 2.97], dtype='float32')
    np.testing.assert_allclose(param.data.numpy(), expected, rtol=1e-5)


def test_isub_sequence(client):
    """Test multiple in-place operations in sequence."""
    a = gt.tensor([10.0], dtype='float32')

    a -= 2.0
    assert np.isclose(a.data.numpy()[0], 8.0)

    a -= 3.0
    assert np.isclose(a.data.numpy()[0], 5.0)

    a -= 1.0
    assert np.isclose(a.data.numpy()[0], 4.0)


def test_isub_independence(client):
    """Test that in-place op on one tensor doesn't affect another."""
    a = gt.tensor([5.0, 10.0], dtype='float32')
    b = gt.tensor([2.0, 3.0], dtype='float32')

    # Store b's data before a's update
    b_before = b.data.numpy().copy()

    # Update a in-place
    a -= b

    # Check that b is unchanged
    b_after = b.data.numpy()
    np.testing.assert_array_equal(b_after, b_before)


def test_isub_with_computation(client):
    """Test -= with a more complex right-hand side."""
    a = gt.tensor([10.0, 20.0], dtype='float32')
    b = gt.tensor([2.0, 3.0], dtype='float32')
    c = gt.tensor([1.0, 1.0], dtype='float32')

    # a -= (b + c)
    a -= (b + c)

    expected = np.array([7.0, 16.0], dtype='float32')
    np.testing.assert_allclose(a.data.numpy(), expected)


def test_gradient_descent_step(client):
    """Test a full gradient descent step with -= (most realistic scenario)."""
    # This is the actual use case: updating parameters during training

    # Create a simple "parameter"
    w = gt.tensor([[1.0, 2.0]], dtype='float32', requires_grad=True)
    x = gt.tensor([[1.0], [1.0]], dtype='float32')
    target = gt.tensor([[5.0]], dtype='float32')

    print(f"Initial weight: {w.data.numpy()}")

    # Forward pass
    pred = w @ x  # Should be [[3.0]]
    print(f"Prediction: {pred.data.numpy()}")

    # Loss
    loss = ((pred - target) ** 2).mean()  # Should be (3-5)^2 = 4
    print(f"Loss: {loss.item()}")

    # Backward
    loss.backward()
    grad = w._grad.data.numpy()
    print(f"Gradient: {grad}")

    # Update with gradient descent: w -= lr * grad
    lr = 0.1
    with gt.no_grad():
        old_weight = w.data.numpy().copy()
        w -= lr * w._grad
        new_weight = w.data.numpy()

        print(f"Old weight: {old_weight}")
        print(f"New weight: {new_weight}")

        # Verify weight changed
        assert not np.allclose(old_weight, new_weight), "Weight should change after update"

        # Verify direction of change (should move toward reducing loss)
        # Gradient points in direction of increasing loss, so -= should decrease it
        expected = old_weight - lr * grad
        np.testing.assert_allclose(new_weight, expected, rtol=1e-5)


def test_multiple_parameter_updates(client):
    """Test updating multiple parameters independently."""
    param1 = gt.tensor([1.0, 2.0], dtype='float32')
    param2 = gt.tensor([3.0, 4.0], dtype='float32')

    grad1 = gt.tensor([0.1, 0.2], dtype='float32')
    grad2 = gt.tensor([0.3, 0.4], dtype='float32')

    lr = 0.1

    param1 -= lr * grad1
    param2 -= lr * grad2

    expected1 = np.array([0.99, 1.98], dtype='float32')
    expected2 = np.array([2.97, 3.96], dtype='float32')

    np.testing.assert_allclose(param1.data.numpy(), expected1, rtol=1e-5)
    np.testing.assert_allclose(param2.data.numpy(), expected2, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
