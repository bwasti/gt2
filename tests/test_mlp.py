"""
Test MLP training with both GT and PyTorch backends.

Usage:
    # Test with GT (default)
    pytest tests/test_mlp.py

    # Test with PyTorch
    pytest tests/test_mlp.py --backend=pytorch

    # Run both
    pytest tests/test_mlp.py --backend=both

Note: The --backend option is configured in conftest.py
"""

import pytest
import numpy as np


# Synthetic dataset
def generate_data(n_samples=100):
    """Generate synthetic classification data."""
    np.random.seed(42)
    X = np.random.randn(n_samples, 4).astype(np.float32)
    y = (X.sum(axis=1) > 0).astype(np.float32).reshape(-1, 1)
    return X, y


def test_mlp_training(backend, client):
    """Test MLP training with specified backend."""

    # Import appropriate backend
    if backend == "gt":
        import gt as torch
        from gt.client import nn
    else:  # pytorch
        import torch
        import torch.nn as nn

    print(f"\n{'='*60}")
    print(f"Testing MLP Training ({backend.upper()})")
    print('='*60)

    # Generate data
    X, y = generate_data(n_samples=100)
    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Positive class: {y.mean():.2%}")

    # Create model
    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(4, 8)
            self.fc2 = nn.Linear(8, 1)

        def forward(self, x):
            if backend == "gt":
                x = self.fc1(x)
                x = nn.relu(x)
                x = self.fc2(x)
                x = nn.sigmoid(x)
            else:  # pytorch
                x = self.fc1(x)
                x = torch.relu(x)
                x = self.fc2(x)
                x = torch.sigmoid(x)
            return x

    model = MLP()

    # Count parameters
    if backend == "gt":
        n_params = sum(p.shape[0] * p.shape[1] if len(p.shape) == 2 else p.shape[0]
                      for p in model.parameters())
    else:
        n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params} parameters")

    # Prepare data
    if backend == "gt":
        X_tensor = torch.from_numpy(X)
        y_tensor = torch.from_numpy(y)
    else:
        X_tensor = torch.from_numpy(X)
        y_tensor = torch.from_numpy(y)

    # Training
    lr = 5.0  # Higher learning rate for faster convergence in tests
    epochs = 20  # Fewer epochs for faster testing

    print(f"\nTraining for {epochs} epochs with lr={lr}...")
    losses = []

    for epoch in range(epochs):
        # Forward
        pred = model(X_tensor)
        loss = ((pred - y_tensor) ** 2).mean()

        # Backward
        loss.backward()

        # Update
        with torch.no_grad():
            for param in model.parameters():
                if backend == "gt":
                    param -= lr * param._grad
                    param._grad.zero_()
                else:
                    param -= lr * param.grad
                    param.grad.zero_()

        # Record loss
        if backend == "gt":
            losses.append(loss.item())
        else:
            losses.append(loss.item())

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:2d}: Loss = {loss.item():.6f}")

    # Final evaluation
    with torch.no_grad():
        pred = model(X_tensor)
        final_loss = ((pred - y_tensor) ** 2).mean()

        if backend == "gt":
            threshold = torch.from_numpy(np.array(0.5, dtype='float32'))
            predictions = (pred > threshold).data.numpy()
            accuracy = (predictions == y).mean()
            final_loss_val = final_loss.item()
        else:
            predictions = (pred > 0.5).float()
            accuracy = (predictions == y_tensor).float().mean().item()
            final_loss_val = final_loss.item()

    print(f"\nFinal Loss: {final_loss_val:.6f}")
    print(f"Accuracy: {accuracy:.2%}")
    print('='*60)

    # Assertions
    assert n_params == 49, f"Expected 49 parameters, got {n_params}"

    # Loss should decrease significantly
    initial_loss = losses[0]
    final_loss_test = losses[-1]
    assert final_loss_test < initial_loss * 0.5, \
        f"Loss did not decrease enough: {initial_loss:.4f} -> {final_loss_test:.4f}"

    # Accuracy should be reasonable (at least random chance)
    assert accuracy >= 0.5, f"Accuracy too low: {accuracy:.2%}"

    print(f"\n✓ {backend.upper()} MLP training test passed!")


if __name__ == "__main__":
    # Run with GT backend when executed directly
    test_mlp_training("gt")
