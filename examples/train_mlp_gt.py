"""
Train a simple MLP on synthetic data (GT version).

This is shot-for-shot compatible with the PyTorch baseline.
Simply replace `import torch` with `import gt as torch` and it works!
"""

import gt as torch
from gt.client import nn
import numpy as np


# Synthetic dataset: learn XOR-like pattern
def generate_data(n_samples=1000):
    """Generate synthetic classification data."""
    np.random.seed(42)
    X = np.random.randn(n_samples, 4).astype(np.float32)
    # Simple pattern: sum of features > 0
    y = (X.sum(axis=1) > 0).astype(np.float32).reshape(-1, 1)
    return X, y


class MLP(nn.Module):
    """Simple 2-layer MLP."""
    def __init__(self, input_dim=4, hidden_dim=8, output_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.relu(x)
        x = self.fc2(x)
        x = nn.sigmoid(x)
        return x


def train(model, X, y, epochs=100, lr=5.0):
    """Train the model."""
    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y)

    for epoch in range(epochs):
        # Forward pass
        pred = model(X_tensor)

        # MSE loss
        loss = ((pred - y_tensor) ** 2).mean()

        # Backward pass
        loss.backward()

        # Update weights (manual SGD)
        with torch.no_grad():
            for param in model.parameters():
                param -= lr * param.grad
                param.grad.zero_()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}: Loss = {loss.item():.6f}")

    return model


def main():
    print("=" * 60)
    print("Training MLP on Synthetic Data (GT)")
    print("=" * 60)

    # Generate data
    X, y = generate_data(n_samples=100)
    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Positive class: {y.mean():.2%}")

    # Create model
    model = MLP(input_dim=4, hidden_dim=8, output_dim=1)
    print(f"\nModel: {sum(p.shape[0] * p.shape[1] if len(p.shape) == 2 else p.shape[0] for p in model.parameters())} parameters")

    # Train
    print("\nTraining...")
    model = train(model, X, y, epochs=100, lr=0.1)

    # Final evaluation
    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y)
    with torch.no_grad():
        pred = model(X_tensor)
        final_loss = ((pred - y_tensor) ** 2).mean()
        accuracy = ((pred > torch.from_numpy(np.array(0.5, dtype='float32'))).data.numpy() == y).mean()

    print(f"\nFinal Loss: {final_loss.item():.6f}")
    print(f"Accuracy: {accuracy:.2%}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
