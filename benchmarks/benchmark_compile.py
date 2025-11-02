"""
Quick benchmark to measure compilation overhead and speedup.
"""

import time
import numpy as np
import gt

def benchmark_mlp(batch_size, compile_enabled):
    """Run MLP training and measure time."""
    from gt.client import nn

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(4, 8)
            self.fc2 = nn.Linear(8, 1)

        def forward(self, x):
            x = self.fc1(x)
            x = nn.relu(x)
            x = self.fc2(x)
            x = nn.sigmoid(x)
            return x

    # Generate data
    np.random.seed(42)
    X = np.random.randn(100, 4).astype(np.float32)
    y = (X.sum(axis=1) > 0).astype(np.float32).reshape(-1, 1)

    X_tensor = gt.from_numpy(X)
    y_tensor = gt.from_numpy(y)

    # Create model
    model = MLP()

    # Training loop
    start = time.time()
    for epoch in range(100):
        pred = model(X_tensor)
        loss = ((pred - y_tensor) ** 2).mean()
        loss.backward()

        with gt.no_grad():
            for param in model.parameters():
                param -= 0.1 * param.grad
                param.grad.zero_()

    end = time.time()
    return end - start

if __name__ == "__main__":
    print("Benchmarking MLP Training (100 epochs)...")
    print("=" * 60)

    # Test different configurations
    configs = [
        ("Eager (batch=1, compile=0)", 1, 0),
        ("Batching (batch=10, compile=0)", 10, 0),
        ("Batching+Compile (batch=10, compile=1)", 10, 1),
    ]

    for name, batch_size, compile_flag in configs:
        import os
        os.environ['GT_WORKER_BATCH_SIZE'] = str(batch_size)
        os.environ['GT_COMPILE'] = str(compile_flag)

        # Restart server for clean config
        import gt
        if hasattr(gt, '_connection'):
            del gt._connection

        elapsed = benchmark_mlp(batch_size, compile_flag)
        print(f"{name:40s}: {elapsed:.3f}s")

    print("=" * 60)
