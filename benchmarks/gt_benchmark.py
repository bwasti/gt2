"""
GT framework benchmarks.

Measures overhead of GT distributed system compared to PyTorch baseline.
"""

import sys
import os

# For benchmarks, we still need this since they're not part of the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import numpy as np
import gt


def benchmark_matmul(size=1000, iterations=100):
    """Benchmark matrix multiplication."""
    a_data = np.random.randn(size, size).astype('float32')
    b_data = np.random.randn(size, size).astype('float32')

    a = gt.tensor(a_data)
    b = gt.tensor(b_data)

    # Warmup
    for _ in range(10):
        c = a @ b
        _ = c.data.numpy()  # Force execution

    # Benchmark
    start = time.time()
    for _ in range(iterations):
        c = a @ b
        _ = c.data.numpy()  # Force execution
    elapsed = time.time() - start

    return elapsed / iterations


def benchmark_linear_forward(batch_size=128, in_features=512, out_features=256, iterations=100):
    """Benchmark linear layer forward pass."""
    from gt.client.nn import Linear

    layer = Linear(in_features, out_features)
    x = gt.tensor(np.random.randn(batch_size, in_features).astype('float32'))

    # Warmup
    for _ in range(10):
        y = layer(x)
        _ = y.data.numpy()

    # Benchmark
    start = time.time()
    for _ in range(iterations):
        y = layer(x)
        _ = y.data.numpy()
    elapsed = time.time() - start

    return elapsed / iterations


def benchmark_mse_loss(size=10000, iterations=100):
    """Benchmark MSE loss computation."""
    from gt.client.nn import mse_loss

    pred = gt.tensor(np.random.randn(size).astype('float32'))
    target = gt.tensor(np.random.randn(size).astype('float32'))

    # Warmup
    for _ in range(10):
        loss = mse_loss(pred, target)
        _ = loss.data.numpy()

    # Benchmark
    start = time.time()
    for _ in range(iterations):
        loss = mse_loss(pred, target)
        _ = loss.data.numpy()
    elapsed = time.time() - start

    return elapsed / iterations


def benchmark_backward(size=1000, iterations=100):
    """Benchmark backward pass."""
    a_data = np.random.randn(size, size).astype('float32')
    b_data = np.random.randn(size, size).astype('float32')

    # Warmup
    for _ in range(10):
        a = gt.tensor(a_data, requires_grad=True)
        b = gt.tensor(b_data, requires_grad=True)
        c = a @ b
        loss = c.sum()
        loss.backward()
        _ = a.grad.data.numpy()

    # Benchmark
    start = time.time()
    for _ in range(iterations):
        a = gt.tensor(a_data, requires_grad=True)
        b = gt.tensor(b_data, requires_grad=True)
        c = a @ b
        loss = c.sum()
        loss.backward()
        _ = a.grad.data.numpy()
    elapsed = time.time() - start

    return elapsed / iterations


def benchmark_activations(size=10000, iterations=100):
    """Benchmark activation functions."""
    x = gt.tensor(np.random.randn(size).astype('float32'))

    results = {}

    # ReLU
    start = time.time()
    for _ in range(iterations):
        y = x.relu()
        _ = y.data.numpy()
    results['relu'] = (time.time() - start) / iterations

    # Sigmoid
    start = time.time()
    for _ in range(iterations):
        y = x.sigmoid()
        _ = y.data.numpy()
    results['sigmoid'] = (time.time() - start) / iterations

    # Tanh
    start = time.time()
    for _ in range(iterations):
        y = x.tanh()
        _ = y.data.numpy()
    results['tanh'] = (time.time() - start) / iterations

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("GT Framework Benchmarks")
    print("=" * 60)

    print(f"\nMatmul (1000x1000): {benchmark_matmul() * 1000:.3f} ms")
    print(f"Linear forward (128x512->256): {benchmark_linear_forward() * 1000:.3f} ms")
    print(f"MSE Loss (10000 elements): {benchmark_mse_loss() * 1000:.3f} ms")
    print(f"Backward pass (1000x1000): {benchmark_backward() * 1000:.3f} ms")

    print("\nActivations (10000 elements):")
    act_results = benchmark_activations()
    for name, time_ms in act_results.items():
        print(f"  {name}: {time_ms * 1000:.3f} ms")
