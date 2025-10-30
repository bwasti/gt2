"""
PyTorch baseline benchmarks.

Pure PyTorch operations to establish performance baseline.
"""

import torch
import time
import numpy as np


def benchmark_matmul(size=1000, iterations=100):
    """Benchmark matrix multiplication."""
    a = torch.randn(size, size)
    b = torch.randn(size, size)

    # Warmup
    for _ in range(10):
        c = torch.matmul(a, b)

    # Benchmark
    start = time.time()
    for _ in range(iterations):
        c = torch.matmul(a, b)
    elapsed = time.time() - start

    return elapsed / iterations


def benchmark_linear_forward(batch_size=128, in_features=512, out_features=256, iterations=100):
    """Benchmark linear layer forward pass."""
    layer = torch.nn.Linear(in_features, out_features)
    x = torch.randn(batch_size, in_features)

    # Warmup
    for _ in range(10):
        y = layer(x)

    # Benchmark
    start = time.time()
    for _ in range(iterations):
        y = layer(x)
    elapsed = time.time() - start

    return elapsed / iterations


def benchmark_mse_loss(size=10000, iterations=100):
    """Benchmark MSE loss computation."""
    pred = torch.randn(size)
    target = torch.randn(size)
    loss_fn = torch.nn.MSELoss()

    # Warmup
    for _ in range(10):
        loss = loss_fn(pred, target)

    # Benchmark
    start = time.time()
    for _ in range(iterations):
        loss = loss_fn(pred, target)
    elapsed = time.time() - start

    return elapsed / iterations


def benchmark_backward(size=1000, iterations=100):
    """Benchmark backward pass."""
    a = torch.randn(size, size, requires_grad=True)
    b = torch.randn(size, size, requires_grad=True)

    # Warmup
    for _ in range(10):
        c = torch.matmul(a, b)
        loss = c.sum()
        loss.backward()
        a.grad = None
        b.grad = None

    # Benchmark
    start = time.time()
    for _ in range(iterations):
        c = torch.matmul(a, b)
        loss = c.sum()
        loss.backward()
        a.grad = None
        b.grad = None
    elapsed = time.time() - start

    return elapsed / iterations


def benchmark_activations(size=10000, iterations=100):
    """Benchmark activation functions."""
    x = torch.randn(size)

    results = {}

    # ReLU
    start = time.time()
    for _ in range(iterations):
        y = torch.relu(x)
    results['relu'] = (time.time() - start) / iterations

    # Sigmoid
    start = time.time()
    for _ in range(iterations):
        y = torch.sigmoid(x)
    results['sigmoid'] = (time.time() - start) / iterations

    # Tanh
    start = time.time()
    for _ in range(iterations):
        y = torch.tanh(x)
    results['tanh'] = (time.time() - start) / iterations

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("PyTorch Baseline Benchmarks")
    print("=" * 60)

    print(f"\nMatmul (1000x1000): {benchmark_matmul() * 1000:.3f} ms")
    print(f"Linear forward (128x512->256): {benchmark_linear_forward() * 1000:.3f} ms")
    print(f"MSE Loss (10000 elements): {benchmark_mse_loss() * 1000:.3f} ms")
    print(f"Backward pass (1000x1000): {benchmark_backward() * 1000:.3f} ms")

    print("\nActivations (10000 elements):")
    act_results = benchmark_activations()
    for name, time_ms in act_results.items():
        print(f"  {name}: {time_ms * 1000:.3f} ms")
