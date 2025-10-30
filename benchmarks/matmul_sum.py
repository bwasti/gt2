"""
Benchmark matmul + sum to minimize data transfer overhead.

Instead of transferring the full result matrix, we only transfer a scalar.
This isolates the computation overhead from data transfer overhead.
"""

import time
import numpy as np
import torch


def benchmark_pytorch(size=1000, iterations=100):
    """PyTorch baseline: matmul + sum."""
    a = torch.randn(size, size)
    b = torch.randn(size, size)

    # Warmup
    for _ in range(10):
        c = a @ b
        result = c.sum().item()

    # Benchmark
    start = time.time()
    for _ in range(iterations):
        c = a @ b
        result = c.sum().item()
    elapsed = time.time() - start

    return elapsed / iterations


def benchmark_gt(size=1000, iterations=100):
    """GT: matmul + sum (only scalar transferred)."""
    import gt

    # Pre-create tensors (randn happens once, not measured)
    print("GT: Creating tensors...")
    a = gt.randn(size, size)
    b = gt.randn(size, size)
    print("GT: Tensors created, starting benchmark...")

    # Warmup
    for _ in range(10):
        c = a @ b
        total = c.sum()
        result = total.data.numpy()

    # Benchmark (only matmul + sum, not randn)
    start = time.time()
    for _ in range(iterations):
        c = a @ b
        total = c.sum()
        result = total.data.numpy()
    elapsed = time.time() - start

    return elapsed / iterations


if __name__ == "__main__":
    print("=" * 80)
    print(" " * 20 + "Matmul + Sum Benchmark")
    print(" " * 15 + "(Minimal data transfer - only scalar)")
    print("=" * 80)

    print("\n--- Matrix Multiplication + Sum (1000x1000) ---")
    pytorch_time = benchmark_pytorch()
    print(f"PyTorch:  {pytorch_time * 1000:.3f} ms")

    gt_time = benchmark_gt()
    print(f"GT:       {gt_time * 1000:.3f} ms")

    overhead = (gt_time / pytorch_time - 1) * 100
    print(f"Overhead: {overhead:.1f}%")

    print("\n" + "=" * 80)
    print("\nNote: With sum(), we only transfer a scalar (~4 bytes) instead of")
    print("      a full 1000×1000 matrix (~4MB). This shows pure computation overhead.")
