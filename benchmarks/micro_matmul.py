"""
Micro-benchmark for profiling: simple matmul operation.

This is the simplest operation to profile and optimize.
"""

import time
import numpy as np
import gt


def benchmark_simple_matmul(size=100, iterations=10):
    """Simplest benchmark: small matmul."""
    print(f"=== Micro Benchmark: Matmul {size}x{size} ===")

    # Create tensors
    a = gt.randn(size, size)
    b = gt.randn(size, size)

    # Warmup
    for _ in range(2):
        c = a @ b
        _ = c.data.numpy()

    # Benchmark
    times = []
    for i in range(iterations):
        start = time.perf_counter()
        c = a @ b
        result = c.data.numpy()
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)  # Convert to ms
        print(f"  Iteration {i+1}: {elapsed * 1000:.3f} ms")

    avg_time = np.mean(times)
    std_time = np.std(times)
    print(f"\nAverage: {avg_time:.3f} ± {std_time:.3f} ms")
    return avg_time


if __name__ == "__main__":
    benchmark_simple_matmul(size=100, iterations=10)
