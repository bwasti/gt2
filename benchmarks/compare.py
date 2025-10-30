"""
Compare PyTorch baseline vs GT framework performance.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytorch_baseline
import gt_benchmark


def compare_benchmarks():
    """Run both benchmarks and show comparison."""
    print("=" * 80)
    print(" " * 25 + "Performance Comparison")
    print("=" * 80)

    # Matmul
    print("\n--- Matrix Multiplication (1000x1000) ---")
    pytorch_matmul = pytorch_baseline.benchmark_matmul()
    gt_matmul = gt_benchmark.benchmark_matmul()
    print(f"PyTorch:  {pytorch_matmul * 1000:.3f} ms")
    print(f"GT:       {gt_matmul * 1000:.3f} ms")
    print(f"Overhead: {(gt_matmul / pytorch_matmul - 1) * 100:.1f}%")

    # Linear forward
    print("\n--- Linear Forward Pass (128x512->256) ---")
    pytorch_linear = pytorch_baseline.benchmark_linear_forward()
    gt_linear = gt_benchmark.benchmark_linear_forward()
    print(f"PyTorch:  {pytorch_linear * 1000:.3f} ms")
    print(f"GT:       {gt_linear * 1000:.3f} ms")
    print(f"Overhead: {(gt_linear / pytorch_linear - 1) * 100:.1f}%")

    # MSE Loss
    print("\n--- MSE Loss (10000 elements) ---")
    pytorch_mse = pytorch_baseline.benchmark_mse_loss()
    gt_mse = gt_benchmark.benchmark_mse_loss()
    print(f"PyTorch:  {pytorch_mse * 1000:.3f} ms")
    print(f"GT:       {gt_mse * 1000:.3f} ms")
    print(f"Overhead: {(gt_mse / pytorch_mse - 1) * 100:.1f}%")

    # Backward pass
    print("\n--- Backward Pass (1000x1000) ---")
    pytorch_backward = pytorch_baseline.benchmark_backward()
    gt_backward = gt_benchmark.benchmark_backward()
    print(f"PyTorch:  {pytorch_backward * 1000:.3f} ms")
    print(f"GT:       {gt_backward * 1000:.3f} ms")
    print(f"Overhead: {(gt_backward / pytorch_backward - 1) * 100:.1f}%")

    # Activations
    print("\n--- Activation Functions (10000 elements) ---")
    pytorch_act = pytorch_baseline.benchmark_activations()
    gt_act = gt_benchmark.benchmark_activations()

    for name in ['relu', 'sigmoid', 'tanh']:
        print(f"\n{name.upper()}:")
        print(f"  PyTorch:  {pytorch_act[name] * 1000:.3f} ms")
        print(f"  GT:       {gt_act[name] * 1000:.3f} ms")
        print(f"  Overhead: {(gt_act[name] / pytorch_act[name] - 1) * 100:.1f}%")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    compare_benchmarks()
