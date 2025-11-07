"""
Benchmark GT_AUTO_SHARD across multiple GPUs.

This benchmark compares single-GPU vs multi-GPU performance with automatic sharding.
Tests large matrix multiplications that actually benefit from distribution.

Usage:
    # Single GPU baseline
    python benchmarks/auto_shard_benchmark.py --gpus 1

    # 8 GPU distributed
    python benchmarks/auto_shard_benchmark.py --gpus 8

    # Run both and compare
    python benchmarks/auto_shard_benchmark.py --gpus 1 8
"""

import os
import time
import argparse
import numpy as np
from collections import defaultdict


def benchmark_matmul(client, m, k, n, warmup=3, iterations=10):
    """
    Benchmark large matrix multiplication: (m, k) @ (k, n)

    Args:
        client: GT client instance
        m, k, n: Matrix dimensions
        warmup: Number of warmup iterations
        iterations: Number of timed iterations

    Returns:
        dict with timing stats
    """
    import gt

    print(f"\n--- Benchmarking matmul ({m}, {k}) @ ({k}, {n}) ---")

    # Warmup
    print(f"Warming up ({warmup} iterations)...")
    for i in range(warmup):
        a = gt.randn(m, k)
        b = gt.randn(k, n)
        c = a @ b
        _ = c.data.numpy()  # Force execution
        print(f"  Warmup {i+1}/{warmup}")

    # Timed iterations
    print(f"Running benchmark ({iterations} iterations)...")
    times = []

    for i in range(iterations):
        start = time.time()

        # Create tensors
        a = gt.randn(m, k)
        b = gt.randn(k, n)

        # Matmul
        c = a @ b

        # Force execution by fetching result
        result = c.data.numpy()

        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Iteration {i+1}/{iterations}: {elapsed:.4f}s")

    # Verify result shape
    assert result.shape == (m, n), f"Expected shape ({m}, {n}), got {result.shape}"

    mean_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)

    # Compute FLOPS (2*m*k*n for matmul)
    flops = 2 * m * k * n
    tflops = flops / mean_time / 1e12

    print(f"  Mean: {mean_time:.4f}s ± {std_time:.4f}s")
    print(f"  Min:  {min_time:.4f}s")
    print(f"  Throughput: {tflops:.2f} TFLOPS")

    return {
        'shape': (m, k, n),
        'mean_time': mean_time,
        'std_time': std_time,
        'min_time': min_time,
        'tflops': tflops,
        'times': times
    }


def benchmark_mlp_forward(client, batch_size, hidden_dims, warmup=3, iterations=10):
    """
    Benchmark MLP forward pass with large batches.

    Architecture: batch_size × hidden[0] -> hidden[1] -> ... -> hidden[-1]
    """
    import gt
    from gt.client import nn

    print(f"\n--- Benchmarking MLP forward pass ---")
    print(f"  Batch size: {batch_size}")
    print(f"  Architecture: {' -> '.join(map(str, hidden_dims))}")

    # Create layers
    layers = []
    for i in range(len(hidden_dims) - 1):
        layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))

    # Warmup
    print(f"Warming up ({warmup} iterations)...")
    for i in range(warmup):
        x = gt.randn(batch_size, hidden_dims[0])
        for layer in layers:
            x = layer(x).relu()
        _ = x.data.numpy()
        print(f"  Warmup {i+1}/{warmup}")

    # Timed iterations
    print(f"Running benchmark ({iterations} iterations)...")
    times = []

    for i in range(iterations):
        start = time.time()

        x = gt.randn(batch_size, hidden_dims[0])
        for layer in layers:
            x = layer(x).relu()
        result = x.data.numpy()

        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Iteration {i+1}/{iterations}: {elapsed:.4f}s")

    mean_time = np.mean(times)
    std_time = np.std(times)

    # Estimate FLOPS (approximate)
    flops = 0
    for i in range(len(hidden_dims) - 1):
        flops += 2 * batch_size * hidden_dims[i] * hidden_dims[i+1]  # matmul
        flops += batch_size * hidden_dims[i+1]  # relu

    tflops = flops / mean_time / 1e12

    print(f"  Mean: {mean_time:.4f}s ± {std_time:.4f}s")
    print(f"  Throughput: {tflops:.2f} TFLOPS")

    return {
        'batch_size': batch_size,
        'hidden_dims': hidden_dims,
        'mean_time': mean_time,
        'std_time': std_time,
        'tflops': tflops,
        'times': times
    }


def run_benchmarks(num_gpus, port=9000):
    """Run all benchmarks for given number of GPUs."""
    import gt

    print("\n" + "="*80)
    print(f"RUNNING BENCHMARKS WITH {num_gpus} GPU(s)")
    print("="*80)

    # Set GT_AUTO_SHARD based on number of GPUs
    if num_gpus > 1:
        os.environ['GT_AUTO_SHARD'] = '1'
        print(f"✓ GT_AUTO_SHARD=1 (auto-sharding across {num_gpus} workers)")
    else:
        os.environ['GT_AUTO_SHARD'] = '0'
        print(f"✓ GT_AUTO_SHARD=0 (single worker)")

    # Connect to GT system
    gt.connect(f'localhost:{port}')
    print(f"✓ Connected to dispatcher at localhost:{port}\n")

    # Initialize
    _ = gt.zeros(1, 1)

    results = {}

    # Benchmark 1: Square matmuls with increasing sizes
    print("\n" + "="*80)
    print("BENCHMARK 1: Square Matrix Multiplications")
    print("="*80)

    square_sizes = [
        4096,   # 4K × 4K
        8192,   # 8K × 8K
        16384,  # 16K × 16K
    ]

    results['square_matmul'] = []
    for size in square_sizes:
        result = benchmark_matmul(gt, size, size, size, warmup=2, iterations=5)
        results['square_matmul'].append(result)

    # Benchmark 2: Large batch matmuls (typical for ML)
    print("\n" + "="*80)
    print("BENCHMARK 2: Large Batch Matrix Multiplications")
    print("="*80)

    batch_configs = [
        (8192, 4096, 4096),   # Large batch × medium hidden
        (16384, 4096, 4096),  # Huge batch × medium hidden
        (32768, 2048, 2048),  # Massive batch × smaller hidden
    ]

    results['batch_matmul'] = []
    for m, k, n in batch_configs:
        result = benchmark_matmul(gt, m, k, n, warmup=2, iterations=5)
        results['batch_matmul'].append(result)

    # Benchmark 3: MLP forward passes
    print("\n" + "="*80)
    print("BENCHMARK 3: MLP Forward Passes")
    print("="*80)

    mlp_configs = [
        (4096, [4096, 4096, 4096]),      # 4K batch, 3-layer 4K hidden
        (8192, [4096, 8192, 4096]),      # 8K batch, 3-layer mixed
        (16384, [2048, 4096, 2048]),     # 16K batch, 3-layer mixed
    ]

    results['mlp'] = []
    for batch_size, hidden_dims in mlp_configs:
        result = benchmark_mlp_forward(gt, batch_size, hidden_dims, warmup=2, iterations=5)
        results['mlp'].append(result)

    return results


def print_comparison(results_by_gpu):
    """Print side-by-side comparison of results."""
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON")
    print("="*80)

    gpu_counts = sorted(results_by_gpu.keys())

    # Square matmuls
    print("\n--- Square Matrix Multiplications ---")
    print(f"{'Size':<12} | " + " | ".join([f"{n} GPU{'s' if n > 1 else '':<12}" for n in gpu_counts]))
    print("-" * 80)

    for i, result in enumerate(results_by_gpu[gpu_counts[0]]['square_matmul']):
        size = result['shape'][0]
        times_str = []
        for n_gpu in gpu_counts:
            res = results_by_gpu[n_gpu]['square_matmul'][i]
            times_str.append(f"{res['mean_time']:6.3f}s ({res['tflops']:5.1f} TF)")
        print(f"{size}x{size:<6} | " + " | ".join(times_str))

        # Print speedup
        if len(gpu_counts) > 1:
            baseline = results_by_gpu[gpu_counts[0]]['square_matmul'][i]['mean_time']
            speedups = []
            for n_gpu in gpu_counts[1:]:
                current = results_by_gpu[n_gpu]['square_matmul'][i]['mean_time']
                speedup = baseline / current
                speedups.append(f"{speedup:.2f}x")
            print(f"{'Speedup:':<12} | {'baseline':<18} | " + " | ".join([f"{s:<18}" for s in speedups]))

    # Batch matmuls
    print("\n--- Large Batch Matrix Multiplications ---")
    print(f"{'Shape':<20} | " + " | ".join([f"{n} GPU{'s' if n > 1 else '':<12}" for n in gpu_counts]))
    print("-" * 80)

    for i, result in enumerate(results_by_gpu[gpu_counts[0]]['batch_matmul']):
        m, k, n = result['shape']
        shape_str = f"({m},{k})@({k},{n})"
        times_str = []
        for n_gpu in gpu_counts:
            res = results_by_gpu[n_gpu]['batch_matmul'][i]
            times_str.append(f"{res['mean_time']:6.3f}s ({res['tflops']:5.1f} TF)")
        print(f"{shape_str:<20} | " + " | ".join(times_str))

        # Print speedup
        if len(gpu_counts) > 1:
            baseline = results_by_gpu[gpu_counts[0]]['batch_matmul'][i]['mean_time']
            speedups = []
            for n_gpu in gpu_counts[1:]:
                current = results_by_gpu[n_gpu]['batch_matmul'][i]['mean_time']
                speedup = baseline / current
                speedups.append(f"{speedup:.2f}x")
            print(f"{'Speedup:':<20} | {'baseline':<18} | " + " | ".join([f"{s:<18}" for s in speedups]))

    # MLPs
    print("\n--- MLP Forward Passes ---")
    print(f"{'Config':<30} | " + " | ".join([f"{n} GPU{'s' if n > 1 else '':<12}" for n in gpu_counts]))
    print("-" * 80)

    for i, result in enumerate(results_by_gpu[gpu_counts[0]]['mlp']):
        batch_size = result['batch_size']
        hidden_dims = result['hidden_dims']
        config_str = f"batch={batch_size}, {hidden_dims}"
        times_str = []
        for n_gpu in gpu_counts:
            res = results_by_gpu[n_gpu]['mlp'][i]
            times_str.append(f"{res['mean_time']:6.3f}s ({res['tflops']:5.1f} TF)")
        print(f"{config_str:<30} | " + " | ".join(times_str))

        # Print speedup
        if len(gpu_counts) > 1:
            baseline = results_by_gpu[gpu_counts[0]]['mlp'][i]['mean_time']
            speedups = []
            for n_gpu in gpu_counts[1:]:
                current = results_by_gpu[n_gpu]['mlp'][i]['mean_time']
                speedup = baseline / current
                speedups.append(f"{speedup:.2f}x")
            print(f"{'Speedup:':<30} | {'baseline':<18} | " + " | ".join([f"{s:<18}" for s in speedups]))

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    if len(gpu_counts) > 1:
        for n_gpu in gpu_counts[1:]:
            # Compute average speedup
            all_speedups = []

            baseline_results = results_by_gpu[gpu_counts[0]]
            current_results = results_by_gpu[n_gpu]

            for i in range(len(baseline_results['square_matmul'])):
                baseline_time = baseline_results['square_matmul'][i]['mean_time']
                current_time = current_results['square_matmul'][i]['mean_time']
                all_speedups.append(baseline_time / current_time)

            for i in range(len(baseline_results['batch_matmul'])):
                baseline_time = baseline_results['batch_matmul'][i]['mean_time']
                current_time = current_results['batch_matmul'][i]['mean_time']
                all_speedups.append(baseline_time / current_time)

            for i in range(len(baseline_results['mlp'])):
                baseline_time = baseline_results['mlp'][i]['mean_time']
                current_time = current_results['mlp'][i]['mean_time']
                all_speedups.append(baseline_time / current_time)

            avg_speedup = np.mean(all_speedups)
            print(f"\n{n_gpu} GPU{'s' if n_gpu > 1 else ''} vs 1 GPU:")
            print(f"  Average speedup: {avg_speedup:.2f}x")
            print(f"  Min speedup: {np.min(all_speedups):.2f}x")
            print(f"  Max speedup: {np.max(all_speedups):.2f}x")


def main():
    parser = argparse.ArgumentParser(description="Benchmark GT_AUTO_SHARD with multiple GPUs")
    parser.add_argument('--gpus', type=int, nargs='+', default=[1, 8],
                        help='Number of GPUs to test (default: 1 8)')
    parser.add_argument('--port', type=int, default=9000,
                        help='Dispatcher port (default: 9000)')
    args = parser.parse_args()

    print("\n" + "="*80)
    print("GT_AUTO_SHARD Multi-GPU Benchmark")
    print("="*80)
    print(f"\nTesting with: {args.gpus} GPU configurations")
    print(f"Dispatcher port: {args.port}")
    print("\nIMPORTANT:")
    print("  1. Start dispatcher: python -m gt.server -p 9000")
    print("  2. Start workers (one per GPU):")
    for i in range(max(args.gpus)):
        print(f"     CUDA_VISIBLE_DEVICES={i} python -m gt.worker --host localhost -p 9000")
    print(f"\n  3. Run this benchmark after all {max(args.gpus)} workers are connected")

    input("\nPress Enter to start benchmarks (make sure workers are running)...")

    results_by_gpu = {}

    for num_gpus in sorted(args.gpus):
        try:
            results = run_benchmarks(num_gpus, port=args.port)
            results_by_gpu[num_gpus] = results

            # Disconnect to allow reconnect with different settings
            import gt
            if gt._client is not None:
                gt._client = None
                gt._connected = False

            # Wait a bit between runs
            time.sleep(2)

        except Exception as e:
            print(f"\nERROR running benchmark with {num_gpus} GPUs: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Print comparison if we have multiple results
    if len(results_by_gpu) > 1:
        print_comparison(results_by_gpu)

    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
