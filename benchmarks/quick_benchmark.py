"""
Quick benchmark runner that starts GT system automatically.

Usage:
    # Test with 8 GPUs (auto-starts system)
    python benchmarks/quick_benchmark.py --gpus 8

    # Compare 1 vs 4 vs 8 GPUs
    python benchmarks/quick_benchmark.py --gpus 1 4 8

    # Just test huge matmuls
    python benchmarks/quick_benchmark.py --gpus 8 --quick
"""

import os
import sys
import time
import signal
import subprocess
import argparse
from contextlib import contextmanager


@contextmanager
def gt_system(num_gpus, port=9000):
    """Context manager that starts and stops GT system with N GPUs."""

    print(f"\n{'='*80}")
    print(f"Starting GT system with {num_gpus} GPU(s)")
    print(f"{'='*80}\n")

    processes = []

    try:
        # Start dispatcher
        print(f"Starting dispatcher on port {port}...")
        dispatcher = subprocess.Popen(
            ['python', '-m', 'gt.server', '-p', str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        processes.append(('dispatcher', dispatcher))
        time.sleep(2)

        # Start workers
        print(f"Starting {num_gpus} worker(s)...")
        for i in range(num_gpus):
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(i)

            worker = subprocess.Popen(
                ['python', '-m', 'gt.worker', '--host', 'localhost', '-p', str(port)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env
            )
            processes.append((f'worker_{i}', worker))
            print(f"  Started worker {i} on GPU {i}")
            time.sleep(0.5)

        print(f"\nWaiting for system to initialize...")
        time.sleep(3)
        print("✓ GT system ready!\n")

        yield port

    finally:
        # Cleanup
        print(f"\n{'='*80}")
        print("Shutting down GT system...")
        print(f"{'='*80}\n")

        for name, proc in processes:
            print(f"Stopping {name}...")
            proc.terminate()

        # Wait for graceful shutdown
        time.sleep(1)

        # Force kill if needed
        for name, proc in processes:
            if proc.poll() is None:
                print(f"Force killing {name}...")
                proc.kill()

        print("✓ Cleanup complete\n")


def quick_benchmark(num_gpus, port=9000):
    """Run a quick benchmark with the most important tests."""
    import gt

    print(f"\n{'='*80}")
    print(f"QUICK BENCHMARK - {num_gpus} GPU(s)")
    print(f"{'='*80}\n")

    # Enable auto-shard if multiple GPUs
    if num_gpus > 1:
        os.environ['GT_AUTO_SHARD'] = '1'
        print(f"✓ GT_AUTO_SHARD=1")
    else:
        os.environ['GT_AUTO_SHARD'] = '0'
        print(f"✓ GT_AUTO_SHARD=0")

    # Connect
    gt.connect(f'localhost:{port}')
    _ = gt.zeros(1, 1)  # Initialize

    results = {}

    # Test 1: Huge matmul (16K x 16K) @ (16K x 16K)
    print("\n--- Test 1: Massive Square Matmul (16K x 16K) ---")
    times = []
    for i in range(3):
        start = time.time()
        a = gt.randn(16384, 16384)
        b = gt.randn(16384, 16384)
        c = a @ b
        result = c.data.numpy()
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Iteration {i+1}: {elapsed:.4f}s")

    mean_time = sum(times) / len(times)
    flops = 2 * 16384 * 16384 * 16384
    tflops = flops / mean_time / 1e12
    print(f"  Mean: {mean_time:.4f}s ({tflops:.2f} TFLOPS)")
    results['huge_square'] = mean_time

    # Test 2: Large batch matmul (32K batch)
    print("\n--- Test 2: Large Batch Matmul (32K x 4K) @ (4K x 4K) ---")
    times = []
    for i in range(3):
        start = time.time()
        a = gt.randn(32768, 4096)
        b = gt.randn(4096, 4096)
        c = a @ b
        result = c.data.numpy()
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Iteration {i+1}: {elapsed:.4f}s")

    mean_time = sum(times) / len(times)
    flops = 2 * 32768 * 4096 * 4096
    tflops = flops / mean_time / 1e12
    print(f"  Mean: {mean_time:.4f}s ({tflops:.2f} TFLOPS)")
    results['large_batch'] = mean_time

    # Test 3: Element-wise ops on huge tensors
    print("\n--- Test 3: Element-wise Ops on Huge Tensor (32K x 8K) ---")
    times = []
    for i in range(3):
        start = time.time()
        a = gt.randn(32768, 8192)
        b = a.relu()
        c = b * 2.0
        result = c.data.numpy()
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Iteration {i+1}: {elapsed:.4f}s")

    mean_time = sum(times) / len(times)
    print(f"  Mean: {mean_time:.4f}s")
    results['element_wise'] = mean_time

    return results


def main():
    parser = argparse.ArgumentParser(description="Quick GT_AUTO_SHARD benchmark")
    parser.add_argument('--gpus', type=int, nargs='+', default=[1, 8],
                        help='Number of GPUs to test (default: 1 8)')
    parser.add_argument('--port', type=int, default=9000,
                        help='Dispatcher port (default: 9000)')
    parser.add_argument('--quick', action='store_true',
                        help='Run quick benchmark instead of full suite')
    args = parser.parse_args()

    print("\n" + "="*80)
    print("GT_AUTO_SHARD Multi-GPU Benchmark")
    print("="*80)
    print(f"\nTesting configurations: {args.gpus} GPU(s)")
    print(f"Port: {args.port}")

    all_results = {}

    for num_gpus in sorted(args.gpus):
        try:
            with gt_system(num_gpus, port=args.port):
                if args.quick:
                    results = quick_benchmark(num_gpus, port=args.port)
                else:
                    # Import and run full benchmark
                    from auto_shard_benchmark import run_benchmarks
                    results = run_benchmarks(num_gpus, port=args.port)

                all_results[num_gpus] = results

                # Disconnect for next iteration
                import gt
                if gt._client is not None:
                    gt._client = None
                    gt._connected = False

            # Wait between runs
            time.sleep(2)

        except KeyboardInterrupt:
            print("\n\nBenchmark interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"\n\nERROR running benchmark with {num_gpus} GPUs: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Print comparison
    if len(all_results) > 1:
        print("\n" + "="*80)
        print("COMPARISON")
        print("="*80)

        if args.quick:
            # Quick comparison
            gpu_counts = sorted(all_results.keys())
            baseline = all_results[gpu_counts[0]]

            print(f"\n{'Test':<30} | " + " | ".join([f"{n} GPU{'s' if n>1 else '':<10}" for n in gpu_counts]))
            print("-" * 80)

            for test_name in baseline.keys():
                times_str = []
                for n_gpu in gpu_counts:
                    time_val = all_results[n_gpu][test_name]
                    times_str.append(f"{time_val:8.4f}s")
                print(f"{test_name:<30} | " + " | ".join(times_str))

                # Speedup
                if len(gpu_counts) > 1:
                    baseline_time = baseline[test_name]
                    speedups = ['baseline']
                    for n_gpu in gpu_counts[1:]:
                        current_time = all_results[n_gpu][test_name]
                        speedup = baseline_time / current_time
                        speedups.append(f"{speedup:.2f}x")
                    print(f"{'Speedup:':<30} | " + " | ".join([f"{s:<10}" for s in speedups]))
                print()

        else:
            # Full comparison
            from auto_shard_benchmark import print_comparison
            print_comparison(all_results)

    print("\n" + "="*80)
    print("✓ BENCHMARK COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
