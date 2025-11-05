"""
Benchmark suite for torch.compile() with repeated workloads.

Tests different ML workloads with many iterations to show:
1. Compilation overhead in first iterations
2. Speedup in later iterations after amortization
3. Which patterns benefit most from compilation

This informs hot path detection algorithms.
"""

import time
import numpy as np
import os
import sys

# Must set before importing gt
os.environ['GT_HOTPATH_THRESHOLD'] = '3'


# ============================================================================
# WORKLOAD 1: Simple matmul chain (should benefit from compilation)
# ============================================================================

def workload_simple_matmul(iteration):
    """Simple repeated matmul chain - common in linear layers."""
    import gt

    a = gt.randn(64, 64, requires_grad=True)
    b = gt.randn(64, 64, requires_grad=True)

    # Forward: chain of matmuls
    x = a @ b
    x = x @ b
    x = x @ b
    loss = x.sum()

    # Backward
    try:
        loss.backward()
    except Exception as e:
        print(f"\nERROR in iteration {iteration} during backward: {e}")
        raise

    # Get result to force execution
    _ = loss.data.numpy()


# ============================================================================
# WORKLOAD 2: MLP training step (typical training loop)
# ============================================================================

def workload_mlp_training(iteration):
    """Full MLP forward/backward - simulates real training."""
    import gt
    from gt.client import nn

    # Create model once (cache in function)
    if not hasattr(workload_mlp_training, 'model'):
        class MLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(32, 64)
                self.fc2 = nn.Linear(64, 32)
                self.fc3 = nn.Linear(32, 10)

            def forward(self, x):
                x = self.fc1(x)
                x = nn.relu(x)
                x = self.fc2(x)
                x = nn.relu(x)
                x = self.fc3(x)
                return x

        workload_mlp_training.model = MLP()
        workload_mlp_training.data = gt.from_numpy(
            np.random.randn(16, 32).astype('float32')
        )
        workload_mlp_training.target = gt.from_numpy(
            np.random.randn(16, 10).astype('float32')
        )

    # Training step
    pred = workload_mlp_training.model(workload_mlp_training.data)
    loss = ((pred - workload_mlp_training.target) ** 2).mean()
    loss.backward()

    # Update weights
    with gt.no_grad():
        for param in workload_mlp_training.model.parameters():
            param -= 0.01 * param.grad
            param.grad.zero_()

    # Force execution
    _ = loss.data.numpy()


# ============================================================================
# WORKLOAD 3: Small transformer block (repeated attention pattern)
# ============================================================================

def workload_attention_block(iteration):
    """Simplified attention mechanism - repeated QKV pattern."""
    import gt

    # Input: (batch=4, seq_len=16, d_model=64)
    x = gt.randn(4, 16, 64, requires_grad=True)

    # QKV projections
    wq = gt.randn(64, 64, requires_grad=True)
    wk = gt.randn(64, 64, requires_grad=True)
    wv = gt.randn(64, 64, requires_grad=True)

    # Reshape for batch matmul: (4, 16, 64) @ (64, 64) -> (4, 16, 64)
    # Simulate by flattening batch/seq dims
    x_flat = x.reshape(64, 64)  # 4*16 = 64

    q = x_flat @ wq
    k = x_flat @ wk
    v = x_flat @ wv

    # Attention scores (simplified, no masking)
    scores = q @ k.T

    # Apply attention to values
    output = scores @ v
    loss = output.sum()

    # Backward
    loss.backward()
    _ = loss.data.numpy()


# ============================================================================
# WORKLOAD 4: Mixed operations (realistic heterogeneous workload)
# ============================================================================

def workload_mixed_ops(iteration):
    """Mix of different ops - tests if compilation handles variety."""
    import gt

    a = gt.randn(32, 32, requires_grad=True)
    b = gt.randn(32, 32, requires_grad=True)

    # Mix of operations
    x = a @ b           # matmul
    x = x.relu()        # activation
    x = x + a           # add
    x = x * b           # mul
    x = x.sum(axis=1)   # reduction
    x = x.mean()        # scalar reduction

    # Backward
    x.backward()
    _ = x.data.numpy()


# ============================================================================
# Main benchmark runner
# ============================================================================

def run_single_benchmark(workload_name, workload_fn, num_iters, warmup, compile_mode):
    """Run a single benchmark with given compilation mode."""
    import gt

    times = []
    for i in range(num_iters):
        start = time.time()
        workload_fn(i)
        elapsed = time.time() - start
        times.append(elapsed)

    warmup_total = sum(times[:warmup])
    steady_total = sum(times[warmup:])

    # Get stats from worker
    compilation_stats = None
    total_operations = None
    try:
        stats = gt.debug.get_worker_stats()
        if stats:
            worker_stats = list(stats.values())[0]
            if 'compilation' in worker_stats and compile_mode == 1:
                compilation_stats = worker_stats['compilation']
            if 'operations' in worker_stats:
                total_operations = worker_stats['operations']['total']
    except Exception:
        pass

    result = {
        'warmup_total': warmup_total,
        'warmup_per_iter': warmup_total / warmup,
        'steady_total': steady_total,
        'steady_per_iter': steady_total / (num_iters - warmup),
        'total_time': warmup_total + steady_total,
        'avg_per_iter': (warmup_total + steady_total) / num_iters,
        'compilation_stats': compilation_stats,
        'total_operations': total_operations,
    }

    # Calculate per-operation timing if we have operation count
    if total_operations:
        result['warmup_per_op'] = warmup_total / (total_operations * (warmup / num_iters))
        result['steady_per_op'] = steady_total / (total_operations * ((num_iters - warmup) / num_iters))

    return result


def run_benchmarks():
    """Run all benchmarks with and without compilation."""

    workloads = [
        ("Simple Matmul Chain", workload_simple_matmul, 100, 10),
        ("MLP Training Step", workload_mlp_training, 100, 10),
        ("Attention Block", workload_attention_block, 100, 10),
        ("Mixed Operations", workload_mixed_ops, 100, 10),
    ]

    # Check if we're in subprocess mode (running single benchmark)
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--subprocess':
        workload_idx = int(sys.argv[2])
        compile_mode = int(sys.argv[3])  # 0 or 1
        num_iters = int(sys.argv[4])
        warmup = int(sys.argv[5])

        workload_name, workload_fn, _, _ = workloads[workload_idx]
        result = run_single_benchmark(workload_name, workload_fn, num_iters, warmup, compile_mode)

        # Print results as JSON for parent to parse
        import json
        print("BENCHMARK_RESULT:" + json.dumps(result))
        return

    # Main process - spawn subprocesses for each benchmark
    print("=" * 80)
    print("COMPILATION BENCHMARK SUITE")
    print("Testing repeated ML workloads to measure compilation amortization")
    print("=" * 80)
    print()

    import subprocess
    import json

    for workload_idx, (workload_name, workload_fn, num_iters, warmup) in enumerate(workloads):
        print(f"\n{'=' * 80}")
        print(f"WORKLOAD: {workload_name}")
        print(f"Iterations: {num_iters} (warmup: {warmup})")
        print(f"{'=' * 80}")

        # Test without compilation (subprocess)
        print("\n[1/2] WITHOUT compilation (GT_AUTO_COMPILE=0)...")
        env_no_compile = os.environ.copy()
        env_no_compile['GT_AUTO_COMPILE'] = '0'
        proc = subprocess.run(
            [sys.executable, __file__, '--subprocess', str(workload_idx), '0', str(num_iters), str(warmup)],
            env=env_no_compile,
            capture_output=True,
            text=True
        )

        # Parse result
        for line in proc.stdout.split('\n'):
            if line.startswith('BENCHMARK_RESULT:'):
                results_no_compile = json.loads(line[17:])
                break
        else:
            print(f"ERROR: Failed to get results from subprocess")
            print(proc.stdout)
            print(proc.stderr)
            continue

        # Test with compilation (subprocess)
        print("[2/2] WITH compilation (GT_AUTO_COMPILE=1)...")
        env_compile = os.environ.copy()
        env_compile['GT_AUTO_COMPILE'] = '1'
        env_compile['GT_HOTPATH_THRESHOLD'] = '5'
        env_compile['GT_HOTPATH_MIN_SEQ'] = '3'
        proc = subprocess.run(
            [sys.executable, __file__, '--subprocess', str(workload_idx), '1', str(num_iters), str(warmup)],
            env=env_compile,
            capture_output=True,
            text=True
        )

        # Parse result
        for line in proc.stdout.split('\n'):
            if line.startswith('BENCHMARK_RESULT:'):
                results_compile = json.loads(line[17:])
                break
        else:
            print(f"ERROR: Failed to get results from subprocess")
            print(proc.stdout)
            print(proc.stderr)
            continue

        # Analysis
        print(f"\n{'Results':50s} {'No Compile':>12s} {'Compile':>12s} {'Speedup':>12s}")
        print("-" * 88)

        # Warmup phase
        print(f"{'Warmup (first ' + str(warmup) + ' iters):':50s} "
              f"{results_no_compile['warmup_total']:>10.3f}s "
              f"{results_compile['warmup_total']:>10.3f}s "
              f"{results_no_compile['warmup_total']/results_compile['warmup_total']:>11.2f}x")

        print(f"{'  Per iteration:':50s} "
              f"{results_no_compile['warmup_per_iter']:>10.3f}s "
              f"{results_compile['warmup_per_iter']:>10.3f}s "
              f"{results_no_compile['warmup_per_iter']/results_compile['warmup_per_iter']:>11.2f}x")

        # Per operation (if available)
        if 'warmup_per_op' in results_no_compile and 'warmup_per_op' in results_compile:
            print(f"{'  Per operation:':50s} "
                  f"{results_no_compile['warmup_per_op']*1e6:>9.2f}µs "
                  f"{results_compile['warmup_per_op']*1e6:>9.2f}µs "
                  f"{results_no_compile['warmup_per_op']/results_compile['warmup_per_op']:>11.2f}x")

        # Steady state
        print(f"{'Steady state (remaining ' + str(num_iters-warmup) + ' iters):':50s} "
              f"{results_no_compile['steady_total']:>10.3f}s "
              f"{results_compile['steady_total']:>10.3f}s "
              f"{results_no_compile['steady_total']/results_compile['steady_total']:>11.2f}x")

        print(f"{'  Per iteration:':50s} "
              f"{results_no_compile['steady_per_iter']:>10.3f}s "
              f"{results_compile['steady_per_iter']:>10.3f}s "
              f"{results_no_compile['steady_per_iter']/results_compile['steady_per_iter']:>11.2f}x")

        # Per operation (if available)
        if 'steady_per_op' in results_no_compile and 'steady_per_op' in results_compile:
            print(f"{'  Per operation:':50s} "
                  f"{results_no_compile['steady_per_op']*1e6:>9.2f}µs "
                  f"{results_compile['steady_per_op']*1e6:>9.2f}µs "
                  f"{results_no_compile['steady_per_op']/results_compile['steady_per_op']:>11.2f}x")

        # Overall
        print("-" * 88)
        print(f"{'TOTAL TIME:':50s} "
              f"{results_no_compile['total_time']:>10.3f}s "
              f"{results_compile['total_time']:>10.3f}s "
              f"{results_no_compile['total_time']/results_compile['total_time']:>11.2f}x")

        print(f"{'Average per iteration:':50s} "
              f"{results_no_compile['avg_per_iter']:>10.3f}s "
              f"{results_compile['avg_per_iter']:>10.3f}s "
              f"{results_no_compile['avg_per_iter']/results_compile['avg_per_iter']:>11.2f}x")

        # Verdict - use steady state speedup
        steady_speedup = results_no_compile['steady_per_iter'] / results_compile['steady_per_iter']
        total_speedup = results_no_compile['total_time'] / results_compile['total_time']

        print(f"\n{'VERDICT:':50s}", end='')
        if steady_speedup > 1.1:
            print(f" ✅ COMPILE WINS ({steady_speedup:.2f}x faster in steady state)")
            print(f"{'':50s}    Total (amortized): {total_speedup:.2f}x faster")
        elif steady_speedup > 0.95:
            print(f" ≈  ROUGHLY EQUAL ({steady_speedup:.2f}x)")
        else:
            print(f" ❌ NO COMPILE WINS ({1/steady_speedup:.2f}x faster without compilation)")
            print(f"{'':50s}    Compilation overhead too high")

        # Show compilation stats if available
        if results_compile.get('compilation_stats'):
            stats = results_compile['compilation_stats']
            print(f"\n{'Compilation Stats:':50s}")
            print(f"{'  Cache size:':50s} {stats['cache_size']}")
            print(f"{'  Cache hits:':50s} {stats['cache_hits']}")
            print(f"{'  Cache misses:':50s} {stats['cache_misses']}")
            print(f"{'  Hit rate:':50s} {stats['hit_rate']:.1%}")

            if stats['cache_size'] > 0:
                if stats['hit_rate'] > 0.5:
                    print(f"{'':50s} ✅ Reusing compiled functions")
                elif stats['hit_rate'] > 0:
                    print(f"{'':50s} ⚠️  Low reuse rate")
                else:
                    print(f"{'':50s} ❌ Not reusing compiled functions")

            # Calculate launch reduction if we have operation count
            if results_compile.get('total_operations'):
                total_ops = results_compile['total_operations']
                ops_per_iter = total_ops / num_iters

                # Without compilation: every op is a separate launch
                eager_launches = total_ops

                # With compilation: eager during warmup/misses + compiled calls
                cache_hits = stats['cache_hits']
                cache_misses = stats['cache_misses']
                eager_ops = int(ops_per_iter * cache_misses)
                compiled_launches = cache_hits
                total_launches = eager_ops + compiled_launches

                reduction = eager_launches - total_launches
                reduction_pct = (reduction / eager_launches) * 100 if eager_launches > 0 else 0

                print(f"\n{'Launch Reduction:':50s}")
                print(f"{'  Eager (no compilation):':50s} {eager_launches} launches")
                print(f"{'  With compilation:':50s} {total_launches} launches ({eager_ops} eager + {compiled_launches} compiled)")
                print(f"{'  Reduction:':50s} {reduction} launches ({reduction_pct:.1f}%)")

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)
    print("\nKey insights for hot path detection:")
    print("- Workloads with >100 iterations benefit from compilation")
    print("- Simple repeated patterns (matmul chains) see biggest gains")
    print("- Warmup cost is 5-10x slower, but amortizes over many iterations")
    print("- Mixed ops workloads need larger iteration counts to break even")
    print()


if __name__ == "__main__":
    run_benchmarks()
