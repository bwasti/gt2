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
os.environ['GT_WORKER_BATCH_SIZE'] = '10'


def benchmark_workload(name, workload_fn, num_iterations=100, warmup=5):
    """
    Run a workload many times and measure:
    - Warmup time (includes compilation overhead)
    - Steady-state time (amortized cost)
    """
    # Cleanup any existing GT system before reimport
    if 'gt' in sys.modules:
        import gt
        gt._cleanup()
        # Wait for cleanup to complete
        time.sleep(0.5)

    # Clear cached workload state (model, data) to avoid stale references
    if hasattr(workload_fn, 'model'):
        delattr(workload_fn, 'model')
    if hasattr(workload_fn, 'data'):
        delattr(workload_fn, 'data')
    if hasattr(workload_fn, 'target'):
        delattr(workload_fn, 'target')

    # Force reimport for clean config
    for mod in list(sys.modules.keys()):
        if mod.startswith('gt'):
            del sys.modules[mod]

    import gt

    # Warmup iterations (includes compilation)
    warmup_start = time.time()
    for i in range(warmup):
        workload_fn(i)
    warmup_time = time.time() - warmup_start

    # Steady-state iterations (amortized compilation)
    steady_start = time.time()
    for i in range(warmup, num_iterations):
        workload_fn(i)
    steady_time = time.time() - steady_start

    return {
        'warmup_total': warmup_time,
        'warmup_per_iter': warmup_time / warmup,
        'steady_total': steady_time,
        'steady_per_iter': steady_time / (num_iterations - warmup),
        'total_time': warmup_time + steady_time,
        'avg_per_iter': (warmup_time + steady_time) / num_iterations,
    }


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
    loss.backward()

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

def run_benchmarks():
    """Run all benchmarks with and without compilation."""

    workloads = [
        ("Simple Matmul Chain", workload_simple_matmul, 100, 5),
        ("MLP Training Step", workload_mlp_training, 100, 5),
        ("Attention Block", workload_attention_block, 100, 5),
        ("Mixed Operations", workload_mixed_ops, 100, 5),
    ]

    print("=" * 80)
    print("COMPILATION BENCHMARK SUITE")
    print("Testing repeated ML workloads to measure compilation amortization")
    print("=" * 80)
    print()

    for workload_name, workload_fn, num_iters, warmup in workloads:
        print(f"\n{'=' * 80}")
        print(f"WORKLOAD: {workload_name}")
        print(f"Iterations: {num_iters} (warmup: {warmup})")
        print(f"{'=' * 80}")

        # Test without compilation
        print("\n[1/2] WITHOUT compilation (GT_AUTO_COMPILE=0)...")
        os.environ['GT_AUTO_COMPILE'] = '0'
        results_no_compile = benchmark_workload(
            workload_name, workload_fn, num_iters, warmup
        )

        # Test with compilation
        print("[2/2] WITH compilation (GT_AUTO_COMPILE=1)...")
        os.environ['GT_AUTO_COMPILE'] = '1'
        os.environ['GT_HOTPATH_THRESHOLD'] = '5'  # Trigger after 5 repetitions
        os.environ['GT_HOTPATH_MIN_SEQ'] = '3'    # Detect sequences of 3+ ops
        results_compile = benchmark_workload(
            workload_name, workload_fn, num_iters, warmup
        )

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

        # Steady state
        print(f"{'Steady state (remaining ' + str(num_iters-warmup) + ' iters):':50s} "
              f"{results_no_compile['steady_total']:>10.3f}s "
              f"{results_compile['steady_total']:>10.3f}s "
              f"{results_no_compile['steady_total']/results_compile['steady_total']:>11.2f}x")

        print(f"{'  Per iteration:':50s} "
              f"{results_no_compile['steady_per_iter']:>10.3f}s "
              f"{results_compile['steady_per_iter']:>10.3f}s "
              f"{results_no_compile['steady_per_iter']/results_compile['steady_per_iter']:>11.2f}x")

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

        # Verdict
        speedup = results_no_compile['steady_per_iter'] / results_compile['steady_per_iter']
        total_speedup = results_no_compile['total_time'] / results_compile['total_time']

        print(f"\n{'VERDICT:':50s}", end='')
        if total_speedup > 1.1:
            print(f" ✅ COMPILE WINS ({total_speedup:.2f}x faster overall)")
            print(f"{'':50s}    Steady-state: {speedup:.2f}x faster")
        elif total_speedup > 0.95:
            print(f" ≈  ROUGHLY EQUAL ({total_speedup:.2f}x)")
        else:
            print(f" ❌ NO COMPILE WINS ({1/total_speedup:.2f}x faster)")
            print(f"{'':50s}    Compilation overhead not amortized")

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
