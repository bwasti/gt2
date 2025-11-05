"""
Signal-based compilation benchmark.

Uses explicit gt.signal() contexts with compile=True configuration.
This is the user-facing API for compilation.
"""

import os
import time
import numpy as np

# Enable compilation but set high threshold to avoid auto-detection
os.environ['GT_AUTO_COMPILE'] = '1'
os.environ['GT_HOTPATH_THRESHOLD'] = '1000'  # Very high to avoid auto-trigger

import gt
from gt.config import SignalConfig

print("=" * 80)
print("SIGNAL-BASED COMPILATION BENCHMARK")
print("=" * 80)
print()

# Initialize
gt.zeros(1, 1)

# Register a signal with compilation enabled
gt.register_config("training_step", SignalConfig(compile=True))
print("Registered signal 'training_step' with compile=True\n")

# Define the workload
def training_iteration():
    """Training step within signal context."""
    x = gt.randn(50, 50, requires_grad=True)
    w = gt.randn(50, 50, requires_grad=True)

    # Forward pass
    h = x @ w
    h = h.relu()
    h = h @ w
    loss = h.sum()

    # Backward pass
    loss.backward()

    return loss.data.numpy().item()

# Phase 1: Non-compiled baseline
print("Phase 1: Running 20 iterations WITHOUT compilation (baseline)...\n")
baseline_times = []
for i in range(20):
    start = time.time()
    # Run WITHOUT signal context - no compilation
    result = training_iteration()
    elapsed = time.time() - start
    baseline_times.append(elapsed)

    if i < 5 or i % 5 == 0:
        print(f"  Baseline iteration {i+1:2d}: {elapsed:.4f}s")

# Phase 2: Compiled version (with warmup)
print("\nPhase 2: Running iterations WITH compilation...\n")

# Warmup: compile the pattern (exclude from timing)
print("  Warmup: compiling pattern...")
for i in range(3):
    with gt.signal.context("training_step"):
        result = training_iteration()
print("  Compilation complete!\n")

# Now measure compiled performance
print("  Measuring compiled performance (20 iterations)...\n")
compiled_times = []
for i in range(20):
    start = time.time()
    # Run WITH signal context - uses compiled version
    with gt.signal.context("training_step"):
        result = training_iteration()
    elapsed = time.time() - start
    compiled_times.append(elapsed)

    if i < 5 or i % 5 == 0:
        print(f"  Compiled iteration {i+1:2d}: {elapsed:.4f}s")

print("\n" + "=" * 80)
print("RESULTS: Non-Compiled vs Compiled")
print("=" * 80)

baseline_avg = np.mean(baseline_times)
compiled_avg = np.mean(compiled_times)

print(f"\nNon-compiled (baseline):  {baseline_avg:.4f}s avg")
print(f"Compiled version:         {compiled_avg:.4f}s avg")

if compiled_avg < baseline_avg * 0.9:
    speedup = baseline_avg / compiled_avg
    print(f"\n✅ SPEEDUP: {speedup:.2f}x faster with compilation")
elif compiled_avg < baseline_avg * 1.1:
    print(f"\n≈ ROUGHLY SAME (compilation overhead ~= benefit)")
else:
    slowdown = compiled_avg / baseline_avg
    print(f"\n❌ SLOWER with compilation ({slowdown:.2f}x)")

# Show worker stats
print("\n" + "=" * 80)
print("WORKER STATS")
print("=" * 80)
stats = gt.debug.get_worker_stats()
worker_stats = list(stats.values())[0]

print(f"\nCompilation:")
print(f"  Cache size:      {worker_stats['compilation']['cache_size']}")
print(f"  Cache hits:      {worker_stats['compilation']['cache_hits']}")
print(f"  Cache misses:    {worker_stats['compilation']['cache_misses']}")
print(f"  Hit rate:        {worker_stats['compilation']['hit_rate']:.1%}")

if worker_stats['compilation']['cache_size'] > 0:
    print(f"\n✅ Compilation occurred: {worker_stats['compilation']['cache_size']} function(s) cached")

    hit_rate = worker_stats['compilation']['hit_rate']
    if hit_rate > 0.5:
        print(f"✅ Reusing compiled versions! Hit rate: {hit_rate:.1%}")
    elif hit_rate > 0:
        print(f"⚠️  Some reuse: {hit_rate:.1%} hit rate")
    else:
        print(f"❌ NOT reusing compiled versions (0% hit rate)")
        print(f"   Issue: Always executing eagerly, never using cached functions")
else:
    print(f"\n❌ No compilation detected")

print("\n" + "=" * 80)
print("DIAGNOSIS")
print("=" * 80)

if worker_stats['compilation']['cache_size'] > 0 and worker_stats['compilation']['hit_rate'] == 0:
    print("\nCurrent behavior:")
    print("  1. ✅ Signal contexts send CompileStart/End")
    print("  2. ✅ Engine records and compiles operations")
    print("  3. ✅ Compiled functions stored in cache")
    print("  4. ❌ But always executes eagerly, never uses cached versions")
    print()
    print("Next step: Modify engine to CHECK cache before eager execution")
    print("  - If pattern matches cached function: use it")
    print("  - If not: fall back to eager (current behavior)")
elif worker_stats['compilation']['hit_rate'] > 0.5:
    print("\n✅ Compilation is WORKING!")
    print(f"   Successfully reusing compiled functions ({worker_stats['compilation']['hit_rate']:.1%} hit rate)")
else:
    print("\nUnexpected state - needs investigation")

print("\n" + "=" * 80)
