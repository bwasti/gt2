"""
Automatic hotpath detection benchmark.

Tests that repeated operation patterns are automatically detected and compiled
without requiring explicit signal contexts or manual configuration.

Example pattern:
    for _ in range(100):
        c = a + b
        d = c + b
        e = d + c
        f = a + e

This 4-operation pattern repeats 100 times and should be automatically:
1. Detected after ~5 iterations (GT_HOTPATH_THRESHOLD)
2. Compiled into a single function
3. Reused for remaining iterations (95+ cache hits)
"""

import os
import time

os.environ['GT_AUTO_COMPILE'] = '1'
os.environ['GT_HOTPATH_THRESHOLD'] = '5'
os.environ['GT_HOTPATH_MIN_SEQ'] = '4'

import gt

print("=" * 80)
print("AUTOMATIC HOTPATH DETECTION BENCHMARK")
print("=" * 80)
print()
print("Pattern:")
print("  c = a + b")
print("  d = c + b")
print("  e = d + c")
print("  f = a + e")
print()

# Initialize
gt.zeros(1, 1)

a = gt.randn(100, 100)
b = gt.randn(100, 100)

print("Running 100 iterations with automatic hotpath detection...")
start = time.time()

for _ in range(100):
    c = a + b
    d = c + b
    e = d + c
    f = a + e

result = f.data.numpy()
elapsed = time.time() - start

print(f"✓ Complete in {elapsed*1000:.1f}ms ({elapsed/100*1000:.2f}ms per iteration)\n")

# Get stats
stats = gt.debug.get_worker_stats()
worker_stats = list(stats.values())[0]

if 'compilation' in worker_stats:
    comp = worker_stats['compilation']
    print("Compilation Stats:")
    print(f"  Functions compiled:  {comp['cache_size']}")
    print(f"  Cache hits:          {comp['cache_hits']}")
    print(f"  Cache misses:        {comp['cache_misses']}")
    print(f"  Hit rate:            {comp['hit_rate']:.1%}")

    if comp['cache_size'] > 0:
        print(f"\n  Ops per compilation:")
        print(f"    Average:           {comp['avg_ops_per_compilation']:.1f} ops")
        if comp['min_ops_per_compilation'] != comp['max_ops_per_compilation']:
            print(f"    Range:             {comp['min_ops_per_compilation']}-{comp['max_ops_per_compilation']} ops")

    # Calculate launch reduction
    ops_per_iteration = 4  # c = a + b, d = c + b, e = d + c, f = a + e
    total_iterations = 100

    # Without compilation: every op is a separate launch
    eager_launches = ops_per_iteration * total_iterations

    # With compilation: eager during warmup + compiled calls after
    warmup_iterations = comp['cache_misses']
    compiled_iterations = comp['cache_hits']
    eager_ops = ops_per_iteration * warmup_iterations
    compiled_launches = compiled_iterations  # 1 launch per compiled call
    total_launches = eager_ops + compiled_launches

    reduction = eager_launches - total_launches
    reduction_pct = (reduction / eager_launches) * 100

    print(f"\nLaunch Reduction:")
    print(f"  Eager (without compilation):     {eager_launches} launches")
    print(f"  With compilation:")
    print(f"    Eager warmup ({warmup_iterations} iters):       {eager_ops} launches")
    print(f"    Compiled ({compiled_iterations} iters):         {compiled_launches} launches")
    print(f"  Total with compilation:          {total_launches} launches")
    print(f"  Reduction:                       {reduction} launches ({reduction_pct:.1f}%)")

    if comp['hit_rate'] > 0.9:
        print(f"\n✅ SUCCESS: Pattern automatically detected and compiled!")
        print(f"   Reduced from {eager_launches} operations to {compiled_launches} compiled function calls")
    else:
        print(f"\n⚠️  Low cache hit rate")
