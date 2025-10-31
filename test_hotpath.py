#!/usr/bin/env python3
"""
Test hot path detection.

This script performs repeated operation patterns to trigger hot path detection.
Run with GT_AUTO_COMPILE=1 to enable.
"""

import gt
import numpy as np

print("="*80)
print("Testing Hot Path Detection")
print("="*80)

# Initialize GT (auto-starts server)
gt.zeros(1, 1)  # Force initialization

print("\nPerforming 20 iterations of: matmul -> relu -> mul")
print("Hot path should be detected after 10 iterations...\n")

# Create input tensors
a = gt.randn(100, 100)
b = gt.randn(100, 100)
c = gt.randn(100, 100)

# Perform repeated pattern: matmul -> relu -> mul
results = []
for i in range(20):
    # This pattern should be detected as a hot path
    x = a @ b  # matmul
    y = x.relu()  # relu
    z = y * c  # mul

    # Sync point - triggers sequence reset
    result = z.sum().data.numpy()
    results.append(float(result))

    if i == 9:
        print(f"[Iteration {i+1}] *** Hot path should be detected around now ***")
    elif i % 5 == 4:
        print(f"[Iteration {i+1}] Result: {result:.2f}")

print(f"\nCompleted {len(results)} iterations")
print(f"First result: {results[0]:.2f}")
print(f"Last result: {results[-1]:.2f}")

# Get worker stats to see hot path detection results
try:
    all_stats = gt.debug.get_worker_stats()

    # Stats are returned per-worker
    found_hotpath = False
    for worker_id, stats in all_stats.items():
        if 'hotpath' in stats:
            found_hotpath = True
            hp = stats['hotpath']
            print("\n" + "="*80)
            print(f"Hot Path Detection Stats (Worker: {worker_id}):")
            print("="*80)
            print(f"Total instructions:     {hp['total_instructions']}")
            print(f"Hot instructions:       {hp['hot_instructions']}")
            print(f"Unique sequences:       {hp['unique_sequences']}")
            print(f"Hot sequences detected: {hp['hot_sequences']}")
            print(f"Detection threshold:    {hp['hot_threshold']}")

            if hp['hot_sequences'] > 0:
                print(f"\n✅ SUCCESS: Hot path detected after {hp['hot_threshold']} repetitions!")
                print("\nTop sequences:")
                for seq, count in hp['top_sequences'][:3]:
                    print(f"  {seq}: {count} times")
            else:
                print(f"\n⚠️  No hot paths detected yet (need {hp['hot_threshold']} repetitions)")
            break

    if not found_hotpath:
        print("\n⚠️  Hot path detection not enabled (set GT_AUTO_COMPILE=1)")

except Exception as e:
    print(f"\n⚠️  Could not retrieve stats: {e}")

print("\n" + "="*80)
print("Done!")
print("="*80)
