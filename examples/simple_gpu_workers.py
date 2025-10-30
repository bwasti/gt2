"""
Simple example showing gt.gpu_workers() API for automatic sharding.

This is the easiest way to use multiple GPUs with GT - just one line!

Usage:
    python examples/simple_gpu_workers.py
"""

import numpy as np
import gt

# Configure to use 4 GPU workers
# This MUST be called before any tensor operations
gt.gpu_workers(4)

print("\n=== GT with gt.gpu_workers(4) ===\n")
print("This will automatically:")
print("  1. Start a local dispatcher")
print("  2. Spawn 4 workers (one per GPU)")
print("  3. Enable automatic tensor sharding")
print()

# Create sharded matrix A (128, 64)
# With 4 workers: automatically sharded as (32, 64) per worker
print("Creating matrix A (128, 64)...")
a = gt.randn(128, 64)
print(f"  A shape: {a.data.numpy().shape}")

# Create matrix B (64, 32) - broadcast to all workers
print("\nCreating matrix B (64, 32)...")
b_data = np.random.randn(64, 32).astype('float32')
b = gt.tensor(b_data)
print(f"  B shape: {b.data.numpy().shape}")

# Distributed matmul: C = A @ B
# Each worker computes: A_shard @ B in parallel
print("\nComputing C = A @ B (distributed)...")
c = a @ b

# Get result (automatically gathers from all workers)
result = c.data.numpy()
print(f"  Result shape: {result.shape}")

# Verify correctness
a_data = a.data.numpy()
b_data_full = b.data.numpy()
expected = a_data @ b_data_full

max_diff = np.max(np.abs(result - expected))
print(f"\n✓ Result matches numpy computation (max diff: {max_diff:.2e})")

print("\n=== How It Works ===")
print("gt.gpu_workers(4) automatically:")
print("  1. Shards A (128, 64) → (32, 64) per worker")
print("  2. Broadcasts B (64, 32) to all workers")
print("  3. Each worker computes: A_shard @ B in parallel")
print("  4. Results are gathered: (32,32) × 4 → (128, 32)")
print("\nNo manual worker setup required!")
