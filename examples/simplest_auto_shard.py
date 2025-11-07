"""
Simplest possible GT_AUTO_SHARD example.

This is the EASIEST way to use GT with multiple GPUs:
1. Set GT_AUTO_SHARD=1
2. Just use gt.tensor() - everything is automatic!

No server setup needed. No worker management. Just pure ML code.
"""

import os
os.environ['GT_AUTO_SHARD'] = '1'  # Enable auto-sharding
os.environ['GT_VERBOSE'] = '1'     # See what's happening

import gt

print("\n" + "="*80)
print("Simplest GT_AUTO_SHARD Example")
print("="*80)
print("\nJust set GT_AUTO_SHARD=1 and use gt.tensor()!")
print("GT automatically:")
print("  • Detects all your GPUs")
print("  • Starts workers on each GPU")
print("  • Shards tensors across all GPUs")
print("  • Runs operations in parallel")
print("\n" + "="*80 + "\n")

# Example 1: Large batch matmul
print("Example 1: Large Batch Matmul\n")

# Just create tensors - GT handles everything!
a = gt.randn(16384, 4096)  # Automatically sharded across all GPUs
b = gt.randn(4096, 2048)
c = a @ b                  # Runs in parallel on all GPUs

result = c.data.numpy()
print(f"✓ Result shape: {result.shape}")

# Example 2: MLP forward pass
print("\n" + "="*80)
print("Example 2: MLP Forward Pass\n")

from gt.client import nn

# Define a simple 3-layer MLP
fc1 = nn.Linear(4096, 8192)
fc2 = nn.Linear(8192, 4096)
fc3 = nn.Linear(4096, 2048)

# Forward pass with large batch
x = gt.randn(16384, 4096)  # Automatically sharded
h1 = fc1(x).relu()
h2 = fc2(h1).relu()
out = fc3(h2)

result = out.data.numpy()
print(f"✓ Output shape: {result.shape}")

# Show statistics
print("\n" + "="*80)
print("Worker Statistics\n")

stats = gt.debug.get_worker_stats()
for worker_id, worker_stats in sorted(stats.items()):
    total_ops = worker_stats.get('operations', {}).get('total', 0)
    print(f"  {worker_id}: {total_ops} operations")

num_workers = len(stats)
print(f"\n✓ Used {num_workers} worker(s) (one per GPU)")

print("\n" + "="*80)
print("Complete!")
print("="*80)
print("\nThat's it! No server setup, no worker management.")
print("Just GT_AUTO_SHARD=1 and pure ML code. 🚀\n")
