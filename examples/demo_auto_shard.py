"""
Demo showing automatic tensor sharding with GT_AUTO_SHARD=1.

This example demonstrates how GT automatically distributes tensors across
multiple GPUs without requiring manual sharding configuration.

Requirements:
- Multiple GPU workers connected to the dispatcher

Setup:
    # Terminal 1: Start dispatcher
    python -m gt.server -p 9000

    # Terminal 2-N: Start workers (one per GPU)
    CUDA_VISIBLE_DEVICES=0 python -m gt.worker --host localhost -p 9000
    CUDA_VISIBLE_DEVICES=1 python -m gt.worker --host localhost -p 9000
    CUDA_VISIBLE_DEVICES=2 python -m gt.worker --host localhost -p 9000
    # ... etc

    # Terminal N+1: Run this demo
    python examples/demo_auto_shard.py
"""

import os
import numpy as np

# Enable automatic sharding
os.environ['GT_AUTO_SHARD'] = '1'
os.environ['GT_VERBOSE'] = '1'

import gt

print("\n" + "="*80)
print("GT_AUTO_SHARD Demo")
print("="*80)
print("\nThis demo shows automatic tensor sharding across multiple GPUs.")
print("With GT_AUTO_SHARD=1, tensors are automatically distributed without")
print("requiring manual sharding configuration.\n")

# Connect to GT system
print("Connecting to GT system...")
gt.connect('localhost:9000')

# Initialize
print("Initializing system...")
_ = gt.zeros(1, 1)

# Get worker stats to see how many workers we have
stats = gt.debug.get_worker_stats()
num_workers = len(stats)
print(f"\n✓ Connected to {num_workers} worker(s)\n")

if num_workers == 1:
    print("⚠️  Only 1 worker detected. For best results, connect multiple workers.")
    print("   Each tensor will be on a single worker (no sharding).\n")
else:
    print(f"✓ With {num_workers} workers, tensors will be automatically sharded!\n")

print("="*80)
print("Example 1: Large Batch Matrix Multiplication")
print("="*80)

# Create large batch tensor (will be sharded across workers)
batch_size = 16384
hidden_dim = 4096

print(f"\nCreating tensor A with shape ({batch_size}, {hidden_dim})...")
if num_workers > 1:
    shard_size = batch_size // num_workers
    print(f"→ Auto-sharded as ({shard_size}, {hidden_dim}) per worker")
a = gt.randn(batch_size, hidden_dim)

print(f"\nCreating tensor B with shape ({hidden_dim}, {hidden_dim})...")
if num_workers > 1:
    print(f"→ Auto-sharded as ({shard_size}, {hidden_dim}) per worker")
b = gt.randn(hidden_dim, hidden_dim)

print(f"\nComputing C = A @ B...")
if num_workers > 1:
    print(f"→ Each worker computes its shard in parallel")
    print(f"→ Results are automatically gathered")
c = a @ b

print(f"\nFetching result...")
result = c.data.numpy()
print(f"✓ Result shape: {result.shape}")
print(f"✓ Expected shape: ({batch_size}, {hidden_dim})")

print("\n" + "="*80)
print("Example 2: Element-wise Operations")
print("="*80)

print(f"\nCreating tensor X with shape ({batch_size}, {hidden_dim})...")
x = gt.randn(batch_size, hidden_dim)

print(f"\nApplying element-wise operations: relu(X) * 2.0 + 1.0")
if num_workers > 1:
    print(f"→ Each operation runs on shards independently")
y = x.relu() * 2.0 + 1.0

result = y.data.numpy()
print(f"✓ Result shape: {result.shape}")

print("\n" + "="*80)
print("Example 3: Creating Tensor from NumPy Array")
print("="*80)

print(f"\nCreating NumPy array with shape ({batch_size}, 128)...")
np_data = np.random.randn(batch_size, 128).astype(np.float32)
print(f"Data range: [{np_data.min():.2f}, {np_data.max():.2f}]")

print(f"\nCopying to GT (will be automatically sharded)...")
if num_workers > 1:
    shard_size = batch_size // num_workers
    print(f"→ Data split into {num_workers} shards of ({shard_size}, 128)")
z = gt.from_numpy(np_data)

print(f"\nFetching back from GT...")
result = z.data.numpy()
print(f"✓ Result shape: {result.shape}")
print(f"✓ Data matches: {np.allclose(result, np_data)}")

print("\n" + "="*80)
print("Example 4: MLP Forward Pass")
print("="*80)

from gt.client import nn

print(f"\nCreating 3-layer MLP: {hidden_dim} -> 8192 -> 4096 -> 2048")
fc1 = nn.Linear(hidden_dim, 8192)
fc2 = nn.Linear(8192, 4096)
fc3 = nn.Linear(4096, 2048)

print(f"\nForward pass with batch size {batch_size}...")
if num_workers > 1:
    print(f"→ Input automatically sharded as ({batch_size // num_workers}, {hidden_dim}) per worker")

x = gt.randn(batch_size, hidden_dim)
h1 = fc1(x).relu()
print(f"  After layer 1: shape {h1.data.numpy().shape}")

h2 = fc2(h1).relu()
print(f"  After layer 2: shape {h2.data.numpy().shape}")

out = fc3(h2)
result = out.data.numpy()
print(f"  Output: shape {result.shape}")
print(f"✓ Final shape: {result.shape}")

print("\n" + "="*80)
print("Summary")
print("="*80)

# Show statistics
print(f"\nWorker Statistics:")
stats = gt.debug.get_worker_stats()
for worker_id, worker_stats in stats.items():
    total_ops = worker_stats.get('operations', {}).get('total', 0)
    print(f"  {worker_id}: {total_ops} operations")

print(f"\n✓ All examples completed successfully!")
print(f"\nWith GT_AUTO_SHARD=1:")
print(f"  • Tensors automatically distributed across {num_workers} worker(s)")
print(f"  • Operations run in parallel on each shard")
print(f"  • Results automatically gathered when needed")
print(f"  • No manual sharding configuration required!")

if num_workers == 1:
    print(f"\n💡 Tip: Start more workers to see automatic sharding in action!")
    print(f"   Example: CUDA_VISIBLE_DEVICES=1 python -m gt.worker --host localhost -p 9000")

print("\n" + "="*80)
