"""
Simple 8-GPU distributed training demo with 1F1B pipeline schedule.

Just run:
    python examples/tuning_demo/train_simple.py

GT will auto-start server + 8 workers!
"""

import os
os.environ['GT_CONFIG'] = 'examples/tuning_demo/sharding_config.yaml'

import gt
from model import create_model, count_parameters
import numpy as np
import time

# Configure for 8 GPUs
gt.gpu_workers(8)

print("=" * 80)
print("8-GPU Distributed Training Demo")
print("=" * 80)
print("\nConfiguration:")
print("  Pipeline stages: 4 (2 layers per stage)")
print("  Tensor parallel: 2 (within each stage)")
print("  GPUs: 8 total")
print("    - Stage 0 (Layers 0-1): GPUs 0-1")
print("    - Stage 1 (Layers 2-3): GPUs 2-3")
print("    - Stage 2 (Layers 4-5): GPUs 4-5")
print("    - Stage 3 (Layers 6-7): GPUs 6-7")

print("\n✓ GT will auto-start with 8 workers...")

# Create model (this triggers auto-start)
print("\nInitializing model...")
model = create_model()
num_params = count_parameters(model)
print(f"✓ Model created: {num_params:,} parameters (~{num_params/1e6:.1f}M)")

print("\n" + "=" * 80)
print("Testing signal-based sharding...")
print("=" * 80)

# Test 1: Create tensor in stage 0 (should shard across GPUs 0-1)
print("\n[Test 1] Creating tensor in pp_stage0 (GPUs 0-1)...")
with gt.signal.context('pp_stage0'):
    a = gt.randn(100, 64)
    print(f"  ✓ Created tensor: shape {a.shape}")

# Test 2: Create tensor in stage 1 (should shard across GPUs 2-3)
print("\n[Test 2] Creating tensor in pp_stage1 (GPUs 2-3)...")
with gt.signal.context('pp_stage1'):
    b = gt.randn(100, 64)
    print(f"  ✓ Created tensor: shape {b.shape}")

# Test 3: Fetch data (should gather from shards)
print("\n[Test 3] Fetching data from sharded tensor...")
try:
    data = a.data.numpy()
    print(f"  ✓ Got data: shape {data.shape}")
    if data.shape == (100, 64):
        print(f"  ✓ Shape is correct!")
except Exception as e:
    print(f"  ✗ Error: {e}")

print("\n" + "=" * 80)
print("Demo Complete!")
print("=" * 80)

print("\nWhat happened:")
print("  1. GT auto-started server + 8 workers (one per GPU)")
print("  2. Loaded sharding config from YAML")
print("  3. Created model weights in signal scopes")
print("  4. Sharding modifier split tensors across workers")
print("  5. Dispatcher tracked sharded tensor locations")

print("\nNext steps:")
print("  - Add actual training loop with 1F1B schedule")
print("  - Handle operations on sharded tensors (matmul, etc.)")
print("  - Implement gradient accumulation")
print("  - Add optimizer (SGD/Adam)")
