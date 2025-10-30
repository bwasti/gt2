"""
Demo of signal-based configuration for sharding.

This demonstrates how to use signals to control tensor sharding across workers.
"""

import os
os.environ['GT_CONFIG'] = 'examples/config_sharding.yaml'

import gt
import numpy as np

print("=" * 60)
print("Signal-Based Sharding Demo")
print("=" * 60)

# Load sharding configuration
gt.load_config('examples/config_sharding.yaml')

print("\n1. Context Manager API - Shards tensors AND compute")
print("-" * 60)

with gt.signal.signal('forward_layer1'):
    # These tensors will be created according to 'forward_layer1' config
    # (sharded across workers [0,1,2,3] along axis 0)
    a = gt.randn(100, 64)
    b = gt.randn(100, 64)

    # This computation happens in sharded mode
    c = a + b
    d = c * 2.0

    print(f"Created tensors within 'forward_layer1' signal scope")
    print(f"  a.shape = {a.shape}, b.shape = {b.shape}")
    print(f"  c.shape = {c.shape}, d.shape = {d.shape}")

print("\n2. Function Call API - Only copy-shards the tensor")
print("-" * 60)

# Create a tensor without signal
x = gt.randn(50, 32)
print(f"Created x without signal: shape = {x.shape}")

# Now apply signal to copy-shard it
x_sharded = gt.signal_tensor(x, name='feature_parallel')
print(f"Applied 'feature_parallel' signal to x")
print(f"  x_sharded will be sharded along axis 1 across workers [0,1]")

print("\n3. Backward Pass Signals")
print("-" * 60)

with gt.signal.signal('pipeline_stage_1', backward='pipeline_stage_1_bwd'):
    # Forward pass uses 'pipeline_stage_1' config (workers [0,1])
    # Backward pass will use 'pipeline_stage_1_bwd' config (workers [2,3])
    y = gt.randn(32, 16)
    z = y * 3.0

    print(f"Created tensors with forward/backward signal split")
    print(f"  Forward:  pipeline_stage_1 (workers [0,1])")
    print(f"  Backward: pipeline_stage_1_bwd (workers [2,3])")

print("\n4. Alternative API - enter/exit")
print("-" * 60)

gt.signal.enter('replicated_weights')
# Code here uses 'replicated_weights' config
w = gt.randn(64, 64)
print(f"Created w within 'replicated_weights' signal scope")
print(f"  Will be replicated across all workers")
gt.signal.exit('replicated_weights')

print("\n" + "=" * 60)
print("Demo completed!")
print("=" * 60)

print("\nNote: Actual sharding logic is not yet implemented.")
print("This demo shows the API for specifying sharding configuration.")
print("The dispatcher will use this information to shard tensors across workers.")
