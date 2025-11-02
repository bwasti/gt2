"""
Example: Programmatic Configuration API

Demonstrates how to configure signal-based sharding from Python
instead of using YAML files.

This is useful for:
- Dynamic configuration based on runtime conditions
- Programmatic experimentation with different sharding strategies
- Embedded configurations without external files
"""

import gt
from gt import SignalConfig, ShardConfig

# Configure multiple workers for this demo
gt.gpu_workers(4)

print("=" * 70)
print("GT Example: Programmatic Configuration API")
print("=" * 70)
print()

# Example 1: Simple data parallelism
print("1. Data Parallelism Configuration")
print("-" * 70)

# Create configuration for data-parallel training
data_parallel_config = SignalConfig(
    shard=ShardConfig(
        axis=0,           # Shard along batch dimension
        workers=[0, 1, 2, 3]  # Use all 4 workers
    ),
    compile=False      # Disable compilation for this example
)

# Register the configuration
gt.register_config('data_parallel', data_parallel_config)

# Use the configuration
print("Creating sharded tensor with data_parallel config...")
with gt.signal.context('data_parallel'):
    x = gt.randn(128, 64)  # Batch size 128, sharded across 4 workers (32 per worker)
    print(f"Tensor shape: {x.data.numpy().shape}")  # Should be (128, 64)

print()

# Example 2: Pipeline parallelism with separate forward/backward configs
print("2. Pipeline Parallelism Configuration")
print("-" * 70)

# Stage 1: Forward on workers [0, 1], backward on workers [2, 3]
stage1_forward = SignalConfig(
    shard=ShardConfig(axis=0, workers=[0, 1]),
    backward_signal='stage1_backward',
    compile=False
)

stage1_backward = SignalConfig(
    shard=ShardConfig(axis=0, workers=[2, 3]),
    compile=False
)

# Register both configs
gt.register_config('stage1_forward', stage1_forward)
gt.register_config('stage1_backward', stage1_backward)

print("Forward pass on workers [0, 1], backward on workers [2, 3]...")
with gt.signal.context('stage1_forward'):
    a = gt.randn(100, 50, requires_grad=True)
    b = a.relu()
    loss = b.sum()

print("Computing gradients...")
loss.backward()
print(f"Gradient shape: {a.grad.data.numpy().shape}")

print()

# Example 3: Model parallelism (feature sharding)
print("3. Model Parallelism Configuration")
print("-" * 70)

model_parallel_config = SignalConfig(
    shard=ShardConfig(
        axis=1,           # Shard along feature dimension
        workers=[0, 1]    # Use 2 workers
    )
)

gt.register_config('model_parallel', model_parallel_config)

print("Creating sharded weight matrix (model parallelism)...")
with gt.signal.context('model_parallel'):
    weight = gt.randn(1024, 512)  # Features sharded across 2 workers
    print(f"Weight shape: {weight.data.numpy().shape}")

print()

# Example 4: Dynamic configuration based on tensor size
print("4. Dynamic Configuration (Size-Based)")
print("-" * 70)

def configure_based_on_size(batch_size: int):
    """Dynamically configure sharding based on batch size."""
    if batch_size >= 256:
        # Large batch: use all workers
        config = SignalConfig(
            shard=ShardConfig(axis=0, workers=[0, 1, 2, 3]),
            compile=False
        )
        gt.register_config('dynamic', config)
        print(f"Batch size {batch_size}: Using 4 workers")
    elif batch_size >= 128:
        # Medium batch: use 2 workers
        config = SignalConfig(
            shard=ShardConfig(axis=0, workers=[0, 1]),
            compile=False
        )
        gt.register_config('dynamic', config)
        print(f"Batch size {batch_size}: Using 2 workers")
    else:
        # Small batch: single worker
        config = SignalConfig(
            shard=ShardConfig(axis=0, workers=[0]),
            compile=False
        )
        gt.register_config('dynamic', config)
        print(f"Batch size {batch_size}: Using 1 worker")

# Test with different batch sizes
for batch_size in [64, 128, 256]:
    configure_based_on_size(batch_size)
    with gt.signal.context('dynamic'):
        x = gt.randn(batch_size, 32)
        print(f"  Created tensor: {x.data.numpy().shape}")

print()

# Example 5: Combining Python and YAML configs
print("5. Mixing Python and YAML Configurations")
print("-" * 70)

# You can register Python configs alongside YAML configs
# If GT_CONFIG environment variable is set, YAML configs are loaded
# Python configs registered with gt.register_config() are added/override

python_config = SignalConfig(
    shard=ShardConfig(axis=0, workers=[0, 1, 2, 3]),
    compile=False
)

gt.register_config('python_defined', python_config)

print("Registered 'python_defined' config from Python")
print("This works alongside any YAML configs loaded via GT_CONFIG")

with gt.signal.context('python_defined'):
    result = gt.randn(200, 100)
    print(f"Result shape: {result.data.numpy().shape}")

print()
print("=" * 70)
print("Complete!")
print()
print("Key Takeaways:")
print("- Use SignalConfig and ShardConfig to define configurations in Python")
print("- Register configs with gt.register_config(name, config)")
print("- Use configs with gt.signal.context(name)")
print("- Python configs can be dynamic, conditional, and programmatic")
print("- Mix Python and YAML configs as needed")
print("=" * 70)
