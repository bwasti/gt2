# Signal-Based Sharding Configuration

This guide shows how to use GT's signal-based sharding API to control tensor placement across workers.

## Quick Start

### 1. Create a Config File (`config.yaml`)

```yaml
# Shard layer 1 across all 4 workers
forward_layer1:
  shard:
    axis: 0              # Shard along batch dimension
    workers: [0, 1, 2, 3]  # Use workers 0-3
  backward: backward_layer1  # Use this config for gradients

backward_layer1:
  shard:
    axis: 0
    workers: [0, 1, 2, 3]

# Pipeline parallelism - different workers for different stages
pipeline_stage_1:
  shard:
    axis: 0
    workers: [0, 1]      # First 2 workers
  backward: pipeline_stage_1_bwd

pipeline_stage_1_bwd:
  shard:
    axis: 0
    workers: [2, 3]      # Last 2 workers (backward on different GPUs)
```

### 2. Load Config and Use Signals

```python
import os
# Option 1: Set GT_CONFIG environment variable (auto-loaded on import)
os.environ['GT_CONFIG'] = 'config.yaml'
import gt

# Option 2: Load config explicitly
# gt.load_config('config.yaml')

# Context manager - shards tensors AND compute
with gt.signal.context('forward_layer1'):
    x = gt.randn(100, 64)  # Sharded per config
    y = x + 1.0            # Compute happens in sharded mode

# Function call - only copy-shards the tensor
x_sharded = gt.signal.tensor(x, name='forward_layer1')

# Alternative enter/exit API
gt.signal.enter('forward_layer1')
z = gt.randn(50, 32)
gt.signal.exit('forward_layer1')
```

## API Reference

### Three Ways to Use Signals

#### 1. Context Manager (Recommended)
```python
with gt.signal.context('signal_name'):
    # Tensors created here inherit the signal
    # Operations inherit the signal
    x = gt.randn(100, 64)
    y = x + 1.0
```

**Use when**: You want both tensor creation AND compute to be sharded.

#### 2. Function Call
```python
x = gt.randn(100, 64)  # Created without signal
x_sharded = gt.signal.tensor(x, name='signal_name')
```

**Use when**: You want to copy-shard an existing tensor without affecting compute.

#### 3. Enter/Exit API
```python
gt.signal.enter('signal_name')
x = gt.randn(100, 64)
gt.signal.exit('signal_name')
```

**Use when**: Context manager is awkward (e.g., in class methods).

### Backward Pass Handling

Specify different sharding for forward and backward passes:

```python
with gt.signal.context('forward_signal', backward='backward_signal'):
    loss = model(input)
    loss.backward()  # Gradients use 'backward_signal' config
```

This enables pipeline parallelism where forward and backward passes run on different workers.

## Config File Format

### Basic Structure

```yaml
signal_name:
  shard:
    axis: <int>              # Which axis to shard (0=batch, 1=features, etc.)
    workers: [<int>, ...]    # List of worker IDs (null = all workers)
    replicated: <bool>       # If true, replicate instead of sharding
  backward: <str>            # Optional: signal name for backward pass
```

### Common Patterns

#### Data Parallelism (Batch Sharding)
```yaml
data_parallel:
  shard:
    axis: 0                  # Shard batch dimension
    workers: [0, 1, 2, 3]    # All workers
```

#### Model Parallelism (Feature Sharding)
```yaml
model_parallel:
  shard:
    axis: 1                  # Shard feature dimension
    workers: [0, 1]
```

#### Pipeline Parallelism (Different Workers for Stages)
```yaml
stage_1_fwd:
  shard:
    axis: 0
    workers: [0, 1]
  backward: stage_1_bwd

stage_1_bwd:
  shard:
    axis: 0
    workers: [2, 3]          # Different workers for backward

stage_2_fwd:
  shard:
    axis: 0
    workers: [2, 3]          # Picks up where stage 1 left off
  backward: stage_2_bwd

stage_2_bwd:
  shard:
    axis: 0
    workers: [0, 1]
```

#### Replicated Parameters
```yaml
replicated_weights:
  shard:
    workers: null            # All workers
    replicated: true         # Don't shard, replicate
```

## Environment Variables

- `GT_CONFIG`: Path to YAML config file (auto-loaded on `import gt`)
- `GT_INSTRUCTION_LOG`: Path to instruction stream log file (optional)
- `GT_WORKER_BATCH_SIZE`: Batch size for instruction batching (default: 1)

## Example: Training with Signals

```python
import os
os.environ['GT_CONFIG'] = 'my_sharding.yaml'

import gt
from gt.nn import Module, Linear

class MyModel(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(784, 256)
        self.fc2 = Linear(256, 10)

    def forward(self, x):
        with gt.signal.context('layer1'):
            x = self.fc1(x).relu()

        with gt.signal.context('layer2'):
            x = self.fc2(x)

        return x

# Config file (my_sharding.yaml):
# layer1:
#   shard:
#     axis: 0
#     workers: [0, 1]
#
# layer2:
#   shard:
#     axis: 0
#     workers: [2, 3]

model = MyModel()
optimizer = gt.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(10):
    # Forward pass uses signal configs
    output = model(input)
    loss = ((output - target) ** 2).mean()

    # Backward pass uses backward signal configs (if specified)
    loss.backward()
    optimizer.step()
```

## Current Status

**Implemented:**
- ✅ Signal API (context manager, function call, enter/exit)
- ✅ YAML config loading
- ✅ Signal metadata propagation to dispatcher
- ✅ Auto-loading from GT_CONFIG environment variable
- ✅ Backward signal tracking

**Not Yet Implemented:**
- ⏳ Dispatcher-side sharding logic (reads config, creates sharded tensors)
- ⏳ Gradient signal application in autograd backward pass
- ⏳ Actual tensor data sharding across workers

The API is complete and functional. The dispatcher will use signal metadata
to make sharding decisions once the sharding implementation is added.

## Next Steps

See the full demo in `examples/signal_demo.py`:
```bash
python examples/signal_demo.py
```

For more information on instruction batching and torch.compile:
```bash
cat docs/BATCHING_AND_COMPILATION.md
```
