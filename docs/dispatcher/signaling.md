# Signal-Based Sharding

Control tensor placement and compute distribution across workers using named signals and YAML configuration.

## Overview

Signals allow you to declaratively specify:
- Which workers to use for operations
- How to shard tensors across workers (data/model parallelism)
- Different configurations for forward and backward passes (pipeline parallelism)
- Compilation boundaries for torch.compile

## Quick Example

```yaml
# config.yaml
forward_layer1:
  shard:
    axis: 0                  # Shard batch dimension
    workers: [0, 1, 2, 3]    # Use workers 0-3
  backward: backward_layer1  # Different config for gradients

backward_layer1:
  shard:
    axis: 0
    workers: [0, 1, 2, 3]
```

```python
import os
os.environ['GT_CONFIG'] = 'config.yaml'
import gt

# Shards tensors AND compute across workers [0,1,2,3]
with gt.signal.context('forward_layer1'):
    x = gt.randn(100, 64)  # Sharded on axis 0
    y = x @ w              # Compute in sharded mode
```

## Configuration Format

### Basic Sharding

```yaml
signal_name:
  shard:
    axis: 0              # Which axis to shard (0 = batch, 1 = features)
    workers: [0, 1]      # List of worker IDs to use
```

### Pipeline Parallelism

```yaml
forward_stage:
  shard:
    axis: 0
    workers: [0, 1]
  backward: backward_stage  # Use different workers for gradients

backward_stage:
  shard:
    axis: 0
    workers: [2, 3]          # Backward on different workers
```

### With Compilation

```yaml
hot_path:
  shard:
    axis: 0
    workers: [0, 1, 2, 3]
  compile: true              # Enable torch.compile for this signal
```

## Python Configuration API

Instead of YAML files, you can configure signals programmatically from Python:

```python
import gt
from gt import SignalConfig, ShardConfig

# Create configuration
config = SignalConfig(
    shard=ShardConfig(
        axis=0,
        workers=[0, 1, 2, 3]
    ),
    backward_signal='backward_layer1',
    compile=False
)

# Register it
gt.register_config('forward_layer1', config)

# Use it
with gt.signal.context('forward_layer1'):
    x = gt.randn(128, 64)
```

### Benefits of Python API

- **Dynamic configuration** - Adjust based on runtime conditions
- **Programmatic experimentation** - Easy to test different strategies
- **No external files** - Embedded configurations in code
- **Mix with YAML** - Python configs work alongside YAML

### Examples

**Data Parallelism:**
```python
from gt import SignalConfig, ShardConfig

config = SignalConfig(
    shard=ShardConfig(axis=0, workers=[0, 1, 2, 3])
)
gt.register_config('data_parallel', config)
```

**Pipeline Parallelism:**
```python
# Forward on workers [0, 1]
forward_config = SignalConfig(
    shard=ShardConfig(axis=0, workers=[0, 1]),
    backward_signal='backward_stage'
)
gt.register_config('forward_stage', forward_config)

# Backward on workers [2, 3]
backward_config = SignalConfig(
    shard=ShardConfig(axis=0, workers=[2, 3])
)
gt.register_config('backward_stage', backward_config)
```

**Dynamic Configuration:**
```python
def configure_for_batch_size(batch_size):
    if batch_size >= 256:
        workers = [0, 1, 2, 3]  # All workers
    else:
        workers = [0, 1]  # Fewer workers

    config = SignalConfig(
        shard=ShardConfig(axis=0, workers=workers)
    )
    gt.register_config('training', config)

# Adjust based on batch size
configure_for_batch_size(batch_size)
with gt.signal.context('training'):
    output = model(batch)
```

See `examples/config_from_python.py` for complete examples.

## API Reference

### Context Manager (Recommended)

Shards both tensors and compute:

```python
with gt.signal.context('signal_name'):
    # All tensor creations are sharded
    x = gt.randn(100, 64)

    # All operations use sharded tensors
    y = x @ weight
    z = y.relu()
```

With backward signal:

```python
with gt.signal.context('forward', backward='backward'):
    loss = model(input)
    # Forward uses 'forward' config

    loss.backward()
    # Backward uses 'backward' config
```

### Function Call

Shard a single tensor (copy-shard):

```python
# Create tensor normally
x = gt.randn(100, 64)

# Shard it according to signal config
x_sharded = gt.signal.tensor(x, name='signal_name')
```

### Manual Enter/Exit

For more control:

```python
# Enter signal scope
gt.signal.enter('signal_name')

# Operations in this scope use signal config
x = gt.randn(100, 64)

# Exit signal scope
gt.signal.exit('signal_name')
```

### Loading Configuration

**Option 1: YAML File**

```python
# Environment variable (recommended)
import os
os.environ['GT_CONFIG'] = 'config.yaml'
import gt

# Explicit load
import gt
gt.load_config('config.yaml')
```

**Option 2: Python API**

```python
import gt
from gt import SignalConfig, ShardConfig

# Register configuration
config = SignalConfig(
    shard=ShardConfig(axis=0, workers=[0, 1, 2, 3])
)
gt.register_config('my_signal', config)
```

**Option 3: Mix Both**

Python configs work alongside YAML - use YAML for static configs and Python for dynamic ones:

```python
# Load YAML base config
gt.load_config('base_config.yaml')

# Override or add with Python
dynamic_config = SignalConfig(
    shard=ShardConfig(axis=0, workers=[0, 1])
)
gt.register_config('dynamic_layer', dynamic_config)
```

## Sharding Strategies

### Data Parallelism

Shard batch dimension across all workers:

```yaml
data_parallel:
  shard:
    axis: 0                    # Batch dimension
    workers: [0, 1, 2, 3]      # All workers
```

```python
with gt.signal.context('data_parallel'):
    # Batch size 128, sharded as 32 per worker
    batch = gt.randn(128, 784)
    output = model(batch)
    loss = loss_fn(output, targets)
    loss.backward()
```

### Model Parallelism

Shard feature dimension:

```yaml
model_parallel:
  shard:
    axis: 1                    # Feature dimension
    workers: [0, 1]
```

```python
with gt.signal.context('model_parallel'):
    # Hidden size 1024, sharded as 512 per worker
    weight = gt.randn(1024, 512)
```

### Pipeline Parallelism

Different workers for different stages:

```yaml
stage_1:
  shard:
    axis: 0
    workers: [0, 1]
  backward: stage_1_backward

stage_1_backward:
  shard:
    axis: 0
    workers: [2, 3]

stage_2:
  shard:
    axis: 0
    workers: [2, 3]
  backward: stage_2_backward

stage_2_backward:
  shard:
    axis: 0
    workers: [0, 1]
```

```python
# Forward on workers [0,1], backward on [2,3]
with gt.signal.context('stage_1'):
    hidden1 = layer1(input)

# Forward on workers [2,3], backward on [0,1]
with gt.signal.context('stage_2'):
    output = layer2(hidden1)
```

### Hybrid: Data + Model Parallelism

```yaml
layer1_data_parallel:
  shard:
    axis: 0                    # Data parallel
    workers: [0, 1, 2, 3]

layer2_model_parallel:
  shard:
    axis: 1                    # Model parallel
    workers: [0, 1]
```

```python
# Data parallel for layer 1
with gt.signal.context('layer1_data_parallel'):
    hidden = layer1(input)

# Model parallel for layer 2
with gt.signal.context('layer2_model_parallel'):
    output = layer2(hidden)
```

## Distributed Operations

### Embarrassingly Parallel Matmul

When left operand (A) is sharded:

```python
# A sharded on axis 0, B not sharded
# Each worker computes: A_shard @ B
# Result is sharded on axis 0 (same as A)

with gt.signal.context('data_parallel'):
    A = gt.randn(128, 64)     # Sharded: 32 per worker
    B = gt.randn(64, 32)      # Not sharded (replicated)
    C = A @ B                 # Result sharded: 32 rows per worker
```

### All-Gather Pattern

When both operands are sharded:

```python
# Both A and B sharded on axis 0
# Dispatcher all-gathers B, then each worker computes: A_shard @ B_full

with gt.signal.context('both_sharded'):
    A = gt.randn(128, 64)     # Sharded
    B = gt.randn(64, 32)      # Sharded
    C = A @ B                 # All-gather B, compute sharded matmul
```

### Distributed Reductions

Reductions (sum, mean) on sharded tensors:

```python
with gt.signal.context('data_parallel'):
    x = gt.randn(128, 64)     # Sharded across workers
    total = x.sum()           # All-reduce: sum partial results
    avg = x.mean()            # Weighted average across shards
```

## Complete Example

```yaml
# training_config.yaml
forward_pass:
  shard:
    axis: 0
    workers: [0, 1, 2, 3]
  backward: backward_pass
  compile: true

backward_pass:
  shard:
    axis: 0
    workers: [0, 1, 2, 3]
  compile: true

parameter_updates:
  shard:
    axis: 0
    workers: [0, 1, 2, 3]
```

```python
import os
os.environ['GT_CONFIG'] = 'training_config.yaml'
import gt
from gt.nn import Module, Linear, SGD

class Model(Module):
    def __init__(self):
        super().__init__()
        self.layer = Linear(784, 10)

    def forward(self, x):
        return self.layer(x)

model = Model()
optimizer = SGD(model.parameters(), lr=0.01)

# Training loop with sharding
for epoch in range(100):
    # Forward pass - data parallel
    with gt.signal.context('forward_pass'):
        pred = model(X_train)
        loss = ((pred - y_train) ** 2).mean()

    # Backward pass - same sharding
    loss.backward()

    # Parameter updates - data parallel
    with gt.signal.context('parameter_updates'):
        optimizer.step()
        optimizer.zero_grad()
```

## Best Practices

1. **Use context managers** - Clearer than manual enter/exit
2. **Name signals descriptively** - `forward_layer1` not `signal1`
3. **Match forward/backward sharding** - Usually same worker set
4. **Start simple** - Data parallelism first, then optimize
5. **Profile first** - Use monitoring tools to identify bottlenecks
6. **Test locally** - Verify with `gt.gpu_workers(4)` before cluster deployment

## Debugging

### View Applied Sharding

```bash
# Enable dispatcher debug output
GT_DEBUG_DISPATCHER=1 python train.py
```

### Verify Tensor Placement

```bash
# Log all instructions
GT_INSTRUCTION_LOG=debug.log python train.py

# Check which workers received tensors
grep "WorkerCreateTensor" debug.log
```

### Visualize Distribution

```bash
# Capture trace
python -m gt.scripts.trace -s 5 --dir traces/

# Generate timeline
python -m gt.scripts.visualize traces/trace_*.log --output timeline.png
```

## Next Steps

- [Tuning Guide](tuning.md) - Performance optimization
- [Monitoring](monitoring.md) - Real-time system monitoring
- [Examples](../../examples/README_SIGNALS.md) - Complete examples
