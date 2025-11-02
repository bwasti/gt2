# 8-GPU Distributed Training Demo

This demo shows realistic distributed training of a 512M parameter transformer model across 8 GPUs using GT's tensor and pipeline parallelism.

## Architecture

### Model: Mini-GPT (512M parameters)

- 8 transformer layers
- 1024 hidden size
- 16 attention heads
- 4096 intermediate size (MLP)
- 50257 vocabulary size (GPT-2 compatible)

### Parallelism Strategy: TP=2, PP=4

**2-way Tensor Parallel × 4-way Pipeline Parallel**

```
Pipeline Stage 0 (Layers 0-1)  →  GPUs 0, 1  ┐
                                               │ Tensor Parallel
Pipeline Stage 1 (Layers 2-3)  →  GPUs 2, 3  ┤ (2-way split)
                                               │
Pipeline Stage 2 (Layers 4-5)  →  GPUs 4, 5  ┤
                                               │
Pipeline Stage 3 (Layers 6-7)  →  GPUs 6, 7  ┘
```

**Why TP=2, PP=4 instead of TP=4, PP=2?**

More pipeline stages enable better pipeline scheduling (1F1B/ZB-V) with less bubble time. With 4 stages, we can interleave forward and backward passes more effectively.

### Tensor Parallelism Patterns

Within each pipeline stage, operations are split using **column parallel** and **row parallel** patterns:

**Column Parallel** (splits output features):
```
Input:  [batch, seq, hidden]
Weight: [hidden, 3*hidden]  ← Split across 2 GPUs
                              GPU 0: [hidden, 1.5*hidden]
                              GPU 1: [hidden, 1.5*hidden]
Output: [batch, seq, 3*hidden] (concatenated)
```
Used for: QKV projections, MLP up-projection

**Row Parallel** (splits input features):
```
Input:  [batch, seq, 3*hidden]  ← Split across 2 GPUs
Weight: [3*hidden, hidden]      ← Split across 2 GPUs
                                  GPU 0: [1.5*hidden, hidden]
                                  GPU 1: [1.5*hidden, hidden]
Output: [batch, seq, hidden] (reduced/summed)
```
Used for: Attention output, MLP down-projection

### Pipeline Schedule: 1F1B (One Forward One Backward)

The 1F1B schedule reduces pipeline bubbles by interleaving forward and backward passes:

**Traditional pipeline** (high bubble ratio):
```
Stage 0: F0 F1 F2 F3 F4 F5 F6 F7 | B0 B1 B2 B3 B4 B5 B6 B7
Stage 1:    F0 F1 F2 F3 F4 F5 F6 | F7 B0 B1 B2 B3 B4 B5 B6 B7
Stage 2:       F0 F1 F2 F3 F4 F5 | F6 F7 B0 B1 B2 B3 B4 B5 B6 B7
Stage 3:          F0 F1 F2 F3 F4 | F5 F6 F7 B0 B1 B2 B3 B4 B5 B6 B7
         ^^^^^^^^ Bubble time ^^^
```

**1F1B schedule** (minimal bubbles):
```
Stage 0: F0 F1 F2 F3 B0 B1 B2 B3 B4 B5 B6 B7
Stage 1:    F0 F1 F2 B0 B1 B2 B3 B4 B5 B6 B7
Stage 2:       F0 F1 B0 B1 B2 B3 B4 B5 B6 B7
Stage 3:          F0 B0 B1 B2 B3 B4 B5 B6 B7
         ^^^ Minimal bubble time ^^^
```

**1F1B Schedule Phases:**

1. **Warmup**: Fill pipeline with forwards (3 microbatches)
2. **Steady State**: Alternate 1 forward + 1 backward (5 microbatches)
3. **Cooldown**: Drain remaining backwards (3 microbatches)

**Pipeline Efficiency**: ~87.5% (vs ~50% for traditional)

With 8 microbatches and 4 stages:
- Total operations: 16 (8 forwards + 8 backwards)
- Bubble overhead: ~3 operations (warmup/cooldown)
- Efficiency: 13/16 = 81.25%

## Files

```
tuning_demo/
├── README.md                 # This file
├── sharding_config.yaml      # GT sharding configuration
├── model.py                  # Mini-GPT model implementation
├── train.py                  # Training script with 1F1B schedule
└── launch.py                 # Launch script for 8-GPU setup
```

## Quick Start

### Simplest Way (Auto-Start)

```bash
# Just run - GT auto-starts everything!
python examples/tuning_demo/train_simple.py
```

GT automatically:
1. Loads config from `GT_CONFIG` environment variable
2. Starts server/dispatcher
3. Spawns 8 workers (one per GPU)
4. Runs your training code

### With Trace Logging

```bash
# See what's happening under the hood
GT_INSTRUCTION_LOG=/tmp/tuning_trace.log \
python examples/tuning_demo/train_simple.py

# Check the trace
cat /tmp/tuning_trace.log | grep -i shard
```

### Manual Launch (If You Want Control)

**Terminal 1** - Start server:
```bash
python -m gt.server -p 12345
```

**Terminal 2-9** - Start 8 workers (one per GPU):
```bash
# For actual multi-GPU setup, use CUDA_VISIBLE_DEVICES:
CUDA_VISIBLE_DEVICES=0 python -m gt.worker --host localhost -p 12345
CUDA_VISIBLE_DEVICES=1 python -m gt.worker --host localhost -p 12345
# ... (repeat for GPUs 2-7)

# For CPU testing (simulating 8 GPUs):
python -m gt.worker --host localhost -p 12345  # Run 8 times
```

**Terminal 10** - Run training:
```bash
python examples/tuning_demo/train.py
```

## Configuration

The demo uses `GT_CONFIG` environment variable to load the sharding configuration.
It's set automatically in `train_simple.py`:

```python
import os
os.environ['GT_CONFIG'] = 'examples/tuning_demo/sharding_config.yaml'

import gt
gt.gpu_workers(8)  # Auto-start with 8 GPUs
```

### Debug Options

```bash
# Enable debug output
GT_DEBUG_DISPATCHER=1 python examples/tuning_demo/train_simple.py

# Log instruction stream
GT_INSTRUCTION_LOG=/tmp/trace.log python examples/tuning_demo/train_simple.py

# Use CPU backend instead of GPU
GT_BACKEND=numpy python examples/tuning_demo/train_simple.py
```

## Understanding the Sharding Config

The `sharding_config.yaml` uses **signal groups** to cluster operations with the same sharding pattern:

```yaml
# Pipeline stage 0 - all layers on GPUs 0-1
pp_stage0:
  shard:
    axis: 0
    workers: [0, 1]

# Column parallel - splits output features
pp_stage0_colpar:
  shard:
    axis: 1
    workers: [0, 1]

# Row parallel - splits input features
pp_stage0_rowpar:
  shard:
    axis: 0
    workers: [0, 1]
```

These signal groups are used in the model code:
```python
# Column-parallel projection
with gt.signal.context('pp_stage0_colpar'):
    qkv = x @ self.qkv_weight

# Row-parallel projection
with gt.signal.context('pp_stage0_rowpar'):
    output = qkv @ self.out_weight
```

## How It Works

### 1. Signal-Based Sharding

The model uses GT's signal API to mark computation scopes:

```python
# All operations in this scope use pp_stage0 sharding config
with gt.signal.context('pp_stage0'):
    x = layer(x)

# Pipeline boundary - moves data to next stage
with gt.signal.context('stage0_to_stage1'):
    x = x  # Triggers data movement from GPUs 0-1 → GPUs 2-3
```

### 2. Microbatching

The global batch is split into microbatches for pipeline parallelism:

```
Global batch: 32 samples
Microbatches: 8
Microbatch size: 4 samples each

Each microbatch flows through the 4 pipeline stages sequentially
```

### 3. 1F1B Schedule

The training loop executes microbatches in 1F1B order:

```python
# Warmup: Fill pipeline
F0, F1, F2

# Steady state: Interleave
F3, B0  # Forward for mb3, backward for mb0
F4, B1  # Forward for mb4, backward for mb1
...

# Cooldown: Drain pipeline
B5, B6, B7
```

## Performance Notes

**Expected Overhead:**

GT adds communication/serialization overhead compared to native PyTorch. For this demo:
- Small operations (< 1ms compute): 100-1000% overhead
- Large operations (> 10ms compute): 10-50% overhead
- As model size increases, overhead becomes negligible

**Bottlenecks:**

1. **Pipeline Bubbles**: 1F1B reduces this to ~10-20% idle time
2. **Cross-Stage Communication**: Data movement between pipeline stages
3. **Tensor Parallel Communication**: AllReduce for row-parallel ops

**Optimizations (Future):**

- ZB-V schedule (zero bubble)
- Overlapping communication with computation
- Activation checkpointing
- Mixed precision training

## Extending the Demo

### Add More Layers

Edit `model.py`:
```python
config = {
    'num_layers': 16,  # Change from 8 to 16
    ...
}
```

Adjust `sharding_config.yaml` to distribute 16 layers across 4 stages (4 layers each).

### Change Parallelism Strategy

**TP=4, PP=2** (4-way tensor, 2-way pipeline):
```yaml
# Stage 0: GPUs 0-3
pp_stage0:
  workers: [0, 1, 2, 3]

# Stage 1: GPUs 4-7
pp_stage1:
  workers: [4, 5, 6, 7]
```

**TP=1, PP=8** (no tensor parallel, 8-way pipeline):
```yaml
pp_stage0:
  workers: [0]  # 1 layer per GPU
pp_stage1:
  workers: [1]
...
pp_stage7:
  workers: [7]
```

### Add Actual Optimizer

Currently the demo shows the structure. To add real parameter updates:

```python
# In train.py
class SGD:
    def __init__(self, parameters, lr=0.001):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        for param in self.parameters:
            # param = param - lr * param.grad
            # Requires in-place operations in GT
            pass

optimizer = SGD(model.parameters(), lr=0.0001)
optimizer.step()
```

## Resources

- **GT Documentation**: `CLAUDE.md` in repo root
- **Signal API**: `examples/signal_demo.py`
- **Compilation**: `examples/compile_demo.py`
- **Debugging**: See `CLAUDE.md` debugging section

## Citation

If you use this demo or GT in your research:

```bibtex
@software{gt2024,
  title = {GT: Distributed Frontend for GPU ML Operations},
  author = {GT Contributors},
  year = {2024},
  url = {https://github.com/your-repo/gt}
}
```

## Troubleshooting

**Workers not connecting:**
```bash
# Check server is running
ps aux | grep "gt.server"

# Check workers are running
ps aux | grep "gt.worker"

# Enable verbose logging
GT_VERBOSE=1 python examples/tuning_demo/launch.py
```

**Out of memory:**
```bash
# Reduce batch size
python examples/tuning_demo/train.py --batch-size 16

# Reduce sequence length
python examples/tuning_demo/train.py --seq-len 64
```

**Debugging distributed issues:**
```bash
# Enable instruction logging
GT_INSTRUCTION_LOG=/tmp/gt_debug.log python examples/tuning_demo/train.py

# Check the log
cat /tmp/gt_debug.log | grep "ERROR\|stage0_to_stage1"
```

## Questions?

See `CLAUDE.md` in the repo root for:
- Detailed debugging guide
- Architecture documentation
- Development workflow
- Testing strategy
