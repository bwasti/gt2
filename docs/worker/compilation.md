# Compilation

GT supports JIT (Just-In-Time) compilation of repeated operation patterns using PyTorch's `torch.compile`.

## Overview

Compilation improves performance by:
1. **Detecting hot paths** - Identifying frequently executed operation sequences
2. **Fusing operations** - Combining multiple ops into single kernel
3. **Optimizing memory** - Reducing intermediate allocations
4. **GPU kernel tuning** - Auto-tuning for specific hardware

## Enabling Compilation

### Automatic Hot Path Detection

Enable automatic detection and compilation:

```bash
GT_AUTO_COMPILE=1 python train.py
```

GT automatically:
- Tracks operation sequences
- Identifies patterns executed 10+ times
- Compiles hot paths with `torch.compile`
- Reports compilation activity

### Force Compile Everything

Compile all operations (not recommended for development):

```bash
GT_COMPILE=1 python train.py
```

### YAML Configuration

Specify compilation boundaries in config:

```yaml
# config.yaml
training_loop:
  shard:
    axis: 0
    workers: [0, 1, 2, 3]
  compile: true              # Enable compilation for this signal
```

```python
import os
os.environ['GT_CONFIG'] = 'config.yaml'
import gt

with gt.signal.context('training_loop'):
    # Operations in this context will be compiled
    loss = model(input)
    loss.backward()
```

## Debug Compilation

View compilation activity:

```bash
GT_AUTO_COMPILE=1 GT_DEBUG_COMPILE=1 python train.py
```

Output shows:
- Hot path detection
- Compilation triggers
- Sequence patterns
- Performance improvements

Example output:
```
[COMPILE] Detected hot sequence: matmul -> relu -> sum (executed 15 times)
[COMPILE] Compiling sequence...
[COMPILE] Compilation complete (2.3s)
[COMPILE] Speedup: 3.2x on subsequent executions
```

## How It Works

### 1. Operation Recording

Workers track operation sequences:

```python
# First iteration
x = gt.randn(100, 100)    # matmul_1
y = x @ w                 # matmul_2
z = y.relu()              # relu_1
loss = z.sum()            # sum_1

# Sequence recorded: [matmul, relu, sum]
```

### 2. Pattern Detection

After N executions (default: 10), pattern marked as "hot":

```
Sequence hash: abc123def456
Execution count: 15
Status: HOT PATH ✓
```

### 3. Compilation

Hot path compiled with `torch.compile`:

```python
# Before: Individual operations
y = torch.matmul(x, w)
z = torch.relu(y)
loss = torch.sum(z)

# After: Fused kernel
loss = compiled_forward(x, w)  # Single fused operation
```

### 4. Execution

Subsequent calls use compiled version:

```
Iteration 1-10: Normal execution (recording)
Iteration 11: Compilation triggered (slow)
Iteration 12+: Compiled execution (fast)
```

## Performance Impact

### Warmup Overhead

First compilation is slow:

```bash
# Example timing
Iteration 1: 100ms   # Normal
Iteration 2-10: 100ms # Recording
Iteration 11: 2500ms # Compilation (25x slower!)
Iteration 12+: 30ms  # Compiled (3x faster)
```

**Strategy:**
- Always warmup before benchmarking
- Accept slow first iterations in training
- Compilation is one-time cost

### Expected Speedup

Typical speedups:

| Pattern | Speedup | Reason |
|---------|---------|--------|
| Matmul + ReLU | 2-3x | Kernel fusion |
| Element-wise chain | 3-5x | Memory optimization |
| Reduction sequence | 2-4x | Fused kernels |
| Simple ops | 1.5-2x | Dispatch overhead reduction |

**Best speedups:**
- Long operation sequences
- Element-wise operations
- Predictable patterns

**Limited speedups:**
- Single operations
- Memory-bound operations
- Irregular patterns

## Best Practices

### 1. Consistent Shapes

Ensure shapes are consistent across iterations:

```python
# Good: Consistent shapes
for epoch in range(100):
    x = gt.randn(128, 784)  # Always 128x784
    y = model(x)

# Bad: Varying shapes (causes recompilation)
for batch_size in [64, 128, 256]:
    x = gt.randn(batch_size, 784)  # Different shapes
    y = model(x)
```

### 2. Use Compilation Boundaries

Define clear boundaries:

```yaml
forward:
  compile: true
  shard:
    axis: 0
    workers: [0, 1, 2, 3]

backward:
  compile: true
  shard:
    axis: 0
    workers: [0, 1, 2, 3]
```

```python
# Forward compiled separately
with gt.signal.context('forward'):
    pred = model(input)
    loss = loss_fn(pred, target)

# Backward compiled separately
loss.backward()
```

### 3. Warmup Before Benchmarking

```python
import time

# Warmup (compilation happens here)
for _ in range(20):
    loss = model(input)
    loss.backward()

# Now benchmark
start = time.time()
for _ in range(100):
    loss = model(input)
    loss.backward()
end = time.time()

print(f"Average: {(end - start) / 100 * 1000:.1f}ms")
```

### 4. Profile Compilation Impact

```bash
# Without compilation
time python train.py

# With compilation
GT_AUTO_COMPILE=1 time python train.py

# Detailed compilation stats
GT_AUTO_COMPILE=1 GT_DEBUG_COMPILE=1 python train.py
```

## Troubleshooting

### Frequent Recompilation

**Symptom:** Many "Compiling..." messages

**Causes:**
1. Varying input shapes
2. Dynamic control flow
3. Changing operation patterns

**Solutions:**
1. Fix input shapes
2. Use static control flow
3. Separate dynamic code from hot paths

**Debug:**
```bash
GT_DEBUG_COMPILE=1 python script.py | grep "Recompiling"
```

### Compilation Errors

**Symptom:** Compilation fails, falls back to normal execution

**Causes:**
1. Unsupported operations
2. Dynamic shapes
3. Python control flow

**Solutions:**
1. Check PyTorch version (2.0+)
2. Simplify operation sequence
3. Disable compilation for problematic sections

**Debug:**
```bash
# Show full error traces
TORCH_LOGS=recompiles GT_AUTO_COMPILE=1 python script.py
```

### No Speedup

**Symptom:** Compilation enabled but no performance improvement

**Causes:**
1. Memory-bound operations (large transfers)
2. Short operation sequences
3. Irregular patterns

**Solutions:**
1. Profile to identify bottlenecks
2. Focus on compute-heavy sections
3. Use longer operation sequences

**Analysis:**
```bash
# Capture before/after traces
python -m gt.scripts.trace -s 5 --dir before/
GT_AUTO_COMPILE=1 python -m gt.scripts.trace -s 5 --dir after/

# Compare timelines
python -m gt.scripts.visualize before/trace_*.log --output before.png
python -m gt.scripts.visualize after/trace_*.log --output after.png
```

## Advanced Configuration

### Torch Compile Modes

PyTorch supports different compilation modes:

```python
# In worker code (for custom backends)
import torch

# Default mode
torch.compile(model)

# Reduce compilation time (faster compile, slower execution)
torch.compile(model, mode="reduce-overhead")

# Maximum performance (slow compile, fast execution)
torch.compile(model, mode="max-autotune")
```

### Selective Compilation

Compile only specific modules:

```python
from gt.nn import Module

class Model(Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.compile(Encoder())  # Compiled
        self.decoder = Decoder()                  # Not compiled

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
```

## Instruction Stream with Compilation

View compilation in logs:

```bash
GT_INSTRUCTION_LOG=/tmp/compile.log GT_AUTO_COMPILE=1 python script.py
```

Log shows:
```
  0.100s | #0010 | RECV       | CLIENT | BinaryOp | op=matmul
  0.101s | #0011 | RECV       | CLIENT | UnaryOp  | op=relu
  0.102s | #0012 | RECV       | CLIENT | UnaryOp  | op=sum
... (repeated pattern) ...
  1.500s | #0150 | COMPILE    | WORKER | Starting | sequence=abc123def456
  3.800s | #0151 | COMPILE    | WORKER | Complete | sequence=abc123def456
  3.801s | #0152 | RECV       | CLIENT | BinaryOp | op=matmul (COMPILED)
```

## Performance Example

Real-world training loop:

```python
# Without compilation: ~150ms/iteration
# With compilation: ~50ms/iteration (3x speedup)

import gt
from gt.nn import Module, Linear, SGD

class MLP(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(784, 256)
        self.fc2 = Linear(256, 10)

    def forward(self, x):
        x = self.fc1(x).relu()
        return self.fc2(x)

model = MLP()
optimizer = SGD(model.parameters(), lr=0.01)

# Warmup
for _ in range(20):
    pred = model(X)
    loss = ((pred - y) ** 2).mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# Benchmark
import time
start = time.time()
for _ in range(100):
    pred = model(X)
    loss = ((pred - y) ** 2).mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
end = time.time()

print(f"Average iteration: {(end - start) / 100 * 1000:.1f}ms")
```

Run:
```bash
# Baseline
python benchmark.py
# Output: Average iteration: 150ms

# With compilation
GT_AUTO_COMPILE=1 python benchmark.py
# Output: Average iteration: 50ms (3x speedup)
```

## Next Steps

- [Backends Guide](backends.md) - PyTorch backend features
- [Tuning Guide](../dispatcher/tuning.md) - Overall performance optimization
- [Monitoring](../dispatcher/monitoring.md) - Profile compilation impact
