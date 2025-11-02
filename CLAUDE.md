# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GT is a distributed frontend for GPU ML operations that multiplexes users to work on the same cluster simultaneously. It automatically shards and places tensors, schedules operations to maximize GPU utilization.

**Target Audience**: Researchers (not engineers). Code must be SUPER simple, VERY readable, and maintainable by non-engineers.

## User Focus

The API is designed to be **trivial** and **obvious**. There are two ways to use GT:

### 1. Simplest: Auto-Server (Local Development)

```python
import gt

# No connection needed - server auto-starts!
a = gt.tensor([1, 2, 3, 4])
b = gt.tensor([5, 6, 7, 8])
c = a + b

# Get data back (PyTorch-like API)
data = c.data.numpy()
```

### 2. Distributed: Manual Server (Production/Multi-GPU)

Terminal 1 - Start server:
```bash
python -m gt.server -p 12345
```

Terminal 2, 3, ... - Start workers (1 per GPU):
```bash
python -m gt.worker --host localhost -p 12345
python -m gt.worker --host localhost -p 12345
# ... scale to ~72 GPUs across 36 nodes
```

Terminal N - Run client:
```python
import gt

gt.connect('localhost:12345')
a = gt.tensor([1, 2, 3, 4])
# ... work continues as normal
```

**Key Design Principle**: The API should feel like PyTorch but work across distributed GPUs without users thinking about it.

## Features

### Tensor Operations (PyTorch Functional API)

All standard PyTorch operations are supported:

**Element-wise**:
- Arithmetic: `+`, `-`, `*`, `/`
- Activation: `relu()`, `sigmoid()`, `tanh()`
- Math: `exp()`, `log()`, `sum()`, `mean()`

**Matrix Operations**:
- Matrix multiplication: `@` (matmul)
- Comparison: `>` (for masking/relu gradients)

**Autograd Support**:
All operations automatically track gradients when `requires_grad=True`:
```python
a = gt.tensor([1, 2, 3], requires_grad=True)
b = a.relu()
loss = b.sum()
loss.backward()  # Computes gradients
print(a.grad.data.numpy())  # Access gradients
```

### Neural Network Module (`gt.client.nn`)

**Base Classes**:
- `nn.Module` - Base class for all neural network modules
- `nn.Linear` - Fully connected layer

**Loss Functions**:
- `nn.mse_loss` - Mean Squared Error
- `nn.binary_cross_entropy` - Binary Cross Entropy
- `nn.cross_entropy_loss` - Cross Entropy (with logits)

**Activation Functions**:
- `nn.relu`, `nn.sigmoid`, `nn.tanh`

Example usage:
```python
import gt
from gt.client import nn

# Define a model
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 1)
        self._parameters = self.fc1.parameters() + self.fc2.parameters()

    def forward(self, x):
        x = self.fc1(x).relu()
        x = self.fc2(x)
        return x

# Train
model = MyModel()
x = gt.randn(10)
y_pred = model(x)
loss = nn.mse_loss(y_pred, target)
loss.backward()

# Update parameters (manual SGD)
for param in model.parameters():
    # param = param - lr * param.grad
    pass
```

## Architecture

The system is organized under the `gt/` package with four main components:

### gt/client/
The user-facing API with two core abstractions:
- **Tensor**: Container for data (usually remote on workers, but transparent to users). When `.data` is accessed, client requests data from dispatcher.
- **AutogradGraph**: Tape-based autograd system (like PyTorch) that is purely abstract with ZERO interaction with computation. Only creates more tensors.

Client creates purely functional operation streams like:
```
TENSOR[3] := ADD TENSOR[0] TENSOR[2]
TENSOR[4] := USER_TENSOR([serialized data], dtype=blah)
```

Client doesn't care where/how things run - only wants correct results. Uses Python's native garbage collection to emit instructions telling dispatcher when tensors can be freed.

### gt/dispatcher/
Takes commands from MULTIPLE clients and schedules them to run. Rewrites incoming "tape" to distribute operations or inject "move" operations between workers.

Main abstraction: **TensorHandle** - maps each `client:tensor` to physical locations (including sharding/replication details).

More precise protocol than client<->dispatcher for talking to workers.

### gt/worker/
Dumb executors that take dispatcher instructions and run them. Backed by different engines (numpy, pytorch, jax - don't write custom kernels).

**Backend Support**:
- `pytorch` (default): Uses PyTorch for all operations - fast and GPU-capable
- `numpy`: Reference implementation - slower but simpler

To change backend, edit `simple_launch.py` or pass `backend='numpy'` to `Worker()` constructor.

### gt/transport/
Defines protocols for communication between components:
- Client <-> Dispatcher protocol (simpler)
- Dispatcher <-> Worker protocol (more precise)

## Key Design Principles

1. **Simplicity First**: Keep everything simple and readable. Researchers need to understand this.
2. **Purely Functional**: All operations are functional (lazily constructed tape).
3. **Location Transparency**: Client never knows where tensors live.
4. **Multiple Clients**: Dispatcher handles concurrent clients.
5. **Async Streams**: Client input, dispatch stream, and worker execution are all async (enables future optimizations).

**CRITICAL: NEVER AVOID FIXING BUGS IN THE CORE GT**

When encountering bugs in the GT framework:
- Always fix the bug in the core framework
- Do NOT create workarounds in example code
- Do NOT skip functionality because it's "not implemented yet"
- Investigate and fix the root cause

The goal is to build a robust, production-ready framework. Avoiding bugs only hides problems and prevents progress.

## Future Considerations

- **Compilation via Batching**: Don't implement now, but multiple async streams will allow "slack" for batching instruction streams. This will enable better use of pytorch/jax backends.
- Keep architecture compatible with future batching/compilation optimizations.

## Development Commands

### Running the System

Start dispatcher, workers, and test (all in one terminal):
```bash
python simple_launch.py
```

This automatically runs the test after starting the system.

### Running Without Test

If you want to run the system and connect your own client:
```bash
python simple_launch.py --no-test
```

### Running with Multiple Workers

```bash
python simple_launch.py --workers 2
```

### Running Test Separately

If the system is already running, you can run the test in another terminal:
```bash
python test_simple.py
```

### Running Pytest Tests

Run numeric correctness tests with pytest:
```bash
pytest tests/ -v
```

This automatically starts a test GT system on a different port and runs comprehensive numeric tests.

### Debug Output

GT provides fine-grained control over debug output through environment variables:

```bash
# Enable worker debug output
GT_DEBUG_WORKER=1 python my_script.py

# Enable dispatcher debug output
GT_DEBUG_DISPATCHER=1 python my_script.py

# Enable client debug output
GT_DEBUG_CLIENT=1 python my_script.py

# Enable compilation/hot path debug output
GT_DEBUG_COMPILE=1 python my_script.py

# Combine multiple flags
GT_DEBUG_WORKER=1 GT_DEBUG_COMPILE=1 python my_script.py
```

**Available Flags:**
- `GT_VERBOSE=1` - Enable framework status messages (startup, connections, registration)
- `GT_DEBUG_CLIENT=1` - Client-side debug messages (tensor operations, connections)
- `GT_DEBUG_DISPATCHER=1` - Dispatcher debug messages (worker registration, client handling)
- `GT_DEBUG_WORKER=1` - Worker debug messages (command processing, registration)
- `GT_DEBUG_COMPILE=1` - Compilation debug messages (hot path detection, compilation progress)

**Note:** By default, GT is completely silent - no framework output at all. Only errors are shown. Use `GT_VERBOSE=1` to see startup and connection messages.

## Debugging Guide

### Core Debugging Philosophy

**ALWAYS USE TAPE-BASED DEBUGGING FOR DISTRIBUTED ISSUES**

GT is a distributed system with multiple components. When debugging:
1. Start with the **instruction stream** (the "tape")
2. Verify correct instruction flow through the system
3. Identify where behavior diverges from expected

### Instruction Stream Logging

The most powerful debugging tool is **instruction stream logging**:

```bash
GT_INSTRUCTION_LOG=/tmp/gt_debug.log python your_script.py
```

This creates a detailed log of ALL instructions flowing through the dispatcher:
- Client commands received (RECV)
- Responses sent back to clients (SEND)
- Worker commands sent (WORKER_SEND)
- Worker responses received (WORKER_RECV)
- Connection events (CONNECT/DISCONNECT)

**Example log output:**
```
  0.123s | #0042 | RECV         | CLIENT 127.0.0.1:12345 | BinaryOp        | 123KB | result=42 op=add
  0.125s | #0043 | WORKER_SEND  | WORKER worker_0        | WorkerBinaryOp  | 123KB | op=add
  0.127s | #0044 | WORKER_RECV  | WORKER worker_0        | WorkerResponse  | 45B   | success=True
  0.128s | #0045 | SEND         | CLIENT 127.0.0.1:12345 | BinaryOp        | 45B   | success=True
```

**When to use:**
- Distributed operations behaving incorrectly
- Sharding bugs
- Worker communication issues
- Tensor not found errors
- Cross-worker operation failures

### Debug Output Hierarchy

Use debug flags in this order (from least to most verbose):

1. **GT_VERBOSE=1** - System startup and connections only
2. **GT_DEBUG_CLIENT=1** - Client-side operations
3. **GT_DEBUG_DISPATCHER=1** - Dispatcher scheduling decisions
4. **GT_DEBUG_WORKER=1** - Worker execution details
5. **GT_INSTRUCTION_LOG=file.log** - Complete instruction stream

**Combine flags for focused debugging:**
```bash
# Debug dispatcher + worker interaction:
GT_DEBUG_DISPATCHER=1 GT_DEBUG_WORKER=1 python script.py

# Debug everything + save instruction log:
GT_VERBOSE=1 GT_DEBUG_DISPATCHER=1 GT_INSTRUCTION_LOG=/tmp/gt.log python script.py
```

### Worker Stats API

Query live system statistics:

```python
import gt

# Run some operations...
x = gt.randn(100, 100)
y = x @ x

# Get stats
stats = gt.debug.get_worker_stats()

# Stats include:
# - total_instructions: Total commands executed
# - hot_instructions: Instructions in hot paths
# - unique_sequences: Number of unique operation patterns
# - hot_sequences: Patterns detected as hot paths
# - compilation stats (if GT_AUTO_COMPILE=1)
```

Example script: `examples/show_stats.py`

### Common Bug Patterns

#### 1. Tuple Unpacking Issues

**Symptom:** `'tuple' object has no attribute 'success'`

**Cause:** Function returns `(response, size)` but code expects just `response`

**Fix:**
```python
# Before (wrong):
response = self._recv_from_worker(worker)

# After (correct):
response, _ = self._recv_from_worker(worker)
```

**How to find:** Search for `_recv_from_worker` calls without tuple unpacking

#### 2. Cross-Worker Operations

**Symptom:** `"Cross-worker ops not yet supported"` or `"Tensor not found"`

**Cause:** Tensors on different workers, operation requires co-location

**Debug approach:**
1. Enable `GT_INSTRUCTION_LOG` to see tensor placement
2. Check TensorHandle registry: where is each tensor?
3. Verify dispatcher moves tensors correctly

**Fix:** Implement cross-worker tensor movement in dispatcher

#### 3. Sharding Issues

**Symptom:** Wrong result shapes, `"Shape mismatch"`, numerical errors

**Cause:** Sharding logic incorrectly computing shard boundaries or concatenation

**Debug approach:**
1. Write a minimal reproduction script
2. Use `GT_INSTRUCTION_LOG` to trace shard creation
3. Verify shard shapes at each step
4. Check ShardInfo metadata (axis, num_shards, shard_index)

**Example debug script:**
```python
import gt

gt.zeros(1, 1)  # Force init

# Create sharded tensor (will be split across workers)
a = gt.randn(128, 64)  # With 4 workers: 4 × (32, 64)

# Verify shape
print(f"Result shape: {a.data.numpy().shape}")  # Should be (128, 64)
```

#### 4. Gradient/Autograd Bugs

**Symptom:** Gradients are `None`, wrong values, or wrong shapes

**Debug approach:**
1. Verify `requires_grad=True` is set
2. Check gradient function is recorded in AutogradGraph
3. Test against PyTorch reference:

```python
import torch
import numpy as np
import gt

# PyTorch reference
x_pt = torch.randn(10, 10, requires_grad=True)
y_pt = torch.relu(x_pt)
y_pt.sum().backward()

# GT version
x_gt = gt.from_numpy(x_pt.detach().numpy(), requires_grad=True)
y_gt = x_gt.relu()
y_gt.sum().backward()

# Compare
print(np.allclose(x_pt.grad.numpy(), x_gt.grad.data.numpy()))
```

**Fix locations:**
- `gt/client/tensor.py`: Gradient functions in `_binary_op`, `_unary_op`
- `gt/client/autograd.py`: Tape recording and backward pass

### Debugging Workflow

#### Step 1: Reproduce Minimally

Create the smallest script that reproduces the bug:

```python
# minimal_repro.py
import gt

# Minimal reproduction here
x = gt.randn(10, 10)
y = x.sum()
print(y.data.numpy())
```

#### Step 2: Add Instrumentation

```bash
# Run with instruction logging
GT_INSTRUCTION_LOG=/tmp/gt_debug.log python minimal_repro.py

# Examine the log
cat /tmp/gt_debug.log | grep "ERROR\|FAILED\|tensor_id"
```

#### Step 3: Compare Against Reference

For numerical bugs, always compare against PyTorch:

```python
import torch
import numpy as np
import gt

data = np.random.randn(10, 10).astype(np.float32)

# PyTorch
x_pt = torch.from_numpy(data)
result_pt = x_pt.sum().item()

# GT
x_gt = gt.from_numpy(data)
result_gt = x_gt.sum().data.numpy()

print(f"PyTorch: {result_pt}")
print(f"GT:      {result_gt}")
print(f"Match:   {np.allclose(result_gt, result_pt)}")
```

#### Step 4: Write a Failing Test

Convert your reproduction into a pytest test:

```python
# tests/test_my_bug.py
def test_my_bug_reproduction(client):
    """Reproduces bug XYZ"""
    x = gt.randn(10, 10)
    y = x.sum()

    # This should pass but doesn't
    assert y.data.numpy().shape == ()
```

Run it:
```bash
pytest tests/test_my_bug.py -v -s
```

#### Step 5: Fix and Verify

1. Identify root cause using logs/stats
2. Implement fix
3. Run your new test: `pytest tests/test_my_bug.py`
4. Run full test suite: `pytest tests/ -v`
5. Verify fix doesn't break other tests

### Testing Strategy

**Test Pyramid:**
1. Unit tests (operation correctness)
2. Integration tests (distributed operations)
3. Regression tests (previously fixed bugs)

**Key test files:**
- `tests/test_numerics.py` - Basic operation correctness
- `tests/test_sharding.py` - Distributed matmul patterns
- `tests/test_distributed_matmul_patterns.py` - Multi-GPU patterns
- `tests/test_autograd.py` - Gradient computation
- `tests/test_varying_batch_sizes.py` - Batch size regressions

**Run tests frequently:**
```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_numerics.py -v

# Run specific test
pytest tests/test_numerics.py::test_addition -v

# Run with output (see prints)
pytest tests/test_numerics.py::test_addition -v -s
```

### Common Error Messages

| Error | Likely Cause | Debug Command |
|-------|-------------|---------------|
| `Tensor not found` | Wrong worker, freed tensor, or bad tensor ID | `GT_INSTRUCTION_LOG=file.log` |
| `Cross-worker ops not yet supported` | Tensors on different workers | Check tensor placement in logs |
| `Worker not found` | Worker disconnected or not registered | `GT_DEBUG_DISPATCHER=1` |
| `Shape mismatch` | Sharding bug or incorrect operation | Compare against PyTorch |
| `'tuple' object has no attribute` | Missing tuple unpacking | Search for function call |
| `mat1 and mat2 shapes cannot be multiplied` | Wrong matmul dimensions, often from sharding | Check shard shapes in logs |

### Debugging Tools Quick Reference

```bash
# See what's happening:
GT_VERBOSE=1 python script.py

# Debug distributed operations:
GT_INSTRUCTION_LOG=/tmp/gt.log python script.py
cat /tmp/gt.log | grep "WORKER_SEND\|WORKER_RECV"

# Debug specific component:
GT_DEBUG_DISPATCHER=1 python script.py

# Get live stats:
python examples/show_stats.py

# Run test with output:
pytest tests/test_file.py::test_name -v -s

# Verify against PyTorch:
# (Write comparison script as shown above)
```

### Advanced Debugging: Tape Analysis

For complex distributed bugs, analyze the instruction tape:

```bash
# 1. Generate tape
GT_INSTRUCTION_LOG=/tmp/tape.log python buggy_script.py

# 2. Analyze patterns
grep "tensor_id" /tmp/tape.log | head -20      # Tensor creation
grep "WORKER_SEND.*matmul" /tmp/tape.log       # Matmul operations
grep "shard" /tmp/tape.log                     # Sharding operations
grep "ERROR\|FAILED" /tmp/tape.log             # Errors

# 3. Compare expected vs actual flow
# Expected: RECV BinaryOp → WORKER_SEND WorkerBinaryOp → WORKER_RECV Response → SEND Response
# Check if this pattern appears in the logs
```

**Example analysis session:**
```python
# Create debug script that logs everything
import gt
import os

os.environ['GT_INSTRUCTION_LOG'] = '/tmp/debug.log'

# Reproduce bug
a = gt.randn(128, 64)  # Should be sharded
b = gt.randn(64, 32)   # Should be sharded
c = a @ b              # This fails

# Check the log to see:
# 1. How were a and b created? (Should be multiple WORKER_SEND CreateTensor)
# 2. How many workers got shards?
# 3. What matmul commands were sent?
# 4. Did any worker return an error?
```

## Communication Flow

1. Client connects to dispatcher via TCP
2. Client sends stream of commands
3. Each command receives either:
   - "ack" for successful operation
   - Response data (e.g., for `.data` requests, dispatcher fetches from worker)
4. Dispatcher schedules operations to workers
5. Workers execute and return results
6. Client can issue tensor free instructions via garbage collection

## Benchmarks

The `benchmarks/` directory contains performance comparisons between pure PyTorch and GT framework:

```bash
cd benchmarks
python pytorch_baseline.py     # PyTorch baseline
python gt_benchmark.py          # GT framework (requires system running)
python compare.py               # Side-by-side comparison
```

These benchmarks measure:
- Matrix multiplication (1000x1000)
- Linear layer forward pass
- MSE loss computation
- Backward pass (autograd)
- Activation functions (ReLU, Sigmoid, Tanh)

**Expected Overhead**: GT adds communication/serialization overhead. For small operations this can be 100-1000%+. For larger operations, overhead becomes negligible as compute dominates.

## Code Organization

```
gt/                     # Main package
├── client/            # User-facing API
│   ├── tensor.py      # Tensor abstraction with autograd
│   ├── client.py      # Client connection
│   ├── autograd.py    # Autograd tape system
│   └── nn.py          # Neural network modules (Module, Linear, losses)
├── dispatcher/        # Coordination layer
│   ├── dispatcher.py  # Main dispatcher
│   └── tensor_handle.py # Tensor location mapping
├── worker/            # Execution layer
│   └── worker.py      # Worker implementation (pytorch/numpy backend)
├── transport/         # Communication
│   ├── protocol.py    # Protocol definitions
│   └── connection.py  # TCP helpers
└── server/            # Server module
    └── __main__.py    # Server CLI

benchmarks/            # Performance benchmarks
├── README.md          # Benchmark documentation
├── pytorch_baseline.py # Pure PyTorch baseline
├── gt_benchmark.py     # GT framework benchmarks
└── compare.py          # Side-by-side comparison

tests/                 # Pytest tests
├── conftest.py        # Test fixtures
├── test_numerics.py   # Numeric correctness tests
├── test_autograd.py   # Autograd/gradient tests
├── test_operations.py # Operation tests (relu, sigmoid, etc)
├── test_nn.py         # Neural network module tests
└── test_multi_client.py # Multi-client concurrency tests

simple_launch.py       # Launcher for manual testing
demo.py                # Simple API demo
test_grad.py           # Standalone gradient test
```

Keep files focused and simple:
- One abstraction per file when possible
- Clear separation between protocol definitions and implementations
- Minimal dependencies between components
- Explicit > implicit
- All imports use `gt.` prefix for clarity
