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
