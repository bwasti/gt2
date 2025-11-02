# GT

A multiplexing tensor framework.

```bash
pip install git+https://github.com/bwasti/gt.git
python -c 'import gt; print(gt.randn(2,2))'
```

## Features

- **High-performance transport** - ZeroMQ (ZMQ) with automatic message batching and efficient DEALER/ROUTER pattern
- **Stream processing** - Workers process operations one at a time for simplicity
- **Signal-based sharding** - Configure data/model/pipeline parallelism via YAML
- **Autograd support** - Tape-based automatic differentiation
- **Distributed execution** - Client-dispatcher-worker architecture
- **PyTorch-compatible API** - Familiar syntax for tensor operations
- **AI-assisted development** - Optimized for collaboration with AI coding assistants. See [AI Development](#optimized-for-ai-development)

## Quick Start

```python
import gt

# Auto-starts local server
a = gt.randn(1000, 1000)
b = gt.randn(1000, 1000)
c = a @ b
result = c.data.numpy()
```

For distributed training, see [Distributed Setup](#distributed-setup).

## Stream Processing

Workers process operations one at a time as they arrive from the dispatcher, which is fed a stream of instructions from the user client.
This keeps the architecture simple and easy to reason over.


## Signal-Based Sharding Configuration

Control tensor placement across workers using named signals and YAML configs:

```yaml
# config.yaml
forward_layer1:
  shard:
    axis: 0                  # Shard batch dimension
    workers: [0, 1, 2, 3]    # Use workers 0-3
  backward: backward_layer1  # Different config for gradients

pipeline_stage_1:
  shard:
    axis: 0
    workers: [0, 1]          # Stage 1 on workers 0-1
  backward: pipeline_stage_1_bwd

pipeline_stage_1_bwd:
  shard:
    axis: 0
    workers: [2, 3]          # Backward on workers 2-3
```

```python
import os
os.environ['GT_CONFIG'] = 'config.yaml'
import gt

# Context manager - shards tensors AND compute
with gt.signal.context('forward_layer1'):
    x = gt.randn(100, 64)  # Sharded across workers [0,1,2,3]
    y = x @ w              # Compute happens in sharded mode

# Function call - copy-shard tensor only
x_sharded = gt.signal.tensor(x, name='forward_layer1')

# Pipeline parallelism - forward/backward on different workers
with gt.signal.context('pipeline_stage_1', backward='pipeline_stage_1_bwd'):
    loss = model(input)
    loss.backward()  # Gradients computed on workers [2,3]
```

**Supported configurations:**
- Data parallelism (batch sharding)
- Model parallelism (feature sharding)
- Pipeline parallelism (stage-wise worker assignment)
- Replicated parameters
- Per-layer sharding strategies
- Compilation directives (`compile: true` for torch.compile boundaries)

See [examples/README_SIGNALS.md](examples/README_SIGNALS.md) for comprehensive guide.

## Autograd Support

Tape-based automatic differentiation:

```python
from gt.nn import Module, Linear, SGD

class MLP(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(784, 256)
        self.fc2 = Linear(256, 10)

    def forward(self, x):
        x = self.fc1(x).relu()
        return self.fc2(x)

# Training loop
model = MLP()
optimizer = SGD(model.parameters(), lr=0.01)

for epoch in range(100):
    pred = model(X_train)
    loss = ((pred - y_train) ** 2).mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

**Implemented:**
- Tape-based autograd (PyTorch-style)
- Gradient computation with broadcasting
- In-place parameter updates
- SGD optimizer

## Instruction Stream Logging

Operations can be logged with timestamps:

```bash
GT_INSTRUCTION_LOG=ops.log python your_script.py
```

Output format:
```
[TAPE 0.003s] RECV         | 127.0.0.1:59712 | BinaryOp  | result=2 op=matmul left=0 right=1
[TAPE 0.003s] WORKER_SEND  | auto_worker     | WorkerOp  | op=matmul result=127.0.0.1:59712_2
[TAPE 0.004s] WORKER_RECV  | auto_worker     | WorkerOp  | success=True
[TAPE 0.004s] SEND         | 127.0.0.1:59712 | BinaryOp  | success=True
```

Useful for:
- Debugging hangs (see last operation before timeout)
- Identifying slow operations (large timestamp gaps)
- Understanding distributed execution flow

## Instruction Tape Visualization

Generate timeline visualizations showing operation flow through the system:

```bash
# 1. Run with instruction logging enabled
GT_INSTRUCTION_LOG=/tmp/debug.log python your_script.py

# 2. Generate visualization
python -m gt.scripts.visualize /tmp/debug.log --output timeline.png --dpi 200
```

The visualizer is included with GT (matplotlib is a core dependency).

Output shows:
- Timeline lanes for Client, Dispatcher, and each Worker
- Color-coded operations (MatMul, BinaryOp, UnaryOp, etc.)
- Event types indicated by marker shapes (RECV, WORKER_SEND, WORKER_RECV)
- Data transfer sizes indicated by marker sizes
- Communication arrows showing instruction flow between components
- Instruction IDs annotated on key events

Use cases:
- Identify idle workers or unbalanced load
- Visualize distributed operation patterns (embarrassingly parallel, all-gather, all-reduce)
- Find communication bottlenecks
- Debug distributed execution issues

See `gt/scripts/README.md` for complete documentation.

## Real-Time Monitoring

<img width="862" height="235" alt="Screenshot 2025-11-02 at 1 54 47 AM" src="https://github.com/user-attachments/assets/e5d2d810-d366-4390-8138-ab4bfec5dd6f" />

Monitor running dispatchers with htop-style worker activity visualization:

```bash
# Auto-attach to running dispatcher
python -m gt.scripts.top

# Attach to specific dispatcher
python -m gt.scripts.top --port 9000 --host localhost
```

The monitor is included with GT (pyzmq, rich, and psutil are core dependencies).

Features:
- **Real-time EMA-smoothed activity bars** showing operation breakdown per worker
- **Color-coded operations** (matmul, add, relu, etc.)
- **Idle time tracking** to identify underutilized workers
- **Auto-detection** of running dispatchers
- **Non-intrusive** - connects via ZMQ monitoring socket without affecting performance

### Trace Capture

Capture event streams for later analysis:

```bash
# Capture 2 seconds of activity
python -m gt.scripts.trace -s 2 --dir traces/

# Capture first 100 events (with 10 second timeout)
python -m gt.scripts.trace -s 10 -n 100 --dir traces/

# Then visualize the captured trace
python -m gt.scripts.visualize traces/trace_*.log --output timeline.png
```

Options:
- `-s, --seconds DURATION` - Maximum capture duration (required)
- `-n, --max-events N` - Stop after N events (optional)
- `--port PORT` - Dispatcher port (auto-detected by default)
- `--dir DIR` - Output directory (default: current directory)

Workflow:
1. **Run your workload** - Normal GT script execution
2. **Capture trace** - Record events for specified duration or event count
3. **Visualize** - Generate timeline diagrams from captured data

This complements the monitoring tools:
- `gt.scripts.top` - Real-time monitoring (htop-style)
- `gt.scripts.trace` - Capture events to file
- `gt.scripts.visualize` - Generate timeline diagrams

## Debug Utilities

Inspect internal state:

```python
import gt

# Create tensors with gradients
a = gt.randn(10, 10, requires_grad=True)
b = gt.randn(10, 10, requires_grad=True)
loss = (a + b).sum()

# View autograd tape
gt.debug.print_tape()

# View worker statistics
gt.debug.print_worker_stats()
```

## Multiple Backends

- **PyTorch** (default): GPU support, compilation, distributed primitives
- **NumPy**: CPU-only reference implementation (for testing)

Workers use PyTorch by default for both GPU and CPU execution.

## High-Performance Transport Layer

GT uses **ZeroMQ (ZMQ)** for client-dispatcher-worker communication:

**Benefits:**
- **Automatic message batching** - ZMQ queues and batches messages at the transport layer
- **Higher throughput** - More efficient than raw TCP for high-frequency small messages
- **Built-in patterns** - DEALER/ROUTER pattern handles multiple connections efficiently
- **Scalability** - Supports many concurrent clients and workers without manual connection management
- **IPC optimization** - Uses Unix domain sockets (IPC) for localhost connections, bypassing TCP/IP stack for lower latency

**Architecture:**
- **Dispatcher** - Single ZMQ ROUTER socket handles all connections
- **Clients/Workers** - DEALER sockets for async communication
- **Worker Registration** - Workers send registration message on startup
- **Transport selection** - Automatically uses IPC (`ipc://`) for localhost, TCP (`tcp://`) for remote hosts

This replaces the previous TCP implementation and provides better performance for the high message rate typical in distributed training workloads.


## Installation

### From GitHub (pip)

```bash
# Install GT (batteries included: PyTorch, visualization, monitoring)
pip install git+https://github.com/bwasti/gt.git
```

### From Source (editable)

```bash
# Clone repository
git clone https://github.com/bwasti/gt.git
cd gt

# Install in editable mode
pip install -e .
```

**Included by default:**
- PyTorch (GPU/CPU support, compilation)
- matplotlib (timeline visualizations)
- rich + psutil (real-time monitoring)
- NumPy, pytest, pyzmq, pyyaml

## Usage

### Auto-Server Mode

```python
import gt

# Auto-starts local server with 1 worker
a = gt.tensor([1, 2, 3, 4])
b = gt.tensor([5, 6, 7, 8])
c = a + b
data = c.data.numpy()  # [6, 8, 10, 12]
```

Configure multiple workers:
```python
import gt
gt.gpu_workers(4)  # Must call before tensor operations

a = gt.randn(1000, 1000)
```

### Distributed Setup

**Terminal 1 - Start dispatcher:**
```bash
python -m gt.server -p 12345
```

**Terminal 2-N - Start workers (1 per GPU):**
```bash
# Worker 0 (GPU 0)
CUDA_VISIBLE_DEVICES=0 python -m gt.worker --host localhost -p 12345

# Worker 1 (GPU 1)
CUDA_VISIBLE_DEVICES=1 python -m gt.worker --host localhost -p 12345

# Worker on another machine
CUDA_VISIBLE_DEVICES=0 python -m gt.worker --host <dispatcher_host> -p 12345
```

**Terminal N - Run code:**
```python
import gt

gt.connect('localhost:12345')
a = gt.randn(10000, 10000)
b = gt.randn(10000, 10000)
c = a @ b
```

## Environment Variables

### Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `GT_CONFIG` | Path to sharding config YAML | None |
| `GT_AUTO_COMPILE` | Enable automatic hot path detection and compilation | 0 |
| `GT_COMPILE` | Force compile all operations | 0 |
| `GT_WORKER_BATCH_SIZE` | Number of operations to batch per worker | 1 |

### Debug Output

| Variable | Description | Default |
|----------|-------------|---------|
| `GT_VERBOSE` | Enable framework status messages (startup, connections) | 0 |
| `GT_DEBUG_CLIENT` | Enable client-side debug messages | 0 |
| `GT_DEBUG_DISPATCHER` | Enable dispatcher debug messages | 0 |
| `GT_DEBUG_WORKER` | Enable worker debug messages | 0 |
| `GT_DEBUG_COMPILE` | Enable compilation debug messages | 0 |

### Logging

| Variable | Description | Default |
|----------|-------------|---------|
| `GT_INSTRUCTION_LOG` | Path to instruction stream log file | None |

By default, GT produces no output except errors. Use `GT_VERBOSE=1` to see startup messages.

Example:
```bash
GT_CONFIG=sharding.yaml \
GT_INSTRUCTION_LOG=debug.log \
GT_VERBOSE=1 \
python train.py
```

## API Reference

### Tensor Operations

**Arithmetic:** `+`, `-`, `*`, `/`, `@` (matmul)
**Activations:** `relu()`, `sigmoid()`, `tanh()`
**Reductions:** `sum()`, `mean()`
**Math:** `exp()`, `log()`
**Shape:** `transpose()`, `.T`
**In-place:** `-=`, `zero_()`

### Neural Network Modules

```python
from gt.nn import Module, Linear, SGD

class MyModel(Module):
    def __init__(self):
        super().__init__()
        self.layer1 = Linear(input_size, hidden_size)
        self.layer2 = Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.layer1(x).relu()
        return self.layer2(x)

optimizer = SGD(model.parameters(), lr=0.01)
```

### Autograd

```python
# Enable gradients
a = gt.randn(3, 4, requires_grad=True)
b = gt.from_numpy(np.array(...), requires_grad=True)

# Forward pass
loss = (a * 2).sum()

# Backward pass
loss.backward()

# Access gradients
print(a.grad.data.numpy())  # [2, 2, 2, ...]
```

### Signal-Based Sharding

```python
# Load config
gt.load_config('sharding.yaml')
# Or: os.environ['GT_CONFIG'] = 'sharding.yaml'

# Context manager
with gt.signal.context('layer1'):
    x = gt.randn(100, 64)

# Function call
x_sharded = gt.signal.tensor(x, name='layer1')

# Enter/exit API
gt.signal.enter('layer1')
x = gt.randn(100, 64)
gt.signal.exit('layer1')
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test suite
pytest tests/test_basic_ops.py -v
pytest tests/test_autograd.py -v
pytest tests/test_mlp.py -v
pytest tests/test_compilation.py -v
```

Tests auto-start a local GT system and verify numeric correctness.

## Benchmarks

```bash
cd benchmarks
python compare.py
```

Note: GT adds communication/serialization overhead. For small operations this overhead is significant. For large operations (training, large matmuls), this overhead becomes negligible.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          User Code                              │
│  import gt                                                      │
│  with gt.signal.context('layer1'):                              │
│      x = gt.randn(100, 64)                                      │
│      loss = model(x)                                            │
│      loss.backward()                                            │
└──────────────────────┬──────────────────────────────────────────┘
                       │ PyTorch-like API + Signal Metadata
                       │
┌──────────────────────▼──────────────────────────────────────────┐
│                      gt/client/                                 │
│  ┌──────────────┐  ┌─────────────┐  ┌──────────────┐            │
│  │   Tensor     │  │  Autograd   │  │  nn.Module   │            │
│  │ (Remote Data)│  │   (Tape)    │  │  (Layers)    │            │
│  └──────────────┘  └─────────────┘  └──────────────┘            │
│                                                                 │
│  ┌─────────────────────────────────────────────────┐            │
│  │ Signal API: Tracks current signal scope         │            │
│  │ Config: Loads YAML sharding strategies          │            │
│  └─────────────────────────────────────────────────┘            │
└──────────────────────┬──────────────────────────────────────────┘
                       │ ZMQ (DEALER → ROUTER with signal metadata)
                       │
┌──────────────────────▼──────────────────────────────────────────┐
│                    gt/dispatcher/                               │
│  • ZMQ ROUTER socket handles all connections                    │
│  • Reads signal configs from YAML                               │
│  • Routes operations based on sharding strategy                 │
│  • Logs instruction stream to file                              │
│  • Handles multiple clients concurrently                        │
└───────┬──────────────┬──────────────┬───────────────────────────┘
        │              │              │ ZMQ (DEALER ← ROUTER)
        │              │              │
    ┌───▼────┐    ┌───▼────┐    ┌───▼────┐
    │Worker 0│    │Worker 1│    │Worker N│ (1 per GPU)
    │        │    │        │    │        │
    │ Stream │    │ Stream │    │ Stream │ Stream processing
    │Process │    │Process │    │Process │ One op at a time
    │        │    │        │    │        │
    │PyTorch │    │PyTorch │    │PyTorch │ Backend
    │  GPU   │    │  GPU   │    │  GPU   │
    └────────┘    └────────┘    └────────┘
      Host 0        Host 0        Host N
```

**Components:**

- **gt/client/** - User-facing API with location-transparent tensors, tape-based autograd, signal tracking, and neural network modules
- **gt/dispatcher/** - Coordinates clients and schedules operations using ZMQ ROUTER socket. Maps client tensors to physical locations, reads signal configs for sharding decisions, and logs instruction streams
- **gt/worker/** - Executes operations using backends. Connects via ZMQ DEALER socket. Processes operations one at a time (stream processing). Supports multiple backends (PyTorch/NumPy). One worker per GPU.
- **gt/transport/** - ZeroMQ-based communication layer with DEALER/ROUTER pattern for high-performance message passing
- **gt/signal.py** - Signal-based sharding API with context managers, thread-local signal stack, and backward signal support
- **gt/config.py** - YAML config loading that parses sharding strategies and maps signal names to worker assignments

## Documentation

- [Signal-Based Sharding Guide](examples/README_SIGNALS.md) - Complete guide to sharding API
- [CLAUDE.md](CLAUDE.md) - Detailed architecture documentation
- [Hot Path Detection](benchmarks/README_COMPILATION.md) - Automatic compilation for repeated patterns (future work)

## Design Philosophy

The code prioritizes:

1. Clarity over performance in initial implementation
2. PyTorch-compatible API
3. Declarative configuration via YAML
4. Simple stream processing (one operation at a time)

## Examples

See [examples/](examples/) directory:

- `signal_demo.py` - Signal-based sharding demonstration
- `compile_demo.py` - Compilation directives demonstration
- `debug_demo.py` - Debug utilities demonstration
- `visualize_demo.py` - Instruction tape visualization demonstration
- `config_sharding.yaml` - Example sharding configuration
- `config_compile.yaml` - Example compilation configuration
- `demo.py` - Basic tensor operations
- `simple_launch.py` - Manual server/worker launch

## Performance Considerations

1. **Signals**: Use to control sharding strategies via configuration
2. **Multiple Workers**: Scale across GPUs for data/model parallelism
3. **Logging**: Use instruction logging to identify bottlenecks
4. **Transport**: ZeroMQ provides efficient message batching at transport layer

## Contributing

Contributions welcome. This is a research prototype focused on simplicity and readability.

## Optimized for AI Development

GT is designed to be understood, modified, and debugged with AI coding assistants:

### 1. **Architecture Documentation for AI**
- [CLAUDE.md](CLAUDE.md) provides detailed architectural context optimized for Claude and other AI assistants
- Explicit codebase structure, design decisions, and implementation patterns
- Helps AI quickly understand the system and make consistent changes

### 2. **Declarative Configuration via YAML**
- Sharding strategies defined in human-readable YAML configs
- Easy for AI to parse, understand, and generate configurations
- Clear mapping between signals and worker assignments
- See [Signal-Based Sharding](#signal-based-sharding-configuration)

### 3. **Debugging Utilities**
- **Tape-based autograd** - Inspect gradient computation graph with `gt.debug.print_tape()`
- **Instruction stream logging** - Track every operation with timestamps via `GT_INSTRUCTION_LOG`
- **Worker statistics** - View operation counts and performance metrics
- Makes it easy to identify bugs and understand execution flow

### 4. **Comprehensive Test Suite**
- 50+ tests covering tensor operations, autograd, distributed execution
- Tests serve as executable documentation and specifications
- Easy for AI to understand intended behavior and verify changes
- See [Running Tests](#running-tests)

### 5. **Standard API**
- PyTorch-compatible API that AI models are already trained on
- Familiar patterns like `Module`, `Linear`, `SGD`, `backward()`
- Extensive inline documentation and type hints
- Reduces cognitive load when making changes

## License

MIT
