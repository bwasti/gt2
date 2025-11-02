# Usage

GT offers two modes: auto-server for local development and distributed setup for multi-GPU clusters.

## Auto-Server Mode

For local development and testing, GT automatically starts a dispatcher and worker:

```python
import gt

# Auto-starts local server with 1 worker
a = gt.tensor([1, 2, 3, 4])
b = gt.tensor([5, 6, 7, 8])
c = a + b
data = c.data.numpy()  # [6, 8, 10, 12]
```

### Multiple Workers

Configure multiple workers for local multi-GPU:

```python
import gt
gt.gpu_workers(4)  # Must call before tensor operations

a = gt.randn(1000, 1000)
```

**Important:** Call `gt.gpu_workers()` before any tensor operations.

## Distributed Setup

For production use across multiple machines or GPUs.

### Terminal 1 - Start Dispatcher

```bash
python -m gt.server -p 12345
```

The dispatcher coordinates all clients and workers.

### Terminal 2-N - Start Workers (1 per GPU)

```bash
# Worker 0 (GPU 0)
CUDA_VISIBLE_DEVICES=0 python -m gt.worker --host localhost -p 12345

# Worker 1 (GPU 1)
CUDA_VISIBLE_DEVICES=1 python -m gt.worker --host localhost -p 12345

# Worker on another machine
CUDA_VISIBLE_DEVICES=0 python -m gt.worker --host <dispatcher_host> -p 12345
```

**Note:** Start one worker per GPU. Use `CUDA_VISIBLE_DEVICES` to assign each worker to a specific GPU.

### Terminal N - Run Code

```python
import gt

gt.connect('localhost:12345')
a = gt.randn(10000, 10000)
b = gt.randn(10000, 10000)
c = a @ b
```

## Basic Operations

### Creating Tensors

```python
import gt
import numpy as np

# Random tensors
a = gt.randn(100, 100)         # Normal distribution
b = gt.zeros(10, 10)           # All zeros
c = gt.ones(5, 5)              # All ones

# From NumPy
data = np.array([[1, 2], [3, 4]])
d = gt.from_numpy(data)

# Direct creation
e = gt.tensor([1, 2, 3, 4])
```

### Arithmetic Operations

```python
# Element-wise operations
c = a + b
c = a - b
c = a * b
c = a / b

# Matrix multiplication
c = a @ b
```

### Getting Data Back

```python
# Get NumPy array
result = c.data.numpy()

# Print tensor (automatically fetches data)
print(c)
```

## Environment Variables

Control GT behavior with environment variables:

### Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `GT_CONFIG` | Path to sharding config YAML | None |
| `GT_AUTO_COMPILE` | Enable automatic hot path detection | 0 |
| `GT_COMPILE` | Force compile all operations | 0 |
| `GT_WORKER_BATCH_SIZE` | Operations to batch per worker | 1 |

### Debug Output

| Variable | Description | Default |
|----------|-------------|---------|
| `GT_VERBOSE` | Framework status messages | 0 |
| `GT_DEBUG_CLIENT` | Client-side debug messages | 0 |
| `GT_DEBUG_DISPATCHER` | Dispatcher debug messages | 0 |
| `GT_DEBUG_WORKER` | Worker debug messages | 0 |
| `GT_DEBUG_COMPILE` | Compilation debug messages | 0 |

### Logging

| Variable | Description | Default |
|----------|-------------|---------|
| `GT_INSTRUCTION_LOG` | Instruction stream log file path | None |

**Example:**

```bash
GT_CONFIG=sharding.yaml \
GT_INSTRUCTION_LOG=debug.log \
GT_VERBOSE=1 \
python train.py
```

## Examples

GT includes several example scripts in the `examples/` directory:

- `demo.py` - Basic tensor operations
- `signal_demo.py` - Signal-based sharding
- `compile_demo.py` - Compilation directives
- `debug_demo.py` - Debug utilities
- `visualize_demo.py` - Timeline visualizations

Run examples:

```bash
python examples/demo.py
python examples/signal_demo.py
```

## Next Steps

- [Tensor API Reference](client/tensor-api.md) - Complete operation list
- [Autograd Guide](client/autograd.md) - Automatic differentiation
- [Sharding Configuration](dispatcher/signaling.md) - Distributed training
- [Monitoring Tools](dispatcher/monitoring.md) - Real-time system monitoring
