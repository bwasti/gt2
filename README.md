# GT - Distributed GPU ML Operations

A distributed frontend for GPU ML operations that multiplexes users to work on the same cluster simultaneously.

**PyTorch-like API that scales from 1 to 72+ GPUs without code changes.**

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run a simple example:
```bash
python demo.py
```

That's it! No server setup needed - it auto-starts.

## Usage

### Simple (Auto-Server)

For local development, just import and use:

```python
import gt

a = gt.tensor([1, 2, 3, 4])
b = gt.tensor([5, 6, 7, 8])
c = a + b
data = c.data.numpy()  # [6, 8, 10, 12]
```

### Distributed (Manual Server)

For production or multi-GPU setups:

**Terminal 1 - Start server:**
```bash
python -m gt.server -p 12345
```

**Terminal 2+ - Start workers (1 per GPU):**
```bash
python -m gt.worker --host localhost -p 12345
```

**Terminal N - Run your code:**
```python
import gt

gt.connect('localhost:12345')
a = gt.tensor([1, 2, 3, 4])
# ... same API as before
```

Scale to ~72 GPUs across 36 nodes by starting more workers!

## Features

### PyTorch-Compatible Operations

All standard operations supported with autograd:
- **Arithmetic**: `+`, `-`, `*`, `/`, `@` (matmul)
- **Activations**: `relu()`, `sigmoid()`, `tanh()`
- **Reductions**: `sum()`, `mean()`
- **Math**: `exp()`, `log()`

### Neural Networks

```python
from gt.client import nn

# Define model
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 1)

    def forward(self, x):
        x = self.fc1(x).relu()
        return self.fc2(x)

# Train
model = MyModel()
pred = model(x)
loss = nn.mse_loss(pred, target)
loss.backward()

# Update parameters
for p in model.parameters():
    # p -= lr * p.grad
    pass
```

### Autograd

Fully automatic differentiation:
```python
a = gt.tensor([1, 2, 3], requires_grad=True)
b = a * 2
loss = b.sum()
loss.backward()
print(a.grad.data.numpy())  # [2, 2, 2]
```

## Running Tests

Run pytest to test numeric correctness:
```bash
pytest tests/ -v
```

This will automatically start a test GT system and verify that all operations produce correct results compared to numpy.

## Benchmarks

Compare performance with pure PyTorch:
```bash
cd benchmarks
python compare.py
```

Note: GT adds communication/serialization overhead. For small operations this can be 100-1000%+. For large operations, overhead becomes negligible.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          User Code                               │
│  import gt                                                       │
│  a = gt.tensor([1, 2, 3])                                       │
│  b = a.relu()                                                   │
│  loss = b.sum()                                                 │
│  loss.backward()                                                │
└──────────────────────┬──────────────────────────────────────────┘
                       │ PyTorch-like API
                       │
┌──────────────────────▼──────────────────────────────────────────┐
│                      gt/client/                                  │
│  ┌──────────────┐  ┌─────────────┐  ┌──────────────┐          │
│  │   Tensor     │  │  Autograd   │  │  nn.Module   │          │
│  │ (Remote Data)│  │   (Tape)    │  │  (Layers)    │          │
│  └──────────────┘  └─────────────┘  └──────────────┘          │
│                                                                  │
│  Creates functional operation streams:                          │
│    TENSOR[3] := ADD TENSOR[0] TENSOR[2]                        │
│    TENSOR[4] := RELU TENSOR[3]                                 │
└──────────────────────┬──────────────────────────────────────────┘
                       │ TCP (Protocol)
                       │
┌──────────────────────▼──────────────────────────────────────────┐
│                    gt/dispatcher/                                │
│  ┌────────────────────────────────────────────────────┐        │
│  │  TensorHandle: client:tensor → physical location   │        │
│  │  Schedules operations to workers (round-robin)     │        │
│  └────────────────────────────────────────────────────┘        │
│                                                                  │
│  Handles multiple clients concurrently                          │
└───────┬──────────────┬──────────────┬──────────────────────────┘
        │              │              │ TCP (WorkerProtocol)
        │              │              │
    ┌───▼────┐    ┌───▼────┐    ┌───▼────┐
    │Worker 0│    │Worker 1│    │Worker N│ (1 per GPU)
    │        │    │        │    │        │
    │PyTorch │    │PyTorch │    │PyTorch │ Backend
    │        │    │        │    │        │
    │  GPU   │    │  GPU   │    │  GPU   │
    └────────┘    └────────┘    └────────┘
      Host 0        Host 0        Host N
```

**Key Components:**

- **`gt/client/`** - User-facing API with Tensor and AutogradGraph abstractions
  - Location-transparent tensors (may be remote, user never knows)
  - Tape-based autograd (like PyTorch)
  - Neural network modules (nn.Module, nn.Linear, losses)

- **`gt/dispatcher/`** - Coordinates clients and schedules operations to workers
  - Maps `client:tensor` to physical locations (TensorHandle)
  - Round-robin scheduling to workers
  - Handles multiple concurrent clients

- **`gt/worker/`** - Executes operations using PyTorch/numpy backends
  - Dumb executors that run dispatcher instructions
  - Backend: PyTorch (default, fast) or numpy (reference)
  - One worker per GPU

- **`gt/transport/`** - Communication protocols between components
  - TCP-based with pickle serialization
  - Client ↔ Dispatcher: Simpler protocol
  - Dispatcher ↔ Worker: More precise protocol

- **`benchmarks/`** - Performance comparisons with pure PyTorch
- **`tests/`** - Pytest-based numeric correctness tests

See `CLAUDE.md` for detailed architecture documentation.

## Design Philosophy

This code is designed to be SIMPLE and READABLE for researchers. We prioritize clarity over performance in this initial implementation.
