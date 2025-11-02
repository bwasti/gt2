# GT

An experimental multiplexing tensor framework for distributed GPU computing.

![gt_viz](https://github.com/user-attachments/assets/309cc78c-3572-44aa-be94-48b54996d24f)


```bash
pip install git+https://github.com/bwasti/gt.git
python -c 'import gt; print(gt.randn(2,2))'
```

## General Idea

The motivation for this project is a rejection of the clunky lock-step paradigm
ML researchers tend to use. GT attempts to pull some of the ideas
that are present in the decades of development done on multi-core operating systems.
It fully embraces dynamic scheduling and heavily asynchronous execution
while presenting a familiar eager frontend.

- **Three components**
    - N × clients (as many users as you want!)
    - 1 × dispatcher (for coordinating)
    - N × workers (1 per GPU)
- **Everything communicates with a stream of instructions**
   - Clients deal with math. They emit (GPU-unaware) pure functional instructions
   - The dispatcher **rewrites** these instructions on the fly to be GPU-aware and sends them to the workers
   - Workers asynchronously process these instructions, optionally JIT compiling
- **Instruction streams are annotated**
   - Clients can send "signals" which allow the dispatcher to more appropriately shard the tensors
   - Dispatchers annotate "hot" paths to give hints to workers about JIT compiling
   - Annotations are supplemented with YAML configs that specify sharding and compilation information
   - Every annotation can be safely ignored, so the same code can run anywhere (just remove the YAML)

## Quick Start

```python
import gt

a = gt.randn(1000, 1000)
b = gt.randn(1000, 1000)
c = a @ b
result = c[:4, :4]
print(result)
```

It may not look like it, but in the background GT automatically spins up an asynchronous dispatching server and GPU worker.

## Features

- **High-performance transport** - ZeroMQ (ZMQ) with automatic message batching and efficient DEALER/ROUTER pattern
- **Autograd support** - Tape-based automatic differentiation exclusively at the client layer
- **PyTorch-compatible API** - Familiar syntax for tensor operations
- **Signal-based sharding** - Declarative YAML configuration for distributed training
- **Real-time monitoring** - htop-style visualization of worker activity
- **Instruction logging** - Debug distributed execution with timeline visualizations
- **AI-assisted development** - Optimized for collaboration with AI coding assistants

## Documentation

📚 **[Read the full documentation](https://bwasti.github.io/gt)**

### Getting Started
- [Installation](https://bwasti.github.io/gt/#/installation) - Install GT and verify setup
- [Usage Guide](https://bwasti.github.io/gt/#/usage) - Auto-server mode and distributed setup

### Client API
- [Tensor Operations](https://bwasti.github.io/gt/#/client/tensor-api) - Complete operation reference
- [Autograd](https://bwasti.github.io/gt/#/client/autograd) - Automatic differentiation

### Distributed Training
- [Signal-Based Sharding](https://bwasti.github.io/gt/#/dispatcher/signaling) - Configure parallelism strategies
- [Tuning & Performance](https://bwasti.github.io/gt/#/dispatcher/tuning) - Optimize performance
- [Monitoring Tools](https://bwasti.github.io/gt/#/dispatcher/monitoring) - Real-time monitoring and debugging

### Workers
- [Backends](https://bwasti.github.io/gt/#/worker/backends) - PyTorch and NumPy backends
- [Compilation](https://bwasti.github.io/gt/#/worker/compilation) - JIT compilation with torch.compile

### Contributing
- [Contributing Guide](https://bwasti.github.io/gt/#/contributing) - Development workflow, testing, and PR guidelines

## Examples

See [examples/](examples/) directory for demonstrations:

- `demo.py` - Basic tensor operations
- `signal_demo.py` - Signal-based sharding
- `compile_demo.py` - Compilation directives
- `debug_demo.py` - Debug utilities
- `visualize_demo.py` - Instruction tape visualization

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
└──────────────────────┬──────────────────────────────────────────┘
                       │ ZMQ (DEALER → ROUTER)
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
    │PyTorch │    │PyTorch │    │PyTorch │
    │  GPU   │    │  GPU   │    │  GPU   │
    └────────┘    └────────┘    └────────┘
```

## Optimized for AI Development

GT is designed to be understood, modified, and debugged with AI coding assistants:

- **[CLAUDE.md](CLAUDE.md)** - Detailed architecture documentation for AI assistants
- **Declarative YAML configs** - Easy for AI to parse and generate
- **Tape-based debugging** - Inspect computation graphs with `gt.debug.print_tape()`
- **Instruction logging** - Track every operation with timestamps
- **Comprehensive test suite** - 50+ tests serving as executable specifications

## Contributing

Contributions welcome! This is a research prototype focused on simplicity and readability.

See [Contributing Guide](docs/contributing.md) for development workflow, testing, code style, and PR guidelines.

## License

MIT

See [License](docs/license.md) for details.
