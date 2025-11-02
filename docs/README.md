# GT Documentation

Welcome to GT - a multiplexing tensor framework for distributed GPU computing.

## What is GT?

GT is a distributed frontend for GPU ML operations that multiplexes users to work on the same cluster simultaneously. It automatically shards and places tensors, schedules operations to maximize GPU utilization.

```bash
pip install git+https://github.com/bwasti/gt.git
python -c 'import gt; print(gt.randn(2,2))'
```

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

## General Idea

The motivation for this project is a rejection of the clunky lock-step paradigm ML researchers tend to use. GT attempts to pull some of the ideas that are present in the decades of development done on multi-core operating systems. It fully embraces dynamic scheduling and heavily asynchronous execution while presenting a familiar eager frontend.

**Three components:**
- N × clients (as many users as you want!)
- 1 × dispatcher (for coordinating)
- N × workers (1 per GPU)

**Everything communicates with a stream of instructions:**
- Clients deal with math. They emit (GPU-unaware) pure functional instructions
- The dispatcher **rewrites** these instructions on the fly to be GPU-aware and sends them to the workers
- Workers asynchronously process these instructions, optionally JIT compiling

**Instruction streams are annotated:**
- Clients can send "signals" which allow the dispatcher to more appropriately shard the tensors
- Dispatchers annotate "hot" paths to give hints to workers about JIT compiling
- Annotations are supplemented with YAML configs that specify sharding and compilation information
- Every annotation can be safely ignored, so the same code can run anywhere (just remove the YAML)

## Key Features

- **High-performance transport** - ZeroMQ (ZMQ) with automatic message batching
- **Autograd support** - Tape-based automatic differentiation
- **PyTorch-compatible API** - Familiar syntax for tensor operations
- **Signal-based sharding** - Control tensor placement with YAML configs
- **Real-time monitoring** - htop-style visualization of worker activity
- **Instruction logging** - Debug distributed execution with timeline visualizations

## Examples

See [examples/](https://github.com/bwasti/gt/tree/main/examples) directory for demonstrations.

## License

[MIT](license.md)
