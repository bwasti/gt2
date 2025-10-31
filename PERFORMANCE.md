# GT Performance Characteristics

## Current Status

GT is currently **~4-6x slower** than pure PyTorch for training workloads due to client-server communication overhead.

### Benchmark (Qwen3 tiny model, batch_size=2)
- **PyTorch**: 0.7s total, 0.03s per batch
- **GT (no batching)**: 4.1s total, 0.08s per batch
- **Overhead**: Each operation requires 4 network round-trips

## Root Cause

Without batching, every operation goes through the full client→dispatcher→worker→dispatcher→client cycle:

```
Training one batch (~2,880 operations):
  - 2,880 ops × 4 round-trips = ~11,520 network messages
  - Even on localhost, this adds significant latency
```

You can visualize this with:
```bash
GT_INSTRUCTION_LOG=/tmp/gt.log python train.py --epochs 1
wc -l /tmp/gt.log  # Shows total instructions
```

## Solution: Operation Batching

The framework supports batching operations together, but it's currently **broken**:

```bash
# This fails with NoneType errors:
GT_WORKER_BATCH_SIZE=10 python train.py
```

**Problem**: The graph builder in `worker.py` doesn't correctly track intermediate tensor dependencies. When operations reference tensors created by previous operations in the same batch, those tensors are `None`.

### What Needs to be Fixed

1. **Tensor Dependency Tracking** in `gt/worker/worker.py`:
   - When batching operations, need to ensure intermediate results are available
   - Build a proper dependency graph before execution
   - Or execute in topological order within the batch

2. **PyTorch Backend Batching** in `gt/worker/engine/pytorch.py`:
   - The `execute_batch()` method needs to handle intermediate tensors
   - Currently has TODO comment about this issue (line 184-188)

### Expected Speedup with Batching

With batching enabled and working:
- **10x-20x speedup** by amortizing network overhead
- Batch of 10-100 operations → 1 round-trip instead of 40-400

## Current Workarounds

For now, use `--pytorch` flag to benchmark against pure PyTorch:
```bash
python train.py --pytorch  # Fast, no overhead
python train.py           # GT backend (slower but distributed-ready)
```

## Future Optimizations

1. **Fix batching** - highest priority, biggest impact
2. **torch.compile()** - PyTorch backend already supports this (needs batching to be effective)
3. **Async operations** - Pipeline client/worker communication
4. **Local-first execution** - Auto-detect local-only workload and skip dispatcher
