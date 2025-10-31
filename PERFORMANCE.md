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

## Solution: Two-Level Batching

GT supports two independent types of batching:

### 1. Worker-Level Message Batching ✅ WORKING

Batch operations at the worker to reduce dispatcher↔worker round-trips:

```bash
# Batch 10 operations together, execute eagerly (no compilation)
GT_WORKER_BATCH_SIZE=10 GT_COMPILE=0 python train.py
```

**How it works:**
- Worker buffers up to N operations
- Executes them as a batch (either eagerly or compiled)
- Reduces worker round-trips by ~10x

**Status:** Working! Operations execute eagerly without dependency tracking issues.

### 2. Worker Compilation 🚧 BROKEN

Compile batched operations with torch.compile():

```bash
# Enable compilation (currently broken):
GT_WORKER_BATCH_SIZE=10 GT_COMPILE=1 python train.py
```

**Problem:** Graph building doesn't track intermediate tensor dependencies correctly, causing NoneType errors.

**What needs fixing:**
- `gt/worker/engine/pytorch.py` execute_batch() dependency tracking
- Operations in a batch that depend on each other fail to resolve

### 3. Client-Level Message Batching 📋 TODO

Buffer operations at the client before sending to dispatcher:

```bash
# Future feature:
GT_CLIENT_BATCH_SIZE=100 python train.py
```

**Would provide:**
- Reduce client→dispatcher round-trips
- Biggest potential speedup (eliminates most network overhead)

**Implementation complexity:**
- Need to buffer operations before sending
- Handle sync points (.item(), .data, .backward()) that force flush
- Map batch responses back to pending tensors

### Expected Speedup

- **Worker batching alone**: 1-2x (reduces worker communication)
- **Client batching**: 5-10x (reduces client↔dispatcher messages)
- **Both + compilation**: 10-20x (network + compute optimization)

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
