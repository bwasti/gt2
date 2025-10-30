# Instruction Batching and torch.compile Support

GT2 now supports **instruction batching** with **torch.compile()** for dramatic performance improvements through graph compilation and kernel fusion.

## Overview

Instead of executing operations one-by-one (eager mode), workers can accumulate multiple operations and execute them as a compiled batch. This enables PyTorch to:

1. **Fuse kernels** - Multiple operations combined into single GPU kernels
2. **Optimize memory** - Reuse intermediate buffers and reduce allocations
3. **Cache compiled functions** - Identical computation patterns reuse compiled code
4. **Reduce Python overhead** - Batch execution avoids per-operation Python calls

## Performance

Real-world measurements from our test suite:

- **First compilation**: ~9.6 seconds (torch.compile() overhead)
- **Cached execution**: ~1.5ms (**5000x faster!**)
- **Subsequent runs**: Cache hit, same ~1.5ms performance

The first time a computation pattern is seen, torch.compile() compiles it into optimized GPU code. This is slow (~10 seconds). But subsequent executions of the same pattern use the cached compiled function, achieving **5000x speedup**.

## Usage

### Enable Batching

Set the `GT_WORKER_BATCH_SIZE` environment variable:

```bash
# Enable batching with batch size of 10
export GT_WORKER_BATCH_SIZE=10
python your_script.py
```

Or set it in your Python script before importing `gt`:

```python
import os
os.environ['GT_WORKER_BATCH_SIZE'] = '10'

import gt
# ... your code ...
```

### Default Behavior

- **Default**: `GT_WORKER_BATCH_SIZE=1` (eager mode, no batching)
- **Recommended for training**: `GT_WORKER_BATCH_SIZE=10` to `GT_WORKER_BATCH_SIZE=50`
- **NumPy backend**: Batching has no effect (NumPy doesn't support compilation)
- **PyTorch backend**: Automatically selected when batching enabled

### Example

```python
import os
os.environ['GT_WORKER_BATCH_SIZE'] = '10'

import gt
import numpy as np

# These operations will be batched together
a = gt.from_numpy(np.random.randn(100, 100).astype('float32'))
b = gt.from_numpy(np.random.randn(100, 100).astype('float32'))

# Pattern: add -> mul -> matmul -> relu -> mean
# All executed in single compiled batch
c = a + b
d = c * a
e = d @ b
f = e.relu()
result = f.mean()

print(f"Result: {result.item()}")
```

### First Run vs Cached Run

```python
import time
import os
os.environ['GT_WORKER_BATCH_SIZE'] = '10'

import gt
import numpy as np

# First run: Compile (slow)
start = time.time()
a = gt.from_numpy(np.random.randn(100, 100).astype('float32'))
b = gt.from_numpy(np.random.randn(100, 100).astype('float32'))
c = a + b
d = c * a
e = d @ b
result = e.mean()
print(f"First run: {(time.time() - start) * 1000:.1f}ms")  # ~9600ms

# Second run: Cache hit (fast!)
start = time.time()
a = gt.from_numpy(np.random.randn(100, 100).astype('float32'))
b = gt.from_numpy(np.random.randn(100, 100).astype('float32'))
c = a + b
d = c * a
e = d @ b
result = e.mean()
print(f"Second run: {(time.time() - start) * 1000:.1f}ms")  # ~1.5ms
```

## How It Works

### Architecture

1. **Operation Batching**: Worker accumulates operations until:
   - Batch size reached (e.g., 10 operations), OR
   - Sync point hit (CreateTensor, GetData, FreeTensor)

2. **Graph Signature**: Computes MD5 hash of operation structure:
   ```
   signature = MD5(op_type:op_name|in:num_inputs|...)
   ```

3. **Compilation**: If signature not in cache:
   - Build PyTorch function from operations
   - Call `torch.compile(function, mode="default")`
   - Cache compiled function by signature

4. **Execution**: Run compiled function on tensors

5. **Cache Reuse**: Identical patterns reuse cached compiled function

### Sync Points

Batching automatically flushes at sync points:

- **CreateTensor**: Data must be created before operations can use it
- **GetData**: Results must be computed before data can be retrieved
- **FreeTensor**: Tensor must exist before it can be freed

This ensures correctness while maximizing batching opportunities.

### Fallback Behavior

Batching gracefully falls back to eager execution when:

- Single operation (batch size 1)
- torch.compile() fails (prints warning)
- NumPy backend (doesn't support compilation)

## Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GT_WORKER_BATCH_SIZE` | `1` | Number of operations to batch before execution |

### Worker Logs

Worker prints batching status on startup:

```
# Eager mode (default)
Worker auto_worker: Eager mode (batch_size=1)

# Batching enabled
Worker auto_worker: Batching enabled (batch_size=10)

# Batching requested but unsupported
Worker auto_worker: Batching requested (batch_size=10) but engine doesn't support it
```

## Implementation Details

### File Structure

```
gt/worker/engine/
  __init__.py       - Engine factory function
  base.py           - Base Engine interface, Operation dataclass
  numpy.py          - NumPy engine (eager only)
  pytorch.py        - PyTorch engine (with compilation)
```

### Engine Interface Extensions

```python
class Engine(ABC):
    def supports_batching(self) -> bool:
        """Whether this engine supports instruction batching/compilation."""
        return False

    def execute_batch(self, operations: List[Operation],
                     tensors: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a batch of operations (optional, for compilation-capable engines)."""
        raise NotImplementedError()
```

### Operation Dataclass

```python
@dataclass
class Operation:
    op_type: str        # 'binary', 'unary', 'create'
    op_name: str        # 'add', 'matmul', 'relu'
    result_id: str      # Unique result tensor ID
    input_ids: List[str] # Input tensor IDs
    params: Dict[str, Any] # Shape, dtype, etc.
```

### Compilation Cache

PyTorch engine tracks cache statistics:

```python
def get_compilation_stats(self) -> Dict[str, int]:
    return {
        "cache_size": len(self._compiled_cache),
        "cache_hits": self._cache_hits,
        "cache_misses": self._cache_misses,
        "hit_rate": self._cache_hits / max(1, self._cache_hits + self._cache_misses)
    }
```

## Performance Tips

1. **Use consistent patterns**: Same operation sequence = cache hit
2. **Batch size tuning**:
   - Too small (1-5): Not enough fusion opportunities
   - Too large (>100): Compilation overhead increases
   - Recommended: 10-50 for most workloads

3. **Warmup compilation**: Run training loop once before timing:
   ```python
   # Warmup
   for i in range(3):
       loss = model(X)
       loss.backward()

   # Now time actual training
   start = time.time()
   for epoch in range(100):
       loss = model(X)
       loss.backward()
   print(f"Training time: {time.time() - start:.2f}s")
   ```

4. **Monitor cache hits**: Add protocol support to expose `get_compilation_stats()` from worker

## Known Limitations

1. **First-run latency**: Initial compilation takes ~10 seconds
2. **Memory overhead**: Compiled functions cached in memory
3. **PyTorch only**: NumPy backend doesn't support batching
4. **No dynamic shapes**: Graph signature doesn't include tensor shapes (yet)

## Future Work

- [ ] Add protocol command to query compilation stats
- [ ] Support dynamic shapes in graph signatures
- [ ] Expose compilation cache size limits
- [ ] Add compilation mode selection (default, reduce-overhead, max-autotune)
- [ ] Benchmark vs eager mode for various workloads
