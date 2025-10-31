# Hot Path Detection Integration

## Overview

The hot path detector automatically enables torch.compile() when it detects repeated batch patterns at the worker level.

## Integration with Worker

### 1. Initialize detector in Worker.__init__

```python
# In gt/worker/worker.py

from gt.worker.hotpath_detector import HotPathDetector

class Worker:
    def __init__(self, worker_id: str, backend="pytorch", batch_size: int = None):
        # ... existing init code ...

        # Hot path detection (only if enabled)
        self.hotpath_detector = None
        if os.environ.get('GT_AUTO_COMPILE', '0') == '1':
            self.hotpath_detector = HotPathDetector(
                hot_threshold=int(os.environ.get('GT_HOTPATH_THRESHOLD', '10')),
                enable_auto_compile=True
            )
            print(f"Worker {worker_id}: Hot path detection enabled")
```

### 2. Check detector in _flush_batch()

```python
# In gt/worker/worker.py

def _flush_batch(self):
    """Execute all pending operations as a batch."""
    if not self.pending_operations:
        return

    # Convert commands to Operation objects
    operations = []
    for cmd in self.pending_operations:
        # ... existing conversion code ...
        operations.append(Operation(...))

    # Hot path detection: should we compile this batch?
    should_compile_batch = False
    if self.hotpath_detector:
        should_compile_batch = self.hotpath_detector.record_batch(operations)

    # Execute batch with dynamic compilation decision
    try:
        # Option 1: Override engine's compilation setting for this batch
        if should_compile_batch and not self.engine.enable_compilation:
            # Temporarily enable compilation for this hot batch
            original_setting = self.engine.enable_compilation
            self.engine.enable_compilation = True
            results = self.engine.execute_batch(operations, self.tensors)
            self.engine.enable_compilation = original_setting
        else:
            results = self.engine.execute_batch(operations, self.tensors)

        # Update tensor storage
        for result_id, tensor in results.items():
            self.tensors[result_id] = tensor

        # Record completion for timing
        if self.hotpath_detector:
            self.hotpath_detector.record_batch_completion(operations)

    except Exception as e:
        print(f"Worker {self.worker_id}: Batch execution failed: {e}")
        raise
    finally:
        self.pending_operations.clear()
```

### 3. Add stats endpoint

```python
# In gt/worker/worker.py

def _handle_get_stats(self, cmd: WorkerGetStats) -> WorkerResponse:
    """Handle stats request."""
    stats = {
        'operations': self.stats['operations'].copy()
    }

    # Compilation stats from engine
    if hasattr(self.engine, 'get_compilation_stats'):
        stats['compilation'] = self.engine.get_compilation_stats()

    # Hot path detection stats
    if self.hotpath_detector:
        stats['hotpath'] = self.hotpath_detector.get_stats()

    return WorkerResponse(success=True, data=stats)
```

## Environment Variables

```bash
# Enable automatic hot path detection and compilation
GT_AUTO_COMPILE=1 python train.py

# Customize detection threshold (default: 10)
GT_AUTO_COMPILE=1 GT_HOTPATH_THRESHOLD=20 python train.py

# Combine with message batching
GT_WORKER_BATCH_SIZE=10 GT_AUTO_COMPILE=1 python train.py
```

## Behavior

**Without GT_AUTO_COMPILE:**
- Worker uses GT_COMPILE setting (manual control)
- Compile all batches or none

**With GT_AUTO_COMPILE=1:**
- First 10 iterations: No compilation (fast startup)
- Iterations 11+: Compilation enabled for detected hot paths
- Different batch patterns can have different compilation settings

## Benefits

1. **Zero overhead for short runs** - Scripts with <10 iterations run fast
2. **Automatic optimization for training** - Long loops get compiled automatically
3. **Per-pattern compilation** - Only hot paths pay compilation cost
4. **No manual tuning** - GT_COMPILE flag becomes optional

## Example Output

```
Worker worker_0: Hot path detection enabled (threshold=10)
Worker worker_0: Message batching enabled (batch_size=10)
[HotPath] Worker detected hot batch after 10 repetitions: Batch(a3f2e8:15ops)
```

Then in stats:
```python
{
    'hotpath': {
        'total_batches': 100,
        'hot_batches_executed': 90,  # Last 90 after threshold
        'unique_patterns': 1,
        'hot_paths': 1,
        'hot_threshold': 10
    }
}
```

## Testing

```bash
# Test with MLP training
GT_WORKER_BATCH_SIZE=10 GT_AUTO_COMPILE=1 python examples/train_mlp_gt.py

# Should see:
# - First 10 epochs fast (no compilation)
# - "[HotPath] Worker detected..." message
# - Remaining epochs compiled
```

## Future Enhancements

1. **Per-batch compilation override** - Let detector control compilation per-batch rather than globally
2. **Performance-based disabling** - If compilation makes things slower, revert
3. **Persistent hot path cache** - Save detected patterns across runs
4. **Shape-agnostic signatures** - Group patterns by structure, not exact shapes
