# Hot Path Detection Integration (Stream-Based)

## Key Insight

**Hot path detection operates on the INSTRUCTION STREAM**, not on batches.

- **Batching** = transport optimization (reducing network round-trips)
- **Stream** = semantic structure (the actual sequence of operations)

The detector should see individual instructions as they arrive.

## Stream-Based Architecture

```
Dispatcher → Network → Worker._process_command()
                          ↓
                    [record_instruction]  ← Hot path detector sees this
                          ↓
                    Execute operation
```

## Integration with Worker

### 1. Initialize detector in Worker.__init__

```python
# In gt/worker/worker.py

from gt.worker.hotpath_detector import HotPathDetector

class Worker:
    def __init__(self, worker_id: str, backend="pytorch", batch_size: int = None):
        # ... existing init code ...

        # Hot path detection (stream-based)
        self.hotpath_detector = None
        if os.environ.get('GT_AUTO_COMPILE', '0') == '1':
            self.hotpath_detector = HotPathDetector(
                window_size=20,  # Track last 20 instructions
                hot_threshold=10,  # Detect after 10 repetitions
                min_sequence_length=5  # Minimum 5-op sequences
            )
            print(f"Worker {worker_id}: Stream-based hot path detection enabled")
```

### 2. Record instructions as they arrive

```python
# In gt/worker/worker.py

def _process_command(self, cmd: WorkerCommand) -> WorkerResponse:
    """Process a command from dispatcher."""
    try:
        # Record instruction in stream (BEFORE batching logic)
        if self.hotpath_detector:
            if isinstance(cmd, WorkerBinaryOp):
                self.hotpath_detector.record_instruction(
                    'binary', cmd.op, input_count=2
                )
            elif isinstance(cmd, WorkerUnaryOp):
                self.hotpath_detector.record_instruction(
                    'unary', cmd.op, input_count=1
                )

        # Now handle the command (with or without batching)
        if isinstance(cmd, WorkerBinaryOp):
            return self._handle_binary_op(cmd)
        elif isinstance(cmd, WorkerUnaryOp):
            return self._handle_unary_op(cmd)
        # ...

    except Exception as e:
        return WorkerResponse(success=False, error=str(e))
```

### 3. Reset at sync points

```python
# In gt/worker/worker.py

def _handle_get_data(self, cmd: WorkerGetData) -> WorkerResponse:
    """Get tensor data - this is a sync point."""

    # Reset sequence tracking at sync points
    if self.hotpath_detector:
        self.hotpath_detector.reset_sequence()

    # ... rest of get_data logic ...
```

## Simplifying: Remove Batching?

Since batching is just transport optimization and adds complexity, consider:

**Option 1: Keep simple message batching, stream-based detection**
```python
# Batch messages for transport, but detector sees stream
GT_WORKER_BATCH_SIZE=10  # Reduce network overhead
GT_AUTO_COMPILE=1         # Stream-based hot path detection
```

**Option 2: Remove batching entirely (simplest)**
```python
# Process one instruction at a time
# No batching complexity, just stream processing
GT_AUTO_COMPILE=1  # Pure stream-based detection
```

The stream-based detector works regardless of batching setting!

## Environment Variables

```bash
# Enable stream-based hot path detection
GT_AUTO_COMPILE=1 python train.py

# Customize detection parameters
GT_AUTO_COMPILE=1 \
GT_HOTPATH_WINDOW=20 \
GT_HOTPATH_THRESHOLD=10 \
GT_HOTPATH_MIN_SEQ=5 \
python train.py
```

## How It Works

**Instruction stream:**
```
matmul → relu → matmul → relu → matmul → relu → ...
```

**After 10 repetitions of "matmul → relu" sequence:**
```
[HotPath] Detected hot sequence after 10 reps: Stream(a3f2e8:2ops)
```

**Then compilation is enabled for that pattern going forward.**

## Benefits of Stream-Based Approach

1. **Simpler** - No confusion between transport batching and semantic grouping
2. **More accurate** - Sees actual instruction sequences as they execute
3. **Flexible** - Works with or without message batching
4. **Natural** - Matches how code actually executes (instruction by instruction)

## Example Output

```
Worker worker_0: Stream-based hot path detection enabled
[Processing instruction stream...]
[HotPath] Detected hot sequence after 10 reps: Stream(a3f2e8:12ops)
    binary:matmul:in2 → unary:relu:in1 → binary:mul:in2 → ...
```

Then in stats:
```python
{
    'hotpath': {
        'total_instructions': 1000,
        'hot_instructions': 900,  # After detection
        'unique_sequences': 1,
        'hot_sequences': 1,
        'hot_threshold': 10,
        'window_size': 20
    }
}
```

## Next Steps

1. **Implement stream recording in worker** - Call `record_instruction()` for each op
2. **Test with training loops** - Verify sequences are detected
3. **Consider removing batching** - Simplify if not providing real benefit
4. **Measure impact** - Run benchmarks to see if it improves performance

## Discussion: Should We Remove Batching?

**Arguments for removing:**
- Adds complexity
- Minimal benefit (just network overhead reduction)
- Confuses semantics (transport vs grouping)
- Stream-based is simpler and clearer

**Arguments for keeping:**
- Does reduce network round-trips
- Can batch multiple ops into single response
- Already implemented and tested

**Recommendation:** Start with stream-based detection, measure benefits, then decide on batching.
