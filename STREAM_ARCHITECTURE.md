# GT Stream Architecture

## Overview

GT uses a clean stream-based architecture where instructions flow through multiple layers:

```
Client → Dispatcher → Worker
```

Each layer can have **stream modifiers** that transform the instruction stream.

## Stream Modifiers

Stream modifiers follow a consistent pattern:

### Pattern (from HotPathDetector)
```python
class StreamModifier:
    def __init__(self, ...):
        # Configuration
        # Statistics tracking

    def process(self, cmd) -> Iterator[cmd]:
        """Process command and yield transformed commands."""
        # Transform the stream
        yield cmd  # or transformed version

    def get_stats(self) -> dict:
        """Return statistics about transformations."""
        return {...}
```

**Key principles:**
- Pure stream transformers (no buffering, no execution)
- Take commands in, yield commands out
- May inject additional commands (markers, moves, etc.)
- Track statistics, not tapes
- Tapes are logged by InstructionStream

## Existing Stream Modifiers

### 1. HotPathDetector (Worker)
- **Location**: `gt/worker/hotpath_detector.py`
- **Purpose**: Detect repeated instruction sequences
- **Transformation**: Injects `WorkerHotPathStart` and `WorkerHotPathEnd` markers
- **Stats**: Tracks hot sequences, repetitions

### 2. ShardingStreamModifier (Dispatcher) - NEW
- **Location**: `gt/dispatcher/sharding_modifier.py`
- **Purpose**: Shard large operations across workers
- **Transformation**: Will inject move/reduce/copy operations (not yet implemented)
- **Stats**: Tracks sharded operations, tensor sizes
- **Status**: **Disabled by default** (`enable_sharding=False`)

## Tape Dumping

### Dispatcher: InstructionStream

The primary tape dumping mechanism is `InstructionStream` in the dispatcher:

```bash
# Enable tape dumping to file
export GT_INSTRUCTION_LOG=/tmp/gt_instructions.log
python my_script.py
```

This logs ALL instructions:
- Client commands (RECV/SEND)
- Worker operations (WORKER_SEND/WORKER_RECV)
- Connection events (CONNECT/DISCONNECT)

Format:
```
<elapsed> | #<seq> | <event_type> | <source> | <command> | <details>
```

Example:
```
1.608s | #0011 | RECV         | CLIENT 006b8b4568           | BinaryOp        |       117B | result=2 op=add left=0 right=1
1.608s | #0012 | WORKER_SEND  | WORKER auto_worker          | WorkerBinaryOp  |       152B | op=add result=006b8b4568_2
1.608s | #0013 | WORKER_RECV  | WORKER auto_worker          | WorkerBinaryOp  |        91B | success=True
```

### Client: Autograd Tape

The client maintains an autograd tape for gradients:

```python
import gt

# Get autograd tape
tape = gt.debug.get_tape()

# Pretty print
gt.debug.print_tape()
```

### Worker: Stats

Workers provide stats via:

```python
import gt

# Get worker compilation/execution stats
gt.debug.print_worker_stats()
```

## Adding New Stream Modifiers

To add a new stream modifier:

1. **Create the class** following the pattern:
   ```python
   class MyModifier:
       def __init__(self, enabled=False):
           self.enabled = enabled
           # stats

       def process(self, cmd) -> Iterator[cmd]:
           if not self.enabled:
               yield cmd
               return
           # Transform stream
           yield cmd

       def get_stats(self) -> dict:
           return {...}
   ```

2. **Integrate into dispatcher/worker**:
   ```python
   self.my_modifier = MyModifier(enabled=False)
   ```

3. **Use in processing loop**:
   ```python
   for cmd in self.my_modifier.process(incoming_cmd):
       # Process transformed command
       pass
   ```

4. **Stream modifiers automatically logged** by InstructionStream - no need to implement custom tape dumping!

## Debug Environment Variables

```bash
# Enable instruction logging to file
GT_INSTRUCTION_LOG=/tmp/gt.log python script.py

# Enable verbose framework messages
GT_VERBOSE=1 python script.py

# Enable debug output
GT_DEBUG_CLIENT=1 python script.py
GT_DEBUG_DISPATCHER=1 python script.py
GT_DEBUG_WORKER=1 python script.py
GT_DEBUG_COMPILE=1 python script.py
```

## Testing Tape Dumping

See `test_tape_dump.py` and `test_hotpath_in_logs.py` for examples.

## Future Work

### ShardingStreamModifier Implementation

The ShardingStreamModifier is currently a stub. To implement:

1. **Detect large operations** in `process()`:
   - Check tensor sizes against `shard_threshold`
   - Identify operations that benefit from sharding (matmul, reduce)

2. **Inject sharding operations**:
   ```python
   # Instead of: MATMUL A, B -> C
   # Yield:
   yield ShardOp(tensor=A, num_shards=3)
   yield ShardOp(tensor=B, num_shards=3)
   yield MatmulOp(A_0, B_0, result=C_0)
   yield MatmulOp(A_1, B_1, result=C_1)
   yield MatmulOp(A_2, B_2, result=C_2)
   yield ReduceOp([C_0, C_1, C_2], result=C)
   ```

3. **Track sharded tensors** in `self.tensor_sharded`

4. **Update stats** in `get_stats()`
