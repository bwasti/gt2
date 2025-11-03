# Monitoring Tools

GT provides comprehensive monitoring and debugging tools for distributed systems.

## Real-Time Monitoring

Monitor running dispatchers with htop-style worker activity visualization.

### Quick Start

<img width="862" height="235" alt="Screenshot 2025-11-02 at 1 54 47 AM" src="https://github.com/user-attachments/assets/8040d59d-6869-42c8-bb2d-ee36d916935b" />

```bash
# Auto-attach to running dispatcher
python -m gt.scripts.top

# Attach to specific dispatcher
python -m gt.scripts.top --port 9000 --host localhost
```

### Features

- **Real-time EMA-smoothed activity bars** - Operation breakdown per worker
- **Color-coded operations** - MatMul, Add, ReLU, etc.
- **Idle time tracking** - Identify underutilized workers
- **Auto-detection** - Finds running dispatchers automatically
- **Non-intrusive** - Connects via ZMQ monitoring socket without affecting performance
- **Responsive layout** - Adapts to terminal width

### Screenshot

```
┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Worker        ┃ Activity                       ┃ Details                      ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ worker_0      │ ████████░░░░░░░░░░░░░░░░░░░░░░ │ 127 ops (15 matmul, 32 add) │
│ worker_1      │ ███████████░░░░░░░░░░░░░░░░░░░ │ 143 ops (18 matmul, 28 add) │
│ worker_2      │ ██████░░░░░░░░░░░░░░░░░░░░░░░░ │ 98 ops (12 matmul, 19 add)  │
│ worker_3      │ █████████░░░░░░░░░░░░░░░░░░░░░ │ 115 ops (14 matmul, 24 add) │
└───────────────┴────────────────────────────────┴──────────────────────────────┘
```

### Usage Tips

- **Check for idle workers** - Bars should be balanced
- **Watch operation mix** - High matmul % = good GPU utilization
- **Identify bottlenecks** - One idle worker suggests load imbalance
- **Monitor over time** - Patterns emerge during training

## Trace Capture

Capture event streams for offline analysis and visualization.

### Basic Usage

```bash
# Capture 2 seconds of activity
python -m gt.scripts.trace -s 2 --dir traces/

# Capture first 100 events (with 10 second timeout)
python -m gt.scripts.trace -s 10 -n 100 --dir traces/

# Specify dispatcher
python -m gt.scripts.trace -s 5 --port 9000 --host localhost --dir traces/
```

### Options

| Flag | Description | Example |
|------|-------------|---------|
| `-s, --seconds` | Maximum capture duration (required) | `-s 10` |
| `-n, --max-events` | Stop after N events | `-n 1000` |
| `--port` | Dispatcher port (auto-detected) | `--port 9000` |
| `--host` | Dispatcher host | `--host 192.168.1.100` |
| `--dir` | Output directory | `--dir traces/` |

### Output

Creates timestamped log file:
```
traces/trace_20250102_143022.log
```

Format matches `GT_INSTRUCTION_LOG` for compatibility with visualizer.

## Timeline Visualization

Generate performance timeline diagrams from instruction logs.

### Basic Usage

```bash
# From instruction log
GT_INSTRUCTION_LOG=/tmp/debug.log python your_script.py
python -m gt.scripts.visualize /tmp/debug.log --output timeline.png

# From captured trace
python -m gt.scripts.trace -s 2 --dir traces/
python -m gt.scripts.visualize traces/trace_*.log --output timeline.png
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `--output` | Output image file | `timeline.png` |
| `--dpi` | Image resolution | `150` |
| `--width` | Figure width (inches) | `16` |
| `--height` | Figure height (inches) | `10` |

### Output Features

The timeline visualization shows:

- **Timeline lanes** - One per component (Client, Dispatcher, Workers)
- **Color-coded operations** - MatMul, BinaryOp, UnaryOp, etc.
- **Event markers** - Different shapes for RECV, WORKER_SEND, WORKER_RECV
- **Data transfer sizes** - Marker size indicates message size
- **Communication flow** - Arrows showing instruction routing
- **Instruction IDs** - Annotated on key events
- **Time axis** - Elapsed time from start

### Use Cases

1. **Identify idle workers** - Gaps in worker lanes
2. **Visualize patterns** - Embarrassingly parallel, all-gather, all-reduce
3. **Find bottlenecks** - Long gaps or single-threaded sections
4. **Debug distribution** - Verify sharding and placement
5. **Measure latency** - Time between request and response

### Example Workflow

```bash
# 1. Run workload with logging
GT_INSTRUCTION_LOG=/tmp/train.log python train.py

# 2. Generate visualization
python -m gt.scripts.visualize /tmp/train.log \
    --output training_timeline.png \
    --dpi 200

# 3. Open and analyze
open training_timeline.png
```

## Instruction Stream Logging

Log all operations with timestamps for debugging and analysis.

### Enable Logging

```bash
# Set environment variable
GT_INSTRUCTION_LOG=/tmp/debug.log python your_script.py
```

### Log Format

```
  0.123s | #0042 | RECV         | CLIENT 127.0.0.1:12345 | BinaryOp        | 123KB | result=42 op=add
  0.125s | #0043 | WORKER_SEND  | WORKER worker_0        | WorkerBinaryOp  | 123KB | op=add
  0.127s | #0044 | WORKER_RECV  | WORKER worker_0        | WorkerResponse  | 45B   | success=True
  0.128s | #0045 | SEND         | CLIENT 127.0.0.1:12345 | BinaryOp        | 45B   | success=True
```

### Columns

- **Time** - Elapsed seconds since dispatcher start (millisecond precision)
- **Sequence** - Instruction number (increments for each event)
- **Event** - Event type (RECV, SEND, WORKER_SEND, WORKER_RECV, CONNECT, DISCONNECT)
- **Source** - Client or worker identifier
- **Command** - Command type (CreateTensor, BinaryOp, UnaryOp, etc.)
- **Size** - Message size (B, KB, MB)
- **Details** - Additional context (tensor IDs, operation names, etc.)

### Analysis Commands

```bash
# View operations
cat /tmp/debug.log | less

# Find specific operation
grep "matmul" /tmp/debug.log

# Find errors
grep "ERROR\|FAILED" /tmp/debug.log

# Show worker activity
grep "WORKER_SEND" /tmp/debug.log

# Time between operations
cat /tmp/debug.log | awk '{print $1, $7}' | less

# Count operation types
grep "RECV" /tmp/debug.log | awk '{print $9}' | sort | uniq -c
```

### Use Cases

1. **Debug hangs** - See last operation before timeout
2. **Identify slow operations** - Large timestamp gaps
3. **Understand execution flow** - Follow instructions through system
4. **Verify sharding** - Check tensor placement across workers
5. **Performance analysis** - Feed into visualizer

## Debug Utilities

Python API for inspecting system state.

### Print Autograd Tape

View computational graph:

```python
import gt

# Build computation
a = gt.randn(10, 10, requires_grad=True)
b = gt.randn(10, 10, requires_grad=True)
loss = (a + b).sum()

# View tape
gt.debug.print_tape()
```

Output:
```
Autograd Tape:
1. ADD: inputs=[tensor_0, tensor_1] -> tensor_2
2. SUM: inputs=[tensor_2] -> tensor_3
```

### Get Worker Statistics

Query live system stats:

```python
import gt

# Run some operations
x = gt.randn(100, 100)
y = x @ x

# Get stats
stats = gt.debug.get_worker_stats()
print(stats)
```

Output includes:
- **total_instructions** - Total operations executed
- **hot_instructions** - Operations in hot paths
- **unique_sequences** - Number of distinct patterns
- **hot_sequences** - Patterns detected as hot paths
- **compilation_stats** - If `GT_AUTO_COMPILE=1`

### Print Worker Stats

Formatted output:

```python
gt.debug.print_worker_stats()
```

Output:
```
Worker Statistics:
  worker_0: 127 instructions (15 matmul, 32 add, ...)
  worker_1: 143 instructions (18 matmul, 28 add, ...)
  ...
```

## Environment Variables for Debugging

### Debug Output Flags

```bash
# Framework status (startup, connections)
GT_VERBOSE=1 python script.py

# Client-side operations
GT_DEBUG_CLIENT=1 python script.py

# Dispatcher scheduling
GT_DEBUG_DISPATCHER=1 python script.py

# Worker execution
GT_DEBUG_WORKER=1 python script.py

# Compilation activity
GT_DEBUG_COMPILE=1 python script.py

# Combine multiple
GT_VERBOSE=1 GT_DEBUG_DISPATCHER=1 GT_DEBUG_WORKER=1 python script.py
```

### When to Use Each Flag

| Flag | Use When |
|------|----------|
| `GT_VERBOSE` | Verifying system startup |
| `GT_DEBUG_CLIENT` | Debugging tensor operations |
| `GT_DEBUG_DISPATCHER` | Understanding scheduling decisions |
| `GT_DEBUG_WORKER` | Investigating worker execution |
| `GT_DEBUG_COMPILE` | Debugging compilation issues |
| `GT_INSTRUCTION_LOG` | Complete execution trace |

## Complete Monitoring Workflow

### 1. Development Phase

Monitor live during development:

```bash
# Terminal 1: Run code with verbose output
GT_VERBOSE=1 python train.py

# Terminal 2: Monitor workers
python -m gt.scripts.top
```

### 2. Performance Analysis

Capture and analyze:

```bash
# Capture trace
python -m gt.scripts.trace -s 10 --dir traces/

# Visualize
python -m gt.scripts.visualize traces/trace_*.log --output perf.png

# Analyze
open perf.png
```

### 3. Debugging Issues

Full instrumentation:

```bash
# Enable all debugging
GT_INSTRUCTION_LOG=/tmp/debug.log \
GT_VERBOSE=1 \
GT_DEBUG_DISPATCHER=1 \
GT_DEBUG_WORKER=1 \
python buggy_script.py

# Analyze log
grep "ERROR" /tmp/debug.log
python -m gt.scripts.visualize /tmp/debug.log --output debug.png
```

### 4. Production Monitoring

Lightweight monitoring:

```bash
# Production: minimal logging
GT_INSTRUCTION_LOG=/var/log/gt/production.log python train.py

# Monitor remotely
python -m gt.scripts.top --host production-server --port 9000
```

## Next Steps

- [Tuning Guide](tuning.md) - Optimize performance based on monitoring data
- [Signal-Based Sharding](signaling.md) - Configure parallelism strategies
- [Contributing](../contributing.md) - Add monitoring features
