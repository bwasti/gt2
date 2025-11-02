# Tuning & Performance

Optimize GT performance through configuration, environment variables, and monitoring.

## Environment Variables

### Configuration

| Variable | Description | Values | Default |
|----------|-------------|--------|---------|
| `GT_CONFIG` | Path to sharding config YAML | File path | None |
| `GT_AUTO_COMPILE` | Enable automatic hot path detection | 0 or 1 | 0 |
| `GT_COMPILE` | Force compile all operations | 0 or 1 | 0 |
| `GT_WORKER_BATCH_SIZE` | Operations to batch per worker | Integer | 1 |

### Debug Output

| Variable | Description | Values | Default |
|----------|-------------|--------|---------|
| `GT_VERBOSE` | Framework status messages | 0 or 1 | 0 |
| `GT_DEBUG_CLIENT` | Client-side debug messages | 0 or 1 | 0 |
| `GT_DEBUG_DISPATCHER` | Dispatcher debug messages | 0 or 1 | 0 |
| `GT_DEBUG_WORKER` | Worker debug messages | 0 or 1 | 0 |
| `GT_DEBUG_COMPILE` | Compilation debug messages | 0 or 1 | 0 |

### Logging

| Variable | Description | Values | Default |
|----------|-------------|--------|---------|
| `GT_INSTRUCTION_LOG` | Instruction stream log file | File path | None |

### Usage Example

```bash
# Production training with compilation
GT_CONFIG=sharding.yaml \
GT_AUTO_COMPILE=1 \
GT_WORKER_BATCH_SIZE=10 \
python train.py

# Debugging with full logging
GT_CONFIG=sharding.yaml \
GT_INSTRUCTION_LOG=/tmp/debug.log \
GT_VERBOSE=1 \
GT_DEBUG_DISPATCHER=1 \
python train.py
```

## Performance Optimization

### 1. Automatic Compilation

Enable hot path detection and JIT compilation:

```bash
GT_AUTO_COMPILE=1 python train.py
```

GT automatically identifies repeated operation sequences and compiles them with `torch.compile`.

**When to use:**
- Training loops with consistent patterns
- Repeated forward/backward passes
- Operations executing 10+ times

**Configuration:**

```yaml
# config.yaml
hot_path:
  shard:
    axis: 0
    workers: [0, 1, 2, 3]
  compile: true  # Mark as compilation boundary
```

### 2. Worker Batching

Process multiple operations per worker round-trip:

```bash
GT_WORKER_BATCH_SIZE=10 python train.py
```

Reduces communication overhead by batching operations.

**Trade-offs:**
- **Higher throughput** - Fewer round-trips
- **Higher latency** - Must wait for batch to fill
- **More memory** - Batched operations held in memory

**Recommended values:**
- Interactive use: `1` (default)
- Training: `10-50`
- Batch inference: `100+`

### 3. Sharding Strategy

Choose appropriate parallelism strategy:

**Data Parallelism** (Recommended for most workloads)
```yaml
data_parallel:
  shard:
    axis: 0              # Batch dimension
    workers: [0, 1, 2, 3]
```

**Pros:**
- Simple to configure
- Works for most models
- Good scaling to many workers

**Cons:**
- Each worker needs full model in memory
- Limited by model size

**Model Parallelism** (For large models)
```yaml
model_parallel:
  shard:
    axis: 1              # Feature dimension
    workers: [0, 1]
```

**Pros:**
- Distributes model memory across workers
- Enables larger models

**Cons:**
- More communication overhead
- Harder to configure

**Pipeline Parallelism** (For very deep models)
```yaml
stage_1:
  shard:
    axis: 0
    workers: [0, 1]
  backward: stage_1_backward

stage_2:
  shard:
    axis: 0
    workers: [2, 3]
  backward: stage_2_backward
```

**Pros:**
- Good for deep models
- Balances compute across stages

**Cons:**
- Complex configuration
- Potential pipeline bubbles

### 4. Worker Placement

Optimize worker-to-GPU assignment:

```bash
# Separate GPUs per worker
CUDA_VISIBLE_DEVICES=0 python -m gt.worker --host localhost -p 9000 &
CUDA_VISIBLE_DEVICES=1 python -m gt.worker --host localhost -p 9000 &
CUDA_VISIBLE_DEVICES=2 python -m gt.worker --host localhost -p 9000 &
CUDA_VISIBLE_DEVICES=3 python -m gt.worker --host localhost -p 9000 &
```

**Best practices:**
- One worker per GPU
- Pin each worker to specific GPU with `CUDA_VISIBLE_DEVICES`
- Use high-bandwidth interconnect (NVLink, InfiniBand)

### 5. Message Batching

ZeroMQ automatically batches messages at transport layer. No configuration needed.

## Profiling

### 1. Instruction Stream Logging

Identify slow operations:

```bash
GT_INSTRUCTION_LOG=/tmp/profile.log python train.py
```

Analyze the log:
```bash
# Find operations with large time gaps
cat /tmp/profile.log | awk '{print $1, $7, $9}' | less

# Look for:
# - Long gaps between operations (> 0.1s)
# - Repeated patterns (candidates for compilation)
# - Cross-worker operations (expensive)
```

### 2. Real-Time Monitoring

Monitor live system:

```bash
# In separate terminal
python -m gt.scripts.top
```

Look for:
- **Idle workers** - Unbalanced load
- **High matmul %** - Good GPU utilization
- **High overhead** - Too many small operations

### 3. Timeline Visualization

Generate performance timeline:

```bash
# Capture trace
python -m gt.scripts.trace -s 10 --dir traces/

# Visualize
python -m gt.scripts.visualize traces/trace_*.log --output timeline.png
```

Identify:
- Idle workers (gaps in timeline)
- Communication bottlenecks (arrows)
- Unbalanced distribution

### 4. Worker Statistics

Query system stats:

```python
import gt

# ... run workload ...

stats = gt.debug.get_worker_stats()
print(stats)

# Shows:
# - total_instructions: Total operations
# - hot_instructions: Repeated patterns
# - unique_sequences: Distinct patterns
# - compilation_stats: If GT_AUTO_COMPILE=1
```

## Performance Patterns

### High Communication Overhead

**Symptoms:**
- Many small operations
- High percentage of non-compute operations
- Timeline shows mostly arrows (communication)

**Solutions:**
1. Enable batching: `GT_WORKER_BATCH_SIZE=10`
2. Reduce operation frequency
3. Use larger batch sizes
4. Enable compilation: `GT_AUTO_COMPILE=1`

### Unbalanced Load

**Symptoms:**
- Some workers idle
- Uneven operation distribution in monitor
- Timeline shows gaps

**Solutions:**
1. Adjust sharding: Use more/fewer workers
2. Balance batch sizes across workers
3. Use data parallelism for uniform load

### Memory Bottleneck

**Symptoms:**
- OOM errors
- Worker crashes
- Slow garbage collection

**Solutions:**
1. Reduce batch size
2. Use model parallelism
3. Enable gradient checkpointing (future feature)
4. Free unused tensors explicitly

### Compilation Overhead

**Symptoms:**
- First iterations very slow
- Then steady-state fast
- `GT_DEBUG_COMPILE=1` shows many recompiles

**Solutions:**
1. Warmup: Run a few iterations before timing
2. Use compilation boundaries in YAML
3. Ensure consistent shapes

## Benchmarking

### Compare Against Baseline

```python
# benchmark.py
import time
import gt

# Warmup
for _ in range(10):
    x = gt.randn(1000, 1000)
    y = x @ x

# Benchmark
start = time.time()
for _ in range(100):
    x = gt.randn(1000, 1000)
    y = x @ x
    result = y.data.numpy()  # Force synchronization
end = time.time()

print(f"Time: {end - start:.3f}s")
print(f"Ops/sec: {100 / (end - start):.1f}")
```

Run with different configurations:
```bash
# Baseline
python benchmark.py

# With compilation
GT_AUTO_COMPILE=1 python benchmark.py

# With batching
GT_WORKER_BATCH_SIZE=10 python benchmark.py

# Both
GT_AUTO_COMPILE=1 GT_WORKER_BATCH_SIZE=10 python benchmark.py
```

### Scaling Test

Measure performance vs. worker count:

```python
# scaling_test.py
import gt
import time

for num_workers in [1, 2, 4, 8]:
    gt.gpu_workers(num_workers)

    start = time.time()
    # Run workload
    end = time.time()

    print(f"{num_workers} workers: {end - start:.3f}s")
```

## Troubleshooting

### High Latency

```bash
# Check operation latency
GT_INSTRUCTION_LOG=/tmp/debug.log python script.py

# Look for large time gaps
grep "RECV\|SEND" /tmp/debug.log
```

### Low GPU Utilization

```bash
# Monitor workers
python -m gt.scripts.top

# If idle time > 50%:
# 1. Increase batch size
# 2. Reduce number of workers
# 3. Enable batching
```

### Frequent Recompilation

```bash
# Debug compilation
GT_AUTO_COMPILE=1 GT_DEBUG_COMPILE=1 python script.py

# If many "Recompiling" messages:
# 1. Fix inconsistent shapes
# 2. Use explicit compilation boundaries
# 3. Disable auto-compile
```

## Best Practices

1. **Start simple** - Data parallelism, no compilation
2. **Profile first** - Measure before optimizing
3. **Optimize bottlenecks** - Don't guess
4. **Test configurations** - A/B test different settings
5. **Monitor production** - Use `gt.scripts.top` in deployment
6. **Log issues** - Use `GT_INSTRUCTION_LOG` for debugging
7. **Iterate** - Performance tuning is iterative

## Next Steps

- [Monitoring Guide](monitoring.md) - Real-time monitoring tools
- [Signal-Based Sharding](signaling.md) - Configure parallelism strategies
- [Contributing](../contributing.md) - Add performance improvements
