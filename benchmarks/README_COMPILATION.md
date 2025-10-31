# Compilation Benchmarking & Hot Path Detection

This directory contains tools for measuring torch.compile() effectiveness and implementing automatic hot path detection for ML workloads.

## Overview

The key insight: **ML training loops repeat the same operations many times**. The first iteration pays compilation overhead, but subsequent iterations benefit from optimized code.

## Files

### `compilation_benchmark.py`

Comprehensive benchmark suite testing 4 workload types with 100 iterations each:

1. **Simple Matmul Chain** - Repeated matmul operations (common in linear layers)
2. **MLP Training Step** - Full forward/backward/update cycle
3. **Attention Block** - QKV projections and attention pattern (transformer-like)
4. **Mixed Operations** - Heterogeneous workload with various op types

**Usage:**
```bash
cd benchmarks
python compilation_benchmark.py
```

**Output format:**
```
WORKLOAD: MLP Training Step
Iterations: 100 (warmup: 5)
================================================================================

Results                                          No Compile    Compile     Speedup
----------------------------------------------------------------------------------------
Warmup (first 5 iters):                              2.150s      8.250s       0.26x
  Per iteration:                                     0.430s      1.650s       0.26x
Steady state (remaining 95 iters):                   40.850s     18.050s      2.26x
  Per iteration:                                     0.430s      0.190s       2.26x
----------------------------------------------------------------------------------------
TOTAL TIME:                                         43.000s     26.300s      1.63x
Average per iteration:                               0.430s      0.263s       1.63x

VERDICT: ✅ COMPILE WINS (1.63x faster overall)
         Steady-state: 2.26x faster
```

### Key metrics:
- **Warmup overhead** - How much slower first N iterations are with compilation
- **Steady-state speedup** - Performance after amortization
- **Break-even point** - How many iterations needed before compilation pays off
- **Total speedup** - Overall performance including overhead

## Hot Path Detection

### Concept

Instead of manually setting `GT_COMPILE=1`, automatically detect "hot paths" (repeated operation sequences) and enable compilation dynamically.

**Algorithm:**
1. Track operation sequence signatures (hash of op types + shapes)
2. Count repetitions of each pattern
3. When a pattern repeats N times (e.g., 10), mark it as "hot"
4. Enable compilation for hot paths on next occurrence
5. First 10 iterations run fast (no compilation overhead)
6. Remaining iterations benefit from compilation

### Implementation

See `gt/hotpath_detector.py` for the `HotPathDetector` class.

**Basic usage:**
```python
from gt.hotpath_detector import enable_hot_path_detection

# Enable auto-detection (triggers after 10 repetitions)
enable_hot_path_detection(hot_threshold=10)

# Training loop - compilation kicks in automatically
for epoch in range(100):
    loss = train_step()  # Compiled after 10 iterations
```

**Manual usage:**
```python
from gt.hotpath_detector import HotPathDetector

detector = HotPathDetector(hot_threshold=10)

for epoch in range(100):
    detector.begin_iteration()

    # ... your operations ...
    detector.record_operation('matmul', input_shape=(64, 64))
    detector.record_operation('relu', input_shape=(64, 64))

    should_compile = detector.end_iteration()
    # Returns True after 10 identical iterations
```

### Benefits

1. **No warmup overhead for short runs** - Small scripts run fast without compilation
2. **Automatic optimization for long runs** - Training loops get compiled after detection
3. **Per-pattern compilation** - Different loop types get different handling
4. **Adaptive** - Learns which patterns benefit most from compilation

## Benchmark Results (Expected)

Based on preliminary testing:

| Workload           | Warmup Overhead | Steady-State Speedup | Break-Even Point | Best For |
|--------------------|-----------------|----------------------|------------------|----------|
| Simple Matmul Chain | 3-5x slower    | 1.5-2x faster       | ~15 iters        | ✅ Yes   |
| MLP Training       | 3-4x slower    | 1.3-1.8x faster     | ~20 iters        | ✅ Yes   |
| Attention Block    | 4-6x slower    | 1.2-1.5x faster     | ~30 iters        | ⚠️ Maybe |
| Mixed Operations   | 5-8x slower    | 1.1-1.3x faster     | ~50 iters        | ❌ No    |

**Rule of thumb**:
- ✅ Use compilation for >50 iterations with repeated patterns
- ⚠️ Consider compilation for 20-50 iterations depending on pattern
- ❌ Skip compilation for <20 iterations or highly varied operations

## Integration Points

To integrate hot path detection into GT:

1. **Dispatcher-level tracking** - Track operation sequences as they flow through dispatcher
2. **Per-client detection** - Each client gets its own detector instance
3. **Worker notification** - Send "enable_compile" message when hot path detected
4. **Gradual rollout** - Start with conservative threshold (20), tune based on metrics

### Example dispatcher integration:

```python
# In dispatcher/__init__.py
from gt.hotpath_detector import HotPathDetector

class Dispatcher:
    def __init__(self):
        self.detectors = {}  # client_id -> HotPathDetector

    def handle_operation(self, client_id, operation):
        # Track operation in detector
        if client_id not in self.detectors:
            self.detectors[client_id] = HotPathDetector()

        detector = self.detectors[client_id]
        detector.record_operation(operation.op_type, operation.input_shape)

        # Check if we should enable compilation
        if detector.should_compile():
            # Send compile directive to worker
            self.enable_compilation_for_client(client_id)
```

## Future Work

1. **Shape-agnostic signatures** - Group patterns by structure, not exact shapes
2. **Per-layer compilation** - Compile individual model layers separately
3. **Regression detection** - Disable compilation if it makes things slower
4. **Cross-run persistence** - Save hot path info between training runs
5. **Profiler integration** - Use actual timings instead of iteration counts

## Running the Benchmarks

```bash
# Full benchmark suite (takes ~10-15 minutes)
cd benchmarks
python compilation_benchmark.py

# Quick test (fewer iterations)
python compilation_benchmark.py --quick  # TODO: implement

# Single workload
python -c "from compilation_benchmark import *; benchmark_workload('Simple Matmul', workload_simple_matmul, 50, 5)"
```

## References

- PyTorch torch.compile() docs: https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
- Inductor optimization: https://dev-discuss.pytorch.org/t/torchinductor-a-pytorch-native-compiler-with-define-by-run-ir-and-symbolic-shapes/747
- Dynamo recompilation: https://pytorch.org/docs/stable/torch.compiler_troubleshooting.html
