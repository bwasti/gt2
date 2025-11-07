# GT_AUTO_SHARD Benchmarks

Benchmarks for measuring GT's automatic sharding performance across multiple GPUs.

## Quick Start (8 GPUs)

The easiest way to benchmark your 8 GPUs:

```bash
# Quick benchmark (auto-starts system, ~5 minutes)
python benchmarks/quick_benchmark.py --gpus 8 --quick

# Full benchmark comparing 1 vs 8 GPUs (~15 minutes)
python benchmarks/quick_benchmark.py --gpus 1 8

# Or use the bash launcher
bash benchmarks/launch_8gpu_benchmark.sh
```

The `quick_benchmark.py` script automatically:
- Starts the dispatcher
- Launches 8 GPU workers (one per GPU)
- Runs benchmarks
- Cleans up everything

## Manual Setup

If you prefer manual control:

### Terminal 1 - Start Dispatcher
```bash
python -m gt.server -p 9000
```

### Terminals 2-9 - Start Workers (one per GPU)
```bash
# Terminal 2
CUDA_VISIBLE_DEVICES=0 python -m gt.worker --host localhost -p 9000

# Terminal 3
CUDA_VISIBLE_DEVICES=1 python -m gt.worker --host localhost -p 9000

# Terminal 4
CUDA_VISIBLE_DEVICES=2 python -m gt.worker --host localhost -p 9000

# ... continue for all 8 GPUs ...

# Terminal 9
CUDA_VISIBLE_DEVICES=7 python -m gt.worker --host localhost -p 9000
```

### Terminal 10 - Run Benchmark
```bash
# Test with different GPU counts
python benchmarks/auto_shard_benchmark.py --gpus 1 4 8

# Or just test 8 GPUs
python benchmarks/auto_shard_benchmark.py --gpus 8
```

## What Gets Benchmarked

### 1. Square Matrix Multiplications
Large square matmuls that benefit from GPU parallelism:
- 4K × 4K @ 4K × 4K
- 8K × 8K @ 8K × 8K
- 16K × 16K @ 16K × 16K

### 2. Large Batch Matmuls (ML-style)
Typical ML workload patterns with huge batch sizes:
- (8K, 4K) @ (4K, 4K) - Large batch
- (16K, 4K) @ (4K, 4K) - Huge batch
- (32K, 2K) @ (2K, 2K) - Massive batch

### 3. MLP Forward Passes
Multi-layer perceptron forward passes with large batches:
- 4K batch, 3-layer 4K hidden dims
- 8K batch, 3-layer mixed hidden dims
- 16K batch, 3-layer mixed hidden dims

## Expected Results

With 8 GPUs and `GT_AUTO_SHARD=1`, you should see:

### Linear Scaling (Best Case)
For large batch operations where work is easily parallelizable:
- **1 GPU → 8 GPUs**: ~7-8x speedup
- Examples: Large batch matmuls (32K batch size)

### Partial Scaling (Communication Overhead)
For operations with more communication:
- **1 GPU → 8 GPUs**: ~4-6x speedup
- Examples: Medium-sized matmuls with gathering

### Limited Scaling (Small Operations)
For operations too small to benefit:
- **1 GPU → 8 GPUs**: ~1-2x speedup
- Examples: Small tensors, element-wise ops

## Understanding the Output

```
BENCHMARK 1: Square Matrix Multiplications
---
16384x16384 | 1 GPU: 2.345s (91.2 TF) | 8 GPUs: 0.312s (684.5 TF)
Speedup:     | baseline              | 7.52x
```

- **Time**: Total time including creation, compute, and data retrieval
- **TFLOPS**: Tera-floating-point-operations per second (higher is better)
- **Speedup**: How much faster 8 GPUs are vs 1 GPU

## Tips for Best Performance

### 1. Use Large Batch Sizes
Auto-sharding works best with large batch sizes (axis 0):
- ✅ Good: `(32768, 4096)` - 32K batch, easily sharded
- ❌ Bad: `(128, 4096)` - Too small to benefit from 8-way split

### 2. Ensure Even Division
Tensor dimensions should be evenly divisible by number of GPUs:
- ✅ Good: `32768 / 8 = 4096` per GPU
- ⚠️ OK: `13000 / 8 = not even` → Falls back to replication

### 3. Monitor GPU Utilization
```bash
# In another terminal, watch GPU usage
watch -n 0.5 nvidia-smi
```

All 8 GPUs should show activity during benchmarks.

### 4. Enable Debug Output (if needed)
```bash
# See what's happening under the hood
GT_DEBUG_DISPATCHER=1 python benchmarks/quick_benchmark.py --gpus 8 --quick
```

## Benchmark Options

### `quick_benchmark.py`
Auto-manages GT system (starts/stops automatically)

```bash
# Quick test (3 tests, 3 iterations each)
python benchmarks/quick_benchmark.py --gpus 8 --quick

# Full comparison
python benchmarks/quick_benchmark.py --gpus 1 4 8

# Custom port
python benchmarks/quick_benchmark.py --gpus 8 --port 9001
```

### `auto_shard_benchmark.py`
Assumes GT system is already running

```bash
# Compare 1, 2, 4, and 8 GPUs
python benchmarks/auto_shard_benchmark.py --gpus 1 2 4 8

# Just test 8 GPUs
python benchmarks/auto_shard_benchmark.py --gpus 8
```

## Troubleshooting

### "Connection refused"
The dispatcher isn't running. Start it first:
```bash
python -m gt.server -p 9000
```

### "No workers available"
Workers aren't connected. Start them:
```bash
for i in {0..7}; do
  CUDA_VISIBLE_DEVICES=$i python -m gt.worker --host localhost -p 9000 &
done
```

### Poor Speedup
Possible causes:
1. **Tensor too small**: Try larger batch sizes (32K+)
2. **Communication overhead**: Normal for smaller tensors
3. **Not all GPUs used**: Check with `nvidia-smi`
4. **Wrong env var**: Ensure `GT_AUTO_SHARD=1` is set

### GPU Out of Memory
Reduce batch sizes in the benchmark:
```python
# Edit benchmarks/quick_benchmark.py
# Change: gt.randn(32768, 8192)  # 32K batch
# To:     gt.randn(16384, 8192)  # 16K batch
```

## Interpreting Speedup

### Near-Perfect Scaling (7-8x on 8 GPUs)
Your workload is:
- ✅ Large enough to benefit from distribution
- ✅ Communication overhead is negligible
- ✅ Well-balanced across GPUs

### Good Scaling (5-7x on 8 GPUs)
Your workload is:
- ✅ Large enough to benefit
- ⚠️ Some communication overhead
- ✅ Reasonably balanced

### Poor Scaling (2-4x on 8 GPUs)
Your workload is:
- ⚠️ Maybe too small for 8-way distribution
- ⚠️ High communication overhead
- Consider using fewer GPUs (4 instead of 8)

### No Scaling (1-2x on 8 GPUs)
Your workload is:
- ❌ Too small for distribution
- ❌ Communication overhead dominates
- Use single GPU instead

## Example Output

```
====================================================================================
PERFORMANCE COMPARISON
====================================================================================

--- Square Matrix Multiplications ---
Size         | 1 GPU            | 8 GPUs
--------------------------------------------------------------------------------
4096x4096    |  0.234s ( 58.1 TF) |  0.089s (152.8 TF)
Speedup:     | baseline           | 2.63x
8192x8192    |  1.567s ( 70.2 TF) |  0.234s (470.3 TF)
Speedup:     | baseline           | 6.70x
16384x16384  |  11.23s ( 95.3 TF) |  1.498s (714.2 TF)
Speedup:     | baseline           | 7.50x

--- Large Batch Matrix Multiplications ---
Shape                | 1 GPU            | 8 GPUs
--------------------------------------------------------------------------------
(8192,4096)@(4096,4096)  |  0.789s ( 87.4 TF) |  0.156s (442.1 TF)
Speedup:                  | baseline           | 5.06x
(16384,4096)@(4096,4096) |  1.534s ( 89.6 TF) |  0.198s (694.4 TF)
Speedup:                  | baseline           | 7.75x
(32768,2048)@(2048,2048) |  1.123s ( 77.3 TF) |  0.145s (598.6 TF)
Speedup:                  | baseline           | 7.74x

====================================================================================
SUMMARY
====================================================================================

8 GPUs vs 1 GPU:
  Average speedup: 6.23x
  Min speedup: 2.63x
  Max speedup: 7.75x
```

## Next Steps

After benchmarking:

1. **Tune Your Application**: Adjust batch sizes based on benchmark results
2. **Optimize Workloads**: Focus on operations that show good scaling
3. **Monitor Production**: Use `GT_DEBUG_DISPATCHER=1` in development
4. **Profile Further**: Use `GT_INSTRUCTION_LOG=/tmp/gt.log` for detailed analysis

## Related Documentation

- Main README: `../CLAUDE.md`
- Compilation benchmarks: `compilation_benchmark.py`
- PyTorch baseline comparison: `pytorch_baseline.py`
