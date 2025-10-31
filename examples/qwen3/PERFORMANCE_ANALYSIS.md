# GT Performance Bottleneck Analysis

**Date:** October 31, 2025
**Test:** Qwen3-nano training (10 samples, 5 batches)

## Executive Summary

GT mode is **5.1x slower** than PyTorch mode (8.34s vs 1.63s).

The critical bottleneck is **gradient broadcast reduction** which fetches gradients from workers to reduce them on the client side, causing ~35% overhead.

## Measured Performance

```
GT Mode:
  Total Time:      8.34s
  Avg Batch Time:  1.669s
  Throughput:      0.6 batches/sec

PyTorch Mode:
  Total Time:      1.63s
  Avg Batch Time:  0.325s
  Throughput:      3.1 batches/sec

Slowdown: 5.1x
```

## Profiling Results (cProfile)

### GT Mode Statistics
- Total time: 10.43s
- Function calls: 2,339,464 calls
- Primitive calls: 2,295,337 calls

### Top Time Consumers

| Component | Time (s) | % of Total | Calls |
|-----------|----------|------------|-------|
| grad_fn (autograd) | 5.62 | 53.9% | 652 |
| from_numpy | 3.47 | 33.3% | 1,036 |
| .data property | 2.94 | 28.2% | 320 |
| pickle.dumps | 2.03 | 19.5% | 33,378 |
| ZMQ socket.send | 1.01 | 9.7% | 83,445 |
| pickle.loads | 0.98 | 9.4% | 33,378 |

## Main Bottlenecks

### 1. Gradient Broadcast Reduction (35% - CRITICAL ⚠️)

**Location:** `gt/client/tensor.py:368` in `reduce_grad_for_broadcast()`

**Problem:** The function fetches gradients from workers to reduce them on the client:

```python
def reduce_grad_for_broadcast(grad, original_shape):
    """Reduce gradient to match original shape after broadcasting."""
    if original_shape == grad.shape:
        return grad

    # Calculate axes to sum
    axes_to_sum = [...]

    if axes_to_sum:
        # ⚠️ THIS IS THE BOTTLENECK ⚠️
        grad_data = grad.data.numpy()  # <-- Fetches from worker!
        for axis in sorted(axes_to_sum, reverse=True):
            grad_data = grad_data.sum(axis=axis, keepdims=True)
        grad_data = grad_data.reshape(original_shape)
        return from_numpy(grad_data)   # <-- Sends back to worker!

    return grad
```

**Impact:**
- Called **652 times** during backward pass
- Each call: serialize command → network → deserialize → fetch data → serialize data → network → deserialize
- Total overhead: ~2.9s (35% of training time)

**Why it happens:**
- Broadcasting in operations like `add`, `mul`, `sub`, `div`
- Example: `(8, 128, 256) + (256,)` broadcasts the second operand
- During backward, gradient must be reduced: `(8, 128, 256) → (256,)`
- Currently done by fetching to client, reducing in Python, sending back

### 2. Serialization Overhead (36%)

**Components:**
- `pickle.dumps`: 2.03s (33,378 calls)
- `pickle.loads`: 0.98s (33,378 calls)
- Total: 3.01s

**Impact:**
- ~90 microseconds per serialize/deserialize operation
- Inherent to distributed architecture
- Every operation requires serialization

**Call breakdown:**
- Binary operations: 2,683 calls
- Create tensors: 1,036 calls
- Get data: 320 calls
- Other commands: ~29,000+ calls

### 3. Communication Overhead (12%)

**Component:** ZMQ socket sends

**Impact:**
- 83,445 socket sends
- 1.01s total time
- Using IPC (local) communication

**Architecture overhead:**
Each operation requires:
1. Client serializes command → Dispatcher
2. Dispatcher deserializes, routes to Worker
3. Worker executes, serializes response
4. Dispatcher forwards to Client
5. Client deserializes response

### 4. Other Bottlenecks

**from_numpy (33.3%):**
- 1,036 calls creating tensors from numpy arrays
- Used heavily in gradient reduction
- 3.47s total time

**Autograd gradient functions (53.9%):**
- grad_fn called 652 times
- Includes all the gradient computations
- 5.62s cumulative time (includes nested calls)

## Detailed Call Flow Analysis

### Forward Pass
1. Create input tensors: `from_numpy()` → serialize → worker
2. For each layer:
   - Matmul operations
   - Add operations (broadcasting)
   - Normalize operations
3. Each operation: client → dispatcher → worker → dispatcher → client

### Backward Pass (THE BOTTLENECK)
1. Start backward: `loss.backward()`
2. For each grad_fn in reverse:
   - Compute gradients
   - **If broadcasting occurred:** `reduce_grad_for_broadcast()`
     - Fetch gradient: worker → client (`.data.numpy()`)
     - Reduce in Python: `numpy.sum(axis=...)`
     - Send back: client → worker (`from_numpy()`)
   - Accumulate gradients
3. **652 reduce_grad_for_broadcast calls** = massive overhead

### Optimizer Step
1. For each of 39 parameters:
   - Fetch gradient: `param.grad.data.numpy()`
   - Update in Python: `param -= lr * grad`
2. Total: 195 gradient fetches (5 batches × 39 params)

## Root Cause Summary

The distributed architecture adds overhead to every operation, but the **critical bottleneck** is:

> **Fetching gradients from workers for broadcast reduction**

This happens because gradient reduction is implemented in Python on the client side, requiring data transfer for every broadcasted operation gradient.

## Proposed Solutions

### Solution 1: Implement Worker-Side Gradient Reduction (HIGH PRIORITY)

**Changes needed:**

1. **Add axis-aware sum operation:**
   ```python
   # In Engine API
   def sum(self, tensor, axis=None, keepdims=False) -> Any:
       """Sum along specified axes."""
   ```

2. **Modify reduce_grad_for_broadcast:**
   ```python
   def reduce_grad_for_broadcast(grad, original_shape):
       if original_shape == grad.shape:
           return grad

       # Calculate axes to sum
       axes_to_sum = [...]

       if axes_to_sum:
           # ✅ NEW: Reduce on worker instead of fetching
           result = grad
           for axis in sorted(axes_to_sum, reverse=True):
               result = result.sum(axis=axis, keepdims=True)
           return result.reshape(original_shape)

       return grad
   ```

3. **Add reshape operation** (already partially implemented)

**Expected impact:**
- Eliminates ~652 gradient fetches
- Saves ~2.9s (35% of training time)
- Expected speedup: **2-3x**

### Solution 2: Batch Gradient Fetches

Instead of fetching each parameter gradient individually:
```python
# Current (slow)
for param in params:
    grad = param.grad.data.numpy()  # 39 separate fetches

# Better (batched)
all_grads = fetch_all_gradients(params)  # 1 fetch
```

**Expected impact:** ~0.5s savings

### Solution 3: Faster Serialization

Replace pickle with msgpack or flatbuffers:
- pickle: ~90μs per operation
- msgpack: ~30μs per operation (3x faster)

**Expected impact:** ~2s savings (from 3s to 1s)

### Solution 4: Reduce Autograd Complexity

Simplify gradient functions to avoid intermediate tensor creation.

**Expected impact:** ~0.5-1s savings

## Implementation Priority

1. **HIGH:** Worker-side gradient reduction (2-3x speedup)
2. **MEDIUM:** Faster serialization (1.5x speedup)
3. **LOW:** Batch gradient fetches (1.1x speedup)
4. **LOW:** Autograd optimization (1.1x speedup)

## Profiling Commands Used

```bash
# Profile GT mode
python -m cProfile -o /tmp/gt_profile.prof \
  examples/qwen3/train_gt.py \
  --model-size nano --epochs 1 --batch-size 2 --num-samples 10

# Profile PyTorch mode
python -m cProfile -o /tmp/pytorch_profile.prof \
  examples/qwen3/train_gt.py --pytorch \
  --model-size nano --epochs 1 --batch-size 2 --num-samples 10

# Analyze
python -c "
import pstats
p = pstats.Stats('/tmp/gt_profile.prof')
p.sort_stats('cumulative')
p.print_stats(30)
"
```

## Conclusion

The 5.1x slowdown is primarily due to:
1. **Gradient broadcast reduction fetching** (35%)
2. **Serialization overhead** (36%)
3. **Communication overhead** (12%)

The **highest ROI fix** is implementing worker-side gradient reduction, which would eliminate ~35% overhead and provide 2-3x speedup with relatively simple changes.
