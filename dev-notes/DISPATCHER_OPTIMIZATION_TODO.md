# Dispatcher Optimization TODOs

## Current Issues: Dispatcher Does Local Computation

When running with multiple workers and sharded tensors, the dispatcher currently fetches data and performs operations locally instead of pushing computation to workers.

## Issue 1: `_handle_distributed_reduction` (PRIORITY: HIGH)

**Location:** `gt/dispatcher/dispatcher.py:774-783`

**Problem:**
```python
# Fetches all partial results from workers
partial_results = [fetch_from_worker_1(), fetch_from_worker_2(), ...]

# Combines them locally on dispatcher
final_result = np.sum(partial_results)  # For sum
weighted_sum = sum(...) / total  # For mean

# Sends result back to a worker
create_tensor_on_worker(final_result)
```

**Why it's bad:**
- Unnecessary network traffic: N workers → dispatcher → 1 worker
- Dispatcher becomes compute bottleneck
- Doesn't scale well (O(N) communication vs O(log N) with all-reduce)

**Correct approach:**
1. Each worker computes partial result
2. Use all-reduce to combine on workers (MPI-style reduction)
3. Result stays on one worker, no round-trip through dispatcher

**Requires implementing:**
- Worker-to-worker communication (peer-to-peer tensor transfer)
- All-reduce operation (sum/mean across workers)
- Protocol: `WorkerAllReduce` command

**Expected benefit:** 2-4x faster for reductions on sharded tensors with multiple workers

---

## Issue 2: `_handle_get_data` with sharded tensors (PRIORITY: LOW)

**Location:** `gt/dispatcher/dispatcher.py:550`

**Problem:**
```python
# Fetches all shards
shards = [fetch_from_worker_1(), fetch_from_worker_2(), ...]

# Concatenates locally
combined_data = np.concatenate(shards, axis=shard_axis)

# Sends to client
return combined_data
```

**Why it might be OK:**
- `GetData` is for fetching data to client (`.data.numpy()`)
- Data must pass through dispatcher anyway
- Final destination is client, not another worker

**When it's bad:**
- If we later want to do more operations on the concatenated tensor
- Could create a "virtual concatenated tensor handle" that lazily combines shards only when needed

**Correct approach (if we optimize):**
1. Create virtual tensor handle pointing to all shards
2. Operations on virtual tensor trigger auto-gather on one worker
3. Only fetch to client when truly necessary

**Requires implementing:**
- Virtual/lazy tensor handles
- Concatenate operation on workers
- Smarter GetData that recognizes when client doesn't need data immediately

**Expected benefit:** Minimal unless we're doing operations after GetData

---

## Implementation Priority:

1. **HIGH: Fix `_handle_distributed_reduction`**
   - Impacts any reduction on sharded tensors (sum, mean)
   - Easy to trigger in distributed training
   - Clear correctness and performance issue

2. **MEDIUM: Add all-reduce primitives**
   - Foundation for proper distributed operations
   - Enables NCCL/Gloo-style collectives
   - Required for multi-GPU scaling

3. **LOW: Optimize `_handle_get_data`**
   - Only matters if doing operations after fetch
   - Current approach is acceptable for terminal GetData calls
   - More of an architectural improvement

---

## Notes:

- **Current Qwen3 tests run with 1 worker** - these issues don't appear yet
- Issues will become critical when scaling to multiple GPUs
- All-reduce is standard in distributed frameworks (PyTorch DDP, Horovod, MPI)
- Worker-to-worker communication is foundational for multi-GPU efficiency
