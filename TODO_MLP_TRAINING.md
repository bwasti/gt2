# MLP Training Issues

## Status
✅ **FIXED!** In-place operations now working correctly. Root cause was async garbage collection causing response mismatches.
✅ **ADDED!** Operation tape logging - every command with timestamps for debugging.
✅ **FIXED!** Autograd tape memory leak - tape must be cleared after each backward() call.
✅ **WORKING!** MLP training converges! Loss: 0.25 → 0.027, Accuracy: 100%

**Result:** Full MLP training now works end-to-end!

## What Works
- ✅ In-place subtraction (`-=`) with functional ID swapping
- ✅ Gradient computation with correct broadcasting
- ✅ Full MLP training with convergence
- ✅ Forward pass through MLP (produces predictions)
- ✅ Transpose operation and gradients
- ✅ Scalar-tensor operations (0.1 * tensor)
- ✅ Autograd tape management (clears after backward())
- ✅ Queued garbage collection (no deadlocks)

## Key Fixes Applied
1. **Autograd tape clearing**: Added `self.tape.clear()` after backward() to prevent memory leak
2. **Queued GC**: FreeTensor operations queued and processed with connection lock
3. **Broadcasting gradients**: Fixed mean and add gradients to handle broadcasting correctly
4. **Shape inference**: Fixed transpose to swap dimensions correctly
5. **Parameter collection**: Made Module.parameters() recursive for nested modules

## Debugging Observations
1. Simple linear model: weights change but loss INCREASES (0.39 → 1.52 over 3 epochs)
   - Gradient norms: 6.8 → 5.7 → 84.15 (exploding!)
   - Suggests either:
     - Learning rate too high
     - Gradient computation bug
     - Weight update bug

2. MLP model: loss perfectly constant
   - Suggests weights might not be changing at all
   - OR forward pass is broken (always returning 0.5)
   - OR bias updates not working

## Next Steps to Debug

### 1. Check if MLP weights actually change
```python
# Print weights before and after one epoch
print(f"Before: {model.fc1.weight.data.numpy()}")
# ... training step ...
print(f"After: {model.fc1.weight.data.numpy()}")
```

### 2. Check if predictions change
```python
# Check if forward pass produces same output every time
pred1 = model(X_tensor)
# ... update weights ...
pred2 = model(X_tensor)
print(f"Predictions changed: {not np.allclose(pred1.data.numpy(), pred2.data.numpy())}")
```

### 3. Verify matmul gradients
The matmul gradient implementation uses transpose:
```python
grad_left = grad_output @ right.T
grad_right = left.T @ grad_output
```
Need to verify this is correct for the shapes involved in MLP.

### 4. Check bias gradients
Bias gradients might not be computed correctly. The bias gradient should just be the sum of grad_output over the batch dimension.

### 5. Verify gradient accumulation
Make sure gradients from fc2 properly backprop through relu and fc1.

### 6. Check if `zero_()` is actually resetting gradients
Current implementation just detaches the finalizer. Might need to actually set `param._grad = None` in the parameter.

## Implementation Notes

### In-place Operations (Functional Approach)
Instead of mutating tensor data on workers, we:
1. Create new result tensor
2. Swap the client-side tensor's ID to point to new result
3. Detach finalizer from result to prevent double-free
4. Let GC clean up old tensor

This keeps everything functional on the worker side while appearing imperative on the client side.

### Finalizer Management
Must call `tensor._finalizer.detach()` before stealing a tensor's ID to prevent the finalizer from freeing the tensor we just adopted.

## Debugging Infrastructure

### Operation Tape
The dispatcher now logs every operation with timestamps for debugging:

```
[TAPE 0.003s] RECV         | 127.0.0.1:59712      | BinaryOp        | result=2 op=matmul left=0 right=1
[TAPE 0.003s] WORKER_SEND  | auto_worker          | WorkerBinaryOp  | op=matmul result=127.0.0.1:59712_2
[TAPE 0.003s] WORKER_RECV  | auto_worker          | WorkerBinaryOp  | success=True
[TAPE 0.004s] SEND         | 127.0.0.1:59712      | BinaryOp        | success=True
```

Each entry shows:
- Timestamp (seconds since dispatcher start)
- Event type (RECV/SEND from client, WORKER_SEND/WORKER_RECV)
- Client/worker ID
- Command type
- Operation details

This makes it easy to:
- Debug hangs (see the last operation before timeout)
- Identify slow operations (large timestamp gaps)
- Understand the full command flow through the system

## Next Steps / TODO

### 1. Refactor Dispatcher Optimizations
**Goal**: Abstract distributed operation logic into separate, reusable modules

Current issues:
- Distributed matmul, reduction, and sharding logic is embedded in dispatcher.py
- Hard to maintain and extend
- Need to support tensor movement between workers

**Tasks**:
- [ ] Create `gt/dispatcher/distributed_ops.py` for distributed operation handlers
  - Extract `_handle_distributed_matmul()` logic
  - Extract `_handle_distributed_reduction()` logic
  - Extract `_handle_sharded_creation()` logic
- [ ] Create `gt/dispatcher/tensor_placement.py` for tensor movement
  - Implement "move" operation to relocate tensors between workers
  - Automatic insertion of move ops when tensors are on different workers
- [ ] Enhance `ShardInfo` to support:
  - Multi-dimensional sharding (not just axis 0)
  - Replication (for FSDP-style patterns)
  - Add fields: `replicated: bool`, `shard_pattern: List[Optional[int]]`

**Benefits**:
- Cleaner dispatcher.py (focus on routing)
- Reusable distributed primitives
- Easier to add new distributed patterns

### 2. Worker Instruction Batching for Compilation ✅ COMPLETED!
**Goal**: Allow workers to accumulate instructions before execution to enable compilation

**Motivation**:
- PyTorch can compile graphs (torch.compile)
- Batching instructions exposes more opportunities for optimization
- Numpy doesn't support this, so need engine-specific implementations

**Tasks**:
- [x] Refactor engine interface into separate files:
  - Created `worker/engine/base.py` with Operation dataclass and extended Engine interface
  - Created `worker/engine/pytorch.py` with compilation support
  - Created `worker/engine/numpy.py` without compilation
  - Created `worker/engine/__init__.py` with factory function
- [x] Add instruction batching to worker:
  - Worker accumulates operations until batch size reached or sync point hit
  - Configurable batch size (default: 1 for eager mode)
  - Flushes on sync points (CreateTensor, GetData, FreeTensor)
- [x] Implement compilation in PyTorch engine:
  - Builds computation graph from batched Operation objects
  - Uses `torch.compile()` to optimize graph functions
  - Caches compiled functions by graph signature (MD5 hash)
  - Falls back to eager execution for single ops or compilation failures
- [x] Add configuration:
  - Environment variable: `GT_WORKER_BATCH_SIZE` (default: 1)
  - Auto-detects and uses PyTorch backend when batching enabled

**Performance**:
- First compilation: ~9.6 seconds (torch.compile overhead)
- Cached execution: ~1.5ms (**5000x faster!**)
- Cache hit rate tracked with get_compilation_stats()

**Example flow**:
```python
# Worker receives:
1. BinaryOp(result=3, op=add, left=1, right=2)
2. UnaryOp(result=4, op=relu, input=3)
3. BinaryOp(result=5, op=mul, left=4, right=2)

# Instead of executing one-by-one, batch into:
def compiled_graph(t1, t2):
    t3 = t1 + t2
    t4 = relu(t3)
    t5 = t4 * t2
    return t5

# torch.compile(compiled_graph) runs once
```

**Benefits**:
- Faster execution via kernel fusion
- Reduced Python overhead
- Unlocks GPU optimizations (memory reuse, pipelining)

## Known Issues to Fix Later

1. **Exception propagation**: Errors from dispatcher/worker should propagate cleanly to client
2. **Protocol documentation**: Need formal spec for client→dispatcher→worker instruction flow
3. **Transpose metadata**: Transpose operations might need shape metadata updates
4. **Cross-worker operations**: Currently only works when tensors are on same worker
