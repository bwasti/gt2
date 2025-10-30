# MLP Training Issues

## Status
MLP training loop is not converging. Loss stuck at exactly 0.249970 (suspiciously close to 0.25, which is random guessing for binary classification).

## What Works
- ✅ In-place subtraction (`-=`) with functional ID swapping
- ✅ Gradient computation (gradients exist and have reasonable norms ~0.2-0.3)
- ✅ Simple Linear model training (weights change, but loss explodes - might be lr issue)
- ✅ Forward pass through MLP (produces predictions)
- ✅ Transpose operation and gradients
- ✅ Scalar-tensor operations (0.1 * tensor)

## What Doesn't Work
- ❌ MLP loss not decreasing across epochs (stuck at 0.249970)
- ❌ Predictions might be constant (0.5) - need to verify

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

## Known Issues to Fix Later

1. **Exception propagation**: Errors from dispatcher/worker should propagate cleanly to client
2. **Protocol documentation**: Need formal spec for client→dispatcher→worker instruction flow
3. **Transpose metadata**: Transpose operations might need shape metadata updates
4. **Cross-worker operations**: Currently only works when tensors are on same worker
