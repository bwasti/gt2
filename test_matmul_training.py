"""
Test matmul with training loop (forward, backward, optimizer step) and changing batch sizes.
"""

import gt
import numpy as np

print("Testing matmul training loop with changing batch sizes...")

# Create a weight matrix (2D)
weight = gt.from_numpy(np.ones((256, 128), dtype='float32'))
weight = weight.requires_grad_(True)

# Batch 1: size 8
print("\n=== Batch 1 (size 8) ===")
x1 = gt.from_numpy(np.random.randn(8, 256).astype('float32'))
print(f"Forward: x1 shape: {x1.shape}, weight shape: {weight.shape}")
result1 = x1 @ weight
print(f"  result1 shape: {result1.shape}")

loss1 = result1.sum()
print(f"Backward...")
loss1.backward()
print(f"  weight.grad shape: {weight.grad.shape if weight.grad else None}")

print(f"Optimizer step...")
if weight.grad:
    weight -= 0.001 * weight.grad
    weight.grad.zero_()
print(f"  weight shape after step: {weight.shape}")

# Batch 2: size 4 (this might fail)
print("\n=== Batch 2 (size 4) ===")
x2 = gt.from_numpy(np.random.randn(4, 256).astype('float32'))
print(f"Forward: x2 shape: {x2.shape}, weight shape: {weight.shape}")
try:
    result2 = x2 @ weight
    print(f"  result2 shape: {result2.shape}")

    loss2 = result2.sum()
    print(f"Backward...")
    loss2.backward()
    print(f"  weight.grad shape: {weight.grad.shape if weight.grad else None}")

    print(f"Optimizer step...")
    if weight.grad:
        weight -= 0.001 * weight.grad
        weight.grad.zero_()
    print(f"  weight shape after step: {weight.shape}")
    print(f"  Success!")
except Exception as e:
    print(f"  FAILED: {e}")
    import traceback
    traceback.print_exc()

# Batch 3: size 8 again
print("\n=== Batch 3 (size 8 again) ===")
x3 = gt.from_numpy(np.random.randn(8, 256).astype('float32'))
print(f"Forward: x3 shape: {x3.shape}, weight shape: {weight.shape}")
try:
    result3 = x3 @ weight
    print(f"  result3 shape: {result3.shape}")
    print(f"  Success!")
except Exception as e:
    print(f"  FAILED: {e}")

print("\nTest complete!")
