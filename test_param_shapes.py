"""
Test to understand the parameter shape issue.
"""

import gt
import numpy as np

print("Testing parameter shapes after optimizer step...")

# Create a parameter (1D)
param = gt.from_numpy(np.ones(256, dtype='float32'))
param = param.requires_grad_(True)
print(f"Initial param.shape: {param.shape}, param.id: {param.id}")

# Simulate forward pass: param is broadcasted
x = gt.from_numpy(np.random.randn(8, 128, 256).astype('float32'))
result = x * param  # Broadcasting: (8, 128, 256) * (256,) -> (8, 128, 256)
print(f"result.shape: {result.shape}")

# Simulate backward pass
loss = result.sum()
loss.backward()

print(f"After backward - param.grad.shape: {param.grad.shape if param.grad else None}")
print(f"After backward - param.shape: {param.shape}, param.id: {param.id}")

# Simulate optimizer step
if param.grad is not None:
    print(f"\nBefore step:")
    print(f"  param.shape: {param.shape}, param.id: {param.id}")
    print(f"  param.grad.shape: {param.grad.shape}, param.grad.id: {param.grad.id}")

    # This is what SGD does
    update = 0.001 * param.grad
    print(f"  update.shape: {update.shape}, update.id: {update.id}")

    param -= update

    print(f"After step:")
    print(f"  param.shape: {param.shape}, param.id: {param.id}")

# Try using param again
x2 = gt.from_numpy(np.random.randn(8, 128, 256).astype('float32'))
try:
    result2 = x2 * param
    print(f"\nSecond forward pass successful! result2.shape: {result2.shape}")
except Exception as e:
    print(f"\nSecond forward pass FAILED: {e}")

print("\nTest complete!")
