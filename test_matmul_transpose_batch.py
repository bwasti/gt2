"""
Simple test to reproduce batch size bug with matmul + transpose pattern.
Uses numpy backend for simplicity.
"""

import gt
import numpy as np
import os

# Force numpy backend
os.environ['GT_BACKEND'] = 'numpy'

print("Testing matmul with transpose pattern (mimics Qwen3 attention)")
print("=" * 60)

# Create weight matrix (2D) - like attention projection weights
weight = gt.from_numpy(np.ones((64, 64), dtype='float32'))
weight = weight.requires_grad_(True)

def forward_backward(batch_size, batch_num, weight):
    """Run one forward+backward pass"""
    print(f"\n=== Batch {batch_num} (size {batch_size}) ===")

    # Input: (batch, seq, hidden) - mimics attention input
    x = gt.from_numpy(np.random.randn(batch_size, 128, 64).astype('float32'))
    print(f"  x.shape: {x.shape}, weight.shape: {weight.shape}")

    # Forward: matmul
    y = x @ weight  # (batch, 128, 64) @ (64, 64) -> (batch, 128, 64)
    print(f"  y.shape: {y.shape}")

    # Loss
    loss = y.sum()
    print(f"  loss: {loss.item():.4f}")

    # Backward - this will call grad_left = grad_output @ right.T
    print(f"  Running backward...")
    loss.backward()

    print(f"  weight.grad.shape: {weight.grad.shape if weight.grad else None}")

    # Optimizer step
    if weight.grad:
        weight -= 0.001 * weight.grad
        weight.grad.zero_()
        print(f"  Optimizer step done, weight.shape: {weight.shape}")

    return weight

# Batch 1: size 8
try:
    weight = forward_backward(8, 1, weight)
    print("  ✓ SUCCESS")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

# Batch 2: size 4 (this should fail if there's a bug)
try:
    weight = forward_backward(4, 2, weight)
    print("  ✓ SUCCESS")
except Exception as e:
    print(f"  ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

# Batch 3: size 8 again
try:
    weight = forward_backward(8, 3, weight)
    print("  ✓ SUCCESS")
except Exception as e:
    print(f"  ✗ FAILED: {e}")

print("\n" + "=" * 60)
print("Test complete!")
