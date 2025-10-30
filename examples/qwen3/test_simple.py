"""
Very simple test to debug shape issues.
"""

import numpy as np
import gt as torch

print("Testing basic matmul shapes...")

# Test 3D @ 2D matmul
x = torch.from_numpy(np.random.randn(2, 10, 64).astype('float32'))
w = torch.from_numpy(np.random.randn(64, 128).astype('float32'))

print(f"x shape: {x.shape}")
print(f"w shape: {w.shape}")

result = x @ w

print(f"result shape: {result.shape}")
print(f"Expected: (2, 10, 128)")

result_data = result.data.numpy()
print(f"Actual shape from data: {result_data.shape}")

print("\n✓ Test complete!")
