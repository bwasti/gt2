"""
Test matmul with changing batch sizes to isolate the caching bug.
"""

import gt
import numpy as np

print("Testing matmul with changing batch sizes...")

# Create a weight matrix (2D)
weight = gt.from_numpy(np.ones((256, 128), dtype='float32'))
weight = weight.requires_grad_(True)

# Test with batch size 8
print("\n1. Testing with batch size 8...")
x1 = gt.from_numpy(np.random.randn(8, 256).astype('float32'))
result1 = x1 @ weight
print(f"  x1 shape: {x1.shape}, weight shape: {weight.shape}")
print(f"  result1 shape: {result1.shape}")
print(f"  Success!")

# Test with batch size 4 (this might fail)
print("\n2. Testing with batch size 4...")
x2 = gt.from_numpy(np.random.randn(4, 256).astype('float32'))
try:
    result2 = x2 @ weight
    print(f"  x2 shape: {x2.shape}, weight shape: {weight.shape}")
    print(f"  result2 shape: {result2.shape}")
    print(f"  Success!")
except Exception as e:
    print(f"  FAILED: {e}")

# Test with batch size 8 again
print("\n3. Testing with batch size 8 again...")
x3 = gt.from_numpy(np.random.randn(8, 256).astype('float32'))
try:
    result3 = x3 @ weight
    print(f"  x3 shape: {x3.shape}, weight shape: {weight.shape}")
    print(f"  result3 shape: {result3.shape}")
    print(f"  Success!")
except Exception as e:
    print(f"  FAILED: {e}")

print("\nTest complete!")
