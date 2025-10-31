"""
Demonstrate debug logging with a simple operation.

This shows the instruction stream flowing through the dispatcher.
"""

import gt
import numpy as np

print("Creating tensors and performing operations...")
print("(Watch for debug output showing each operation)\n")

# Create some tensors
a = gt.from_numpy(np.array([[1, 2], [3, 4]], dtype='float32'))
print(f"Created tensor a: {a}")

b = gt.from_numpy(np.array([[5, 6], [7, 8]], dtype='float32'))
print(f"Created tensor b: {b}")

# Do some operations
print("\nPerforming c = a @ b...")
c = a @ b
print(f"Result tensor c: {c}")

# Get the data
print("\nFetching result data...")
result = c.data.numpy()
print(f"Result value:\n{result}")

print("\nDone! Check the output above for debug logging of each operation.")
