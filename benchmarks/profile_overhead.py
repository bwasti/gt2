"""
Profile where the framework overhead comes from.
"""

import time
import numpy as np
import gt

print("Creating tensors...")
a = gt.randn(1000, 1000)
b = gt.randn(1000, 1000)

# Warmup
for _ in range(10):
    c = a @ b
    total = c.sum()
    result = total.data.numpy()

print("\nProfiling 100 iterations of matmul + sum...")

# Time just the operations (no data fetch)
start = time.time()
for _ in range(100):
    c = a @ b
    total = c.sum()
ops_only = time.time() - start

# Time operations + data fetch
start = time.time()
for _ in range(100):
    c = a @ b
    total = c.sum()
    result = total.data.numpy()
ops_with_fetch = time.time() - start

print(f"Operations only (no .data.numpy()):  {ops_only * 1000:.3f} ms total, {ops_only * 10:.3f} ms per iter")
print(f"Operations + fetch:                   {ops_with_fetch * 1000:.3f} ms total, {ops_with_fetch * 10:.3f} ms per iter")
print(f"Data fetch overhead:                  {(ops_with_fetch - ops_only) * 1000:.3f} ms total, {(ops_with_fetch - ops_only) * 10:.3f} ms per iter")

print("\nBreakdown per operation:")
print(f"  2 ops (matmul + sum) per iteration")
print(f"  Without fetch: {ops_only * 1000 / 200:.3f} ms per operation")
print(f"  With fetch: {ops_with_fetch * 1000 / 200:.3f} ms per operation")
