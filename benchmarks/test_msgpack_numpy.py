"""
Test if msgpack is interfering with numpy operations.
"""

import time
import numpy as np
import msgpack
import msgpack_numpy as m
m.patch()

# Test numpy matmul performance
a = np.random.randn(1000, 1000).astype('float32')
b = np.random.randn(1000, 1000).astype('float32')

# Warmup
for _ in range(10):
    c = a @ b

# Benchmark
start = time.time()
for _ in range(100):
    c = a @ b
elapsed = time.time() - start

print(f"Numpy matmul: {elapsed * 10:.2f} ms per iteration")
print(f"Expected: ~1ms per iteration")

# Test if msgpack operations affect numpy
data = {"test": 123}
for _ in range(100):
    packed = msgpack.packb(data)
    unpacked = msgpack.unpackb(packed)

# Test again
start = time.time()
for _ in range(100):
    c = a @ b
elapsed2 = time.time() - start

print(f"After msgpack: {elapsed2 * 10:.2f} ms per iteration")
print(f"Difference: {(elapsed2 - elapsed) * 10:.2f} ms")
