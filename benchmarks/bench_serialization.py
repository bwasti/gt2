"""
Benchmark serialization methods for protocol commands.
"""

import time
import pickle
import numpy as np
from dataclasses import dataclass

# Try msgpack if available
try:
    import msgpack
    import msgpack_numpy as m
    m.patch()
    HAS_MSGPACK = True
except ImportError:
    HAS_MSGPACK = False

# Simulate typical protocol commands
@dataclass
class BinaryOp:
    result_id: int
    op: str
    left_id: int
    right_id: int
    signal: str = None

@dataclass
class CreateTensor:
    tensor_id: int
    data: np.ndarray
    dtype: str
    shape: tuple
    signal: str = None

def benchmark_pickle(obj, iterations=10000):
    """Benchmark pickle serialization."""
    start = time.time()
    for _ in range(iterations):
        data = pickle.dumps(obj)
        obj2 = pickle.loads(data)
    elapsed = time.time() - start

    # Get size
    data = pickle.dumps(obj)
    return elapsed / iterations, len(data)

def benchmark_msgpack(obj, iterations=10000):
    """Benchmark msgpack serialization."""
    if not HAS_MSGPACK:
        return None, None

    # Convert dataclass to dict
    if hasattr(obj, '__dataclass_fields__'):
        obj_dict = obj.__dict__
    else:
        obj_dict = obj

    start = time.time()
    for _ in range(iterations):
        data = msgpack.packb(obj_dict)
        obj2 = msgpack.unpackb(data)
    elapsed = time.time() - start

    data = msgpack.packb(obj_dict)
    return elapsed / iterations, len(data)

print("=" * 80)
print("Serialization Benchmark")
print("=" * 80)

# Test 1: Simple command (no data)
print("\n1. Simple Command (BinaryOp - no numpy data):")
cmd = BinaryOp(result_id=42, op="matmul", left_id=1, right_id=2, signal="test")

pickle_time, pickle_size = benchmark_pickle(cmd)
print(f"  Pickle:  {pickle_time * 1e6:.2f} μs, {pickle_size} bytes")

if HAS_MSGPACK:
    msgpack_time, msgpack_size = benchmark_msgpack(cmd)
    print(f"  Msgpack: {msgpack_time * 1e6:.2f} μs, {msgpack_size} bytes")
    print(f"  Speedup: {pickle_time / msgpack_time:.2f}x faster")
else:
    print("  Msgpack: Not installed (pip install msgpack msgpack-numpy)")

# Test 2: Command with small numpy array
print("\n2. Small Tensor Command (100 floats):")
data = np.random.randn(100).astype('float32')
cmd = CreateTensor(tensor_id=1, data=data, dtype="float32", shape=(100,))

pickle_time, pickle_size = benchmark_pickle(cmd, iterations=1000)
print(f"  Pickle:  {pickle_time * 1e6:.2f} μs, {pickle_size} bytes")

if HAS_MSGPACK:
    msgpack_time, msgpack_size = benchmark_msgpack(cmd, iterations=1000)
    print(f"  Msgpack: {msgpack_time * 1e6:.2f} μs, {msgpack_size} bytes")
    print(f"  Speedup: {pickle_time / msgpack_time:.2f}x faster")

# Test 3: Command with large numpy array
print("\n3. Large Tensor Command (1000×1000 floats = 4MB):")
data = np.random.randn(1000, 1000).astype('float32')
cmd = CreateTensor(tensor_id=1, data=data, dtype="float32", shape=(1000, 1000))

pickle_time, pickle_size = benchmark_pickle(cmd, iterations=100)
print(f"  Pickle:  {pickle_time * 1e3:.2f} ms, {pickle_size / 1024 / 1024:.2f} MB")

if HAS_MSGPACK:
    msgpack_time, msgpack_size = benchmark_msgpack(cmd, iterations=100)
    print(f"  Msgpack: {msgpack_time * 1e3:.2f} ms, {msgpack_size / 1024 / 1024:.2f} MB")
    print(f"  Speedup: {pickle_time / msgpack_time:.2f}x faster")

print("\n" + "=" * 80)
print("Recommendation: Use msgpack for metadata, raw numpy arrays for tensor data")
print("=" * 80)
