"""
Simple test for GT2 system.

Tests randn and matmul operations.

Keep this SIMPLE and READABLE.
"""

import sys
import time
sys.path.insert(0, '/home/bwasti/oss/gt2')

from gt.client.client import Client
from gt.client.tensor import randn, from_numpy, zeros
import numpy as np


def test_simple():
    """Test basic operations."""
    print("Connecting to dispatcher...")
    client = Client(dispatcher_host="localhost", dispatcher_port=9000)
    client.connect()

    print("\n=== Test 1: Create random tensors ===")
    a = randn(3, 3)
    print(f"Created tensor a: {a}")
    print(f"Data:\n{a.data}")

    b = randn(3, 3)
    print(f"\nCreated tensor b: {b}")
    print(f"Data:\n{b.data}")

    print("\n=== Test 2: Matrix multiplication ===")
    c = a @ b
    print(f"Created tensor c = a @ b: {c}")
    print(f"Data:\n{c.data}")

    print("\n=== Test 3: Addition ===")
    d = a + b
    print(f"Created tensor d = a + b: {d}")
    print(f"Data:\n{d.data}")

    print("\n=== Test 4: Create from numpy ===")
    np_array = np.array([[1, 2], [3, 4]], dtype=np.float32)
    e = from_numpy(np_array)
    print(f"Created tensor e from numpy: {e}")
    print(f"Data:\n{e.data}")

    print("\n=== Test 5: Zeros ===")
    f = zeros(2, 2)
    print(f"Created tensor f (zeros): {f}")
    print(f"Data:\n{f.data}")

    print("\n=== All tests passed! ===")

    client.disconnect()


if __name__ == "__main__":
    test_simple()
