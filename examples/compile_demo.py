"""
Demo of compilation directives with signal-based configuration.

Shows how to use signals to control compilation boundaries.
"""

import os
os.environ['GT_CONFIG'] = 'examples/config_compile.yaml'

import gt
import time

print("=" * 60)
print("Compilation Directive Demo")
print("=" * 60)

print("\n1. Operations without compilation (eager mode)")
print("-" * 60)

a = gt.randn(100, 100)
b = gt.randn(100, 100)

start = time.time()
c = a + b
d = c * 2.0
e = d @ a
result1 = e.data.numpy()
elapsed1 = time.time() - start
print(f"Eager execution time: {elapsed1*1000:.2f}ms")

print("\n2. Operations with compilation (compileme signal)")
print("-" * 60)

x = gt.randn(100, 100)
y = gt.randn(100, 100)

start = time.time()
with gt.signal.context('compileme'):
    z1 = x + y
    z2 = z1 * 2.0
    z3 = z2 @ x
result2 = z3.data.numpy()
elapsed2 = time.time() - start
print(f"Compiled execution time: {elapsed2*1000:.2f}ms")

print("\n3. Run compiled version again (should be faster - cached)")
print("-" * 60)

start = time.time()
with gt.signal.context('compileme'):
    z1 = x + y
    z2 = z1 * 2.0
    z3 = z2 @ x
result3 = z3.data.numpy()
elapsed3 = time.time() - start
print(f"Cached compiled execution time: {elapsed3*1000:.2f}ms")
print(f"Speedup vs eager: {elapsed1/elapsed3:.1f}x")

print("\n" + "=" * 60)
print("Demo completed!")
print("=" * 60)
