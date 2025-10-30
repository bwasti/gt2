"""
Demo of debug utilities for inspecting tape and compilation stats.

Shows how to:
- View the autograd tape
- Get worker compilation statistics
- See what's happening during execution
"""

import os
os.environ['GT_CONFIG'] = 'examples/config_compile.yaml'

import gt

print("=" * 60)
print("Debug Utilities Demo")
print("=" * 60)

print("\n1. Autograd Tape Inspection")
print("-" * 60)

# Create some tensors with gradients
a = gt.randn(10, 10, requires_grad=True)
b = gt.randn(10, 10, requires_grad=True)

# Perform operations
c = a + b
d = c * 2.0
e = d @ a
loss = e.sum()

print("\nOperations recorded in tape:")
gt.debug.print_tape()

print("\n\n2. Worker Compilation Statistics (before compilation)")
print("-" * 60)
gt.debug.print_worker_stats()

print("\n\n3. Run operations with compilation")
print("-" * 60)

x = gt.randn(100, 100)
y = gt.randn(100, 100)

print("Running first batch (will compile)...")
with gt.signal.context('compileme'):
    z1 = x + y
    z2 = z1 * 2.0
    z3 = z2 @ x
result = z3.data.numpy()

print("\n4. Worker Statistics (after first compilation)")
print("-" * 60)
gt.debug.print_worker_stats()

print("\n\n5. Run again (should hit cache)")
print("-" * 60)

print("Running second batch (should use cached compiled version)...")
with gt.signal.context('compileme'):
    z1 = x + y
    z2 = z1 * 2.0
    z3 = z2 @ x
result = z3.data.numpy()

print("\n6. Worker Statistics (after cache hit)")
print("-" * 60)
gt.debug.print_worker_stats()

# Check tape again after loss
print("\n\n7. Autograd Tape (after new operations)")
print("-" * 60)
gt.debug.print_tape()

print("\n" + "=" * 60)
print("Debug Demo Complete!")
print("=" * 60)
print("\nKey takeaways:")
print("- Use gt.debug.print_tape() to see recorded autograd operations")
print("- Use gt.debug.print_worker_stats() to see compilation statistics")
print("- Cache hits show when compiled graphs are reused")
print("- Operation counts help understand batching behavior")
