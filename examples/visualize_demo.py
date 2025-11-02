#!/usr/bin/env python3
"""
Demonstration of GT instruction tape visualization.

This script generates a workload and shows how to visualize the
instruction flow through the GT system.

Usage:
    GT_INSTRUCTION_LOG=/tmp/demo_tape.log python examples/visualize_demo.py
    python -m gt.scripts.visualize /tmp/demo_tape.log --output demo_timeline.png
"""

import gt
import numpy as np

print("=" * 70)
print("GT Tape Visualization Demo")
print("=" * 70)
print()
print("This demo generates various operations to showcase the visualizer.")
print("After running, visualize with:")
print("  python -m gt.scripts.visualize /tmp/demo_tape.log")
print()

# Create input tensors
print("[1/6] Creating tensors...")
x = gt.randn(200, 200)
y = gt.randn(200, 200)
weights = gt.randn(200, 100)

# Matrix operations
print("[2/6] Matrix multiplication...")
z = x @ y  # Large matmul operation

# Activation functions
print("[3/6] Applying activations...")
a = z.relu()
b = a.sigmoid()
c = b.tanh()

# Element-wise operations
print("[4/6] Element-wise operations...")
d = c + 0.5
e = d * 2.0
f = e - 1.0

# Another matmul with different shape
print("[5/6] Final projection...")
output = f @ weights

# Reductions
print("[6/6] Computing final metrics...")
result_sum = output.sum()
result_mean = output.mean()

# Get results
final_sum = result_sum.item()
final_mean = result_mean.item()

print()
print("=" * 70)
print("Results:")
print(f"  Sum:  {final_sum:.2f}")
print(f"  Mean: {final_mean:.2f}")
print("=" * 70)
print()
print("✓ Workload complete!")
print()
print("Now visualize the tape with:")
print("  python -m gt.scripts.visualize /tmp/demo_tape.log --output demo_timeline.png --dpi 200")
