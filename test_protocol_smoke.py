#!/usr/bin/env python3
"""
Quick smoke test for protocol changes.

Tests a chain of 15 operations and verifies output matches expected result.
Client-only test - just verifies end-to-end correctness.

Run before/after protocol changes:
    python test_protocol_smoke.py

Should complete in < 2 seconds.
"""

import numpy as np
import gt


def test_operation_chain():
    """Chain 15 operations together and verify result."""
    print("Running operation chain test...")

    # Input data
    a_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b_np = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)

    # GT operations
    a = gt.from_numpy(a_np)  # Op 1
    b = gt.from_numpy(b_np)  # Op 2

    x = a @ b                # Op 3: matmul
    x = x + a                # Op 4: add
    x = x * 2.0              # Op 5: scalar mul
    x = x - b                # Op 6: sub
    x = x / 2.0              # Op 7: scalar div
    x = x @ b                # Op 8: matmul
    x = x + a                # Op 9: add
    x = x.T                  # Op 10: transpose
    x = x @ a                # Op 11: matmul
    x = x * b                # Op 12: element-wise mul
    x = x.sum()              # Op 13: sum
    x = x / 10.0             # Op 14: scalar div
    result = x + 100.0       # Op 15: scalar add

    # Get result from GT
    gt_result = result.data.numpy()

    # Compute expected result with numpy
    x_np = a_np @ b_np           # Op 3
    x_np = x_np + a_np           # Op 4
    x_np = x_np * 2.0            # Op 5
    x_np = x_np - b_np           # Op 6
    x_np = x_np / 2.0            # Op 7
    x_np = x_np @ b_np           # Op 8
    x_np = x_np + a_np           # Op 9
    x_np = x_np.T                # Op 10
    x_np = x_np @ a_np           # Op 11
    x_np = x_np * b_np           # Op 12
    x_np = x_np.sum()            # Op 13
    x_np = x_np / 10.0           # Op 14
    expected = x_np + 100.0      # Op 15

    print(f"GT result:       {gt_result}")
    print(f"Expected result: {expected}")
    print(f"Difference:      {abs(gt_result - expected)}")

    # Verify match
    assert np.allclose(gt_result, expected, rtol=1e-5), \
        f"Mismatch! GT={gt_result}, Expected={expected}"

    print("✅ PASS - Operation chain matches expected result")
    return True


def test_backward_chain():
    """Test backward pass with chained operations."""
    print("\nRunning backward chain test...")

    # Create tensors with gradients
    a = gt.randn(4, 4, requires_grad=True)
    b = gt.randn(4, 4, requires_grad=True)

    # Forward pass: 10 operations
    x = a @ b           # Op 1
    x = x.relu()        # Op 2
    x = x + a           # Op 3
    x = x @ b           # Op 4
    x = x * 2.0         # Op 5
    x = x.relu()        # Op 6
    x = x + b           # Op 7
    x = x.sum()         # Op 8
    loss = x / 100.0    # Op 9

    # Backward pass (generates many more operations)
    loss.backward()

    # Verify gradients exist and have correct shape
    assert a.grad is not None, "Gradient for 'a' is None"
    assert b.grad is not None, "Gradient for 'b' is None"
    assert a.grad.data.numpy().shape == (4, 4), f"Wrong gradient shape for 'a'"
    assert b.grad.data.numpy().shape == (4, 4), f"Wrong gradient shape for 'b'"

    print(f"Loss value: {loss.data.numpy()}")
    print(f"Gradient 'a' shape: {a.grad.data.numpy().shape}")
    print(f"Gradient 'b' shape: {b.grad.data.numpy().shape}")
    print("✅ PASS - Backward chain works correctly")
    return True


def main():
    """Run all smoke tests."""
    print("=" * 60)
    print("PROTOCOL SMOKE TEST")
    print("=" * 60)

    try:
        test_operation_chain()
        test_backward_chain()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        return 0

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
