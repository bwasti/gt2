"""
Test tensor subscripting matches PyTorch semantics exactly.
"""

import numpy as np
import torch
import gt

print("Testing GT tensor subscripting vs PyTorch\n")
print("=" * 70)

# Test data
np_data = np.arange(24).reshape(4, 6).astype(np.float32)

# Create tensors
gt_tensor = gt.from_numpy(np_data)
pt_tensor = torch.from_numpy(np_data)

print(f"\nOriginal shape: {np_data.shape}")
print(f"Original data:\n{np_data}\n")

# Test cases
test_cases = [
    # Basic slicing
    ("Single index", lambda t: t[0]),
    ("Slice one dim", lambda t: t[:2]),
    ("Slice with step", lambda t: t[::2]),
    ("Negative index", lambda t: t[-1]),
    ("Negative slice", lambda t: t[-2:]),

    # Multi-dimensional
    ("2D slice", lambda t: t[:2, :3]),
    ("2D mixed", lambda t: t[0, :]),
    ("2D index both", lambda t: t[1, 2]),
    ("Full colon", lambda t: t[:, :]),

    # Advanced
    ("Ellipsis start", lambda t: t[..., 0]),
    ("Ellipsis end", lambda t: t[0, ...]),

    # None (newaxis)
    ("None at start", lambda t: t[None, :]),
    ("None at end", lambda t: t[:, None]),
    ("None middle", lambda t: t[:, None, :]),
]

print("Running tests...\n")

passed = 0
failed = 0

for name, fn in test_cases:
    try:
        # Get PyTorch result
        pt_result = fn(pt_tensor)
        pt_shape = tuple(pt_result.shape)
        pt_data = pt_result.numpy()

        # Get GT result
        gt_result = fn(gt_tensor)

        # For GT, we need to get the data
        if hasattr(gt_result, 'data'):
            gt_data = gt_result.data.numpy()
            gt_shape = gt_data.shape
        elif isinstance(gt_result, (int, float, np.ndarray)):
            # Scalar or numpy array
            gt_data = np.array(gt_result) if not isinstance(gt_result, np.ndarray) else gt_result
            gt_shape = gt_data.shape
        else:
            gt_data = gt_result
            gt_shape = ()

        # Compare shapes
        if gt_shape != pt_shape:
            print(f"❌ {name}: Shape mismatch")
            print(f"   PyTorch: {pt_shape}")
            print(f"   GT:      {gt_shape}")
            failed += 1
            continue

        # Compare data
        if not np.allclose(gt_data, pt_data):
            print(f"❌ {name}: Data mismatch")
            print(f"   PyTorch:\n{pt_data}")
            print(f"   GT:\n{gt_data}")
            failed += 1
            continue

        print(f"✅ {name}: shape={gt_shape}")
        passed += 1

    except Exception as e:
        print(f"❌ {name}: Exception: {e}")
        failed += 1

print("\n" + "=" * 70)
print(f"Results: {passed} passed, {failed} failed")
print("=" * 70)

if failed == 0:
    print("\n🎉 All tests passed! GT matches PyTorch semantics!")
else:
    print(f"\n⚠️  {failed} test(s) failed. Need to fix subscripting.")

# Test __repr__ output
print("\n" + "=" * 70)
print("Testing __repr__ output:")
print("=" * 70)

small_tensor = gt.randn(3, 4)
print(f"\nSmall tensor (3x4):\n{small_tensor}")

# Test with a slice
sliced = small_tensor[:2, :2]
print(f"\nSliced [:2, :2]:\n{sliced}")
