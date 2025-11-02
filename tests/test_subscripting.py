"""
Test tensor subscripting matches PyTorch semantics.

Validates that GT tensor slicing, indexing, and subscripting operations
produce the same results as PyTorch.
"""

import pytest
import numpy as np
import torch
from gt.client.tensor import from_numpy


def test_basic_slicing(client):
    """Test basic 1D slicing operations."""
    np_data = np.arange(10).astype(np.float32)
    gt_tensor = from_numpy(np_data)
    pt_tensor = torch.from_numpy(np_data)

    test_cases = [
        ("Single index", lambda t: t[0]),
        ("Slice", lambda t: t[:5]),
        ("Slice with step", lambda t: t[::2]),
        ("Negative index", lambda t: t[-1]),
        ("Negative slice", lambda t: t[-3:]),
    ]

    for name, fn in test_cases:
        pt_result = fn(pt_tensor).numpy()
        gt_result = fn(gt_tensor).data.numpy()

        np.testing.assert_array_equal(gt_result, pt_result,
            err_msg=f"{name}: GT result doesn't match PyTorch")
        assert gt_result.shape == pt_result.shape, \
            f"{name}: Shape mismatch - GT {gt_result.shape} vs PT {pt_result.shape}"


def test_2d_slicing(client):
    """Test 2D slicing and indexing."""
    np_data = np.arange(24).reshape(4, 6).astype(np.float32)
    gt_tensor = from_numpy(np_data)
    pt_tensor = torch.from_numpy(np_data)

    test_cases = [
        ("2D slice", lambda t: t[:2, :3]),
        ("Row slice", lambda t: t[0, :]),
        ("Column slice", lambda t: t[:, 0]),
        ("Single element", lambda t: t[1, 2]),
        ("Full colon", lambda t: t[:, :]),
        ("Negative 2D", lambda t: t[-2:, -3:]),
    ]

    for name, fn in test_cases:
        pt_result = fn(pt_tensor)
        gt_result = fn(gt_tensor)

        # Handle scalar results
        if pt_result.ndim == 0:
            pt_value = float(pt_result)
            gt_value = float(gt_result.data.numpy())
            assert abs(gt_value - pt_value) < 1e-5, \
                f"{name}: Scalar mismatch - GT {gt_value} vs PT {pt_value}"
        else:
            pt_array = pt_result.numpy()
            gt_array = gt_result.data.numpy()
            np.testing.assert_array_equal(gt_array, pt_array,
                err_msg=f"{name}: GT result doesn't match PyTorch")
            assert gt_array.shape == pt_array.shape, \
                f"{name}: Shape mismatch - GT {gt_array.shape} vs PT {pt_array.shape}"


def test_ellipsis_slicing(client):
    """Test ellipsis (...) in slicing."""
    np_data = np.arange(24).reshape(2, 3, 4).astype(np.float32)
    gt_tensor = from_numpy(np_data)
    pt_tensor = torch.from_numpy(np_data)

    test_cases = [
        ("Ellipsis start", lambda t: t[..., 0]),
        ("Ellipsis end", lambda t: t[0, ...]),
        ("Ellipsis middle", lambda t: t[0, ..., 0]),
    ]

    for name, fn in test_cases:
        pt_result = fn(pt_tensor).numpy()
        gt_result = fn(gt_tensor).data.numpy()

        np.testing.assert_array_equal(gt_result, pt_result,
            err_msg=f"{name}: GT result doesn't match PyTorch")
        assert gt_result.shape == pt_result.shape, \
            f"{name}: Shape mismatch - GT {gt_result.shape} vs PT {pt_result.shape}"


def test_newaxis_slicing(client):
    """Test None (newaxis) in slicing."""
    np_data = np.arange(12).reshape(3, 4).astype(np.float32)
    gt_tensor = from_numpy(np_data)
    pt_tensor = torch.from_numpy(np_data)

    test_cases = [
        ("None at start", lambda t: t[None, :]),
        ("None at end", lambda t: t[:, None]),
        ("None middle", lambda t: t[:, None, :]),
        ("Multiple None", lambda t: t[None, :, None]),
    ]

    for name, fn in test_cases:
        pt_result = fn(pt_tensor).numpy()
        gt_result = fn(gt_tensor).data.numpy()

        np.testing.assert_array_equal(gt_result, pt_result,
            err_msg=f"{name}: GT result doesn't match PyTorch")
        assert gt_result.shape == pt_result.shape, \
            f"{name}: Shape mismatch - GT {gt_result.shape} vs PT {pt_result.shape}"


def test_complex_slicing(client):
    """Test complex combinations of slicing operations."""
    np_data = np.arange(60).reshape(3, 4, 5).astype(np.float32)
    gt_tensor = from_numpy(np_data)
    pt_tensor = torch.from_numpy(np_data)

    test_cases = [
        ("Mixed slice and index", lambda t: t[0, :2, ::2]),
        ("Negative indices", lambda t: t[-1, -2:, -3:]),
        ("Step in multiple dims", lambda t: t[::2, ::2, ::2]),
        ("Skip middle dim", lambda t: t[:, 1, :]),
    ]

    for name, fn in test_cases:
        pt_result = fn(pt_tensor).numpy()
        gt_result = fn(gt_tensor).data.numpy()

        np.testing.assert_array_equal(gt_result, pt_result,
            err_msg=f"{name}: GT result doesn't match PyTorch")
        assert gt_result.shape == pt_result.shape, \
            f"{name}: Shape mismatch - GT {gt_result.shape} vs PT {pt_result.shape}"


def test_slicing_preserves_operations(client):
    """Test that sliced tensors can be used in operations."""
    np_data = np.arange(20).reshape(4, 5).astype(np.float32)
    gt_tensor = from_numpy(np_data)

    # Slice
    sliced = gt_tensor[:2, :3]

    # Use in operation
    result = sliced + sliced
    result_data = result.data.numpy()

    # Verify
    expected = np_data[:2, :3] + np_data[:2, :3]
    np.testing.assert_array_equal(result_data, expected)
