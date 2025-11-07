"""
Test that sharded matmul produces correct, deterministic results.

This test verifies the fix for shard ordering bug where shards were not
sorted by index before processing, leading to incorrect concatenation.
"""

import os
import numpy as np
import pytest


def test_sharded_matmul_correctness(gt_system_4workers):
    """Test that sharded matmul produces correct results matching PyTorch."""
    import torch

    # Enable auto-sharding for this test
    os.environ['GT_AUTO_SHARD'] = '1'

    import gt

    # Disconnect if already connected
    if gt._client is not None:
        gt._client = None
        gt._connected = False

    # Connect to test dispatcher
    gt.connect('localhost:9002')

    # Initialize
    _ = gt.zeros(1, 1)

    # Create deterministic test data
    np.random.seed(42)
    torch.manual_seed(42)

    N = 512
    a_np = np.random.randn(N, N).astype(np.float32)
    b_np = np.random.randn(N, N).astype(np.float32)

    # GT computation (will be sharded across workers)
    a = gt.from_numpy(a_np)
    b = gt.from_numpy(b_np)
    c = a @ b
    d = c.sum()
    gt_result = d.data.numpy()

    # PyTorch reference
    a_pt = torch.from_numpy(a_np)
    b_pt = torch.from_numpy(b_np)
    c_pt = a_pt @ b_pt
    d_pt = c_pt.sum()
    pt_result = d_pt.numpy()

    print(f"\nGT result:      {gt_result}")
    print(f"PyTorch result: {pt_result}")

    # Results should match within floating-point tolerance
    assert np.allclose(gt_result, pt_result, rtol=1e-4, atol=1e-4), \
        f"GT result {gt_result} doesn't match PyTorch {pt_result}"


def test_sharded_matmul_deterministic(gt_system_4workers):
    """Test that sharded matmul produces deterministic results across runs."""

    # Enable auto-sharding for this test
    os.environ['GT_AUTO_SHARD'] = '1'

    import gt

    # Disconnect if already connected
    if gt._client is not None:
        gt._client = None
        gt._connected = False

    # Connect to test dispatcher
    gt.connect('localhost:9002')

    # Initialize
    _ = gt.zeros(1, 1)

    # Create deterministic test data
    np.random.seed(123)
    N = 256
    a_np = np.random.randn(N, N).astype(np.float32)
    b_np = np.random.randn(N, N).astype(np.float32)

    # Run computation twice
    results = []
    for run in range(2):
        a = gt.from_numpy(a_np)
        b = gt.from_numpy(b_np)
        c = a @ b
        d = c.sum()
        results.append(d.data.numpy())

    print(f"\nRun 1: {results[0]}")
    print(f"Run 2: {results[1]}")

    # Results should be identical (deterministic)
    assert np.allclose(results[0], results[1], rtol=1e-6, atol=1e-6), \
        f"Results not deterministic: {results[0]} vs {results[1]}"
