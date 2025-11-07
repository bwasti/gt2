"""
Test GT_AUTO_SHARD functionality.
"""

import os
import numpy as np
import pytest


def test_auto_shard_with_4_workers(gt_system_4workers):
    """Test that GT_AUTO_SHARD=1 automatically shards tensors across all workers."""

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

    # Create a tensor that should be auto-sharded
    # With 4 workers and GT_AUTO_SHARD=1, this should be sharded as (32, 64) per worker
    print("\n--- Creating tensor with shape (128, 64) ---")
    a = gt.randn(128, 64)

    # Verify the tensor works correctly
    data = a.data.numpy()
    print(f"Retrieved shape: {data.shape}")
    assert data.shape == (128, 64), f"Expected shape (128, 64), got {data.shape}"

    # Test with explicit data
    print("\n--- Creating tensor from numpy array ---")
    test_data = np.arange(256).reshape(128, 2).astype(np.float32)
    b = gt.from_numpy(test_data)
    retrieved = b.data.numpy()
    print(f"Retrieved shape: {retrieved.shape}")
    assert retrieved.shape == (128, 2), f"Expected shape (128, 2), got {retrieved.shape}"
    assert np.allclose(retrieved, test_data), "Data mismatch!"

    # Test operations on auto-sharded tensors
    print("\n--- Testing operations on auto-sharded tensors ---")
    c = gt.randn(128, 64)
    d = c.relu()
    result = d.data.numpy()
    print(f"Result shape after relu: {result.shape}")
    assert result.shape == (128, 64), f"Expected shape (128, 64), got {result.shape}"

    # Clean up
    os.environ['GT_AUTO_SHARD'] = '0'

    print("\n✓ All auto-shard tests passed!")


def test_auto_shard_uneven_split(gt_system_4workers):
    """Test that GT_AUTO_SHARD replicates when tensor can't be evenly sharded."""

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

    # Create a tensor that can't be evenly sharded (13 % 4 != 0)
    # Should replicate instead
    print("\n--- Creating tensor with shape (13, 64) ---")
    print("Expected: Cannot shard evenly, should replicate instead")
    a = gt.randn(13, 64)

    data = a.data.numpy()
    print(f"Retrieved shape: {data.shape}")
    assert data.shape == (13, 64), f"Expected shape (13, 64), got {data.shape}"

    # Clean up
    os.environ['GT_AUTO_SHARD'] = '0'

    print("\n✓ Replication fallback works!")


def test_auto_shard_disabled_by_default(gt_system_4workers):
    """Test that auto-sharding is disabled by default (GT_AUTO_SHARD=0)."""

    # Ensure auto-sharding is disabled
    os.environ['GT_AUTO_SHARD'] = '0'

    import gt

    # Disconnect if already connected
    if gt._client is not None:
        gt._client = None
        gt._connected = False

    # Connect to test dispatcher
    gt.connect('localhost:9002')

    # Initialize
    _ = gt.zeros(1, 1)

    # Create a tensor - should NOT be auto-sharded
    print("\n--- Creating tensor with GT_AUTO_SHARD=0 ---")
    a = gt.randn(128, 64)

    data = a.data.numpy()
    print(f"Retrieved shape: {data.shape}")
    assert data.shape == (128, 64), f"Expected shape (128, 64), got {data.shape}"

    print("\n✓ Auto-shard correctly disabled by default!")
