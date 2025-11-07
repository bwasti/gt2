"""
Tests for bugs discovered during LPM (Pixel Transformer) project.

These tests verify fixes for bugs found in /home/bwasti/oss/lpm/MISSING.md
"""
import pytest
import numpy as np


def test_bug4_reshape_negative_dimensions(client):
    """
    Bug 4: Gradient shape can have negative dimensions.
    
    Issue: reshape(-1, N) created result.shape with -1 instead of resolving it.
    This caused gradients to have negative dimensions like (-1, 256).
    
    Fix: Added _resolve_reshape_shape() to compute actual dimensions.
    """
    import gt
    
    # Create tensor with known size
    x = gt.randn(40000, 256, requires_grad=True)
    
    # Reshape with -1
    y = x.reshape(-1, 256)
    
    # Check shape is resolved (no negative dims)
    assert y.shape == (40000, 256), f"Expected (40000, 256), got {y.shape}"
    assert all(d > 0 for d in y.shape), f"Found negative dimension: {y.shape}"
    
    # Backward pass
    z = y.sum()
    z.backward()
    
    # Check gradient shape matches parameter shape
    assert x.grad.shape == x.shape, f"Gradient shape {x.grad.shape} != param shape {x.shape}"
    assert all(d > 0 for d in x.grad.shape), f"Found negative gradient dimension: {x.grad.shape}"


def test_bug4_large_linear_gradient_shapes(client):
    """
    Bug 4: Large Linear layers had gradient shape issues.
    
    Reproduction from LPM OCR model: Linear(40000, 256) gradients had shape (-1, 256).
    This prevented parameter updates: param -= lr * param.grad would fail.
    """
    import gt
    from gt.client import nn
    
    # Create large Linear layer (like in OCR model)
    linear = nn.Linear(40000, 256)
    
    # Forward pass
    x = gt.randn(4, 40000, requires_grad=True)  # Batch of 4
    y = linear(x)
    
    # Backward pass
    loss = y.sum()
    loss.backward()
    
    # Check all parameter gradients have correct shapes
    for param in linear.parameters():
        if param.grad is not None:
            # No negative dimensions
            assert all(d > 0 for d in param.grad.shape), \
                f"Found negative dimension in gradient: {param.grad.shape}"
            
            # Shapes match
            assert param.grad.shape == param.shape, \
                f"Gradient shape {param.grad.shape} != param shape {param.shape}"
    
    # Test parameter update (this used to fail with negative dimensions)
    learning_rate = 0.001
    for param in linear.parameters():
        if param.grad is not None:
            # This should work without fixing negative dimensions manually
            param -= learning_rate * param.grad


def test_bug3_slice_sharded_tensors_basic(client):
    """
    Bug 3: Slicing sharded tensors.
    
    Issue: Slicing sharded tensors would hang (deadlock).
    Cause: Slice handler tried to gather shards synchronously during async batch processing.
    
    Fix: Added SliceOp handling to sharding modifier - injects GatherShards command.
    """
    import gt
    import os
    
    # Enable AUTO_SHARD for this test
    old_autoshard = os.environ.get('GT_AUTO_SHARD')
    os.environ['GT_AUTO_SHARD'] = '1'
    
    try:
        # Create sharded tensor (will be split across workers)
        x = gt.randn(128, 64)
        
        # Slice it
        y = x[:64, :]
        
        # Verify result
        data = y.data.numpy()
        assert data.shape == (64, 64), f"Expected (64, 64), got {data.shape}"
        
    finally:
        # Restore AUTO_SHARD setting
        if old_autoshard is None:
            os.environ.pop('GT_AUTO_SHARD', None)
        else:
            os.environ['GT_AUTO_SHARD'] = old_autoshard


def test_bug3_slice_sharded_tensors_advanced(client):
    """
    Bug 3: Advanced slicing patterns on sharded tensors.
    
    Tests various slicing patterns that use the GatherShards injection.
    """
    import gt
    import os
    
    # Enable AUTO_SHARD
    old_autoshard = os.environ.get('GT_AUTO_SHARD')
    os.environ['GT_AUTO_SHARD'] = '1'
    
    try:
        x = gt.randn(128, 64)
        
        # Test different slicing patterns
        y1 = x[:32, :]
        assert y1.data.numpy().shape == (32, 64)
        
        y2 = x[64:96, :]
        assert y2.data.numpy().shape == (32, 64)
        
        y3 = x[:, :32]
        assert y3.data.numpy().shape == (128, 32)
        
        y4 = x[::2, :]
        assert y4.data.numpy().shape == (64, 64)
        
        # Test with operations after slicing
        y5 = x[:64, :].sum()
        result = y5.data.numpy()
        assert result.shape == ()
        
    finally:
        if old_autoshard is None:
            os.environ.pop('GT_AUTO_SHARD', None)
        else:
            os.environ['GT_AUTO_SHARD'] = old_autoshard
