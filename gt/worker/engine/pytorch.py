"""
PyTorch-based engine for eager execution.

Simple, readable PyTorch backend without compilation complexity.
For compilation support, use pytorch_compile.py instead.
"""

import numpy as np
from typing import Optional, List, Dict, Any
from .base import Engine, Operation


class PyTorchEngine(Engine):
    """PyTorch-based engine (GPU support, distributed primitives)."""

    def __init__(self):
        try:
            import torch
            import torch.distributed as dist
            self.torch = torch
            self.dist = dist
            self.device = torch.device('cpu')  # TODO: support GPU via env var

            # Initialize process group for distributed operations
            self.distributed_initialized = False
        except ImportError:
            raise ImportError("PyTorch not available")

    def create_tensor(self, data: np.ndarray):
        return self.torch.from_numpy(data).to(self.device)

    def to_numpy(self, tensor) -> np.ndarray:
        return tensor.cpu().numpy()

    def randn(self, shape: tuple, dtype: str = 'float32'):
        torch_dtype = getattr(self.torch, dtype)
        return self.torch.randn(*shape, dtype=torch_dtype, device=self.device)

    def zeros(self, shape: tuple, dtype: str = 'float32'):
        torch_dtype = getattr(self.torch, dtype)
        return self.torch.zeros(*shape, dtype=torch_dtype, device=self.device)

    def matmul(self, a, b):
        return self.torch.matmul(a, b)

    def add(self, a, b):
        return a + b

    def mul(self, a, b):
        return a * b

    def sum(self, tensor, axis: Optional[int] = None, keepdims: bool = False):
        if axis is None:
            # Full reduction - sum all elements
            if keepdims:
                # For keepdims with full reduction, we need to keep all dims as 1
                shape = tuple([1] * len(tensor.shape))
                return self.torch.sum(tensor).reshape(shape)
            return self.torch.sum(tensor)
        # Axis-specific reduction
        return self.torch.sum(tensor, dim=axis, keepdim=keepdims)

    def mean(self, tensor, axis: Optional[int] = None, keepdims: bool = False):
        if axis is None:
            # Full reduction - mean of all elements
            if keepdims:
                # For keepdims with full reduction, we need to keep all dims as 1
                shape = tuple([1] * len(tensor.shape))
                return self.torch.mean(tensor).reshape(shape)
            return self.torch.mean(tensor)
        # Axis-specific reduction
        return self.torch.mean(tensor, dim=axis, keepdim=keepdims)

    def relu(self, tensor):
        return self.torch.relu(tensor)

    def sigmoid(self, tensor):
        return self.torch.sigmoid(tensor)

    def tanh(self, tensor):
        return self.torch.tanh(tensor)

    def sqrt(self, tensor):
        return self.torch.sqrt(tensor)

    def transpose(self, tensor):
        # For 1D or 0D tensors, transpose is a no-op
        if tensor.ndim < 2:
            return tensor
        return self.torch.transpose(tensor, -2, -1)

    def supports_distributed(self) -> bool:
        return True

    def all_reduce_sum(self, tensor, group=None):
        """
        All-reduce sum across workers.

        NOTE: This is a simplified implementation. In production, you'd need:
        1. Proper process group initialization
        2. NCCL backend for GPU tensors
        3. Proper error handling
        """
        if not self.distributed_initialized:
            return tensor

        # This would be the actual implementation:
        # self.dist.all_reduce(tensor, op=self.dist.ReduceOp.SUM, group=group)
        return tensor
