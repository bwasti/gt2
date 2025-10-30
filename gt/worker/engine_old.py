"""
Engine abstraction for different backends (NumPy, PyTorch).

Engines handle the low-level tensor operations and provide distributed primitives.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Optional


class Engine(ABC):
    """Abstract base class for computation engines."""

    @abstractmethod
    def create_tensor(self, data: np.ndarray) -> Any:
        """Create a tensor from numpy array."""
        pass

    @abstractmethod
    def to_numpy(self, tensor: Any) -> np.ndarray:
        """Convert tensor to numpy array."""
        pass

    @abstractmethod
    def randn(self, shape: tuple, dtype: str = 'float32') -> Any:
        """Create random normal tensor."""
        pass

    @abstractmethod
    def zeros(self, shape: tuple, dtype: str = 'float32') -> Any:
        """Create zero tensor."""
        pass

    @abstractmethod
    def matmul(self, a: Any, b: Any) -> Any:
        """Matrix multiplication."""
        pass

    @abstractmethod
    def add(self, a: Any, b: Any) -> Any:
        """Element-wise addition."""
        pass

    @abstractmethod
    def mul(self, a: Any, b: Any) -> Any:
        """Element-wise multiplication."""
        pass

    @abstractmethod
    def sum(self, tensor: Any, axis: Optional[int] = None) -> Any:
        """Sum reduction."""
        pass

    @abstractmethod
    def mean(self, tensor: Any, axis: Optional[int] = None) -> Any:
        """Mean reduction."""
        pass

    @abstractmethod
    def relu(self, tensor: Any) -> Any:
        """ReLU activation."""
        pass

    @abstractmethod
    def sigmoid(self, tensor: Any) -> Any:
        """Sigmoid activation."""
        pass

    @abstractmethod
    def tanh(self, tensor: Any) -> Any:
        """Tanh activation."""
        pass

    @abstractmethod
    def transpose(self, tensor: Any) -> Any:
        """Transpose (swap last two dimensions)."""
        pass

    @abstractmethod
    def supports_distributed(self) -> bool:
        """Whether this engine supports distributed operations."""
        pass

    def all_reduce_sum(self, tensor: Any, group=None) -> Any:
        """
        All-reduce sum across workers (distributed primitive).

        Only implemented for engines that support distributed operations.
        Raises NotImplementedError for single-node engines like NumPy.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support distributed all-reduce")


class NumpyEngine(Engine):
    """NumPy-based engine (CPU only, no distributed support)."""

    def __init__(self):
        self.np = np

    def create_tensor(self, data: np.ndarray) -> np.ndarray:
        return data

    def to_numpy(self, tensor: np.ndarray) -> np.ndarray:
        return tensor

    def randn(self, shape: tuple, dtype: str = 'float32') -> np.ndarray:
        return np.random.randn(*shape).astype(dtype)

    def zeros(self, shape: tuple, dtype: str = 'float32') -> np.ndarray:
        return np.zeros(shape, dtype=dtype)

    def matmul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a @ b

    def add(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a + b

    def mul(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a * b

    def sum(self, tensor: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
        return np.sum(tensor, axis=axis)

    def mean(self, tensor: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
        return np.mean(tensor, axis=axis)

    def relu(self, tensor: np.ndarray) -> np.ndarray:
        return np.maximum(0, tensor)

    def sigmoid(self, tensor: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-tensor))

    def tanh(self, tensor: np.ndarray) -> np.ndarray:
        return np.tanh(tensor)

    def transpose(self, tensor: np.ndarray) -> np.ndarray:
        return np.transpose(tensor)

    def supports_distributed(self) -> bool:
        return False


class PyTorchEngine(Engine):
    """PyTorch-based engine (GPU support, distributed primitives)."""

    def __init__(self):
        try:
            import torch
            import torch.distributed as dist
            self.torch = torch
            self.dist = dist
            self.device = torch.device('cpu')  # TODO: support GPU

            # Initialize process group for distributed operations
            # NOTE: This is a simplified version - in production you'd need
            # proper initialization with rank, world_size, backend, etc.
            self.distributed_initialized = False
        except ImportError:
            raise ImportError("PyTorch not available")

    def create_tensor(self, data: np.ndarray):
        return self.torch.from_numpy(data).to(self.device)

    def to_numpy(self, tensor) -> np.ndarray:
        return tensor.cpu().numpy()

    def randn(self, shape: tuple, dtype: str = 'float32'):
        torch_dtype = getattr(self.torch, dtype)
        return self.torch.randn(shape, dtype=torch_dtype, device=self.device)

    def zeros(self, shape: tuple, dtype: str = 'float32'):
        torch_dtype = getattr(self.torch, dtype)
        return self.torch.zeros(shape, dtype=torch_dtype, device=self.device)

    def matmul(self, a, b):
        return self.torch.matmul(a, b)

    def add(self, a, b):
        return a + b

    def mul(self, a, b):
        return a * b

    def sum(self, tensor, axis: Optional[int] = None):
        if axis is None:
            return self.torch.sum(tensor)
        return self.torch.sum(tensor, dim=axis)

    def mean(self, tensor, axis: Optional[int] = None):
        if axis is None:
            return self.torch.mean(tensor)
        return self.torch.mean(tensor, dim=axis)

    def relu(self, tensor):
        return self.torch.relu(tensor)

    def sigmoid(self, tensor):
        return self.torch.sigmoid(tensor)

    def tanh(self, tensor):
        return self.torch.tanh(tensor)

    def transpose(self, tensor):
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

        For now, this is a placeholder that will be implemented when we
        add proper distributed training support.
        """
        if not self.distributed_initialized:
            # For now, just return the tensor unchanged
            # TODO: Implement proper distributed all-reduce
            return tensor

        # This would be the actual implementation:
        # self.dist.all_reduce(tensor, op=self.dist.ReduceOp.SUM, group=group)
        return tensor


def create_engine(backend: str = 'numpy') -> Engine:
    """Factory function to create an engine."""
    if backend == 'numpy':
        return NumpyEngine()
    elif backend == 'pytorch':
        return PyTorchEngine()
    else:
        raise ValueError(f"Unknown backend: {backend}")
