"""
NumPy-based engine (CPU only, eager execution).

No batching or compilation support - operations execute immediately.
"""

import numpy as np
from typing import Optional
from .base import Engine


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

    def sum(self, tensor: np.ndarray, axis: Optional[int] = None, keepdims: bool = False) -> np.ndarray:
        return np.sum(tensor, axis=axis, keepdims=keepdims)

    def mean(self, tensor: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
        return np.mean(tensor, axis=axis)

    def relu(self, tensor: np.ndarray) -> np.ndarray:
        return np.maximum(0, tensor)

    def sigmoid(self, tensor: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-tensor))

    def tanh(self, tensor: np.ndarray) -> np.ndarray:
        return np.tanh(tensor)

    def sqrt(self, tensor: np.ndarray) -> np.ndarray:
        return np.sqrt(tensor)

    def transpose(self, tensor: np.ndarray) -> np.ndarray:
        # Swap last two dimensions (PyTorch/GT semantics)
        # For 1D or 0D arrays, transpose is a no-op
        if tensor.ndim < 2:
            return tensor
        # np.transpose() without args reverses ALL axes, but we only want last 2
        return np.swapaxes(tensor, -2, -1)

    def supports_distributed(self) -> bool:
        return False

    def supports_batching(self) -> bool:
        return False
