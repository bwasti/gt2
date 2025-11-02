"""
Base Engine interface for GT workers.

All computation backends (NumPy, PyTorch, etc.) implement this interface.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Optional, List, Dict
from dataclasses import dataclass


@dataclass
class Operation:
    """Represents a single operation to execute."""
    op_type: str  # 'binary', 'unary', 'create', etc.
    op_name: str  # 'add', 'matmul', 'relu', etc.
    result_id: str
    input_ids: List[str]
    params: Optional[Dict[str, Any]] = None  # Shape, dtype, etc. (optional)


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
    def sum(self, tensor: Any, axis: Optional[int] = None, keepdims: bool = False) -> Any:
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
    def sqrt(self, tensor: Any) -> Any:
        """Square root."""
        pass

    @abstractmethod
    def transpose(self, tensor: Any) -> Any:
        """Transpose (swap last two dimensions)."""
        pass

    @abstractmethod
    def supports_distributed(self) -> bool:
        """Whether this engine supports distributed operations."""
        pass

    def supports_batching(self) -> bool:
        """Whether this engine supports instruction batching/compilation."""
        return False

    def execute_batch(self, operations: List[Operation], tensors: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a batch of operations (optional, for compilation-capable engines).

        Args:
            operations: List of operations to execute
            tensors: Dictionary mapping tensor IDs to tensor objects

        Returns:
            Dictionary of newly created tensors

        Default implementation: execute operations one by one (eager mode).
        """
        results = {}
        for op in operations:
            # This will be overridden by engines that support batching
            raise NotImplementedError(f"Batching not supported by {self.__class__.__name__}")
        return results

    def all_reduce_sum(self, tensor: Any, group=None) -> Any:
        """
        All-reduce sum across workers (distributed primitive).

        Only implemented for engines that support distributed operations.
        Raises NotImplementedError for single-node engines like NumPy.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support distributed all-reduce")

    def handle_hotpath_start(self, sequence_id: str):
        """
        Handle start of hot path marker (no-op by default).

        Compilation-capable engines can override to begin buffering operations.
        """
        pass

    def handle_hotpath_end(self, sequence_id: str):
        """
        Handle end of hot path marker (no-op by default).

        Compilation-capable engines can override to compile and execute buffered operations.
        """
        pass
