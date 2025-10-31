"""
PyTorch-based engine with compilation support.

Supports instruction batching and torch.compile() optimization.
Caches compiled graphs based on operation patterns.
"""

import numpy as np
import hashlib
from typing import Optional, List, Dict, Any, Callable
from .base import Engine, Operation


class PyTorchEngine(Engine):
    """PyTorch-based engine (GPU support, compilation, distributed primitives)."""

    def __init__(self, enable_compilation: bool = True):
        try:
            import torch
            import torch.distributed as dist
            self.torch = torch
            self.dist = dist
            self.device = torch.device('cpu')  # TODO: support GPU via env var

            # Compilation support
            self.enable_compilation = enable_compilation
            self._compiled_cache: Dict[str, Callable] = {}
            self._cache_hits = 0
            self._cache_misses = 0

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

    def sqrt(self, tensor):
        return self.torch.sqrt(tensor)

    def transpose(self, tensor):
        return self.torch.transpose(tensor, -2, -1)

    def supports_distributed(self) -> bool:
        return True

    def supports_batching(self) -> bool:
        return True

    def _compute_graph_signature(self, operations: List[Operation]) -> str:
        """
        Compute a signature for the computation graph.

        The signature captures the structure of the computation:
        - Operation types and names
        - Topology (which operations depend on which)

        Operations with the same signature can share a compiled function.
        """
        # Build signature from operation sequence
        sig_parts = []
        for op in operations:
            # Include operation type and name
            sig_parts.append(f"{op.op_type}:{op.op_name}")
            # Include number and order of inputs
            sig_parts.append(f"in:{len(op.input_ids)}")

        # Hash the signature
        sig_str = "|".join(sig_parts)
        return hashlib.md5(sig_str.encode()).hexdigest()

    def _build_graph_function(self, operations: List[Operation]) -> Callable:
        """
        Build a PyTorch function from a sequence of operations.

        This function will be compiled with torch.compile().
        """
        def graph_fn(tensor_dict: Dict[str, Any]) -> Dict[str, Any]:
            """Execute the operation graph."""
            results = {}

            for op in operations:
                if op.op_type == 'binary':
                    left = tensor_dict.get(op.input_ids[0])
                    right = tensor_dict.get(op.input_ids[1])

                    if op.op_name == 'add':
                        results[op.result_id] = left + right
                    elif op.op_name == 'sub':
                        results[op.result_id] = left - right
                    elif op.op_name == 'mul':
                        results[op.result_id] = left * right
                    elif op.op_name == 'div':
                        results[op.result_id] = left / right
                    elif op.op_name == 'matmul':
                        results[op.result_id] = self.torch.matmul(left, right)
                    elif op.op_name == 'gt':
                        results[op.result_id] = (left > right).float()
                    else:
                        raise ValueError(f"Unknown binary op: {op.op_name}")

                elif op.op_type == 'unary':
                    input_tensor = tensor_dict.get(op.input_ids[0])

                    if op.op_name == 'relu':
                        results[op.result_id] = self.torch.relu(input_tensor)
                    elif op.op_name == 'sigmoid':
                        results[op.result_id] = self.torch.sigmoid(input_tensor)
                    elif op.op_name == 'tanh':
                        results[op.result_id] = self.torch.tanh(input_tensor)
                    elif op.op_name == 'exp':
                        results[op.result_id] = self.torch.exp(input_tensor)
                    elif op.op_name == 'log':
                        results[op.result_id] = self.torch.log(input_tensor)
                    elif op.op_name == 'sum':
                        results[op.result_id] = self.torch.sum(input_tensor)
                    elif op.op_name == 'mean':
                        results[op.result_id] = self.torch.mean(input_tensor)
                    elif op.op_name == 'transpose':
                        results[op.result_id] = self.torch.transpose(input_tensor, -2, -1)
                    elif op.op_name == 'sqrt':
                        results[op.result_id] = self.torch.sqrt(input_tensor)
                    else:
                        raise ValueError(f"Unknown unary op: {op.op_name}")

                # Update tensor_dict with new results for subsequent operations
                tensor_dict[op.result_id] = results[op.result_id]

            return results

        return graph_fn

    def execute_batch(self, operations: List[Operation], tensors: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a batch of operations with compilation support.

        Args:
            operations: List of operations to execute
            tensors: Dictionary mapping tensor IDs to tensor objects

        Returns:
            Dictionary of newly created tensors (result_id -> tensor)

        TODO: Batched execution is currently broken - graph building doesn't handle
        all tensor dependencies correctly, resulting in NoneType errors. Intermediate
        tensors from scalar operations (e.g., from .sum(), .mean()) aren't being
        tracked properly in the operation graph. Need to fix tensor dependency
        tracking in the worker or dispatcher.
        """
        if not self.enable_compilation or len(operations) == 1:
            # Fall back to eager execution for single operations
            return self._execute_eager(operations, tensors)

        # Compute graph signature
        signature = self._compute_graph_signature(operations)

        # Check cache
        if signature in self._compiled_cache:
            compiled_fn = self._compiled_cache[signature]
            self._cache_hits += 1
        else:
            # Build and compile the graph function
            graph_fn = self._build_graph_function(operations)

            # Compile with torch.compile
            try:
                compiled_fn = self.torch.compile(graph_fn, mode="default")
                self._compiled_cache[signature] = compiled_fn
                self._cache_misses += 1
            except Exception as e:
                # Fall back to eager if compilation fails
                print(f"Warning: Compilation failed ({e}), falling back to eager execution")
                return self._execute_eager(operations, tensors)

        # Execute compiled function
        results = compiled_fn(tensors.copy())
        return results

    def _execute_eager(self, operations: List[Operation], tensors: Dict[str, Any]) -> Dict[str, Any]:
        """Execute operations eagerly (one by one) without compilation."""
        results = {}

        for op in operations:
            if op.op_type == 'binary':
                left = tensors.get(op.input_ids[0])
                right = tensors.get(op.input_ids[1])

                if op.op_name == 'add':
                    result = self.add(left, right)
                elif op.op_name == 'sub':
                    result = left - right
                elif op.op_name == 'mul':
                    result = self.mul(left, right)
                elif op.op_name == 'div':
                    result = left / right
                elif op.op_name == 'matmul':
                    result = self.matmul(left, right)
                elif op.op_name == 'gt':
                    result = (left > right).float()
                else:
                    raise ValueError(f"Unknown binary op: {op.op_name}")

                results[op.result_id] = result
                tensors[op.result_id] = result

            elif op.op_type == 'unary':
                input_tensor = tensors.get(op.input_ids[0])

                if op.op_name == 'relu':
                    result = self.relu(input_tensor)
                elif op.op_name == 'sigmoid':
                    result = self.sigmoid(input_tensor)
                elif op.op_name == 'tanh':
                    result = self.tanh(input_tensor)
                elif op.op_name == 'exp':
                    result = self.torch.exp(input_tensor)
                elif op.op_name == 'log':
                    result = self.torch.log(input_tensor)
                elif op.op_name == 'sum':
                    result = self.sum(input_tensor)
                elif op.op_name == 'mean':
                    result = self.mean(input_tensor)
                elif op.op_name == 'transpose':
                    result = self.transpose(input_tensor)
                elif op.op_name == 'sqrt':
                    result = self.sqrt(input_tensor)
                else:
                    raise ValueError(f"Unknown unary op: {op.op_name}")

                results[op.result_id] = result
                tensors[op.result_id] = result

        return results

    def get_compilation_stats(self) -> Dict[str, int]:
        """Get compilation cache statistics."""
        return {
            "cache_size": len(self._compiled_cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": self._cache_hits / max(1, self._cache_hits + self._cache_misses)
        }

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
