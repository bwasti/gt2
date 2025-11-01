"""
PyTorch-based engine with compilation support.

Supports instruction batching and torch.compile() optimization.
Caches compiled graphs based on operation patterns.
"""

import numpy as np
import hashlib
from typing import Optional, List, Dict, Any, Callable
from .base import Engine, Operation
from gt.debug import debug_print_compile


class PyTorchEngine(Engine):
    """PyTorch-based engine (GPU support, compilation, distributed primitives)."""

    def __init__(self, enable_compilation: bool = False):
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
                    # Check results first (for intermediate tensors), then tensor_dict
                    left = results.get(op.input_ids[0])
                    if left is None:
                        left = tensor_dict.get(op.input_ids[0])
                    right = results.get(op.input_ids[1])
                    if right is None:
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
                    # Get input tensor (if operation has inputs)
                    input_tensor = None
                    if op.input_ids:
                        input_tensor = results.get(op.input_ids[0])
                        if input_tensor is None:
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
                        # Use axis and keepdims from params
                        params = op.params or {}
                        axis = params.get('axis', None)
                        keepdims = params.get('keepdims', False)
                        results[op.result_id] = self.sum(input_tensor, axis=axis, keepdims=keepdims)
                    elif op.op_name == 'mean':
                        params = op.params or {}
                        axis = params.get('axis', None)
                        keepdims = params.get('keepdims', False)
                        results[op.result_id] = self.mean(input_tensor, axis=axis, keepdims=keepdims)
                    elif op.op_name == 'transpose':
                        results[op.result_id] = self.torch.transpose(input_tensor, -2, -1)
                    elif op.op_name == 'sqrt':
                        results[op.result_id] = self.torch.sqrt(input_tensor)
                    elif op.op_name == 'zeros':
                        # Creation operation - get shape and dtype from params
                        params = op.params or {}
                        shape = params.get('shape', ())
                        dtype = params.get('dtype', 'float32')
                        results[op.result_id] = self.zeros(shape, dtype)
                    elif op.op_name == 'randn':
                        # Creation operation - get shape and dtype from params
                        params = op.params or {}
                        shape = params.get('shape', ())
                        dtype = params.get('dtype', 'float32')
                        results[op.result_id] = self.randn(shape, dtype)
                    else:
                        raise ValueError(f"Unknown unary op: {op.op_name}")

                # Update tensor_dict with new results for subsequent operations
                tensor_dict[op.result_id] = results[op.result_id]

            # Return full tensor_dict, not just results
            # This ensures ALL intermediates are visible, including ones that
            # might be needed by future operations (like backward pass)
            return tensor_dict

        return graph_fn

    def _normalize_tensor_ids(self, operations: List[Operation], tensors: Dict[str, Any]):
        """
        Normalize tensor IDs to stable indices for compilation.

        Maps all tensor IDs (inputs and results) to positional indices.
        This makes the compiled function independent of actual tensor IDs.

        Returns:
            id_to_index: Dict mapping tensor ID -> index
            input_list: List of input tensors in index order
            result_mapping: Dict mapping result tensor ID -> result index
        """
        id_to_index = {}
        next_index = 0

        # First, assign indices to all input tensors (already exist in tensors dict)
        all_result_ids = set(op.result_id for op in operations)

        for op in operations:
            for input_id in op.input_ids:
                if input_id not in id_to_index and input_id not in all_result_ids:
                    # This is an external input tensor
                    id_to_index[input_id] = next_index
                    next_index += 1

        # Second, assign indices to all result tensors
        result_start_idx = next_index
        result_mapping = {}
        for op in operations:
            if op.result_id not in id_to_index:
                id_to_index[op.result_id] = next_index
                result_mapping[op.result_id] = next_index - result_start_idx
                next_index += 1

        # Build input list in index order
        input_list = [None] * result_start_idx
        for tensor_id, idx in id_to_index.items():
            if idx < result_start_idx:  # External input
                input_list[idx] = tensors[tensor_id]

        return id_to_index, input_list, result_mapping

    def _build_normalized_graph_function(self, operations: List[Operation], id_to_index: Dict[str, int]):
        """
        Build a graph function using normalized indices instead of tensor IDs.

        The function takes a list of tensors and returns a list of result tensors.
        This allows torch.compile to recognize the same pattern across iterations.
        """
        # Pre-compute all operation instructions with normalized indices
        # This avoids capturing the operations list in the closure
        op_instructions = []
        for op in operations:
            instr = {
                'op_type': op.op_type,
                'op_name': op.op_name,
                'result_idx': id_to_index[op.result_id],
                'input_indices': [id_to_index[input_id] for input_id in op.input_ids] if op.input_ids else [],
                'params': op.params
            }
            op_instructions.append(instr)

        def graph_fn(input_tensors: list) -> list:
            """Execute the operation graph with indexed tensors."""
            # tensor_storage holds all tensors by index (inputs + intermediates)
            tensor_storage = list(input_tensors)  # Copy inputs

            for instr in op_instructions:
                if instr['op_type'] == 'binary':
                    left = tensor_storage[instr['input_indices'][0]]
                    right = tensor_storage[instr['input_indices'][1]]

                    if instr['op_name'] == 'add':
                        result = left + right
                    elif instr['op_name'] == 'sub':
                        result = left - right
                    elif instr['op_name'] == 'mul':
                        result = left * right
                    elif instr['op_name'] == 'div':
                        result = left / right
                    elif instr['op_name'] == 'matmul':
                        result = self.torch.matmul(left, right)
                    elif instr['op_name'] == 'gt':
                        result = (left > right).float()
                    else:
                        raise ValueError(f"Unknown binary op: {instr['op_name']}")

                    result_idx = instr['result_idx']
                    # Extend tensor_storage if needed
                    while len(tensor_storage) <= result_idx:
                        tensor_storage.append(None)
                    tensor_storage[result_idx] = result

                elif instr['op_type'] == 'unary':
                    # Get input tensor (if operation has inputs)
                    input_tensor = None
                    if instr['input_indices']:
                        input_tensor = tensor_storage[instr['input_indices'][0]]

                    if instr['op_name'] == 'relu':
                        result = self.torch.relu(input_tensor)
                    elif instr['op_name'] == 'sigmoid':
                        result = self.torch.sigmoid(input_tensor)
                    elif instr['op_name'] == 'tanh':
                        result = self.torch.tanh(input_tensor)
                    elif instr['op_name'] == 'exp':
                        result = self.torch.exp(input_tensor)
                    elif instr['op_name'] == 'log':
                        result = self.torch.log(input_tensor)
                    elif instr['op_name'] == 'sum':
                        params = instr['params'] or {}
                        axis = params.get('axis', None)
                        keepdims = params.get('keepdims', False)
                        result = self.torch.sum(input_tensor, dim=axis, keepdim=keepdims)
                    elif instr['op_name'] == 'mean':
                        params = instr['params'] or {}
                        axis = params.get('axis', None)
                        keepdims = params.get('keepdims', False)
                        result = self.torch.mean(input_tensor, dim=axis, keepdim=keepdims)
                    elif instr['op_name'] == 'sqrt':
                        result = self.torch.sqrt(input_tensor)
                    elif instr['op_name'] == 'transpose':
                        result = input_tensor.transpose(-2, -1)
                    elif instr['op_name'] == 'randn':
                        params = instr['params'] or {}
                        shape = params.get('shape', ())
                        result = self.torch.randn(shape)
                    elif instr['op_name'] == 'zeros':
                        params = instr['params'] or {}
                        shape = params.get('shape', ())
                        result = self.torch.zeros(shape)
                    else:
                        raise ValueError(f"Unknown unary op: {instr['op_name']}")

                    result_idx = instr['result_idx']
                    while len(tensor_storage) <= result_idx:
                        tensor_storage.append(None)
                    tensor_storage[result_idx] = result

                elif instr['op_type'] == 'reshape':
                    input_tensor = tensor_storage[instr['input_indices'][0]]

                    if instr['op_name'] == 'reshape':
                        result = input_tensor.reshape(*instr['params'])
                    elif instr['op_name'] == 'unsqueeze':
                        dim = instr['params'][0]
                        result = input_tensor.unsqueeze(dim)
                    elif instr['op_name'] == 'squeeze':
                        if len(instr['params']) == 0:
                            result = input_tensor.squeeze()
                        else:
                            dim = instr['params'][0]
                            result = input_tensor.squeeze(dim)
                    else:
                        raise ValueError(f"Unknown reshape op: {instr['op_name']}")

                    result_idx = instr['result_idx']
                    while len(tensor_storage) <= result_idx:
                        tensor_storage.append(None)
                    tensor_storage[result_idx] = result

                elif instr['op_type'] == 'slice':
                    input_tensor = tensor_storage[instr['input_indices'][0]]
                    # params contains the slice key
                    result = input_tensor[instr['params']]
                    result_idx = instr['result_idx']
                    while len(tensor_storage) <= result_idx:
                        tensor_storage.append(None)
                    tensor_storage[result_idx] = result

            # Return all newly created tensors (results) in order
            num_inputs = len(input_tensors)
            return tensor_storage[num_inputs:]

        return graph_fn

    def execute_batch(self, operations: List[Operation], tensors: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a batch of operations with compilation support.

        Args:
            operations: List of operations to execute
            tensors: Dictionary mapping tensor IDs to tensor objects

        Returns:
            Dictionary of newly created tensors (result_id -> tensor)
        """
        # Minimum batch size for compilation to be worthwhile
        # Small batches (1-2 ops) have overhead that outweighs compilation benefits
        MIN_COMPILE_BATCH_SIZE = 3

        if not self.enable_compilation or len(operations) < MIN_COMPILE_BATCH_SIZE:
            # Fall back to eager execution for small batches
            return self._execute_eager(operations, tensors)

        # First normalize to determine number of external inputs
        id_to_index, input_list, result_mapping = self._normalize_tensor_ids(operations, tensors)
        num_inputs = len(input_list)

        # Compute graph signature including input count
        # Same operations with different input counts need different compiled functions
        base_signature = self._compute_graph_signature(operations)
        graph_signature = f"{base_signature}:inputs={num_inputs}"

        # Check if we have a cached compiled function
        if graph_signature in self._compiled_cache:
            compiled_fn = self._compiled_cache[graph_signature]
            self._cache_hits += 1
        else:
            # Build graph function with normalized indices (id_to_index already computed above)
            graph_fn = self._build_normalized_graph_function(operations, id_to_index)

            # Compile with torch.compile
            try:
                compiled_fn = self.torch.compile(graph_fn, mode="default")
                self._compiled_cache[graph_signature] = compiled_fn
                self._cache_misses += 1
                debug_print_compile(f"Compiled new graph: {graph_signature} ({len(operations)} ops)")
            except Exception as e:
                # Fall back to eager if compilation fails
                debug_print_compile(f"Warning: Compilation failed ({e}), falling back to eager execution")
                return self._execute_eager(operations, tensors)

        # Execute compiled function with input tensors as a list
        # (input_list and result_mapping already computed above)
        result_list = compiled_fn(input_list)

        # Map results back to original tensor IDs
        results_dict = {}
        for result_id, result_idx in result_mapping.items():
            results_dict[result_id] = result_list[result_idx]
            tensors[result_id] = result_list[result_idx]

        return tensors

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
                # Get input tensor (if operation has inputs)
                input_tensor = None
                if op.input_ids:
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
                    # Use axis and keepdims from params
                    axis = op.params.get('axis', None)
                    keepdims = op.params.get('keepdims', False)
                    result = self.sum(input_tensor, axis=axis, keepdims=keepdims)
                elif op.op_name == 'mean':
                    axis = op.params.get('axis', None)
                    keepdims = op.params.get('keepdims', False)
                    result = self.mean(input_tensor, axis=axis, keepdims=keepdims)
                elif op.op_name == 'transpose':
                    result = self.transpose(input_tensor)
                elif op.op_name == 'sqrt':
                    result = self.sqrt(input_tensor)
                elif op.op_name == 'zeros':
                    # Creation operation - get shape and dtype from params
                    params = op.params or {}
                    shape = params.get('shape', ())
                    dtype = params.get('dtype', 'float32')
                    result = self.zeros(shape, dtype)
                elif op.op_name == 'randn':
                    # Creation operation - get shape and dtype from params
                    params = op.params or {}
                    shape = params.get('shape', ())
                    dtype = params.get('dtype', 'float32')
                    result = self.randn(shape, dtype)
                else:
                    raise ValueError(f"Unknown unary op: {op.op_name}")

                results[op.result_id] = result
                tensors[op.result_id] = result

        # Return full tensors dict for consistency with compiled path
        # This ensures ALL intermediates are visible to future operations
        return tensors

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
