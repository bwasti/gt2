"""
PyTorch engine with torch.compile() support.

Extends the base PyTorchEngine with:
- Operation batching and compilation  
- Graph signature computation
- Compiled function caching
- Tensor ID normalization for stable compilation

Use this when you want automatic hot path compilation (GT_AUTO_COMPILE=1).
"""

import numpy as np
import hashlib
from typing import Optional, List, Dict, Any, Callable
from .pytorch import PyTorchEngine
from .base import Operation
from gt.debug import debug_print_compile


class PyTorchCompileEngine(PyTorchEngine):
    """PyTorch engine with torch.compile() optimization."""

    def __init__(self):
        super().__init__()

        # Compilation cache and stats
        self._compiled_cache: Dict[str, Callable] = {}
        self._cache_hits = 0
        self._cache_misses = 0

        # Hot path buffering state
        self._buffering = False
        self._current_sequence_id: Optional[str] = None
        self._buffer: List = []  # Buffer operations during hot sequence

    def supports_batching(self) -> bool:
        """This engine supports batching for compilation."""
        return True

    def is_buffering(self) -> bool:
        """Check if currently buffering a hot sequence."""
        return self._buffering

    def handle_hotpath_start(self, sequence_id: str):
        """Begin buffering operations for compilation."""
        self._buffering = True
        self._current_sequence_id = sequence_id
        self._buffer = []
        debug_print_compile(f"Engine: Started buffering hot sequence {sequence_id}")

    def handle_hotpath_end(self, sequence_id: str):
        """Compile and execute buffered operations."""
        if not self._buffering:
            debug_print_compile(f"Engine: WARNING - hotpath_end without start")
            return

        if sequence_id != self._current_sequence_id:
            debug_print_compile(f"Engine: WARNING - sequence ID mismatch")
            return

        debug_print_compile(f"Engine: Compiling {len(self._buffer)} buffered operations")

        # Operations are already in self._buffer as (Operation, tensors_dict)
        # We need to extract them and compile
        if self._buffer:
            # Extract operations and merge tensor dicts
            operations = []
            all_tensors = {}
            for op, tensors in self._buffer:
                operations.append(op)
                all_tensors.update(tensors)

            # Execute as compiled batch
            try:
                self.execute_batch(operations, all_tensors)
            except Exception as e:
                debug_print_compile(f"Compilation failed: {e}")
                # Fall back handled in execute_batch

        # Reset buffering state
        self._buffering = False
        self._current_sequence_id = None
        self._buffer = []

    def queue_operation(self, operation: Operation, tensors: Dict[str, Any]):
        """
        Queue an operation for execution (buffer if in hot sequence, execute otherwise).

        This is the main entry point for operations when using compilation.
        """
        if self._buffering:
            # Buffer the operation
            self._buffer.append((operation, tensors))
        else:
            # Execute immediately (eager mode)
            self._execute_eager(operation, tensors)

    def _execute_eager(self, operation: Operation, tensors: Dict[str, Any]):
        """Execute a single operation eagerly (not compiled)."""
        # This is essentially the same as execute_batch with 1 operation
        # but without compilation
        results = {}

        op = operation
        if op.op_type == 'binary':
            left = tensors[op.input_ids[0]]
            right = tensors[op.input_ids[1]]

            if op.op_name == 'add':
                result = left + right
            elif op.op_name == 'sub':
                result = left - right
            elif op.op_name == 'mul':
                result = left * right
            elif op.op_name == 'div':
                result = left / right
            elif op.op_name == 'matmul':
                result = self.torch.matmul(left, right)
            elif op.op_name == 'gt':
                result = (left > right).float()
            else:
                raise ValueError(f"Unknown binary op: {op.op_name}")

            tensors[op.result_id] = result

        elif op.op_type == 'unary':
            input_tensor = tensors.get(op.input_ids[0]) if op.input_ids else None

            if op.op_name == 'relu':
                result = self.torch.relu(input_tensor)
            elif op.op_name == 'sigmoid':
                result = self.torch.sigmoid(input_tensor)
            elif op.op_name == 'tanh':
                result = self.torch.tanh(input_tensor)
            elif op.op_name == 'exp':
                result = self.torch.exp(input_tensor)
            elif op.op_name == 'log':
                result = self.torch.log(input_tensor)
            elif op.op_name == 'sum':
                params = op.params or {}
                axis = params.get('axis', None)
                keepdims = params.get('keepdims', False)
                result = self.torch.sum(input_tensor, dim=axis, keepdim=keepdims)
            elif op.op_name == 'mean':
                params = op.params or {}
                axis = params.get('axis', None)
                keepdims = params.get('keepdims', False)
                result = self.torch.mean(input_tensor, dim=axis, keepdim=keepdims)
            elif op.op_name == 'sqrt':
                result = self.torch.sqrt(input_tensor)
            elif op.op_name == 'transpose':
                result = input_tensor.transpose(-2, -1)
            elif op.op_name == 'randn':
                params = op.params or {}
                shape = params.get('shape', ())
                result = self.torch.randn(shape)
            elif op.op_name == 'zeros':
                params = op.params or {}
                shape = params.get('shape', ())
                result = self.torch.zeros(shape)
            else:
                raise ValueError(f"Unknown unary op: {op.op_name}")

            tensors[op.result_id] = result

        elif op.op_type == 'reshape':
            input_tensor = tensors[op.input_ids[0]]

            if op.op_name == 'reshape':
                result = input_tensor.reshape(*op.params)
            elif op.op_name == 'unsqueeze':
                dim = op.params[0]
                result = input_tensor.unsqueeze(dim)
            elif op.op_name == 'squeeze':
                if len(op.params) == 0:
                    result = input_tensor.squeeze()
                else:
                    dim = op.params[0]
                    result = input_tensor.squeeze(dim)
            else:
                raise ValueError(f"Unknown reshape op: {op.op_name}")

            tensors[op.result_id] = result

        elif op.op_type == 'slice':
            input_tensor = tensors[op.input_ids[0]]
            result = input_tensor[op.params]
            tensors[op.result_id] = result
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

        if len(operations) < MIN_COMPILE_BATCH_SIZE:
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

    def get_compilation_stats(self) -> Dict[str, int]:
        """Get compilation cache statistics."""
        return {
            "cache_size": len(self._compiled_cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": self._cache_hits / max(1, self._cache_hits + self._cache_misses)
        }
