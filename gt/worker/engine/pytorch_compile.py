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
        self._compiled_op_counts: Dict[str, int] = {}  # Track ops per compiled function
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
        import os
        if os.environ.get('GT_DEBUG_WORKER') or os.environ.get('GT_DEBUG_COMPILE'):
            print(f"[ENGINE] Started buffering hot sequence {sequence_id}")
        debug_print_compile(f"Engine: Started buffering hot sequence {sequence_id}")

    def handle_hotpath_end(self, sequence_id: str):
        """Compile buffered operations and use cached version if available."""
        import os
        if os.environ.get('GT_DEBUG_WORKER') or os.environ.get('GT_DEBUG_COMPILE'):
            print(f"[ENGINE] Ending hot sequence {sequence_id}, buffer size: {len(self._buffer) if self._buffering else 'N/A'}")

        if not self._buffering:
            debug_print_compile(f"Engine: WARNING - hotpath_end without start")
            return

        if sequence_id != self._current_sequence_id:
            debug_print_compile(f"Engine: WARNING - sequence ID mismatch")
            return

        debug_print_compile(f"Engine: Processing {len(self._buffer)} buffered operations")

        # Operations were already executed eagerly during buffering
        # Now check if we have a cached compiled version to use
        if self._buffer:
            # Extract operations - all operations share the same tensors dict (worker's self.tensors)
            operations = []
            tensors_dict = None
            for op, tensors in self._buffer:
                operations.append(op)
                tensors_dict = tensors  # All entries point to the same dict (worker's self.tensors)

            import os
            if os.environ.get('GT_DEBUG_COMPILE'):
                print(f"[HOTPATH_END] Processing {len(operations)} operations")

            # Check if pattern exists in cache
            if len(operations) >= 3:  # Only for patterns worth compiling
                # Generate signature
                source_code, input_tensor_ids, output_tensor_ids, id_to_var = self._generate_python_source(operations, tensors_dict)
                base_signature = self._compute_graph_signature(operations)
                graph_signature = f"{base_signature}:inputs={len(input_tensor_ids)}"

                if graph_signature in self._compiled_cache:
                    # Cache HIT! Use compiled version
                    if os.environ.get('GT_DEBUG_COMPILE'):
                        print(f"[CACHE HIT] Using cached compiled function for {len(operations)} ops")

                    self._cache_hits += 1
                    try:
                        # Execute compiled version and update tensors
                        self._execute_compiled_and_update(operations, tensors_dict, graph_signature, input_tensor_ids, output_tensor_ids)
                    except Exception as e:
                        debug_print_compile(f"Compiled execution failed: {e}")
                        if os.environ.get('GT_DEBUG_COMPILE'):
                            print(f"[CACHE HIT ERROR] {e}")
                        # Eager results are already there, so we're safe
                else:
                    # Cache MISS - compile for next time
                    if os.environ.get('GT_DEBUG_COMPILE'):
                        print(f"[CACHE MISS] Compiling {len(operations)} operations for future use")

                    try:
                        self._compile_and_cache(operations, tensors_dict)
                    except Exception as e:
                        debug_print_compile(f"Compilation failed: {e}")
                        # Not a big deal - we already executed eagerly

        # Reset buffering state
        self._buffering = False
        self._current_sequence_id = None
        self._buffer = []

    def queue_operation(self, operation: Operation, tensors: Dict[str, Any]):
        """
        Queue an operation for execution (buffer if in hot sequence, execute otherwise).

        This is the main entry point for operations when using compilation.
        """
        # ALWAYS execute eagerly to create tensors immediately
        # This prevents "tensor not found" errors in backward passes
        self._execute_eager(operation, tensors)

        # If buffering, also record for compilation
        if self._buffering:
            self._buffer.append((operation, tensors))

    def _execute_compiled_and_update(self, operations: List[Operation], tensors: Dict[str, Any],
                                      graph_signature: str, input_tensor_ids: List[str],
                                      output_tensor_ids: List[str]):
        """
        Execute cached compiled function and update tensors dict.

        This overwrites the eagerly-computed results with compiled results.
        Called when we have a cache hit.
        """
        import os

        compiled_fn = self._compiled_cache[graph_signature]

        # Prepare inputs
        input_list = [tensors[tid] for tid in input_tensor_ids]

        # Execute compiled function
        result = compiled_fn(*input_list)

        # Handle single vs multiple return values
        if len(output_tensor_ids) == 1:
            result_tensors = [result]
        else:
            result_tensors = result

        # Update tensors dict with compiled results (overwrite eager results)
        if os.environ.get('GT_DEBUG_COMPILE'):
            print(f"[COMPILED EXEC] Updating {len(output_tensor_ids)} tensors with compiled results")

        for tensor_id, tensor_value in zip(output_tensor_ids, result_tensors):
            tensors[tensor_id] = tensor_value
            if os.environ.get('GT_DEBUG_COMPILE'):
                shape_str = str(tensor_value.shape) if hasattr(tensor_value, 'shape') else 'scalar'
                print(f"  [UPDATE] {tensor_id}: {shape_str}")

    def _compile_and_cache(self, operations: List[Operation], tensors: Dict[str, Any]):
        """
        Compile operations and cache the compiled function (without executing).

        This is called after operations have already been executed eagerly.
        We just want to compile them for future iterations.
        """
        # Minimum batch size for compilation to be worthwhile
        MIN_COMPILE_BATCH_SIZE = 3

        if len(operations) < MIN_COMPILE_BATCH_SIZE:
            import os
            if os.environ.get('GT_DEBUG_COMPILE'):
                print(f"[SKIP COMPILE] Batch too small ({len(operations)} ops < {MIN_COMPILE_BATCH_SIZE})")
            return

        # Generate Python source code
        source_code, input_tensor_ids, output_tensor_ids, id_to_var = self._generate_python_source(operations, tensors)

        # Compute signature for caching
        base_signature = self._compute_graph_signature(operations)
        graph_signature = f"{base_signature}:inputs={len(input_tensor_ids)}"

        import os

        # Skip if already cached
        if graph_signature in self._compiled_cache:
            if os.environ.get('GT_DEBUG_COMPILE'):
                print(f"[ALREADY CACHED] Skipping compilation")
            return

        if os.environ.get('GT_DEBUG_COMPILE'):
            print(f"[TORCH.COMPILE] Compiling {len(operations)} operations...")
            print(f"[GENERATED CODE]:\n{source_code}\n")

        # Save generated code for debugging
        if os.environ.get('GT_SAVE_COMPILED_CODE'):
            import hashlib
            code_hash = hashlib.md5(source_code.encode()).hexdigest()[:8]
            filename = f"/tmp/gt_compiled_{code_hash}.py"
            with open(filename, 'w') as f:
                f.write(source_code)
            if os.environ.get('GT_DEBUG_COMPILE'):
                print(f"[SAVED CODE] {filename}")

        # Create function from source using exec()
        try:
            namespace = {}
            exec(source_code, namespace)
            graph_fn = namespace['compiled_function']

            # Actually run torch.compile - this can take a few seconds!
            import time
            compile_start = time.time()
            compiled_fn = self.torch.compile(graph_fn, mode="default")
            compile_time = time.time() - compile_start

            if os.environ.get('GT_DEBUG_COMPILE'):
                print(f"[TORCH.COMPILE] ✅ Compiled in {compile_time:.3f}s! Cache size now: {len(self._compiled_cache) + 1}")

            self._compiled_cache[graph_signature] = compiled_fn
            self._compiled_op_counts[graph_signature] = len(operations)
            self._cache_misses += 1
            debug_print_compile(f"Compiled new graph: {graph_signature} ({len(operations)} ops)")
        except Exception as e:
            debug_print_compile(f"Warning: Compilation failed ({e})")
            if os.environ.get('GT_DEBUG_COMPILE'):
                print(f"[COMPILE ERROR] {e}")

    def _execute_eager(self, operation: Operation, tensors: Dict[str, Any]):
        """Execute a single operation eagerly (not compiled)."""
        # This is essentially the same as execute_batch with 1 operation
        # but without compilation
        results = {}

        import os
        if os.environ.get('GT_DEBUG_COMPILE'):
            print(f"[EAGER] Executing {operation.op_type}:{operation.op_name} -> {operation.result_id}")
            print(f"[EAGER] Tensors dict id: {id(tensors)}, size: {len(tensors)}")

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
            elif op.op_name == 'ones':
                params = op.params or {}
                shape = params.get('shape', ())
                result = self.torch.ones(shape)
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
                    elif instr['op_name'] == 'ones':
                        params = instr['params'] or {}
                        shape = params.get('shape', ())
                        result = self.torch.ones(shape)
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

    def _generate_python_source(self, operations: List[Operation], tensors: Dict[str, Any]) -> tuple:
        """
        Generate readable Python source code from operations.

        Returns:
            (source_code, input_tensor_ids, output_tensor_ids, id_to_var)
        """
        # Identify which tensors are inputs (already exist) vs outputs (will be created)
        all_result_ids = set(op.result_id for op in operations)
        input_tensor_ids = []

        for op in operations:
            for input_id in op.input_ids:
                if input_id not in all_result_ids and input_id not in input_tensor_ids:
                    input_tensor_ids.append(input_id)

        # Map tensor IDs to clean variable names
        id_to_var = {}
        for i, tid in enumerate(input_tensor_ids):
            id_to_var[tid] = f"t{i}"

        # Generate code lines
        lines = []
        lines.append("def compiled_function(" + ", ".join(f"t{i}" for i in range(len(input_tensor_ids))) + "):")
        lines.append("    # Auto-generated from GT operations")
        lines.append("    import torch")
        lines.append("")

        # Generate operation code
        next_var_idx = len(input_tensor_ids)
        for op in operations:
            result_var = f"t{next_var_idx}"
            id_to_var[op.result_id] = result_var
            next_var_idx += 1

            if op.op_type == 'binary':
                left_var = id_to_var[op.input_ids[0]]
                right_var = id_to_var[op.input_ids[1]]

                if op.op_name == 'add':
                    lines.append(f"    {result_var} = {left_var} + {right_var}")
                elif op.op_name == 'sub':
                    lines.append(f"    {result_var} = {left_var} - {right_var}")
                elif op.op_name == 'mul':
                    lines.append(f"    {result_var} = {left_var} * {right_var}")
                elif op.op_name == 'div':
                    lines.append(f"    {result_var} = {left_var} / {right_var}")
                elif op.op_name == 'matmul':
                    lines.append(f"    {result_var} = torch.matmul({left_var}, {right_var})")
                elif op.op_name == 'gt':
                    lines.append(f"    {result_var} = ({left_var} > {right_var}).float()")
                else:
                    raise ValueError(f"Unknown binary op: {op.op_name}")

            elif op.op_type == 'unary':
                input_var = id_to_var[op.input_ids[0]] if op.input_ids else None

                if op.op_name == 'relu':
                    lines.append(f"    {result_var} = torch.relu({input_var})")
                elif op.op_name == 'sigmoid':
                    lines.append(f"    {result_var} = torch.sigmoid({input_var})")
                elif op.op_name == 'tanh':
                    lines.append(f"    {result_var} = torch.tanh({input_var})")
                elif op.op_name == 'exp':
                    lines.append(f"    {result_var} = torch.exp({input_var})")
                elif op.op_name == 'log':
                    lines.append(f"    {result_var} = torch.log({input_var})")
                elif op.op_name == 'sum':
                    params = op.params or {}
                    axis = params.get('axis', None)
                    keepdims = params.get('keepdims', False)
                    if axis is None:
                        lines.append(f"    {result_var} = torch.sum({input_var})")
                    else:
                        lines.append(f"    {result_var} = torch.sum({input_var}, dim={axis}, keepdim={keepdims})")
                elif op.op_name == 'mean':
                    params = op.params or {}
                    axis = params.get('axis', None)
                    keepdims = params.get('keepdims', False)
                    if axis is None:
                        lines.append(f"    {result_var} = torch.mean({input_var})")
                    else:
                        lines.append(f"    {result_var} = torch.mean({input_var}, dim={axis}, keepdim={keepdims})")
                elif op.op_name == 'sqrt':
                    lines.append(f"    {result_var} = torch.sqrt({input_var})")
                elif op.op_name == 'transpose':
                    lines.append(f"    {result_var} = {input_var}.transpose(-2, -1)")
                else:
                    raise ValueError(f"Unknown unary op: {op.op_name}")

            elif op.op_type == 'reshape':
                input_var = id_to_var[op.input_ids[0]]
                if op.op_name == 'reshape':
                    shape_str = ", ".join(str(s) for s in op.params)
                    lines.append(f"    {result_var} = {input_var}.reshape({shape_str})")
                elif op.op_name == 'unsqueeze':
                    dim = op.params[0]
                    lines.append(f"    {result_var} = {input_var}.unsqueeze({dim})")
                elif op.op_name == 'squeeze':
                    if len(op.params) == 0:
                        lines.append(f"    {result_var} = {input_var}.squeeze()")
                    else:
                        dim = op.params[0]
                        lines.append(f"    {result_var} = {input_var}.squeeze({dim})")
                else:
                    raise ValueError(f"Unknown reshape op: {op.op_name}")

            elif op.op_type == 'slice':
                input_var = id_to_var[op.input_ids[0]]
                # op.params contains the slice key
                lines.append(f"    {result_var} = {input_var}[{op.params}]")

        # Determine output tensors (all created tensors that still exist in tensors dict)
        output_tensor_ids = [op.result_id for op in operations]

        # Return all output tensors
        output_vars = [id_to_var[tid] for tid in output_tensor_ids]
        if len(output_vars) == 1:
            lines.append(f"    return {output_vars[0]}")
        else:
            lines.append(f"    return ({', '.join(output_vars)},)")

        source_code = "\n".join(lines)
        return source_code, input_tensor_ids, output_tensor_ids, id_to_var

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
            import os
            if os.environ.get('GT_DEBUG_COMPILE'):
                print(f"[EAGER] Batch too small ({len(operations)} ops < {MIN_COMPILE_BATCH_SIZE}), using eager execution")
            for op in operations:
                self._execute_eager(op, tensors)
            return tensors

        # Generate Python source code
        source_code, input_tensor_ids, output_tensor_ids, id_to_var = self._generate_python_source(operations, tensors)

        # Compute signature for caching
        base_signature = self._compute_graph_signature(operations)
        graph_signature = f"{base_signature}:inputs={len(input_tensor_ids)}"

        import os

        # Check if we have a cached compiled function
        if graph_signature in self._compiled_cache:
            if os.environ.get('GT_DEBUG_COMPILE'):
                print(f"[CACHE HIT] Using cached compiled function for {len(operations)} ops")
            compiled_fn = self._compiled_cache[graph_signature]
            self._cache_hits += 1
        else:
            if os.environ.get('GT_DEBUG_COMPILE'):
                print(f"[TORCH.COMPILE] Compiling {len(operations)} operations...")
                print(f"[GENERATED CODE]:\n{source_code}\n")

            # Save generated code for debugging
            if os.environ.get('GT_SAVE_COMPILED_CODE'):
                import hashlib
                code_hash = hashlib.md5(source_code.encode()).hexdigest()[:8]
                filename = f"/tmp/gt_compiled_{code_hash}.py"
                with open(filename, 'w') as f:
                    f.write(source_code)
                if os.environ.get('GT_DEBUG_COMPILE'):
                    print(f"[SAVED CODE] {filename}")

            # Create function from source using exec()
            try:
                namespace = {}
                exec(source_code, namespace)
                graph_fn = namespace['compiled_function']

                # Actually run torch.compile - this can take a few seconds!
                import time
                compile_start = time.time()
                compiled_fn = self.torch.compile(graph_fn, mode="default")
                compile_time = time.time() - compile_start

                if os.environ.get('GT_DEBUG_COMPILE'):
                    print(f"[TORCH.COMPILE] ✅ Compiled in {compile_time:.3f}s! Cache size now: {len(self._compiled_cache) + 1}")

                self._compiled_cache[graph_signature] = compiled_fn
                self._compiled_op_counts[graph_signature] = len(operations)
                self._cache_misses += 1
                debug_print_compile(f"Compiled new graph: {graph_signature} ({len(operations)} ops)")
            except Exception as e:
                # Fall back to eager if compilation fails
                debug_print_compile(f"Warning: Compilation failed ({e}), falling back to eager execution")
                if os.environ.get('GT_DEBUG_COMPILE'):
                    print(f"[COMPILE ERROR] {e}")
                for op in operations:
                    self._execute_eager(op, tensors)
                return tensors

        # Prepare input tensors in the correct order
        input_list = [tensors[tid] for tid in input_tensor_ids]

        # Execute compiled function
        try:
            result = compiled_fn(*input_list)

            # Handle single vs multiple return values
            if len(output_tensor_ids) == 1:
                result_tensors = [result]
            else:
                result_tensors = result

            # Map results back to tensor IDs and add to tensors dict
            import os
            if os.environ.get('GT_DEBUG_COMPILE'):
                print(f"[EXECUTE] Adding {len(output_tensor_ids)} tensors: {output_tensor_ids}")

            for tensor_id, tensor_value in zip(output_tensor_ids, result_tensors):
                tensors[tensor_id] = tensor_value
                if os.environ.get('GT_DEBUG_COMPILE'):
                    print(f"  -> {tensor_id}: shape={tensor_value.shape if hasattr(tensor_value, 'shape') else 'N/A'}")

        except Exception as e:
            debug_print_compile(f"Execution error: {e}")
            if os.environ.get('GT_DEBUG_COMPILE'):
                print(f"[EXECUTION ERROR] {e}")
            # Fall back to eager
            for op in operations:
                self._execute_eager(op, tensors)

        return tensors

    def get_compilation_stats(self) -> Dict[str, int]:
        """Get compilation cache statistics."""
        # Calculate average ops per compiled function
        if self._compiled_op_counts:
            avg_ops = sum(self._compiled_op_counts.values()) / len(self._compiled_op_counts)
            min_ops = min(self._compiled_op_counts.values())
            max_ops = max(self._compiled_op_counts.values())
        else:
            avg_ops = min_ops = max_ops = 0

        return {
            "cache_size": len(self._compiled_cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": self._cache_hits / max(1, self._cache_hits + self._cache_misses),
            "avg_ops_per_compilation": avg_ops,
            "min_ops_per_compilation": min_ops,
            "max_ops_per_compilation": max_ops,
        }
