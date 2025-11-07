"""
Signal-driven sharding stream modifier.

This is a PURE STREAM TRANSFORMER:
- Takes instruction stream (commands from clients) as input
- Yields instruction stream as output (with shard operations injected)
- Does NOT buffer or execute anything
- Waits for user signals to know when/how to shard

Design Philosophy:
- DUMB: Only shards when user explicitly signals (default behavior)
- Can enable automatic sharding with GT_AUTO_SHARD=1
- When auto-sharding: distributes tensors across all workers automatically
- Just reads YAML config and follows instructions for signal-based sharding
- Keeps dispatcher simple (just executes commands)

Example (signal-based):
  Input stream (with signals):
    CompileStart(signal='pp_stage0')
    CreateTensor(id=1, shape=(128, 64))   # User wants sharding here
    CompileEnd(signal='pp_stage0')

  Config says: pp_stage0 uses workers [0, 1], axis 0

  Output stream:
    CompileStart(signal='pp_stage0')
    CreateTensor(id=1_shard_0, shape=(64, 64)) on worker 0
    CreateTensor(id=1_shard_1, shape=(64, 64)) on worker 1
    CompileEnd(signal='pp_stage0')

Example (auto-sharding with GT_AUTO_SHARD=1):
  Input stream:
    CreateTensor(id=1, shape=(128, 64))   # Automatically sharded

  Output stream (with 4 workers):
    CreateTensor(id=1_shard_0, shape=(32, 64)) on worker 0
    CreateTensor(id=1_shard_1, shape=(32, 64)) on worker 1
    CreateTensor(id=1_shard_2, shape=(32, 64)) on worker 2
    CreateTensor(id=1_shard_3, shape=(32, 64)) on worker 3
"""

import os
from typing import Iterator, Optional, Dict, List, Callable
from gt.debug import debug_print_dispatcher
from gt.transport.protocol import (
    ClientCommand, BinaryOp, UnaryOp, CreateTensor, GetData, CompileStart, CompileEnd
)


class ShardingStreamModifier:
    """
    Signal-driven stream transformer that injects sharding operations.

    Supports two modes:
    1. Signal-based (default): Only shards inside CompileStart/CompileEnd with config
    2. Auto-sharding (GT_AUTO_SHARD=1): Automatically shards all tensors across workers

    Usage:
        modifier = ShardingStreamModifier(get_workers=lambda: workers)
        for cmd in command_stream:
            for output_cmd in modifier.process(cmd):
                yield output_cmd
    """

    def __init__(self, get_workers: Callable[[], List] = None):
        """
        Initialize sharding stream modifier.

        Args:
            get_workers: Callable that returns list of available workers.
                         Each worker should be a dict with 'id' field.
        """
        # Track active signal scope
        self.active_signal: Optional[str] = None

        # Worker access
        self.get_workers = get_workers if get_workers else lambda: []

        # Auto-sharding configuration
        self.auto_shard_enabled = os.environ.get('GT_AUTO_SHARD', '0') == '1'
        self.auto_shard_axis = 0  # Default to sharding along axis 0

        if self.auto_shard_enabled:
            debug_print_dispatcher("[SHARDING] Auto-sharding enabled (GT_AUTO_SHARD=1)")

        # Tensor location tracking
        # Maps tensor_id -> list of (worker_id, shard_info, shape, dtype)
        # If len > 1, tensor is sharded across workers
        # shard_info is dict with {axis, shard_index, num_shards} or None if not sharded
        self.tensor_locations: Dict[int, List[dict]] = {}

        # Stats
        self.total_commands = 0
        self.sharded_commands = 0

    def process(self, cmd: ClientCommand) -> Iterator[ClientCommand]:
        """
        Process a command and yield output commands.

        If we're in a signal scope with sharding config, transform commands.
        If auto-sharding is enabled and no signal scope, automatically shard.
        Otherwise, pass through unchanged.

        Yields:
            Original command or transformed commands with sharding
        """
        self.total_commands += 1

        # Track signal scope
        if isinstance(cmd, CompileStart):
            self.active_signal = cmd.signal_name
            debug_print_dispatcher(f"[SHARDING] Entered signal scope: {self.active_signal}")
            yield cmd
            return

        if isinstance(cmd, CompileEnd):
            debug_print_dispatcher(f"[SHARDING] Exited signal scope: {self.active_signal}")
            self.active_signal = None
            yield cmd
            return

        # Handle signal-based sharding (takes priority)
        if self.active_signal is not None:
            # Get signal config
            from gt.config import get_signal_config
            signal_config = get_signal_config(self.active_signal)

            if signal_config is not None and signal_config.shard is not None:
                shard_config = signal_config.shard

                # For now, only handle CreateTensor and UnaryOp (for randn/zeros)
                if isinstance(cmd, CreateTensor):
                    yield from self._shard_create_tensor(cmd, shard_config)
                    self.sharded_commands += 1
                    return
                elif isinstance(cmd, UnaryOp) and cmd.input_id is None:
                    # randn, zeros - tensor creation ops
                    yield from self._shard_unary_creation(cmd, shard_config)
                    self.sharded_commands += 1
                    return

            # No sharding config for this signal, pass through
            yield cmd
            return

        # Auto-sharding mode (when not in signal scope)
        if self.auto_shard_enabled and self.active_signal is None:
            if isinstance(cmd, CreateTensor):
                yield from self._auto_shard_create_tensor(cmd)
                return
            elif isinstance(cmd, UnaryOp) and cmd.input_id is None:
                # randn, zeros - tensor creation ops
                yield from self._auto_shard_unary_creation(cmd)
                return

        # Default: pass through but assign worker if it's a creation op
        if isinstance(cmd, CreateTensor):
            # Assign to worker 0 if not already assigned
            if cmd.worker_id is None:
                workers = self.get_workers()
                if workers and len(workers) > 0:
                    cmd.worker_id = workers[0]['id']
                    self._register_tensor_location(cmd.tensor_id, cmd.worker_id, cmd.shape, cmd.dtype)
            yield cmd
            return
        elif isinstance(cmd, UnaryOp) and cmd.input_id is None:
            # Creation op - assign worker
            if cmd.worker_id is None:
                workers = self.get_workers()
                if workers and len(workers) > 0:
                    cmd.worker_id = workers[0]['id']
                    self._register_tensor_location(cmd.result_id, cmd.worker_id, cmd.shape, cmd.dtype)
            yield cmd
            return
        elif isinstance(cmd, BinaryOp):
            # Handle binary operations - need to handle cross-worker and sharded cases
            yield from self._handle_binary_op(cmd)
            return
        elif isinstance(cmd, UnaryOp) and cmd.input_id is not None:
            # UnaryOp with input (transpose, sqrt, relu, etc.)
            yield from self._handle_unary_op(cmd)
            return
        elif isinstance(cmd, GetData):
            # Let dispatcher handle GetData - it has the complete TensorHandle tracking
            # Modifier only handles worker assignment for creation ops, not lookup
            yield cmd
            return

        # Other commands pass through unchanged
        yield cmd

    def _register_tensor_location(self, tensor_id: int, worker_id, shape, dtype, shard_info=None):
        """
        Register where a tensor is located.

        Args:
            tensor_id: Tensor ID
            worker_id: Worker ID where tensor is located
            shape: Tensor shape
            dtype: Tensor dtype
            shard_info: Optional dict with {axis, shard_index, num_shards}
        """
        if tensor_id not in self.tensor_locations:
            self.tensor_locations[tensor_id] = []

        self.tensor_locations[tensor_id].append({
            'worker_id': worker_id,
            'shape': shape,
            'dtype': dtype,
            'shard_info': shard_info
        })

    def _get_tensor_locations(self, tensor_id: int) -> List[dict]:
        """Get all locations where a tensor exists."""
        return self.tensor_locations.get(tensor_id, [])

    def _handle_binary_op(self, cmd: BinaryOp) -> Iterator[ClientCommand]:
        """
        Handle binary operation with cross-worker and sharding logic.

        Injects GatherShards commands when operands are sharded/replicated or on different workers.
        """
        from gt.transport.protocol import GatherShards
        import time

        # Get locations
        left_locs = self._get_tensor_locations(cmd.left_id)
        right_locs = self._get_tensor_locations(cmd.right_id)

        if not left_locs or not right_locs:
            # Tensor not found in tracking - pass through and let dispatcher handle error
            debug_print_dispatcher(f"[BINARY_OP] Tensor {cmd.left_id} or {cmd.right_id} not found in tracking")
            yield cmd
            return

        # Check if sharded/replicated
        left_multi = len(left_locs) > 1
        right_multi = len(right_locs) > 1

        # Case 1: Both on same single worker - simple case
        if not left_multi and not right_multi:
            left_loc = left_locs[0]
            right_loc = right_locs[0]

            if left_loc['worker_id'] == right_loc['worker_id']:
                # Same worker - simple case
                cmd.worker_id = left_loc['worker_id']
                yield cmd
                # Register result location
                self._register_tensor_location(
                    cmd.result_id,
                    left_loc['worker_id'],
                    left_loc['shape'],  # Approximation
                    left_loc['dtype']
                )
                debug_print_dispatcher(f"[BINARY_OP] Same-worker {cmd.op}, registered result {cmd.result_id} on worker {left_loc['worker_id']}")
                return

        # Case 2: Need to gather one or both operands
        # Choose a target worker (prefer left's first worker, or any worker)
        workers = self.get_workers()
        if not workers or len(workers) == 0:
            debug_print_dispatcher(f"[BINARY_OP] No workers available")
            yield cmd
            return

        # Pick target worker (use left's first location if available, else first worker)
        if left_locs:
            target_worker = left_locs[0]['worker_id']
        else:
            target_worker = workers[0]['id']

        debug_print_dispatcher(f"[BINARY_OP] {cmd.op} with left={len(left_locs)} locs, right={len(right_locs)} locs - gathering to worker {target_worker}")

        # Generate temporary IDs for gathered tensors
        left_gathered_id = cmd.left_id
        right_gathered_id = cmd.right_id

        # Gather left if needed
        if left_multi:
            left_gathered_id = cmd.left_id + 1000000 + int(time.time() * 1000000) % 1000000
            debug_print_dispatcher(f"[BINARY_OP] Injecting GatherShards for left: {cmd.left_id} → {left_gathered_id}")
            gather_left = GatherShards(
                result_id=left_gathered_id,
                source_tensor_id=cmd.left_id,
                target_worker_id=target_worker
            )
            yield gather_left
        elif left_locs[0]['worker_id'] != target_worker:
            # Single location but wrong worker - need to gather/copy to target
            left_gathered_id = cmd.left_id + 1000000 + int(time.time() * 1000000) % 1000000
            debug_print_dispatcher(f"[BINARY_OP] Injecting GatherShards for left (cross-worker): {cmd.left_id} → {left_gathered_id}")
            gather_left = GatherShards(
                result_id=left_gathered_id,
                source_tensor_id=cmd.left_id,
                target_worker_id=target_worker
            )
            yield gather_left

        # Gather right if needed
        if right_multi:
            right_gathered_id = cmd.right_id + 2000000 + int(time.time() * 1000000) % 1000000
            debug_print_dispatcher(f"[BINARY_OP] Injecting GatherShards for right: {cmd.right_id} → {right_gathered_id}")
            gather_right = GatherShards(
                result_id=right_gathered_id,
                source_tensor_id=cmd.right_id,
                target_worker_id=target_worker
            )
            yield gather_right
        elif right_locs[0]['worker_id'] != target_worker:
            # Single location but wrong worker - need to gather/copy to target
            right_gathered_id = cmd.right_id + 2000000 + int(time.time() * 1000000) % 1000000
            debug_print_dispatcher(f"[BINARY_OP] Injecting GatherShards for right (cross-worker): {cmd.right_id} → {right_gathered_id}")
            gather_right = GatherShards(
                result_id=right_gathered_id,
                source_tensor_id=cmd.right_id,
                target_worker_id=target_worker
            )
            yield gather_right

        # Inject modified BinaryOp with gathered inputs
        modified_cmd = BinaryOp(
            result_id=cmd.result_id,
            op=cmd.op,
            left_id=left_gathered_id,
            right_id=right_gathered_id
        )
        # Set worker_id to ensure dispatcher routes it to the right worker
        modified_cmd.worker_id = target_worker
        yield modified_cmd

        # Register result location
        # Use left's dtype/shape as approximation (dispatcher will compute actual)
        first_left = left_locs[0]
        self._register_tensor_location(
            cmd.result_id,
            target_worker,
            first_left['shape'],  # Approximation
            first_left['dtype']
        )
        debug_print_dispatcher(f"[BINARY_OP] Registered result {cmd.result_id} on worker {target_worker}")

    def _handle_unary_op(self, cmd: UnaryOp) -> Iterator[ClientCommand]:
        """
        Handle unary operation with input.

        Assigns worker and tells dispatcher where to gather sharded tensors.
        """
        input_locs = self._get_tensor_locations(cmd.input_id)

        if not input_locs:
            # Tensor not found - pass through
            debug_print_dispatcher(f"[UNARY_OP] Input tensor {cmd.input_id} not found in tracking")
            yield cmd
            return

        # If sharded, inject GatherShards command followed by UnaryOp on gathered tensor
        if len(input_locs) > 1:
            # Choose first worker for gathering (could use any strategy here)
            workers = self.get_workers()
            if workers and len(workers) > 0:
                target_worker = workers[0]['id']

                # Generate temporary tensor ID for gathered result
                # Use a large number to avoid conflicts with user tensor IDs
                import time
                gathered_tensor_id = cmd.input_id + 1000000 + int(time.time() * 1000000) % 1000000

                debug_print_dispatcher(f"[UNARY_OP] Sharded {cmd.op} - injecting GatherShards({cmd.input_id} → {gathered_tensor_id}) on worker {target_worker}")

                # Inject GatherShards command
                from gt.transport.protocol import GatherShards
                gather_cmd = GatherShards(
                    result_id=gathered_tensor_id,
                    source_tensor_id=cmd.input_id,
                    target_worker_id=target_worker
                )
                yield gather_cmd

                # Inject modified UnaryOp that operates on the gathered tensor
                modified_cmd = UnaryOp(
                    result_id=cmd.result_id,
                    op=cmd.op,
                    input_id=gathered_tensor_id,
                    shape=cmd.shape,
                    dtype=cmd.dtype,
                    axis=getattr(cmd, 'axis', None),
                    keepdims=getattr(cmd, 'keepdims', False),
                    worker_id=target_worker
                )
                yield modified_cmd

                # Register result location (dispatcher will create it, but we track it)
                # Use the first input shard's shape/dtype as approximation
                first_loc = input_locs[0]
                self._register_tensor_location(
                    cmd.result_id,
                    target_worker,
                    first_loc['shape'],  # Approximation (dispatcher will compute actual)
                    first_loc['dtype']
                )
                debug_print_dispatcher(f"[UNARY_OP] Registered result tensor {cmd.result_id} on worker {target_worker}")
                return
            else:
                debug_print_dispatcher(f"[UNARY_OP] No workers available for sharded {cmd.op}")
                yield cmd
                return

        # Single location - assign to same worker as input
        input_loc = input_locs[0]
        cmd.worker_id = input_loc['worker_id']
        yield cmd

        # Register result location (same worker as input)
        self._register_tensor_location(
            cmd.result_id,
            input_loc['worker_id'],
            input_loc['shape'],  # Approximation
            input_loc['dtype']
        )
        debug_print_dispatcher(f"[UNARY_OP] Registered result tensor {cmd.result_id} on worker {input_loc['worker_id']}")

    def _shard_create_tensor(self, cmd: CreateTensor, shard_config) -> Iterator[ClientCommand]:
        """
        Shard a CreateTensor command across workers.

        Creates multiple tensor shards according to config.
        """
        workers = shard_config.workers
        axis = shard_config.axis
        replicated = shard_config.replicated

        # If no workers specified, can't shard
        if workers is None or len(workers) == 0:
            debug_print_dispatcher(f"[SHARDING] No workers specified for signal, passing through")
            yield cmd
            return

        # If only one worker, no sharding needed
        if len(workers) == 1:
            # Modify command to specify worker
            cmd.worker_id = workers[0]
            yield cmd
            return

        # Check if tensor can be sharded
        if cmd.shape is None or len(cmd.shape) <= axis:
            debug_print_dispatcher(f"[SHARDING] Cannot shard shape {cmd.shape} along axis {axis}")
            yield cmd
            return

        # Replicated: send full tensor to all workers
        if replicated:
            debug_print_dispatcher(f"[SHARDING] Replicating tensor {cmd.tensor_id} across {len(workers)} workers")
            for worker_id in workers:
                shard_cmd = CreateTensor(
                    tensor_id=cmd.tensor_id,  # Same ID, different workers
                    data=cmd.data,
                    dtype=cmd.dtype,
                    shape=cmd.shape,
                    worker_id=worker_id  # Specify target worker
                )
                yield shard_cmd
                # Register location
                self._register_tensor_location(cmd.tensor_id, worker_id, cmd.shape, cmd.dtype, shard_info=None)
            return

        # Sharded: split tensor along axis
        num_shards = len(workers)
        dim_size = cmd.shape[axis]

        # Check if evenly divisible
        if dim_size % num_shards != 0:
            error_msg = f"Cannot shard axis {axis} of size {dim_size} across {num_shards} workers evenly"
            debug_print_dispatcher(f"[SHARDING] {error_msg}")
            # Return error by yielding original command with error attached
            # Actually, we can't attach errors to commands, so just pass through
            # The dispatcher will handle it
            yield cmd
            return

        shard_size = dim_size // num_shards

        debug_print_dispatcher(f"[SHARDING] Sharding tensor {cmd.tensor_id} shape {cmd.shape} "
                             f"across {num_shards} workers along axis {axis}")

        # Create shard for each worker
        import numpy as np
        for shard_idx, worker_id in enumerate(workers):
            # Compute shard shape
            shard_shape = list(cmd.shape)
            shard_shape[axis] = shard_size

            # Slice data for this shard
            start = shard_idx * shard_size
            end = start + shard_size

            if cmd.data is not None:
                # Slice along the sharding axis
                slices = [slice(None)] * len(cmd.shape)
                slices[axis] = slice(start, end)
                shard_data = cmd.data[tuple(slices)].copy()
            else:
                shard_data = None

            shard_cmd = CreateTensor(
                tensor_id=cmd.tensor_id,  # Same ID, sharded across workers
                data=shard_data,
                dtype=cmd.dtype,
                shape=tuple(shard_shape),
                worker_id=worker_id,  # Target specific worker
                shard_info={
                    'axis': axis,
                    'shard_index': shard_idx,
                    'num_shards': num_shards
                }
            )
            yield shard_cmd
            # Register location
            self._register_tensor_location(
                cmd.tensor_id,
                worker_id,
                tuple(shard_shape),
                cmd.dtype,
                shard_info={'axis': axis, 'shard_index': shard_idx, 'num_shards': num_shards}
            )

    def _shard_unary_creation(self, cmd: UnaryOp, shard_config) -> Iterator[ClientCommand]:
        """
        Shard a unary creation op (randn, zeros) across workers.

        Similar to _shard_create_tensor but for ops without data.
        """
        workers = shard_config.workers
        axis = shard_config.axis
        replicated = shard_config.replicated

        if workers is None or len(workers) == 0:
            yield cmd
            return

        if len(workers) == 1:
            cmd.worker_id = workers[0]
            yield cmd
            # Register single-worker tensor location
            self._register_tensor_location(cmd.result_id, workers[0], cmd.shape, cmd.dtype)
            return

        if cmd.shape is None or len(cmd.shape) <= axis:
            yield cmd
            return

        # Replicated: full tensor on all workers
        if replicated:
            debug_print_dispatcher(f"[SHARDING] Replicating {cmd.op} tensor {cmd.result_id} across {len(workers)} workers")
            for worker_id in workers:
                shard_cmd = UnaryOp(
                    result_id=cmd.result_id,
                    op=cmd.op,
                    input_id=cmd.input_id,
                    shape=cmd.shape,
                    dtype=cmd.dtype,
                    axis=getattr(cmd, 'axis', None),
                    keepdims=getattr(cmd, 'keepdims', False),
                    worker_id=worker_id
                )
                yield shard_cmd
                # Register replicated tensor location
                self._register_tensor_location(cmd.result_id, worker_id, cmd.shape, cmd.dtype)
            return

        # Sharded: split along axis
        num_shards = len(workers)
        dim_size = cmd.shape[axis]

        if dim_size % num_shards != 0:
            debug_print_dispatcher(f"[SHARDING] Cannot shard axis {axis} of size {dim_size} "
                                 f"across {num_shards} workers evenly")
            yield cmd
            return

        shard_size = dim_size // num_shards

        debug_print_dispatcher(f"[SHARDING] Sharding {cmd.op} tensor {cmd.result_id} shape {cmd.shape} "
                             f"across {num_shards} workers along axis {axis}")

        for shard_idx, worker_id in enumerate(workers):
            shard_shape = list(cmd.shape)
            shard_shape[axis] = shard_size

            shard_cmd = UnaryOp(
                result_id=cmd.result_id,
                op=cmd.op,
                input_id=cmd.input_id,
                shape=tuple(shard_shape),
                dtype=cmd.dtype,
                axis=getattr(cmd, 'axis', None),
                keepdims=getattr(cmd, 'keepdims', False),
                worker_id=worker_id,
                shard_info={
                    'axis': axis,
                    'shard_index': shard_idx,
                    'num_shards': num_shards
                }
            )
            yield shard_cmd
            # Register shard location
            self._register_tensor_location(
                cmd.result_id,
                worker_id,
                tuple(shard_shape),
                cmd.dtype,
                shard_info={'axis': axis, 'shard_index': shard_idx, 'num_shards': num_shards}
            )

    def get_stats(self) -> dict:
        """Get sharding statistics."""
        return {
            'total_commands': self.total_commands,
            'sharded_commands': self.sharded_commands,
            'active_signal': self.active_signal,
            'auto_shard_enabled': self.auto_shard_enabled,
        }

    def _auto_shard_create_tensor(self, cmd: CreateTensor) -> Iterator[ClientCommand]:
        """
        Automatically shard a CreateTensor command across all available workers.

        Shards along axis 0 by default. Falls back to replication if sharding not possible.
        """
        workers = self.get_workers()

        if not workers or len(workers) == 0:
            debug_print_dispatcher(f"[AUTO-SHARD] No workers available, passing through")
            yield cmd
            return

        if len(workers) == 1:
            # Only one worker, just assign to it
            cmd.worker_id = workers[0]['id']
            yield cmd
            return

        # Check if tensor can be sharded along axis 0
        if cmd.shape is None or len(cmd.shape) == 0:
            debug_print_dispatcher(f"[AUTO-SHARD] Cannot shard scalar, passing through")
            yield cmd
            return

        axis = self.auto_shard_axis
        if len(cmd.shape) <= axis:
            debug_print_dispatcher(f"[AUTO-SHARD] Cannot shard shape {cmd.shape} along axis {axis}, passing through")
            yield cmd
            return

        num_workers = len(workers)
        dim_size = cmd.shape[axis]

        # Check if evenly divisible
        if dim_size % num_workers != 0:
            debug_print_dispatcher(f"[AUTO-SHARD] Cannot evenly shard axis {axis} of size {dim_size} "
                                 f"across {num_workers} workers, replicating instead")
            # Replicate instead
            for worker in workers:
                shard_cmd = CreateTensor(
                    tensor_id=cmd.tensor_id,
                    data=cmd.data,
                    dtype=cmd.dtype,
                    shape=cmd.shape,
                    worker_id=worker['id']
                )
                yield shard_cmd
                self._register_tensor_location(cmd.tensor_id, worker['id'], cmd.shape, cmd.dtype)
            self.sharded_commands += 1
            return

        # Shard along axis 0
        shard_size = dim_size // num_workers

        debug_print_dispatcher(f"[AUTO-SHARD] Sharding tensor {cmd.tensor_id} shape {cmd.shape} "
                             f"across {num_workers} workers along axis {axis}")

        # Create shard for each worker
        import numpy as np
        for shard_idx, worker in enumerate(workers):
            # Compute shard shape
            shard_shape = list(cmd.shape)
            shard_shape[axis] = shard_size

            # Slice data for this shard
            start = shard_idx * shard_size
            end = start + shard_size

            if cmd.data is not None:
                # Slice along the sharding axis
                slices = [slice(None)] * len(cmd.shape)
                slices[axis] = slice(start, end)
                shard_data = cmd.data[tuple(slices)].copy()
            else:
                shard_data = None

            shard_cmd = CreateTensor(
                tensor_id=cmd.tensor_id,
                data=shard_data,
                dtype=cmd.dtype,
                shape=tuple(shard_shape),
                worker_id=worker['id'],
                shard_info={
                    'axis': axis,
                    'shard_index': shard_idx,
                    'num_shards': num_workers
                }
            )
            yield shard_cmd
            self._register_tensor_location(
                cmd.tensor_id,
                worker['id'],
                tuple(shard_shape),
                cmd.dtype,
                shard_info={'axis': axis, 'shard_index': shard_idx, 'num_shards': num_workers}
            )

        self.sharded_commands += 1

    def _auto_shard_unary_creation(self, cmd: UnaryOp) -> Iterator[ClientCommand]:
        """
        Automatically shard a unary creation op (randn, zeros) across all available workers.

        Similar to _auto_shard_create_tensor but for ops without data.
        """
        workers = self.get_workers()

        if not workers or len(workers) == 0:
            yield cmd
            return

        if len(workers) == 1:
            cmd.worker_id = workers[0]['id']
            yield cmd
            # Register single-worker tensor location
            self._register_tensor_location(cmd.result_id, workers[0]['id'], cmd.shape, cmd.dtype)
            return

        if cmd.shape is None or len(cmd.shape) == 0:
            yield cmd
            return

        axis = self.auto_shard_axis
        if len(cmd.shape) <= axis:
            yield cmd
            return

        num_workers = len(workers)
        dim_size = cmd.shape[axis]

        # Check if evenly divisible
        if dim_size % num_workers != 0:
            debug_print_dispatcher(f"[AUTO-SHARD] Cannot evenly shard {cmd.op} axis {axis} of size {dim_size} "
                                 f"across {num_workers} workers, replicating instead")
            # Replicate
            for worker in workers:
                shard_cmd = UnaryOp(
                    result_id=cmd.result_id,
                    op=cmd.op,
                    input_id=cmd.input_id,
                    shape=cmd.shape,
                    dtype=cmd.dtype,
                    axis=getattr(cmd, 'axis', None),
                    keepdims=getattr(cmd, 'keepdims', False),
                    worker_id=worker['id']
                )
                yield shard_cmd
                # Register replicated tensor location
                self._register_tensor_location(cmd.result_id, worker['id'], cmd.shape, cmd.dtype)
                debug_print_dispatcher(f"[AUTO-SHARD] Registered REPLICATED location for tensor {cmd.result_id} on worker {worker['id']}")
            self.sharded_commands += 1
            return

        # Shard along axis
        shard_size = dim_size // num_workers

        debug_print_dispatcher(f"[AUTO-SHARD] Sharding {cmd.op} tensor {cmd.result_id} shape {cmd.shape} "
                             f"across {num_workers} workers along axis {axis}")

        for shard_idx, worker in enumerate(workers):
            shard_shape = list(cmd.shape)
            shard_shape[axis] = shard_size

            shard_cmd = UnaryOp(
                result_id=cmd.result_id,
                op=cmd.op,
                input_id=cmd.input_id,
                shape=tuple(shard_shape),
                dtype=cmd.dtype,
                axis=getattr(cmd, 'axis', None),
                keepdims=getattr(cmd, 'keepdims', False),
                worker_id=worker['id'],
                shard_info={
                    'axis': axis,
                    'shard_index': shard_idx,
                    'num_shards': num_workers
                }
            )
            yield shard_cmd
            # Register shard location
            self._register_tensor_location(
                cmd.result_id,
                worker['id'],
                tuple(shard_shape),
                cmd.dtype,
                shard_info={'axis': axis, 'shard_index': shard_idx, 'num_shards': num_workers}
            )
            debug_print_dispatcher(f"[AUTO-SHARD] Registered location for tensor {cmd.result_id} on worker {worker['id']}")

        self.sharded_commands += 1
