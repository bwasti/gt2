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

        # Default: pass through unchanged
        yield cmd

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

        self.sharded_commands += 1
