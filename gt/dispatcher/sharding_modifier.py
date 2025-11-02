"""
Signal-driven sharding stream modifier.

This is a PURE STREAM TRANSFORMER:
- Takes instruction stream (commands from clients) as input
- Yields instruction stream as output (with shard operations injected)
- Does NOT buffer or execute anything
- Waits for user signals to know when/how to shard

Design Philosophy:
- DUMB: Only shards when user explicitly signals
- No automatic sharding decisions
- Just reads YAML config and follows instructions
- Keeps dispatcher simple (just executes commands)

Example:
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
"""

from typing import Iterator, Optional, Dict, List
from gt.debug import debug_print_dispatcher
from gt.transport.protocol import (
    ClientCommand, BinaryOp, UnaryOp, CreateTensor, GetData, CompileStart, CompileEnd
)


class ShardingStreamModifier:
    """
    Signal-driven stream transformer that injects sharding operations.

    Usage:
        modifier = ShardingStreamModifier()
        for cmd in command_stream:
            for output_cmd in modifier.process(cmd):
                yield output_cmd
    """

    def __init__(self):
        """Initialize sharding stream modifier."""
        # Track active signal scope
        self.active_signal: Optional[str] = None

        # Stats
        self.total_commands = 0
        self.sharded_commands = 0

    def process(self, cmd: ClientCommand) -> Iterator[ClientCommand]:
        """
        Process a command and yield output commands.

        If we're in a signal scope with sharding config, transform commands.
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

        # If no active signal, pass through
        if self.active_signal is None:
            yield cmd
            return

        # Get signal config
        from gt.config import get_signal_config
        signal_config = get_signal_config(self.active_signal)

        if signal_config is None or signal_config.shard is None:
            # No sharding config for this signal
            yield cmd
            return

        shard_config = signal_config.shard

        # For now, only handle CreateTensor and UnaryOp (for randn/zeros)
        # TODO: Handle BinaryOp, matmul, etc.
        if isinstance(cmd, CreateTensor):
            yield from self._shard_create_tensor(cmd, shard_config)
            self.sharded_commands += 1
        elif isinstance(cmd, UnaryOp) and cmd.input_id is None:
            # randn, zeros - tensor creation ops
            yield from self._shard_unary_creation(cmd, shard_config)
            self.sharded_commands += 1
        else:
            # Other ops - just pass through for now
            # TODO: Handle operations on sharded tensors
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
                    signal=cmd.signal,
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
                signal=cmd.signal,
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
                    signal=cmd.signal,
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
                signal=cmd.signal,
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
        }
