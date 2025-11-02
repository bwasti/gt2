"""
Sharding stream modifier - transforms instructions to support distributed execution.

This is a PURE STREAM TRANSFORMER:
- Takes instruction stream (commands from clients) as input
- Yields instruction stream as output (with shard/move/reduce operations injected)
- Does NOT buffer or execute anything
- Dispatcher applies this before routing to workers

The modifier identifies operations that should be sharded and injects:
- Move operations (data movement between workers)
- Reduce operations (gather results from multiple workers)
- Copy operations (replicate data across workers)

Input stream:
  MATMUL A, B -> C    (large tensors)

Output stream (with sharding):
  SHARD A -> A_0, A_1, A_2
  SHARD B -> B_0, B_1, B_2
  MATMUL A_0, B_0 -> C_0
  MATMUL A_1, B_1 -> C_1
  MATMUL A_2, B_2 -> C_2
  REDUCE C_0, C_1, C_2 -> C
"""

from typing import Iterator, Optional, Dict, Set
from gt.debug import debug_print_dispatcher
from gt.transport.protocol import (
    ClientCommand, BinaryOp, UnaryOp, CreateTensor, GetData
)


class ShardingStreamModifier:
    """
    Stream transformer that injects sharding operations.

    Usage:
        modifier = ShardingStreamModifier(enabled=False)
        for cmd in command_stream:
            for output_cmd in modifier.process(cmd):
                yield output_cmd
    """

    def __init__(self, enabled: bool = False, shard_threshold: int = 10000):
        """
        Initialize sharding stream modifier.

        Args:
            enabled: Whether sharding is enabled (default: False)
            shard_threshold: Minimum tensor size to shard (elements)
        """
        self.enabled = enabled
        self.shard_threshold = shard_threshold

        # Track tensor metadata
        self.tensor_shapes: Dict[int, tuple] = {}  # tensor_id -> shape
        self.tensor_sharded: Set[int] = set()  # tensors that have been sharded

        # Stats
        self.total_operations = 0
        self.sharded_operations = 0
        self.injected_ops = 0

    def _should_shard(self, tensor_id: int) -> bool:
        """
        Determine if a tensor should be sharded based on its size.

        Args:
            tensor_id: Tensor to check

        Returns:
            True if tensor should be sharded
        """
        if not self.enabled:
            return False

        if tensor_id not in self.tensor_shapes:
            return False

        shape = self.tensor_shapes[tensor_id]
        size = 1
        for dim in shape:
            size *= dim

        return size >= self.shard_threshold

    def process(self, cmd: ClientCommand) -> Iterator[ClientCommand]:
        """
        Process a command and yield output commands (with potential sharding ops).

        Yields:
            Original command or transformed commands with sharding
        """
        self.total_operations += 1

        # If sharding is disabled, pass through
        if not self.enabled:
            yield cmd
            return

        # Track tensor shapes for sharding decisions
        if isinstance(cmd, CreateTensor):
            self.tensor_shapes[cmd.tensor_id] = cmd.shape
            debug_print_dispatcher(f"[SHARDING] Tracking tensor {cmd.tensor_id} shape: {cmd.shape}")
            yield cmd
            return

        # For now, just pass through all commands
        # Future: detect large matmuls, reductions, etc. and inject sharding ops
        # TODO: Implement actual sharding logic here
        #   - Detect operations that should be sharded (matmul, reduce, etc.)
        #   - Inject move/copy operations
        #   - Inject reduce operations
        #   - Track which tensors are sharded

        yield cmd

    def get_stats(self) -> dict:
        """Get sharding statistics."""
        return {
            'enabled': self.enabled,
            'total_operations': self.total_operations,
            'sharded_operations': self.sharded_operations,
            'injected_ops': self.injected_ops,
            'tracked_tensors': len(self.tensor_shapes),
            'sharded_tensors': len(self.tensor_sharded),
            'shard_threshold': self.shard_threshold,
        }
