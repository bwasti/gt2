"""
TensorHandle manages mapping from client:tensor to worker locations.

Keep this SIMPLE and READABLE.
"""

from typing import Optional, List
from dataclasses import dataclass


@dataclass
class ShardInfo:
    """Describes how a tensor is sharded."""
    axis: int  # Which axis is sharded (-1 means not sharded)
    num_shards: int  # Total number of shards
    shard_index: int  # Which shard this is (0-indexed)


@dataclass
class TensorLocation:
    """Describes where a tensor lives."""
    worker_id: str  # Which worker has this tensor
    worker_tensor_id: str  # The ID used on that worker
    shape: tuple  # Shape on this worker (may be partial if sharded)
    dtype: str
    shard_info: Optional[ShardInfo] = None  # Sharding information


class TensorHandle:
    """
    Maps client tensors to their physical locations.

    Each client:tensor maps to one or more worker locations (for sharding).
    """

    def __init__(self):
        # Map: (client_id, tensor_id) -> List[TensorLocation]
        # Most tensors have 1 location, sharded tensors have multiple
        self.locations = {}

    def register(self, client_id: str, tensor_id: int, worker_id: str,
                 worker_tensor_id: str, shape: tuple, dtype: str,
                 shard_info: Optional[ShardInfo] = None):
        """Register a tensor's location."""
        key = (client_id, tensor_id)
        location = TensorLocation(
            worker_id=worker_id,
            worker_tensor_id=worker_tensor_id,
            shape=shape,
            dtype=dtype,
            shard_info=shard_info
        )

        # For sharding, append to list. For non-sharded, replace any existing location.
        # This handles tensor ID reuse (e.g., from in-place operations like weight -= grad)
        if shard_info:
            # Sharded tensor: append to list
            if key not in self.locations:
                self.locations[key] = []
            self.locations[key].append(location)
        else:
            # Non-sharded tensor: replace any existing location
            self.locations[key] = [location]

    def get_locations(self, client_id: str, tensor_id: int) -> List[TensorLocation]:
        """Get all locations of a tensor (list for sharded tensors)."""
        key = (client_id, tensor_id)
        return self.locations.get(key, [])

    def get_location(self, client_id: str, tensor_id: int) -> Optional[TensorLocation]:
        """Get the first location of a tensor (for backward compatibility)."""
        locations = self.get_locations(client_id, tensor_id)
        return locations[0] if locations else None

    def is_sharded(self, client_id: str, tensor_id: int) -> bool:
        """Check if a tensor is sharded across workers."""
        locations = self.get_locations(client_id, tensor_id)
        return len(locations) > 1

    def get_full_shape(self, client_id: str, tensor_id: int) -> Optional[tuple]:
        """Get the full logical shape of a tensor (combining shards)."""
        locations = self.get_locations(client_id, tensor_id)
        if not locations:
            return None

        if len(locations) == 1:
            return locations[0].shape

        # Reconstruct full shape from sharded pieces
        first_loc = locations[0]
        if first_loc.shard_info:
            full_shape = list(first_loc.shape)
            # Multiply the sharded axis by num_shards
            full_shape[first_loc.shard_info.axis] *= first_loc.shard_info.num_shards
            return tuple(full_shape)

        return first_loc.shape

    def free(self, client_id: str, tensor_id: int):
        """Free a tensor."""
        key = (client_id, tensor_id)
        if key in self.locations:
            del self.locations[key]

    def has_tensor(self, client_id: str, tensor_id: int) -> bool:
        """Check if a tensor is registered."""
        key = (client_id, tensor_id)
        return key in self.locations
