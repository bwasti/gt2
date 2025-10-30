"""
TensorHandle manages mapping from client:tensor to worker locations.

Keep this SIMPLE and READABLE.
"""

from typing import Optional, List
from dataclasses import dataclass


@dataclass
class TensorLocation:
    """Describes where a tensor lives."""
    worker_id: str  # Which worker has this tensor
    worker_tensor_id: str  # The ID used on that worker
    shape: tuple
    dtype: str
    # Future: sharding info, replication info


class TensorHandle:
    """
    Maps client tensors to their physical locations.

    Each client:tensor maps to one or more worker locations.
    For now, keep it simple: one tensor = one location.
    """

    def __init__(self):
        # Map: (client_id, tensor_id) -> TensorLocation
        self.locations = {}

    def register(self, client_id: str, tensor_id: int, worker_id: str,
                 worker_tensor_id: str, shape: tuple, dtype: str):
        """Register a tensor's location."""
        key = (client_id, tensor_id)
        self.locations[key] = TensorLocation(
            worker_id=worker_id,
            worker_tensor_id=worker_tensor_id,
            shape=shape,
            dtype=dtype
        )

    def get_location(self, client_id: str, tensor_id: int) -> Optional[TensorLocation]:
        """Get the location of a tensor."""
        key = (client_id, tensor_id)
        return self.locations.get(key)

    def free(self, client_id: str, tensor_id: int):
        """Free a tensor."""
        key = (client_id, tensor_id)
        if key in self.locations:
            del self.locations[key]

    def has_tensor(self, client_id: str, tensor_id: int) -> bool:
        """Check if a tensor is registered."""
        key = (client_id, tensor_id)
        return key in self.locations
