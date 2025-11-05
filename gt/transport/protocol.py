"""
Protocol definitions for communication between components.

Keep this SIMPLE and READABLE.
"""

from dataclasses import dataclass
from typing import Any, Optional
import pickle


# Client <-> Dispatcher Protocol (simpler)

@dataclass
class ClientCommand:
    """Base class for commands from client to dispatcher."""
    pass


@dataclass
class CreateTensor(ClientCommand):
    """Create a new tensor with data."""
    tensor_id: int
    data: Any  # serialized numpy array or similar
    dtype: str
    shape: tuple
    worker_id: Optional[int] = None  # Specific worker to target (set by sharding modifier)
    shard_info: Optional[dict] = None  # Shard metadata (axis, index, num_shards)


@dataclass
class BinaryOp(ClientCommand):
    """Binary operation: result = op(left, right)."""
    result_id: int
    op: str  # "add", "mul", "matmul", etc.
    left_id: int
    right_id: int


@dataclass
class UnaryOp(ClientCommand):
    """Unary operation: result = op(input)."""
    result_id: int
    op: str  # "randn", "exp", "log", "sum", "mean", etc.
    input_id: Optional[int]  # None for ops like randn that create data
    shape: Optional[tuple] = None  # for randn, etc.
    dtype: Optional[str] = None
    # For reduction operations (sum, mean, etc.)
    axis: Optional[int] = None  # Axis to reduce over (None = all axes)
    keepdims: bool = False  # Whether to keep reduced dimensions
    worker_id: Optional[int] = None  # Specific worker to target (set by sharding modifier)
    shard_info: Optional[dict] = None  # Shard metadata (axis, index, num_shards)


@dataclass
class ReshapeOp(ClientCommand):
    """Reshape operation: result = reshape/unsqueeze/squeeze(input)."""
    result_id: int
    op: str  # "reshape", "unsqueeze", "squeeze"
    input_id: int
    params: tuple  # reshape: new_shape, unsqueeze: (dim,), squeeze: () or (dim,)


@dataclass
class SliceOp(ClientCommand):
    """Slice operation: result = input[key]."""
    result_id: int
    input_id: int
    key: tuple  # Serialized slice key (can be tuple of slice objects, ints, None, Ellipsis)


@dataclass
class GetData(ClientCommand):
    """Request data for a tensor."""
    tensor_id: int


@dataclass
class FreeTensor(ClientCommand):
    """Free a tensor (garbage collection)."""
    tensor_id: int


@dataclass
class CopyTensor(ClientCommand):
    """Copy data from one tensor to another (in-place update)."""
    dest_id: int  # Destination tensor ID
    src_id: int   # Source tensor ID


@dataclass
class CompileStart(ClientCommand):
    """Mark the start of a signal scope (used for sharding, compilation, etc)."""
    signal_name: str


@dataclass
class CompileEnd(ClientCommand):
    """Mark the end of a signal scope."""
    signal_name: str


@dataclass
class GetWorkerStats(ClientCommand):
    """Request compilation and operation statistics from workers."""
    pass


@dataclass
class RegisterWorker(ClientCommand):
    """Register a worker with the dispatcher."""
    worker_id: str


@dataclass
class ClientResponse:
    """Response from dispatcher to client."""
    success: bool
    data: Any = None  # For GetData responses
    error: Optional[str] = None


# Dispatcher <-> Worker Protocol (more precise)

@dataclass
class WorkerCommand:
    """Base class for commands from dispatcher to worker."""
    pass


@dataclass
class WorkerCreateTensor(WorkerCommand):
    """Create tensor on worker."""
    tensor_id: str  # worker-local ID (dispatcher tracks mapping)
    data: Any
    dtype: str
    shape: tuple


@dataclass
class WorkerBinaryOp(WorkerCommand):
    """Execute binary operation on worker."""
    result_id: str
    op: str
    left_id: str
    right_id: str


@dataclass
class WorkerUnaryOp(WorkerCommand):
    """Execute unary operation on worker."""
    result_id: str
    op: str
    input_id: Optional[str]
    shape: Optional[tuple] = None
    dtype: Optional[str] = None
    # For reduction operations (sum, mean, etc.)
    axis: Optional[int] = None  # Axis to reduce over (None = all axes)
    keepdims: bool = False  # Whether to keep reduced dimensions


@dataclass
class WorkerReshapeOp(WorkerCommand):
    """Execute reshape operation on worker."""
    result_id: str
    op: str  # "reshape", "unsqueeze", "squeeze"
    input_id: str
    params: tuple  # reshape: new_shape, unsqueeze: (dim,), squeeze: () or (dim,)


@dataclass
class WorkerSliceOp(WorkerCommand):
    """Execute slice operation on worker."""
    result_id: str
    input_id: str
    key: tuple  # Serialized slice key


@dataclass
class WorkerGetData(WorkerCommand):
    """Get tensor data from worker."""
    tensor_id: str


@dataclass
class WorkerFreeTensor(WorkerCommand):
    """Free tensor on worker."""
    tensor_id: str


@dataclass
class WorkerCompileStart(WorkerCommand):
    """Signal worker to start a signal scope."""
    signal_name: str


@dataclass
class WorkerCompileEnd(WorkerCommand):
    """Signal worker to end a signal scope."""
    signal_name: str


@dataclass
class WorkerHotPathStart(WorkerCommand):
    """Mark the start of a detected hot path sequence."""
    sequence_id: str  # Unique ID for this hot sequence


@dataclass
class WorkerHotPathEnd(WorkerCommand):
    """Mark the end of a detected hot path sequence."""
    sequence_id: str  # Must match the corresponding START


@dataclass
class WorkerGetStats(WorkerCommand):
    """Request statistics from worker."""
    pass


@dataclass
class WorkerBatch(WorkerCommand):
    """Batch of commands to execute on worker.

    This allows sending multiple operations in a single message,
    reducing network overhead. Worker processes commands sequentially.
    """
    commands: list  # List of WorkerCommand objects


@dataclass
class WorkerResponse:
    """Response from worker to dispatcher."""
    success: bool
    data: Any = None
    error: Optional[str] = None


# Serialization helpers

def serialize(obj):
    """Serialize object for network transmission."""
    return pickle.dumps(obj)


def deserialize(data):
    """Deserialize object from network transmission."""
    return pickle.loads(data)
