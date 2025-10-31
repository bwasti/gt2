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
    signal: Optional[str] = None  # Signal name for sharding config


@dataclass
class BinaryOp(ClientCommand):
    """Binary operation: result = op(left, right)."""
    result_id: int
    op: str  # "add", "mul", "matmul", etc.
    left_id: int
    right_id: int
    signal: Optional[str] = None  # Signal name for sharding config


@dataclass
class UnaryOp(ClientCommand):
    """Unary operation: result = op(input)."""
    result_id: int
    op: str  # "randn", "exp", "log", etc.
    input_id: Optional[int]  # None for ops like randn that create data
    shape: Optional[tuple] = None  # for randn, etc.
    dtype: Optional[str] = None
    signal: Optional[str] = None  # Signal name for sharding config


@dataclass
class ReshapeOp(ClientCommand):
    """Reshape operation: result = reshape/unsqueeze/squeeze(input)."""
    result_id: int
    op: str  # "reshape", "unsqueeze", "squeeze"
    input_id: int
    params: tuple  # reshape: new_shape, unsqueeze: (dim,), squeeze: () or (dim,)
    signal: Optional[str] = None  # Signal name for sharding config


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
    """Mark the start of a compilation region."""
    signal_name: str  # Signal name for this compilation region


@dataclass
class CompileEnd(ClientCommand):
    """Mark the end of a compilation region."""
    signal_name: str  # Signal name for this compilation region


@dataclass
class GetWorkerStats(ClientCommand):
    """Request compilation and operation statistics from workers."""
    pass


@dataclass
class RegisterWorker(ClientCommand):
    """Register a worker with the dispatcher."""
    worker_id: str


@dataclass
class BatchCommands(ClientCommand):
    """Batch multiple operations together for efficient communication.

    This reduces network overhead by sending multiple ops in one message.
    Operations are executed in order, and all must succeed or the batch fails.
    """
    commands: list  # List of ClientCommand objects (BinaryOp, UnaryOp, etc.)


@dataclass
class ClientResponse:
    """Response from dispatcher to client."""
    success: bool
    data: Any = None  # For GetData responses
    error: Optional[str] = None


@dataclass
class BatchResponses:
    """Response for a batch of commands."""
    success: bool
    responses: list = None  # List of ClientResponse objects
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


@dataclass
class WorkerGetData(WorkerCommand):
    """Get tensor data from worker."""
    tensor_id: str


@dataclass
class WorkerFreeTensor(WorkerCommand):
    """Free tensor on worker."""
    tensor_id: str


@dataclass
class WorkerMoveTensor(WorkerCommand):
    """Move tensor to another worker."""
    tensor_id: str
    target_worker: str  # worker address


@dataclass
class WorkerCompileStart(WorkerCommand):
    """Signal worker to start a compilation region."""
    signal_name: str


@dataclass
class WorkerCompileEnd(WorkerCommand):
    """Signal worker to end a compilation region and compile."""
    signal_name: str


@dataclass
class WorkerGetStats(WorkerCommand):
    """Request statistics from worker."""
    pass


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
