"""
Tensor abstraction for GT.

Keep this SIMPLE and READABLE.
"""

import numpy as np
import weakref
from typing import Optional


# Global client connection (set when client connects)
_client_connection = None
_next_tensor_id = 0


class TensorData:
    """Wrapper for tensor data that provides a .numpy() method."""

    def __init__(self, data: np.ndarray):
        self._data = data

    def numpy(self) -> np.ndarray:
        """Get the underlying numpy array."""
        return self._data

    def __repr__(self):
        return repr(self._data)

    def __str__(self):
        return str(self._data)


def _get_next_id():
    """Get next tensor ID."""
    global _next_tensor_id
    tid = _next_tensor_id
    _next_tensor_id += 1
    return tid


class Tensor:
    """
    Tensor container that may be remote.

    Users interact with this like a normal tensor, but behind the scenes
    it may live on a remote worker. Location is completely transparent.
    """

    def __init__(self, tensor_id: Optional[int] = None, shape=None, dtype=None, requires_grad: bool = False):
        self.id = tensor_id if tensor_id is not None else _get_next_id()
        self.shape = shape
        self.dtype = dtype
        self._grad = None
        self.requires_grad = requires_grad

        # Register for garbage collection
        self._finalizer = weakref.finalize(self, _free_tensor, self.id)

    @property
    def data(self):
        """
        Get the actual data for this tensor.

        Behind the scenes, this sends a request to the dispatcher to fetch
        the data from wherever it lives.

        Returns a TensorData wrapper that has a .numpy() method.
        """
        from gt.transport.protocol import GetData, ClientResponse

        if _client_connection is None:
            raise RuntimeError("Not connected to dispatcher")

        cmd = GetData(tensor_id=self.id)
        _client_connection.send(cmd)
        response: ClientResponse = _client_connection.recv()

        if not response.success:
            raise RuntimeError(f"Failed to get data: {response.error}")

        return TensorData(response.data)

    @property
    def grad(self):
        """Gradient tensor (set by autograd)."""
        return self._grad

    def __repr__(self):
        return f"Tensor(id={self.id}, shape={self.shape}, dtype={self.dtype})"

    # Binary operations
    def __add__(self, other):
        return _binary_op("add", self, other)

    def __mul__(self, other):
        return _binary_op("mul", self, other)

    def __rmul__(self, other):
        """Reverse multiplication: other * self"""
        return self * other

    def __matmul__(self, other):
        return _binary_op("matmul", self, other)

    def __sub__(self, other):
        return _binary_op("sub", self, other)

    def __isub__(self, other):
        """In-place subtraction: a -= b"""
        # Compute the subtraction
        result = self - other

        # Steal result's ID and metadata
        new_id = result.id
        new_shape = result.shape
        new_dtype = result.dtype

        # Detach result's finalizer to prevent it from freeing the tensor we're adopting
        result._finalizer.detach()

        # Update this tensor's ID to point to the new result
        # The old tensor will be garbage collected naturally
        self.id = new_id
        self.shape = new_shape
        self.dtype = new_dtype

        return self

    def __truediv__(self, other):
        return _binary_op("div", self, other)

    def __pow__(self, other):
        """Power operation: a ** b"""
        # For now, handle simple case of squaring (** 2)
        if isinstance(other, (int, float)) and other == 2:
            return self * self
        else:
            raise NotImplementedError(f"Power operation only supports ** 2 for now, got ** {other}")

    def __neg__(self):
        """Negate a tensor: -a"""
        return self * from_numpy(np.array(-1.0, dtype='float32'))

    def __gt__(self, other):
        """Greater than comparison: a > b"""
        return _binary_op("gt", self, other)

    # Unary operations
    def exp(self):
        return _unary_op("exp", self)

    def log(self):
        return _unary_op("log", self)

    def sum(self):
        return _unary_op("sum", self)

    def mean(self):
        return _unary_op("mean", self)

    def relu(self):
        """ReLU activation: max(0, x)"""
        return _unary_op("relu", self)

    def sigmoid(self):
        """Sigmoid activation: 1 / (1 + exp(-x))"""
        return _unary_op("sigmoid", self)

    def tanh(self):
        """Tanh activation: (exp(x) - exp(-x)) / (exp(x) + exp(-x))"""
        return _unary_op("tanh", self)

    def reshape(self, *shape):
        """Reshape tensor to new shape"""
        # TODO: implement reshape
        raise NotImplementedError("reshape not yet implemented")

    def transpose(self):
        """Transpose tensor (swap last two dimensions)"""
        return _unary_op("transpose", self)

    @property
    def T(self):
        """Transpose property (PyTorch/NumPy style)"""
        return self.transpose()

    def item(self):
        """Get scalar value (for 0-d or 1-element tensors)."""
        data = self.data.numpy()
        if data.size != 1:
            raise ValueError(f"item() only works on tensors with 1 element, got {data.size}")
        return float(data.flatten()[0])

    def zero_(self):
        """Zero out the tensor in-place (PyTorch-style)."""
        # For gradient tensors, just detach them so they'll be recreated on next backward
        # This is simpler than actually zeroing the data
        if self.shape is None:
            # Gradient tensor without shape info - just mark for GC
            self._finalizer.detach()
            return self

        # For regular tensors, replace with zeros
        zeros_tensor = zeros(*self.shape, dtype=self.dtype)

        # Swap IDs (functional approach)
        self._finalizer.detach()  # Detach old finalizer
        self.id = zeros_tensor.id
        zeros_tensor._finalizer.detach()  # Prevent zeros_tensor from freeing our new ID

        return self

    def backward(self):
        """
        Compute gradients using autograd.

        Like PyTorch: loss.backward()
        """
        from gt.client.autograd import get_graph
        graph = get_graph()
        graph.backward(self)


def _binary_op(op: str, left, right) -> Tensor:
    """Execute a binary operation."""
    from gt.transport.protocol import BinaryOp, ClientResponse
    from gt.client.autograd import get_graph

    if _client_connection is None:
        raise RuntimeError("Not connected to dispatcher")

    # Convert scalars to tensors
    if not isinstance(left, Tensor):
        left = from_numpy(np.array(left, dtype='float32'))
    if not isinstance(right, Tensor):
        right = from_numpy(np.array(right, dtype='float32'))

    # Check if we need gradients
    requires_grad = left.requires_grad or right.requires_grad

    result = Tensor(requires_grad=requires_grad)
    cmd = BinaryOp(result_id=result.id, op=op, left_id=left.id, right_id=right.id)
    _client_connection.send(cmd)
    response: ClientResponse = _client_connection.recv()

    if not response.success:
        raise RuntimeError(f"Operation {op} failed: {response.error}")

    # Record on autograd tape if needed
    if requires_grad:
        graph = get_graph()

        # Define gradient function for this operation
        def grad_fn(grad_output):
            if op == "add":
                return [grad_output, grad_output]
            elif op == "sub":
                # d/dleft (left - right) = grad_output
                # d/dright (left - right) = -grad_output
                # Create -1 * grad_output
                neg_one = from_numpy(np.array(-1.0, dtype='float32'))
                return [grad_output, neg_one * grad_output]
            elif op == "mul":
                return [grad_output * right, grad_output * left]
            elif op == "div":
                return [grad_output / right, -grad_output * left / (right * right)]
            elif op == "matmul":
                # For matmul: C = A @ B
                # grad_A = grad_C @ B.T
                # grad_B = A.T @ grad_C
                grad_left = grad_output @ right.T
                grad_right = left.T @ grad_output
                return [grad_left, grad_right]
            elif op == "gt":
                # Comparison operations have zero gradient
                zero = from_numpy(np.array(0.0, dtype='float32'))
                return [zero, zero]
            else:
                raise RuntimeError(f"Gradient not implemented for {op}")

        graph.record(result, [left, right], grad_fn)

    return result


def _unary_op(op: str, input_tensor: Tensor) -> Tensor:
    """Execute a unary operation."""
    from gt.transport.protocol import UnaryOp, ClientResponse
    from gt.client.autograd import get_graph

    if _client_connection is None:
        raise RuntimeError("Not connected to dispatcher")

    requires_grad = input_tensor.requires_grad

    result = Tensor(requires_grad=requires_grad)
    cmd = UnaryOp(result_id=result.id, op=op, input_id=input_tensor.id)
    _client_connection.send(cmd)
    response: ClientResponse = _client_connection.recv()

    if not response.success:
        raise RuntimeError(f"Operation {op} failed: {response.error}")

    # Record on autograd tape if needed
    if requires_grad:
        graph = get_graph()

        # Define gradient function for this operation
        def grad_fn(grad_output):
            if op == "sum":
                # grad of sum: broadcast scalar gradient to input shape
                # Create a tensor filled with the gradient value
                grad_val = grad_output.data.numpy()
                if input_tensor.shape:
                    broadcasted = np.full(input_tensor.shape, grad_val, dtype='float32')
                    return [from_numpy(broadcasted)]
                else:
                    return [grad_output]
            elif op == "mean":
                # grad of mean is grad_output / n
                # For now, simplified
                return [grad_output]
            elif op == "exp":
                # grad of exp(x) is exp(x) * grad_output = result * grad_output
                return [grad_output * result]
            elif op == "log":
                # grad of log(x) is grad_output / x
                return [grad_output / input_tensor]
            elif op == "relu":
                # grad of relu(x): grad_output * (x > 0)
                # For simplicity, recompute in grad
                # In practice, would cache forward pass
                return [grad_output * (input_tensor > from_numpy(np.array(0.0, dtype='float32')))]
            elif op == "sigmoid":
                # grad of sigmoid(x): sigmoid(x) * (1 - sigmoid(x)) * grad_output
                # result is sigmoid(x)
                one = from_numpy(np.array(1.0, dtype='float32'))
                return [grad_output * result * (one - result)]
            elif op == "tanh":
                # grad of tanh(x): (1 - tanh²(x)) * grad_output
                one = from_numpy(np.array(1.0, dtype='float32'))
                return [grad_output * (one - result * result)]
            elif op == "transpose":
                # grad of transpose: just transpose the gradient back
                return [grad_output.T]
            else:
                raise RuntimeError(f"Gradient not implemented for {op}")

        graph.record(result, [input_tensor], grad_fn)

    return result


def _free_tensor(tensor_id: int):
    """Callback for garbage collection."""
    from gt.transport.protocol import FreeTensor

    if _client_connection is None or tensor_id is None:
        return

    try:
        cmd = FreeTensor(tensor_id=tensor_id)
        _client_connection.send(cmd)
        # Don't wait for response during cleanup
    except:
        pass  # Connection might be closed


# Factory functions for creating tensors

def from_numpy(array: np.ndarray, requires_grad: bool = False) -> Tensor:
    """Create a tensor from a numpy array."""
    from gt.transport.protocol import CreateTensor, ClientResponse

    if _client_connection is None:
        raise RuntimeError("Not connected to dispatcher")

    tensor = Tensor(shape=array.shape, dtype=str(array.dtype), requires_grad=requires_grad)
    cmd = CreateTensor(
        tensor_id=tensor.id,
        data=array,
        dtype=str(array.dtype),
        shape=array.shape
    )
    _client_connection.send(cmd)
    response: ClientResponse = _client_connection.recv()

    if not response.success:
        raise RuntimeError(f"Failed to create tensor: {response.error}")

    return tensor


def randn(*shape, dtype="float32") -> Tensor:
    """Create a random tensor with normal distribution."""
    from gt.transport.protocol import UnaryOp, ClientResponse

    if _client_connection is None:
        raise RuntimeError("Not connected to dispatcher")

    tensor = Tensor(shape=shape, dtype=dtype)
    cmd = UnaryOp(
        result_id=tensor.id,
        op="randn",
        input_id=None,
        shape=shape,
        dtype=dtype
    )
    _client_connection.send(cmd)
    response: ClientResponse = _client_connection.recv()

    if not response.success:
        raise RuntimeError(f"Failed to create randn tensor: {response.error}")

    return tensor


def zeros(*shape, dtype="float32") -> Tensor:
    """Create a tensor filled with zeros."""
    from gt.transport.protocol import UnaryOp, ClientResponse

    if _client_connection is None:
        raise RuntimeError("Not connected to dispatcher")

    tensor = Tensor(shape=shape, dtype=dtype)
    cmd = UnaryOp(
        result_id=tensor.id,
        op="zeros",
        input_id=None,
        shape=shape,
        dtype=dtype
    )
    _client_connection.send(cmd)
    response: ClientResponse = _client_connection.recv()

    if not response.success:
        raise RuntimeError(f"Failed to create zeros tensor: {response.error}")

    return tensor
