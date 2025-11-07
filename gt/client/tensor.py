"""
Tensor abstraction for GT.

Keep this SIMPLE and READABLE.
"""

import numpy as np
import weakref
import threading
from typing import Optional
from collections import deque
from gt.errors import not_connected_error, operation_failed_error


# Global client connection (set when client connects)
_client_connection = None
_next_tensor_id = 0
_connection_lock = threading.Lock()  # Ensure serial command/response flow
_free_queue = deque()  # Queue of tensor IDs to free (appended from GC, processed in lock)

# Thread-local gradient tracking
_thread_local = threading.local()


def is_grad_enabled():
    """Check if gradient tracking is enabled."""
    return getattr(_thread_local, 'grad_enabled', True)


def set_grad_enabled(enabled: bool):
    """Set gradient tracking state."""
    _thread_local.grad_enabled = enabled


class TensorData:
    """Wrapper for tensor data that provides a .numpy() method."""

    def __init__(self, data: np.ndarray):
        self._data = data

    def numpy(self) -> np.ndarray:
        """Get the underlying numpy array."""
        return self._data

    @property
    def shape(self):
        """Return the shape of the underlying data."""
        return np.asarray(self._data).shape

    @property
    def dtype(self):
        """Return the dtype of the underlying data."""
        return np.asarray(self._data).dtype

    def __array__(self) -> np.ndarray:
        """Allow numpy to automatically convert this to an array."""
        # Ensure we always return an ndarray (convert scalars to 0-d arrays)
        return np.asarray(self._data)

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

        NOTE: This is a sync point - flushes any pending batched operations.
        """
        from gt.transport.protocol import GetData, ClientResponse

        if _client_connection is None:
            raise RuntimeError("Not connected to dispatcher")

        with _connection_lock:
            # Process any pending frees first
            _process_free_queue()

            cmd = GetData(tensor_id=self.id)
            _client_connection.send(cmd)
            response: ClientResponse = _client_connection.recv()

        if not response.success:
            raise RuntimeError(f"Failed to get data: {response.error}")

        if response.data is None:
            raise RuntimeError(f"Tensor {self.id}: Got success=True but data=None from dispatcher")

        return TensorData(response.data)

    @property
    def grad(self):
        """Gradient tensor (set by autograd)."""
        return self._grad

    def __repr__(self):
        """PyTorch-style repr showing shape, dtype, and data."""
        try:
            # Try to fetch a small preview of the data
            data = self.data.numpy()

            # Format similar to PyTorch
            dtype_str = f"dtype={self.dtype}" if self.dtype else ""
            shape_str = f"shape={self.shape}" if self.shape else ""

            # Use numpy's repr but clean it up
            data_str = np.array2string(data, threshold=10, edgeitems=3,
                                       precision=4, suppress_small=True)

            return f"tensor({data_str}, {dtype_str})" if dtype_str else f"tensor({data_str})"
        except:
            # Fallback if data fetch fails
            return f"Tensor(id={self.id}, shape={self.shape}, dtype={self.dtype})"

    def __str__(self):
        return self.__repr__()

    def __getitem__(self, key):
        """
        Subscript/slice the tensor: a[:4, :4] or a[0] or a[1:3, :]

        Supports:
        - Single indices: a[0]
        - Slices: a[1:5], a[:10], a[::2]
        - Multi-dimensional: a[:4, :4], a[0, :], a[:, 1]
        - Ellipsis: a[..., 0]
        """
        return _slice_op(self, key)

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

        # Steal result's ID
        new_id = result.id

        # Detach result's finalizer to prevent it from freeing the tensor we're adopting
        result._finalizer.detach()

        # Update this tensor's ID to point to the new result
        # The old tensor will be garbage collected naturally
        self.id = new_id

        # IMPORTANT: Do NOT update shape/dtype - these should remain constant!
        # The result of the subtraction should have the same shape as self.
        # If it doesn't, that's a bug in the operation, not something we should hide.

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

    def __eq__(self, other):
        """Element-wise equality: a == b"""
        return _binary_op("eq", self, other)

    # Unary operations
    def exp(self):
        return _unary_op("exp", self)

    def log(self):
        return _unary_op("log", self)

    def sum(self, axis=None, keepdims=False):
        """Sum tensor elements along specified axis.

        Args:
            axis: Axis or tuple of axes to sum over. None means sum all elements.
            keepdims: Whether to keep reduced dimensions (size 1).

        Returns:
            Tensor with summed values
        """
        if axis is None and not keepdims:
            # Legacy behavior: sum all elements to scalar
            return _unary_op("sum", self)
        else:
            # New behavior: axis-aware sum
            return _reduce_op("sum", self, axis=axis, keepdims=keepdims)

    def mean(self, axis=None, keepdims=False):
        """Compute mean reduction.

        Args:
            axis: Axis to reduce. If None, reduces all dimensions.
            keepdims: If True, retains reduced dimensions with size 1.

        Returns:
            Tensor with mean values
        """
        return _reduce_op("mean", self, axis=axis, keepdims=keepdims)

    def max(self, axis=None, keepdims=False):
        """Compute maximum reduction.

        Args:
            axis: Axis to reduce. If None, reduces all dimensions.
            keepdims: If True, retains reduced dimensions with size 1.

        Returns:
            Tensor with maximum values
        """
        return _reduce_op("max", self, axis=axis, keepdims=keepdims)

    def relu(self):
        """ReLU activation: max(0, x)"""
        return _unary_op("relu", self)

    def sigmoid(self):
        """Sigmoid activation: 1 / (1 + exp(-x))"""
        return _unary_op("sigmoid", self)

    def tanh(self):
        """Tanh activation: (exp(x) - exp(-x)) / (exp(x) + exp(-x))"""
        return _unary_op("tanh", self)

    def sqrt(self):
        """Square root: sqrt(x)"""
        return _unary_op("sqrt", self)

    def reshape(self, *shape):
        """Reshape tensor to new shape"""
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return _reshape_op("reshape", self, shape)

    def view(self, *shape):
        """View tensor with new shape (alias for reshape, PyTorch-style)"""
        return self.reshape(*shape)

    def unsqueeze(self, dim):
        """Add a dimension of size 1 at the specified position"""
        return _reshape_op("unsqueeze", self, (dim,))

    def squeeze(self, dim=None):
        """Remove dimensions of size 1"""
        if dim is None:
            return _reshape_op("squeeze", self, ())
        return _reshape_op("squeeze", self, (dim,))

    def transpose(self, dim0=None, dim1=None):
        """
        Transpose tensor.

        If dim0 and dim1 are None, swaps last two dimensions (like .T).
        If dim0 and dim1 are specified, swaps those two dimensions.

        Args:
            dim0: First dimension to swap (default: None, swaps last 2)
            dim1: Second dimension to swap (default: None, swaps last 2)

        Returns:
            Transposed tensor
        """
        if dim0 is None and dim1 is None:
            # Legacy behavior: swap last two dimensions
            return _unary_op("transpose", self)
        elif dim0 is not None and dim1 is not None:
            # New behavior: swap specific dimensions
            return _transpose_op(self, dim0, dim1)
        else:
            raise ValueError("transpose() requires both dim0 and dim1, or neither")

    def permute(self, *dims):
        """
        Permute dimensions of tensor.

        Args:
            *dims: New ordering of dimensions

        Example:
            x = gt.randn(2, 3, 4, 5)
            y = x.permute(0, 2, 1, 3)  # Shape: (2, 4, 3, 5)

        Returns:
            Tensor with permuted dimensions
        """
        if len(dims) == 1 and isinstance(dims[0], (list, tuple)):
            dims = dims[0]
        return _permute_op(self, dims)

    def split(self, split_size, dim=0):
        """
        Split tensor into chunks of size split_size along dimension dim.

        Args:
            split_size: Size of each chunk
            dim: Dimension to split along (default: 0)

        Returns:
            List of tensors
        """
        if not self.shape or len(self.shape) <= dim:
            raise ValueError(f"Cannot split shape {self.shape} along dim {dim}")

        dim_size = self.shape[dim]
        num_chunks = (dim_size + split_size - 1) // split_size  # Ceiling division

        chunks = []
        for i in range(num_chunks):
            start = i * split_size
            end = min(start + split_size, dim_size)

            # Create slice for this chunk
            slices = [slice(None)] * len(self.shape)
            slices[dim] = slice(start, end)
            chunks.append(self[tuple(slices)])

        return chunks

    def chunk(self, chunks, dim=0):
        """
        Split tensor into specific number of chunks along dimension dim.

        Args:
            chunks: Number of chunks to split into
            dim: Dimension to split along (default: 0)

        Returns:
            List of tensors
        """
        if not self.shape or len(self.shape) <= dim:
            raise ValueError(f"Cannot chunk shape {self.shape} along dim {dim}")

        dim_size = self.shape[dim]
        chunk_size = (dim_size + chunks - 1) // chunks  # Ceiling division

        return self.split(chunk_size, dim=dim)

    @property
    def T(self):
        """Transpose property (PyTorch/NumPy style)"""
        return self.transpose()

    def item(self):
        """
        Get scalar value (for 0-d or 1-element tensors).

        NOTE: This is a sync point - flushes any pending batched operations.
        """
        # SYNC POINT: .data property already flushes
        data = self.data.numpy()
        if data.size != 1:
            raise ValueError(f"item() only works on tensors with 1 element, got {data.size}")
        return float(data.flatten()[0])

    def zero_(self):
        """Zero out the tensor in-place (PyTorch-style)."""
        # For gradient tensors, just detach them so they'll be recreated on next backward
        # This is simpler than actually zeroing the data
        if self.shape is None or self.shape == ():
            # Gradient tensor without shape info or scalar - just mark for GC
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

        NOTE: This is a sync point - flushes any pending batched operations.
        """
        from gt.client.autograd import get_graph
        graph = get_graph()
        graph.backward(self)

    def requires_grad_(self, requires_grad=True):
        """
        Set requires_grad flag in-place (PyTorch-compatible).

        Args:
            requires_grad: Whether to track gradients

        Returns:
            self (for chaining)

        Example:
            tensor.requires_grad_(True)
        """
        self.requires_grad = requires_grad
        return self


def _binary_op(op: str, left, right) -> Tensor:
    """Execute a binary operation."""
    from gt.transport.protocol import BinaryOp, ClientResponse
    from gt.client.autograd import get_graph
    import numpy as np

    if _client_connection is None:
        raise RuntimeError(not_connected_error())

    # Convert scalars to tensors
    if not isinstance(left, Tensor):
        left = from_numpy(np.array(left, dtype='float32'))
    if not isinstance(right, Tensor):
        right = from_numpy(np.array(right, dtype='float32'))

    # Check if we need gradients
    requires_grad = is_grad_enabled() and (left.requires_grad or right.requires_grad)

    result = Tensor(requires_grad=requires_grad)

    cmd = BinaryOp(result_id=result.id, op=op, left_id=left.id, right_id=right.id)

    with _connection_lock:
        # Process any pending frees first
        _process_free_queue()

        _client_connection.send(cmd)
        response: ClientResponse = _client_connection.recv()

    if not response.success:
        raise RuntimeError(f"Operation {op} failed: {response.error}")

    # Infer result shape/dtype (TODO: get from dispatcher)
    if op in ["add", "sub", "mul", "div", "gt", "eq"]:
        # Use NumPy broadcasting rules to compute result shape
        import numpy as np
        result.shape = np.broadcast_shapes(left.shape, right.shape)
        result.dtype = left.dtype
    elif op == "matmul":
        # Result shape depends on input dimensions
        # For 2D @ 2D: (m, n) @ (n, p) -> (m, p)
        # For 3D @ 2D: (batch, seq, n) @ (n, p) -> (batch, seq, p)
        # For 3D @ 3D: (batch, m, n) @ (batch, n, p) -> (batch, m, p)
        if left.shape and right.shape:
            if len(left.shape) == 3 and len(right.shape) == 2:
                # 3D @ 2D batched matmul
                result.shape = (left.shape[0], left.shape[1], right.shape[-1])
            elif len(left.shape) >= 2 and len(right.shape) >= 2:
                # Standard matmul: result has left's batch dims and right's last dim
                result.shape = left.shape[:-1] + (right.shape[-1],)
            else:
                # Fallback
                result.shape = (left.shape[0], right.shape[-1])
            result.dtype = left.dtype

    # Record on autograd tape if needed
    if requires_grad:
        graph = get_graph()

        # Helper function to reduce gradients when broadcasting occurred
        def reduce_grad_for_broadcast(grad, original_shape):
            """Reduce gradient to match original shape after broadcasting."""
            import numpy as np

            if original_shape == grad.shape:
                return grad

            # Sum across dimensions that were broadcasted
            axes_to_remove = []  # Dimensions added by broadcasting (orig didn't have them)
            axes_to_reduce = []  # Dimensions where orig had 1 but grad has > 1
            grad_shape = list(grad.shape)
            orig_shape = list(original_shape) if original_shape else []

            # Pad orig_shape with 1s on the left to match grad_shape length
            num_new_dims = len(grad_shape) - len(orig_shape)
            if num_new_dims > 0:
                # These dimensions were added by broadcasting - sum and remove them
                axes_to_remove = list(range(num_new_dims))
                orig_shape = [1] * num_new_dims + orig_shape

            # Find dimensions where original was size 1 but grad is larger
            # These need to be summed with keepdims=True
            for i in range(len(grad_shape)):
                if i not in axes_to_remove and orig_shape[i] == 1 and grad_shape[i] > 1:
                    axes_to_reduce.append(i)

            result = grad

            # First, reduce dimensions where original had size 1 (with keepdims=True)
            for axis in sorted(axes_to_reduce, reverse=True):
                result = result.sum(axis=axis, keepdims=True)

            # Then, remove dimensions that were added by broadcasting (with keepdims=False)
            for axis in sorted(axes_to_remove, reverse=True):
                result = result.sum(axis=axis, keepdims=False)

            return result

        # Define gradient function for this operation
        def grad_fn(grad_output):
            import numpy as np

            if op == "add":
                # For broadcasting, we need to sum grad_output to match input shapes
                grad_left = reduce_grad_for_broadcast(grad_output, left.shape)
                grad_right = reduce_grad_for_broadcast(grad_output, right.shape)
                return [grad_left, grad_right]
            elif op == "sub":
                # d/dleft (left - right) = grad_output
                # d/dright (left - right) = -grad_output
                neg_one = from_numpy(np.array(-1.0, dtype='float32'))
                grad_left = reduce_grad_for_broadcast(grad_output, left.shape)
                grad_right = reduce_grad_for_broadcast(neg_one * grad_output, right.shape)
                return [grad_left, grad_right]
            elif op == "mul":
                # d/dleft (left * right) = grad_output * right
                # d/dright (left * right) = grad_output * left
                grad_left = reduce_grad_for_broadcast(grad_output * right, left.shape)
                grad_right = reduce_grad_for_broadcast(grad_output * left, right.shape)
                return [grad_left, grad_right]
            elif op == "div":
                # d/dleft (left / right) = grad_output / right
                # d/dright (left / right) = -grad_output * left / right^2
                grad_left = reduce_grad_for_broadcast(grad_output / right, left.shape)
                grad_right = reduce_grad_for_broadcast(-grad_output * left / (right * right), right.shape)
                return [grad_left, grad_right]
            elif op == "matmul":
                # For matmul: C = A @ B
                # grad_A = grad_C @ B.T
                # grad_B = A.T @ grad_C
                grad_left = grad_output @ right.T
                grad_right = left.T @ grad_output

                # For batched left and non-batched right, sum grad_right across batch dim
                # Example: (8, 128, 64) @ (64, 64) -> (8, 128, 64)
                # grad_right = (8, 64, 128) @ (8, 128, 64) = (8, 64, 64)
                # But right is (64, 64), so we need to sum: (8, 64, 64) -> (64, 64)
                if len(left.shape) > len(right.shape):
                    # Sum across leading batch dimensions using remote operations
                    for _ in range(len(left.shape) - len(right.shape)):
                        grad_right = grad_right.sum(axis=0, keepdims=False)

                return [grad_left, grad_right]
            elif op in ["gt", "eq"]:
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
        raise RuntimeError(not_connected_error())

    requires_grad = is_grad_enabled() and input_tensor.requires_grad

    result = Tensor(requires_grad=requires_grad)

    cmd = UnaryOp(result_id=result.id, op=op, input_id=input_tensor.id)

    with _connection_lock:
        # Process any pending frees first
        _process_free_queue()

        _client_connection.send(cmd)
        response: ClientResponse = _client_connection.recv()

    if not response.success:
        raise RuntimeError(f"Operation {op} failed: {response.error}")

    # Infer result shape/dtype (TODO: get from dispatcher)
    if op in ["exp", "log", "relu", "sigmoid", "tanh", "sqrt"]:
        # These ops preserve shape
        result.shape = input_tensor.shape
        result.dtype = input_tensor.dtype
    elif op == "transpose":
        # Transpose swaps last two dimensions
        if input_tensor.shape and len(input_tensor.shape) >= 2:
            new_shape = list(input_tensor.shape)
            new_shape[-2], new_shape[-1] = new_shape[-1], new_shape[-2]
            result.shape = tuple(new_shape)
            result.dtype = input_tensor.dtype
        else:
            result.shape = input_tensor.shape
            result.dtype = input_tensor.dtype
    elif op in ["sum", "mean"]:
        # Reductions produce scalars
        result.shape = ()
        result.dtype = input_tensor.dtype

    # Record on autograd tape if needed
    if requires_grad:
        graph = get_graph()

        # Define gradient function for this operation
        def grad_fn(grad_output):
            import numpy as np

            if op == "sum":
                # grad of sum: broadcast scalar gradient to input shape
                # Use broadcasting instead of creating full array:
                # grad_output (scalar) will broadcast when added to existing gradients
                # We need to create a tensor with the right shape filled with ones, then multiply
                if input_tensor.shape and input_tensor.shape != ():
                    # Create ones with input shape, multiply by scalar gradient
                    ones = zeros(*input_tensor.shape, dtype=input_tensor.dtype) + from_numpy(np.array(1.0, dtype='float32'))
                    return [grad_output * ones]
                else:
                    return [grad_output]
            elif op == "mean":
                # mean is now implemented as sum() / n, so this gradient code is unused
                # Keeping for compatibility if called directly
                raise RuntimeError("mean gradient should not be called - mean is implemented via sum/div")
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
            elif op == "sqrt":
                # grad of sqrt(x): grad_output / (2 * sqrt(x)) = grad_output / (2 * result)
                two = from_numpy(np.array(2.0, dtype='float32'))
                return [grad_output / (two * result)]
            elif op == "transpose":
                # grad of transpose: just transpose the gradient back
                return [grad_output.T]
            else:
                raise RuntimeError(f"Gradient not implemented for {op}")

        graph.record(result, [input_tensor], grad_fn)

    return result


def _compute_slice_shape(input_shape: tuple, key: tuple) -> tuple:
    """Compute output shape from input shape and slice key."""
    import numpy as np

    if not input_shape:
        return ()

    # Create a dummy array with the input shape to compute output shape
    try:
        dummy = np.empty(input_shape)
        sliced = dummy[key]
        return sliced.shape
    except:
        # If shape computation fails, return None to defer to runtime
        return None


def _slice_op(input_tensor: Tensor, key) -> Tensor:
    """
    Execute a slicing/subscript operation remotely.

    Converts Python slice notation to serializable format and
    executes on the worker side (100% remote).

    Args:
        input_tensor: Input tensor to slice
        key: Index/slice expression (int, slice, tuple of slices, etc.)

    Returns:
        New tensor representing the sliced view
    """
    from gt.transport.protocol import SliceOp, ClientResponse

    if _client_connection is None:
        raise RuntimeError(not_connected_error())

    # Normalize key to tuple format for serialization
    if not isinstance(key, tuple):
        key = (key,)

    result = Tensor()

    cmd = SliceOp(result_id=result.id, input_id=input_tensor.id, key=key)

    with _connection_lock:
        # Process any pending frees first
        _process_free_queue()

        _client_connection.send(cmd)
        response: ClientResponse = _client_connection.recv()

    if not response.success:
        raise RuntimeError(f"Slice operation failed: {response.error}")

    # Compute result shape from input shape and slice key
    result.shape = _compute_slice_shape(input_tensor.shape, key)
    result.dtype = input_tensor.dtype

    return result


def _reshape_op(op: str, input_tensor: Tensor, params: tuple) -> Tensor:
    """Execute a reshape operation (reshape, unsqueeze, squeeze)."""
    from gt.transport.protocol import ReshapeOp, ClientResponse
    from gt.client.autograd import get_graph
    import numpy as np

    if _client_connection is None:
        raise RuntimeError(not_connected_error())

    requires_grad = is_grad_enabled() and input_tensor.requires_grad

    result = Tensor(requires_grad=requires_grad)

    cmd = ReshapeOp(result_id=result.id, op=op, input_id=input_tensor.id, params=params)

    with _connection_lock:
        # Process any pending frees first
        _process_free_queue()

        _client_connection.send(cmd)
        response: ClientResponse = _client_connection.recv()

    if not response.success:
        raise RuntimeError(f"Operation {op} failed: {response.error}")

    # Compute result shape
    if op == "reshape":
        result.shape = params
        result.dtype = input_tensor.dtype
    elif op == "unsqueeze":
        dim = params[0]
        if input_tensor.shape:
            shape_list = list(input_tensor.shape)
            # Handle negative dimensions
            if dim < 0:
                dim = len(shape_list) + dim + 1
            shape_list.insert(dim, 1)
            result.shape = tuple(shape_list)
        else:
            result.shape = (1,)
        result.dtype = input_tensor.dtype
    elif op == "squeeze":
        if input_tensor.shape:
            if len(params) == 0:
                # Squeeze all dimensions of size 1
                result.shape = tuple(d for d in input_tensor.shape if d != 1)
            else:
                # Squeeze specific dimension
                dim = params[0]
                shape_list = list(input_tensor.shape)
                if shape_list[dim] == 1:
                    shape_list.pop(dim)
                result.shape = tuple(shape_list)
        else:
            result.shape = ()
        result.dtype = input_tensor.dtype

    # Record on autograd tape if needed
    if requires_grad:
        graph = get_graph()

        # Define gradient function for reshape operations
        def grad_fn(grad_output):
            if op == "reshape":
                # Gradient of reshape: reshape back to original shape
                return [grad_output.reshape(input_tensor.shape)]
            elif op == "unsqueeze":
                # Gradient of unsqueeze: squeeze the added dimension
                dim = params[0]
                return [grad_output.squeeze(dim)]
            elif op == "squeeze":
                # Gradient of squeeze: unsqueeze back
                if len(params) == 0:
                    # Need to unsqueeze all removed dimensions - just reshape
                    return [grad_output.reshape(input_tensor.shape)]
                else:
                    dim = params[0]
                    return [grad_output.unsqueeze(dim)]
            else:
                raise RuntimeError(f"Gradient not implemented for {op}")

        graph.record(result, [input_tensor], grad_fn)

    return result


def _reduce_op(op: str, input_tensor: Tensor, axis=None, keepdims=False) -> Tensor:
    """Execute a reduction operation with axis support."""
    from gt.transport.protocol import UnaryOp, ClientResponse
    from gt.client.autograd import get_graph
    import numpy as np

    if _client_connection is None:
        raise RuntimeError(not_connected_error())

    requires_grad = is_grad_enabled() and input_tensor.requires_grad

    result = Tensor(requires_grad=requires_grad)

    cmd = UnaryOp(
        result_id=result.id,
        op=op,
        input_id=input_tensor.id,
        axis=axis,
        keepdims=keepdims
    )

    with _connection_lock:
        # Process any pending frees first
        _process_free_queue()

        _client_connection.send(cmd)
        response: ClientResponse = _client_connection.recv()

    if not response.success:
        raise RuntimeError(f"Operation {op} failed: {response.error}")

    # Compute result shape based on axis and keepdims
    if input_tensor.shape:
        if axis is None:
            # Reduce all axes
            result.shape = () if not keepdims else tuple([1] * len(input_tensor.shape))
        else:
            # Reduce specific axis
            shape_list = list(input_tensor.shape)
            if keepdims:
                shape_list[axis] = 1
            else:
                shape_list.pop(axis)
            result.shape = tuple(shape_list)
    else:
        result.shape = ()
    result.dtype = input_tensor.dtype

    # Record on autograd tape if needed
    if requires_grad:
        graph = get_graph()

        # Define gradient function for axis-aware reductions
        def grad_fn(grad_output):
            if op == "sum":
                # grad of sum: broadcast scalar/reduced gradient to input shape
                if axis is None:
                    # Full reduction - broadcast to full shape
                    # ✅ REMOTE EXECUTION: Use tensor operations instead of fetch-create-upload
                    if input_tensor.shape and input_tensor.shape != ():
                        # Create ones tensor and multiply by gradient (broadcasting happens remotely)
                        ones = zeros(*input_tensor.shape, dtype=input_tensor.dtype) + from_numpy(np.array(1.0, dtype='float32'))
                        return [grad_output * ones]
                    else:
                        return [grad_output]
                else:
                    # Axis-specific reduction - need to unsqueeze if not keepdims
                    if not keepdims:
                        grad_output = grad_output.unsqueeze(axis)
                    # ✅ REMOTE EXECUTION: Broadcast via tensor operations
                    # Create zeros with input shape, add unsqueezed gradient (broadcasting happens remotely)
                    grad_result = zeros(*input_tensor.shape, dtype=input_tensor.dtype) + grad_output
                    return [grad_result]
            elif op == "mean":
                # grad of mean: broadcast gradient / num_elements to input shape
                import numpy as np
                if axis is None:
                    # Full reduction - mean over all elements
                    if input_tensor.shape and input_tensor.shape != ():
                        num_elements = np.prod(input_tensor.shape)
                        ones = zeros(*input_tensor.shape, dtype=input_tensor.dtype) + from_numpy(np.array(1.0 / num_elements, dtype='float32'))
                        return [grad_output * ones]
                    else:
                        return [grad_output]
                else:
                    # Axis-specific reduction
                    if not keepdims:
                        grad_output = grad_output.unsqueeze(axis)
                    # Divide by the size of the reduced dimension
                    reduced_size = input_tensor.shape[axis]
                    scale = from_numpy(np.array(1.0 / reduced_size, dtype='float32'))
                    grad_result = zeros(*input_tensor.shape, dtype=input_tensor.dtype) + (grad_output * scale)
                    return [grad_result]
            elif op == "max":
                # grad of max: gradient flows only to maximum elements
                # For simplicity, we'll use a mask approach
                # Note: this is a simplified implementation - in practice, max should cache the indices
                if axis is None:
                    # Full reduction - max over all elements
                    # Create a mask where input == max value
                    max_val = result  # result is the max value
                    mask = (input_tensor == max_val)
                    # Count how many elements are equal to max (for tie-breaking)
                    # Convert mask to float for counting
                    count = mask.sum()  # This will be a scalar
                    # Normalize gradient by count (so if multiple elements are max, they share the gradient)
                    # grad_output / count * mask
                    return [grad_output * mask / count]
                else:
                    # Axis-specific reduction
                    # Unsqueeze result and grad_output to match input dimensions
                    if not keepdims:
                        max_val = result.unsqueeze(axis)
                        grad_unsqueezed = grad_output.unsqueeze(axis)
                    else:
                        max_val = result
                        grad_unsqueezed = grad_output
                    # Create mask where input == max value along axis
                    mask = (input_tensor == max_val)
                    # Count matches along the axis (for tie-breaking)
                    count = mask.sum(axis=axis, keepdims=True)
                    # Broadcast gradient through mask, normalized by count
                    grad_result = grad_unsqueezed * mask / count
                    return [grad_result]
            else:
                raise RuntimeError(f"Gradient not implemented for axis-aware {op}")

        graph.record(result, [input_tensor], grad_fn)

    return result


def _free_tensor(tensor_id: int):
    """Callback for garbage collection - queue for later processing."""
    if _client_connection is None or tensor_id is None:
        return

    # Just append to queue - no lock needed here since append is atomic
    # Queue will be processed by _process_free_queue() while holding connection lock
    _free_queue.append(tensor_id)


def _process_free_queue():
    """Process queued tensor frees. Must be called while holding _connection_lock."""
    from gt.transport.protocol import FreeTensor

    # Process all pending frees
    while _free_queue:
        tensor_id = _free_queue.popleft()
        try:
            cmd = FreeTensor(tensor_id=tensor_id)
            _client_connection.send(cmd)
            _client_connection.recv()  # Wait for response to maintain sync
        except:
            pass  # Connection might be closed


def _transpose_op(input_tensor: Tensor, dim0: int, dim1: int) -> Tensor:
    """Transpose (swap) two specific dimensions."""
    from gt.transport.protocol import ReshapeOp, ClientResponse
    from gt.client.autograd import get_graph
    import numpy as np

    if _client_connection is None:
        raise RuntimeError(not_connected_error())

    requires_grad = is_grad_enabled() and input_tensor.requires_grad
    result = Tensor(requires_grad=requires_grad)

    # Use ReshapeOp with "transpose_dims" operation
    cmd = ReshapeOp(
        result_id=result.id,
        op="transpose_dims",
        input_id=input_tensor.id,
        params=(dim0, dim1)
    )

    with _connection_lock:
        _process_free_queue()
        _client_connection.send(cmd)
        response: ClientResponse = _client_connection.recv()

    if not response.success:
        raise RuntimeError(f"transpose failed: {response.error}")

    # Compute result shape
    if input_tensor.shape:
        shape_list = list(input_tensor.shape)
        shape_list[dim0], shape_list[dim1] = shape_list[dim1], shape_list[dim0]
        result.shape = tuple(shape_list)
        result.dtype = input_tensor.dtype

    # Record on autograd tape if needed
    if requires_grad:
        graph = get_graph()

        def grad_fn(grad_output):
            # Gradient of transpose is transpose with swapped dims
            return [grad_output.transpose(dim0, dim1)]

        graph.record(result, [input_tensor], grad_fn)

    return result


def _permute_op(input_tensor: Tensor, dims: tuple) -> Tensor:
    """Permute (rearrange) dimensions."""
    from gt.transport.protocol import ReshapeOp, ClientResponse
    from gt.client.autograd import get_graph
    import numpy as np

    if _client_connection is None:
        raise RuntimeError(not_connected_error())

    requires_grad = is_grad_enabled() and input_tensor.requires_grad
    result = Tensor(requires_grad=requires_grad)

    # Use ReshapeOp with "permute" operation
    cmd = ReshapeOp(
        result_id=result.id,
        op="permute",
        input_id=input_tensor.id,
        params=dims
    )

    with _connection_lock:
        _process_free_queue()
        _client_connection.send(cmd)
        response: ClientResponse = _client_connection.recv()

    if not response.success:
        raise RuntimeError(f"permute failed: {response.error}")

    # Compute result shape
    if input_tensor.shape:
        result.shape = tuple(input_tensor.shape[i] for i in dims)
        result.dtype = input_tensor.dtype

    # Record on autograd tape if needed
    if requires_grad:
        graph = get_graph()

        def grad_fn(grad_output):
            # Gradient of permute is inverse permutation
            inverse_dims = [0] * len(dims)
            for i, d in enumerate(dims):
                inverse_dims[d] = i
            return [grad_output.permute(*inverse_dims)]

        graph.record(result, [input_tensor], grad_fn)

    return result


def cat(tensors, axis=0):
    """
    Concatenate tensors along an axis.

    Args:
        tensors: List of tensors to concatenate
        axis: Axis along which to concatenate (default: 0)

    Returns:
        Concatenated tensor

    Example:
        a = gt.randn(10, 20)
        b = gt.randn(10, 30)
        c = gt.cat([a, b], axis=1)  # Shape: (10, 50)
    """
    from gt.transport.protocol import ConcatOp, ClientResponse
    from gt.client.autograd import get_graph
    import numpy as np

    if _client_connection is None:
        raise RuntimeError(not_connected_error())

    if not tensors or len(tensors) == 0:
        raise ValueError("cat() requires at least one tensor")

    if len(tensors) == 1:
        return tensors[0]

    # Check if any tensor requires gradients
    requires_grad = is_grad_enabled() and any(t.requires_grad for t in tensors)

    result = Tensor(requires_grad=requires_grad)

    # Get tensor IDs
    tensor_ids = [t.id for t in tensors]

    cmd = ConcatOp(
        result_id=result.id,
        tensor_ids=tensor_ids,
        axis=axis
    )

    with _connection_lock:
        _process_free_queue()
        _client_connection.send(cmd)
        response: ClientResponse = _client_connection.recv()

    if not response.success:
        raise RuntimeError(f"cat failed: {response.error}")

    # Compute result shape
    if tensors[0].shape:
        shape_list = list(tensors[0].shape)
        # Sum up the sizes along the concatenation axis
        concat_size = sum(t.shape[axis] for t in tensors)
        shape_list[axis] = concat_size
        result.shape = tuple(shape_list)
        result.dtype = tensors[0].dtype

    # Record on autograd tape if needed
    if requires_grad:
        graph = get_graph()

        def grad_fn(grad_output):
            # Gradient of cat is split
            # Split gradient back to original sizes
            grads = []
            start = 0
            for t in tensors:
                size = t.shape[axis]
                slices = [slice(None)] * len(grad_output.shape)
                slices[axis] = slice(start, start + size)
                grads.append(grad_output[tuple(slices)])
                start += size
            return grads

        graph.record(result, tensors, grad_fn)

    return result


# Factory functions for creating tensors

def from_numpy(array: np.ndarray, requires_grad: bool = False) -> Tensor:
    """
    Create a tensor from a numpy array.

    NOTE: This is a sync point - flushes any pending batched operations.
    CreateTensor must execute immediately since subsequent ops may reference it.
    """
    from gt.transport.protocol import CreateTensor, ClientResponse

    if _client_connection is None:
        raise RuntimeError(not_connected_error())

    tensor = Tensor(shape=array.shape, dtype=str(array.dtype), requires_grad=requires_grad)

    cmd = CreateTensor(
        tensor_id=tensor.id,
        data=array,
        dtype=str(array.dtype),
        shape=array.shape
    )
    with _connection_lock:
        # Process any pending frees first
        _process_free_queue()

        _client_connection.send(cmd)
        response: ClientResponse = _client_connection.recv()

    if not response.success:
        raise RuntimeError(f"Failed to create tensor: {response.error}")

    return tensor


def randn(*shape, dtype="float32", requires_grad: bool = False) -> Tensor:
    """Create a random tensor with normal distribution."""
    from gt.transport.protocol import UnaryOp, ClientResponse

    if _client_connection is None:
        raise RuntimeError(not_connected_error())

    tensor = Tensor(shape=shape, dtype=dtype, requires_grad=requires_grad)

    cmd = UnaryOp(
        result_id=tensor.id,
        op="randn",
        input_id=None,
        shape=shape,
        dtype=dtype
    )
    with _connection_lock:
        # Process any pending frees first
        _process_free_queue()

        _client_connection.send(cmd)
        response: ClientResponse = _client_connection.recv()

    if not response.success:
        raise RuntimeError(f"Failed to create randn tensor: {response.error}")

    return tensor


def zeros(*shape, dtype="float32", requires_grad: bool = False) -> Tensor:
    """Create a tensor filled with zeros."""
    from gt.transport.protocol import UnaryOp, ClientResponse

    if _client_connection is None:
        raise RuntimeError(not_connected_error())

    tensor = Tensor(shape=shape, dtype=dtype, requires_grad=requires_grad)

    cmd = UnaryOp(
        result_id=tensor.id,
        op="zeros",
        input_id=None,
        shape=shape,
        dtype=dtype
    )
    with _connection_lock:
        # Process any pending frees first
        _process_free_queue()

        _client_connection.send(cmd)
        response: ClientResponse = _client_connection.recv()

    if not response.success:
        raise RuntimeError(f"Failed to create zeros tensor: {response.error}")

    return tensor


def ones(*shape, dtype="float32", requires_grad: bool = False) -> Tensor:
    """Create a tensor filled with ones."""
    from gt.transport.protocol import UnaryOp, ClientResponse

    if _client_connection is None:
        raise RuntimeError(not_connected_error())

    tensor = Tensor(shape=shape, dtype=dtype, requires_grad=requires_grad)

    cmd = UnaryOp(
        result_id=tensor.id,
        op="ones",
        input_id=None,
        shape=shape,
        dtype=dtype
    )
    with _connection_lock:
        # Process any pending frees first
        _process_free_queue()

        _client_connection.send(cmd)
        response: ClientResponse = _client_connection.recv()

    if not response.success:
        raise RuntimeError(f"Failed to create ones tensor: {response.error}")

    return tensor
