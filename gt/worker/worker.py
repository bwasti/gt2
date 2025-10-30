"""
Worker executes operations using a backend (numpy, pytorch, jax).

Workers are dumb executors - they just run what dispatcher tells them.

Keep this SIMPLE and READABLE.
"""

import numpy as np
from gt.transport.connection import connect, Connection
from gt.transport.protocol import (
    WorkerCommand, WorkerCreateTensor, WorkerBinaryOp, WorkerUnaryOp,
    WorkerGetData, WorkerFreeTensor, WorkerResponse
)


class Worker:
    """
    Worker executes operations.

    Supports multiple backends: 'numpy', 'pytorch'
    """

    def __init__(self, worker_id: str, backend="pytorch"):
        self.worker_id = worker_id
        self.backend = backend
        self.tensors = {}  # Map: tensor_id -> tensor (numpy array or torch tensor)

        # Initialize backend
        if backend == "pytorch":
            try:
                import torch
                self.torch = torch
                self.device = torch.device('cpu')  # TODO: support GPU
            except ImportError:
                print(f"Warning: PyTorch not available, falling back to numpy")
                self.backend = "numpy"
                self.torch = None
        else:
            self.torch = None

    def connect_to_dispatcher(self, dispatcher_host="localhost", dispatcher_port=9000):
        """Connect to dispatcher and start processing."""
        self.conn = connect(dispatcher_host, dispatcher_port)
        print(f"Worker {self.worker_id} connected to dispatcher")

        # Process commands
        while True:
            try:
                cmd = self.conn.recv()
                response = self._process_command(cmd)
                self.conn.send(response)
            except Exception as e:
                print(f"Worker {self.worker_id} error: {e}")
                break

    def _process_command(self, cmd: WorkerCommand) -> WorkerResponse:
        """Process a command from dispatcher."""
        try:
            if isinstance(cmd, WorkerCreateTensor):
                return self._handle_create_tensor(cmd)
            elif isinstance(cmd, WorkerBinaryOp):
                return self._handle_binary_op(cmd)
            elif isinstance(cmd, WorkerUnaryOp):
                return self._handle_unary_op(cmd)
            elif isinstance(cmd, WorkerGetData):
                return self._handle_get_data(cmd)
            elif isinstance(cmd, WorkerFreeTensor):
                return self._handle_free_tensor(cmd)
            else:
                return WorkerResponse(success=False, error=f"Unknown command: {type(cmd)}")
        except Exception as e:
            return WorkerResponse(success=False, error=str(e))

    def _handle_create_tensor(self, cmd: WorkerCreateTensor) -> WorkerResponse:
        """Create a tensor."""
        if self.backend == "pytorch":
            # Convert numpy to torch
            self.tensors[cmd.tensor_id] = self.torch.from_numpy(cmd.data).to(self.device)
        else:
            self.tensors[cmd.tensor_id] = cmd.data
        return WorkerResponse(success=True)

    def _handle_binary_op(self, cmd: WorkerBinaryOp) -> WorkerResponse:
        """Execute binary operation."""
        if cmd.left_id not in self.tensors or cmd.right_id not in self.tensors:
            return WorkerResponse(success=False, error="Input tensor not found")

        left = self.tensors[cmd.left_id]
        right = self.tensors[cmd.right_id]

        # Execute operation
        if cmd.op == "add":
            result = left + right
        elif cmd.op == "sub":
            result = left - right
        elif cmd.op == "mul":
            result = left * right
        elif cmd.op == "div":
            result = left / right
        elif cmd.op == "matmul":
            result = left @ right
        elif cmd.op == "gt":
            if self.backend == "pytorch":
                result = (left > right).float()
            else:
                result = (left > right).astype(np.float32)
        else:
            return WorkerResponse(success=False, error=f"Unknown op: {cmd.op}")

        self.tensors[cmd.result_id] = result
        return WorkerResponse(success=True)

    def _handle_unary_op(self, cmd: WorkerUnaryOp) -> WorkerResponse:
        """Execute unary operation."""
        # Operations that create new data
        if cmd.input_id is None:
            if self.backend == "pytorch":
                if cmd.op == "randn":
                    result = self.torch.randn(*cmd.shape, device=self.device)
                elif cmd.op == "zeros":
                    result = self.torch.zeros(*cmd.shape, device=self.device)
                elif cmd.op == "ones":
                    result = self.torch.ones(*cmd.shape, device=self.device)
                else:
                    return WorkerResponse(success=False, error=f"Unknown creation op: {cmd.op}")
            else:
                if cmd.op == "randn":
                    result = np.random.randn(*cmd.shape).astype(cmd.dtype)
                elif cmd.op == "zeros":
                    result = np.zeros(cmd.shape, dtype=cmd.dtype)
                elif cmd.op == "ones":
                    result = np.ones(cmd.shape, dtype=cmd.dtype)
                else:
                    return WorkerResponse(success=False, error=f"Unknown creation op: {cmd.op}")

            self.tensors[cmd.result_id] = result
            return WorkerResponse(success=True)

        # Operations on existing tensors
        if cmd.input_id not in self.tensors:
            return WorkerResponse(success=False, error="Input tensor not found")

        input_tensor = self.tensors[cmd.input_id]

        if self.backend == "pytorch":
            if cmd.op == "exp":
                result = self.torch.exp(input_tensor)
            elif cmd.op == "log":
                result = self.torch.log(input_tensor)
            elif cmd.op == "sum":
                result = self.torch.sum(input_tensor)
            elif cmd.op == "mean":
                result = self.torch.mean(input_tensor)
            elif cmd.op == "relu":
                result = self.torch.relu(input_tensor)
            elif cmd.op == "sigmoid":
                result = self.torch.sigmoid(input_tensor)
            elif cmd.op == "tanh":
                result = self.torch.tanh(input_tensor)
            else:
                return WorkerResponse(success=False, error=f"Unknown op: {cmd.op}")
        else:
            if cmd.op == "exp":
                result = np.exp(input_tensor)
            elif cmd.op == "log":
                result = np.log(input_tensor)
            elif cmd.op == "sum":
                result = np.sum(input_tensor)
            elif cmd.op == "mean":
                result = np.mean(input_tensor)
            elif cmd.op == "relu":
                result = np.maximum(0, input_tensor)
            elif cmd.op == "sigmoid":
                result = 1.0 / (1.0 + np.exp(-input_tensor))
            elif cmd.op == "tanh":
                result = np.tanh(input_tensor)
            else:
                return WorkerResponse(success=False, error=f"Unknown op: {cmd.op}")

        self.tensors[cmd.result_id] = result
        return WorkerResponse(success=True)

    def _handle_get_data(self, cmd: WorkerGetData) -> WorkerResponse:
        """Get tensor data."""
        if cmd.tensor_id not in self.tensors:
            return WorkerResponse(success=False, error="Tensor not found")

        data = self.tensors[cmd.tensor_id]

        # Convert to numpy for transport
        if self.backend == "pytorch":
            data = data.cpu().numpy()

        return WorkerResponse(success=True, data=data)

    def _handle_free_tensor(self, cmd: WorkerFreeTensor) -> WorkerResponse:
        """Free a tensor."""
        if cmd.tensor_id in self.tensors:
            del self.tensors[cmd.tensor_id]
        return WorkerResponse(success=True)
