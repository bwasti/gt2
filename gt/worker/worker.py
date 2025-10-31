"""
Worker executes operations using an Engine (numpy, pytorch, jax).

Workers are dumb executors - they just run what dispatcher tells them.

Keep this SIMPLE and READABLE.
"""

import os
import numpy as np
from typing import List, Dict, Any
from gt.transport.connection import connect, Connection
from gt.transport.protocol import (
    WorkerCommand, WorkerCreateTensor, WorkerBinaryOp, WorkerUnaryOp, WorkerReshapeOp,
    WorkerGetData, WorkerFreeTensor, WorkerCompileStart, WorkerCompileEnd, WorkerGetStats, WorkerResponse
)
from gt.worker.engine import create_engine, Engine, Operation


class Worker:
    """
    Worker executes operations using an Engine backend.

    Supports multiple backends: 'numpy', 'pytorch'
    Processes instructions one at a time (no batching).
    """

    def __init__(self, worker_id: str, backend="pytorch"):
        self.worker_id = worker_id
        self.backend_name = backend
        self.tensors = {}  # Map: tensor_id -> tensor

        # Compilation disabled for now (has bugs)
        enable_compilation = False

        # Statistics tracking
        self.stats = {
            'operations': {
                'total': 0,
            }
        }

        # Create engine
        self.engine: Engine = create_engine(backend, enable_compilation=enable_compilation)

        print(f"Worker {self.worker_id}: Stream processing mode")

    def connect_to_dispatcher(self, dispatcher_host="localhost", dispatcher_port=9000):
        """Connect to dispatcher and start processing."""
        from gt.transport.protocol import RegisterWorker

        self.conn = connect(dispatcher_host, dispatcher_port)
        print(f"Worker {self.worker_id} connected to dispatcher")

        # Register with dispatcher
        reg_cmd = RegisterWorker(worker_id=self.worker_id)
        self.conn.send(reg_cmd)
        reg_response = self.conn.recv()
        if not reg_response.success:
            print(f"Worker {self.worker_id} failed to register: {reg_response.error}")
            return

        print(f"Worker {self.worker_id} registered successfully")

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
        """Process a command from dispatcher (one at a time, no batching)."""
        try:
            # Execute command immediately
            if isinstance(cmd, WorkerCreateTensor):
                return self._handle_create_tensor(cmd)
            elif isinstance(cmd, WorkerBinaryOp):
                return self._handle_binary_op(cmd)
            elif isinstance(cmd, WorkerUnaryOp):
                return self._handle_unary_op(cmd)
            elif isinstance(cmd, WorkerReshapeOp):
                return self._handle_reshape_op(cmd)
            elif isinstance(cmd, WorkerGetData):
                return self._handle_get_data(cmd)
            elif isinstance(cmd, WorkerFreeTensor):
                return self._handle_free_tensor(cmd)
            elif isinstance(cmd, WorkerGetStats):
                return self._handle_get_stats(cmd)
            else:
                return WorkerResponse(success=False, error=f"Unknown command: {type(cmd)}")
        except Exception as e:
            return WorkerResponse(success=False, error=str(e))

    def _handle_create_tensor(self, cmd: WorkerCreateTensor) -> WorkerResponse:
        """Create a tensor."""
        self.tensors[cmd.tensor_id] = self.engine.create_tensor(cmd.data)
        return WorkerResponse(success=True)

    def _handle_binary_op(self, cmd: WorkerBinaryOp) -> WorkerResponse:
        """Execute binary operation."""
        self.stats['operations']['total'] += 1

        if cmd.left_id not in self.tensors or cmd.right_id not in self.tensors:
            return WorkerResponse(success=False, error="Input tensor not found")

        left = self.tensors[cmd.left_id]
        right = self.tensors[cmd.right_id]

        # Execute operation using engine
        if cmd.op == "add":
            result = self.engine.add(left, right)
        elif cmd.op == "sub":
            result = left - right  # TODO: add to engine
        elif cmd.op == "mul":
            result = self.engine.mul(left, right)
        elif cmd.op == "div":
            result = left / right  # TODO: add to engine
        elif cmd.op == "matmul":
            result = self.engine.matmul(left, right)
        elif cmd.op == "gt":
            result = (left > right)  # TODO: add to engine
            if self.backend_name == "pytorch":
                result = result.float()
            else:
                result = result.astype(np.float32)
        else:
            return WorkerResponse(success=False, error=f"Unknown op: {cmd.op}")

        self.tensors[cmd.result_id] = result
        return WorkerResponse(success=True)

    def _handle_unary_op(self, cmd: WorkerUnaryOp) -> WorkerResponse:
        """Execute unary operation."""
        self.stats['operations']['total'] += 1

        # Operations that create new data
        if cmd.input_id is None:
            if cmd.op == "randn":
                result = self.engine.randn(cmd.shape, cmd.dtype)
            elif cmd.op == "zeros":
                result = self.engine.zeros(cmd.shape, cmd.dtype)
            elif cmd.op == "ones":
                # TODO: add ones() to engine
                if self.backend_name == "pytorch":
                    import torch
                    result = torch.ones(*cmd.shape)
                else:
                    result = np.ones(cmd.shape, dtype=cmd.dtype)
            else:
                return WorkerResponse(success=False, error=f"Unknown creation op: {cmd.op}")

            self.tensors[cmd.result_id] = result
            return WorkerResponse(success=True)

        # Operations on existing tensors
        if cmd.input_id not in self.tensors:
            return WorkerResponse(success=False, error="Input tensor not found")

        input_tensor = self.tensors[cmd.input_id]

        if cmd.op == "exp":
            result = np.exp(input_tensor) if self.backend_name == "numpy" else input_tensor.exp()  # TODO: add to engine
        elif cmd.op == "log":
            result = np.log(input_tensor) if self.backend_name == "numpy" else input_tensor.log()  # TODO: add to engine
        elif cmd.op == "sum":
            # Pass axis and keepdims for axis-aware reductions
            axis = getattr(cmd, 'axis', None)
            keepdims = getattr(cmd, 'keepdims', False)
            result = self.engine.sum(input_tensor, axis=axis, keepdims=keepdims)
        elif cmd.op == "mean":
            axis = getattr(cmd, 'axis', None)
            keepdims = getattr(cmd, 'keepdims', False)
            result = self.engine.mean(input_tensor, axis=axis, keepdims=keepdims)
        elif cmd.op == "relu":
            result = self.engine.relu(input_tensor)
        elif cmd.op == "sigmoid":
            result = self.engine.sigmoid(input_tensor)
        elif cmd.op == "tanh":
            result = self.engine.tanh(input_tensor)
        elif cmd.op == "sqrt":
            result = self.engine.sqrt(input_tensor)
        elif cmd.op == "transpose":
            result = self.engine.transpose(input_tensor)
        else:
            return WorkerResponse(success=False, error=f"Unknown op: {cmd.op}")

        self.tensors[cmd.result_id] = result
        return WorkerResponse(success=True)

    def _handle_reshape_op(self, cmd: WorkerReshapeOp) -> WorkerResponse:
        """Execute reshape operation (reshape, unsqueeze, squeeze)."""
        self.stats['operations']['total'] += 1

        if cmd.input_id not in self.tensors:
            return WorkerResponse(success=False, error="Input tensor not found")

        input_tensor = self.tensors[cmd.input_id]

        # Execute reshape operation using PyTorch/NumPy primitives
        if cmd.op == "reshape":
            new_shape = cmd.params
            if self.backend_name == "pytorch":
                result = input_tensor.reshape(*new_shape)
            else:
                result = input_tensor.reshape(new_shape)
        elif cmd.op == "unsqueeze":
            dim = cmd.params[0]
            if self.backend_name == "pytorch":
                result = input_tensor.unsqueeze(dim)
            else:
                result = np.expand_dims(input_tensor, axis=dim)
        elif cmd.op == "squeeze":
            if len(cmd.params) == 0:
                # Squeeze all dimensions of size 1
                if self.backend_name == "pytorch":
                    result = input_tensor.squeeze()
                else:
                    result = np.squeeze(input_tensor)
            else:
                # Squeeze specific dimension
                dim = cmd.params[0]
                if self.backend_name == "pytorch":
                    result = input_tensor.squeeze(dim)
                else:
                    result = np.squeeze(input_tensor, axis=dim)
        else:
            return WorkerResponse(success=False, error=f"Unknown reshape op: {cmd.op}")

        self.tensors[cmd.result_id] = result
        return WorkerResponse(success=True)

    def _handle_get_data(self, cmd: WorkerGetData) -> WorkerResponse:
        """Get tensor data."""
        if cmd.tensor_id not in self.tensors:
            return WorkerResponse(success=False, error="Tensor not found")

        tensor = self.tensors[cmd.tensor_id]

        # Convert to numpy for transport
        data = self.engine.to_numpy(tensor)

        return WorkerResponse(success=True, data=data)

    def _handle_free_tensor(self, cmd: WorkerFreeTensor) -> WorkerResponse:
        """Free a tensor."""
        if cmd.tensor_id in self.tensors:
            del self.tensors[cmd.tensor_id]
        return WorkerResponse(success=True)

    def _handle_get_stats(self, cmd: WorkerGetStats) -> WorkerResponse:
        """Handle stats request."""
        stats = {
            'operations': self.stats['operations'].copy()
        }
        return WorkerResponse(success=True, data=stats)
