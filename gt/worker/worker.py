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
    WorkerCommand, WorkerCreateTensor, WorkerBinaryOp, WorkerUnaryOp,
    WorkerGetData, WorkerFreeTensor, WorkerCompileStart, WorkerCompileEnd, WorkerResponse
)
from gt.worker.engine import create_engine, Engine, Operation


class Worker:
    """
    Worker executes operations using an Engine backend.

    Supports multiple backends: 'numpy', 'pytorch'
    Supports instruction batching for compilation (PyTorch only).
    """

    def __init__(self, worker_id: str, backend="pytorch", batch_size: int = None):
        self.worker_id = worker_id
        self.backend_name = backend
        self.tensors = {}  # Map: tensor_id -> tensor

        # Instruction batching configuration
        if batch_size is None:
            # Read from environment variable or default to 1 (eager mode)
            batch_size = int(os.environ.get('GT_WORKER_BATCH_SIZE', '1'))
        self.batch_size = batch_size

        # Batch accumulator for compilation
        self.pending_operations: List[tuple] = []  # (cmd, response_placeholder)

        # Compilation region tracking
        self.in_compile_region = False
        self.current_compile_signal = None

        # Create engine
        self.engine: Engine = create_engine(backend)

        # Log batching mode
        if self.batch_size > 1 and self.engine.supports_batching():
            print(f"Worker {self.worker_id}: Batching enabled (batch_size={self.batch_size})")
        elif self.batch_size > 1 and not self.engine.supports_batching():
            print(f"Worker {self.worker_id}: Batching requested (batch_size={self.batch_size}) but engine doesn't support it")
        else:
            print(f"Worker {self.worker_id}: Eager mode (batch_size={self.batch_size})")

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
        """
        Process a command from dispatcher.

        Batching logic:
        - Batchable ops (BinaryOp, UnaryOp on existing tensors) are accumulated
        - Batch is flushed when full or sync point is hit
        - Sync points: CreateTensor, GetData, FreeTensor
        """
        try:
            # Determine if this is a batchable operation
            is_batchable = self._is_batchable(cmd)
            is_sync_point = not is_batchable

            # Flush batch if:
            # 1. Sync point reached, OR
            # 2. Batch is full, OR
            # 3. Engine doesn't support batching
            should_flush = (
                is_sync_point or
                len(self.pending_operations) >= self.batch_size or
                not self.engine.supports_batching()
            )

            if should_flush and self.pending_operations:
                self._flush_batch()

            # Handle the current command
            if isinstance(cmd, WorkerCreateTensor):
                return self._handle_create_tensor(cmd)
            elif isinstance(cmd, WorkerBinaryOp):
                if is_batchable and self.engine.supports_batching():
                    return self._add_to_batch(cmd)
                else:
                    return self._handle_binary_op(cmd)
            elif isinstance(cmd, WorkerUnaryOp):
                if is_batchable and self.engine.supports_batching():
                    return self._add_to_batch(cmd)
                else:
                    return self._handle_unary_op(cmd)
            elif isinstance(cmd, WorkerGetData):
                return self._handle_get_data(cmd)
            elif isinstance(cmd, WorkerFreeTensor):
                return self._handle_free_tensor(cmd)
            elif isinstance(cmd, WorkerCompileStart):
                return self._handle_compile_start(cmd)
            elif isinstance(cmd, WorkerCompileEnd):
                return self._handle_compile_end(cmd)
            else:
                return WorkerResponse(success=False, error=f"Unknown command: {type(cmd)}")
        except Exception as e:
            return WorkerResponse(success=False, error=str(e))

    def _is_batchable(self, cmd: WorkerCommand) -> bool:
        """Check if a command can be batched for compilation."""
        if isinstance(cmd, (WorkerBinaryOp,)):
            return True
        if isinstance(cmd, WorkerUnaryOp):
            # Only batch operations on existing tensors, not creation ops
            return cmd.input_id is not None
        return False

    def _add_to_batch(self, cmd: WorkerCommand) -> WorkerResponse:
        """
        Add an operation to the pending batch.
        Returns success immediately - actual execution is deferred.
        """
        self.pending_operations.append(cmd)

        # Check if batch is now full
        if len(self.pending_operations) >= self.batch_size:
            self._flush_batch()

        return WorkerResponse(success=True)

    def _flush_batch(self):
        """
        Execute all pending operations as a batch using engine.execute_batch().
        """
        if not self.pending_operations:
            return

        # Convert commands to Operation objects
        operations = []
        for cmd in self.pending_operations:
            if isinstance(cmd, WorkerBinaryOp):
                operations.append(Operation(
                    op_type='binary',
                    op_name=cmd.op,
                    result_id=cmd.result_id,
                    input_ids=[cmd.left_id, cmd.right_id],
                    params={}
                ))
            elif isinstance(cmd, WorkerUnaryOp):
                operations.append(Operation(
                    op_type='unary',
                    op_name=cmd.op,
                    result_id=cmd.result_id,
                    input_ids=[cmd.input_id],
                    params={'shape': cmd.shape, 'dtype': cmd.dtype}
                ))

        # Execute batch
        try:
            results = self.engine.execute_batch(operations, self.tensors)

            # Update tensor storage with results
            for result_id, tensor in results.items():
                self.tensors[result_id] = tensor

        except Exception as e:
            print(f"Worker {self.worker_id}: Batch execution failed: {e}")
            raise
        finally:
            # Clear pending operations
            self.pending_operations.clear()

    def _handle_create_tensor(self, cmd: WorkerCreateTensor) -> WorkerResponse:
        """Create a tensor."""
        self.tensors[cmd.tensor_id] = self.engine.create_tensor(cmd.data)
        return WorkerResponse(success=True)

    def _handle_binary_op(self, cmd: WorkerBinaryOp) -> WorkerResponse:
        """Execute binary operation."""
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
            result = self.engine.sum(input_tensor)
        elif cmd.op == "mean":
            result = self.engine.mean(input_tensor)
        elif cmd.op == "relu":
            result = self.engine.relu(input_tensor)
        elif cmd.op == "sigmoid":
            result = self.engine.sigmoid(input_tensor)
        elif cmd.op == "tanh":
            result = self.engine.tanh(input_tensor)
        elif cmd.op == "transpose":
            result = self.engine.transpose(input_tensor)
        else:
            return WorkerResponse(success=False, error=f"Unknown op: {cmd.op}")

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

    def _handle_compile_start(self, cmd: WorkerCompileStart) -> WorkerResponse:
        """Handle compilation region start."""
        self.in_compile_region = True
        self.current_compile_signal = cmd.signal_name
        return WorkerResponse(success=True)

    def _handle_compile_end(self, cmd: WorkerCompileEnd) -> WorkerResponse:
        """Handle compilation region end - flush any pending operations."""
        if cmd.signal_name != self.current_compile_signal:
            return WorkerResponse(success=False,
                error=f"Signal mismatch: expected {self.current_compile_signal}, got {cmd.signal_name}")

        if self.pending_operations:
            self._flush_batch()

        self.in_compile_region = False
        self.current_compile_signal = None
        return WorkerResponse(success=True)
