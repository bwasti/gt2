"""
Worker executes operations using an Engine (numpy, pytorch, jax).

Workers are simple command processors - they execute what dispatcher tells them.
All compilation logic is in the Engine, not the Worker.

Keep this SIMPLE and READABLE.
"""

import os
import numpy as np
from typing import Dict, Any
from gt.transport.connection import connect
from gt.transport.protocol import (
    WorkerCommand, WorkerCreateTensor, WorkerBinaryOp, WorkerUnaryOp, WorkerReshapeOp, WorkerSliceOp,
    WorkerGetData, WorkerFreeTensor, WorkerHotPathStart, WorkerHotPathEnd, WorkerGetStats, WorkerResponse
)
from gt.worker.engine import create_engine, Engine, Operation
from gt.worker.hotpath_detector import HotPathDetector
from gt.debug import debug_print_worker


class Worker:
    """
    Worker executes operations using an Engine backend.

    Supports multiple backends: 'numpy', 'pytorch'
    Processes instructions through optional hot path detector.
    """

    def __init__(self, worker_id: str, backend="pytorch"):
        self.worker_id = worker_id
        self.backend_name = backend
        self.tensors = {}  # Map: tensor_id -> tensor

        # Hot path detection configuration
        auto_compile = os.environ.get('GT_AUTO_COMPILE', '0') == '1'
        enable_compilation = auto_compile

        # Statistics tracking
        self.stats = {
            'operations': {
                'total': 0,
                'compiled': 0,
                'eager': 0,
            }
        }

        # Create engine
        self.engine: Engine = create_engine(backend, enable_compilation=enable_compilation)

        # Initialize hot path detector if enabled
        self.hotpath_detector = None
        if auto_compile:
            window_size = int(os.environ.get('GT_HOTPATH_WINDOW', '20'))
            hot_threshold = int(os.environ.get('GT_HOTPATH_THRESHOLD', '10'))
            min_seq_length = int(os.environ.get('GT_HOTPATH_MIN_SEQ', '3'))

            self.hotpath_detector = HotPathDetector(
                window_size=window_size,
                hot_threshold=hot_threshold,
                min_sequence_length=min_seq_length,
            )
            debug_print_worker(f"Worker {self.worker_id}: Hot path compilation enabled")
        else:
            debug_print_worker(f"Worker {self.worker_id}: Eager execution mode")

    def connect_to_dispatcher(self, dispatcher_host="localhost", dispatcher_port=9000):
        """Connect to dispatcher and start processing."""
        from gt.transport.protocol import RegisterWorker

        self.conn = connect(dispatcher_host, dispatcher_port)
        debug_print_worker(f"Worker {self.worker_id} connected to dispatcher")

        # Register with dispatcher
        reg_cmd = RegisterWorker(worker_id=self.worker_id)
        self.conn.send(reg_cmd)
        reg_response = self.conn.recv()
        if not reg_response.success:
            print(f"Worker {self.worker_id} failed to register: {reg_response.error}")
            return

        debug_print_worker(f"Worker {self.worker_id} registered successfully")

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
            # If hot path detection enabled, pass through transformer
            if self.hotpath_detector is not None:
                # Process command through detector (yields markers + original command)
                for output_cmd in self.hotpath_detector.process(cmd):
                    self._execute_command(output_cmd)
                return WorkerResponse(success=True)
            else:
                # Direct execution (no hot path detection)
                return self._execute_command(cmd)

        except Exception as e:
            return WorkerResponse(success=False, error=str(e))

    def _execute_command(self, cmd: WorkerCommand) -> WorkerResponse:
        """Execute a single command (after hot path processing)."""
        # Hot path markers
        if isinstance(cmd, WorkerHotPathStart):
            self.engine.handle_hotpath_start(cmd.sequence_id)
            return WorkerResponse(success=True)

        elif isinstance(cmd, WorkerHotPathEnd):
            self.engine.handle_hotpath_end(cmd.sequence_id)
            return WorkerResponse(success=True)

        # Non-compute commands
        elif isinstance(cmd, WorkerCreateTensor):
            return self._handle_create_tensor(cmd)

        elif isinstance(cmd, WorkerGetData):
            return self._handle_get_data(cmd)

        elif isinstance(cmd, WorkerFreeTensor):
            return self._handle_free_tensor(cmd)

        elif isinstance(cmd, WorkerGetStats):
            return self._handle_get_stats(cmd)

        # Compute operations
        elif isinstance(cmd, WorkerBinaryOp):
            return self._handle_binary_op(cmd)

        elif isinstance(cmd, WorkerUnaryOp):
            return self._handle_unary_op(cmd)

        elif isinstance(cmd, WorkerReshapeOp):
            return self._handle_reshape_op(cmd)

        elif isinstance(cmd, WorkerSliceOp):
            return self._handle_slice_op(cmd)

        else:
            return WorkerResponse(success=False, error=f"Unknown command: {type(cmd)}")

    def _handle_create_tensor(self, cmd: WorkerCreateTensor) -> WorkerResponse:
        """Create a tensor."""
        self.tensors[cmd.tensor_id] = self.engine.create_tensor(cmd.data)
        return WorkerResponse(success=True)

    def _handle_binary_op(self, cmd: WorkerBinaryOp) -> WorkerResponse:
        """Execute binary operation."""
        self.stats['operations']['total'] += 1

        # If engine is buffering (hot sequence), queue the operation
        # Don't validate inputs - they might be results from earlier buffered ops
        if hasattr(self.engine, 'is_buffering') and self.engine.is_buffering():
            op = Operation(
                op_type='binary',
                op_name=cmd.op,
                result_id=cmd.result_id,
                input_ids=[cmd.left_id, cmd.right_id]
            )
            self.engine.queue_operation(op, self.tensors)
            return WorkerResponse(success=True)

        # Otherwise execute eagerly - validate inputs
        if cmd.left_id not in self.tensors or cmd.right_id not in self.tensors:
            return WorkerResponse(success=False, error="Input tensor not found")

        left = self.tensors[cmd.left_id]
        right = self.tensors[cmd.right_id]

        if cmd.op == "add":
            result = self.engine.add(left, right)
        elif cmd.op == "sub":
            result = left - right
        elif cmd.op == "mul":
            result = self.engine.mul(left, right)
        elif cmd.op == "div":
            result = left / right
        elif cmd.op == "matmul":
            result = self.engine.matmul(left, right)
        elif cmd.op == "gt":
            result = (left > right)
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

        # Operations that create new data - execute eagerly (not buffered)
        if cmd.input_id is None:
            if cmd.op == "randn":
                result = self.engine.randn(cmd.shape, cmd.dtype)
            elif cmd.op == "zeros":
                result = self.engine.zeros(cmd.shape, cmd.dtype)
            elif cmd.op == "ones":
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
        # If engine is buffering (hot sequence), queue the operation
        # Don't validate inputs - they might be results from earlier buffered ops
        if hasattr(self.engine, 'is_buffering') and self.engine.is_buffering():
            params = {}
            if hasattr(cmd, 'axis') and cmd.axis is not None:
                params['axis'] = cmd.axis
            if hasattr(cmd, 'keepdims'):
                params['keepdims'] = cmd.keepdims

            op = Operation(
                op_type='unary',
                op_name=cmd.op,
                result_id=cmd.result_id,
                input_ids=[cmd.input_id],
                params=params if params else None
            )
            self.engine.queue_operation(op, self.tensors)
            return WorkerResponse(success=True)

        # Otherwise execute eagerly - validate inputs
        if cmd.input_id not in self.tensors:
            return WorkerResponse(success=False, error="Input tensor not found")

        input_tensor = self.tensors[cmd.input_id]

        if cmd.op == "exp":
            result = np.exp(input_tensor) if self.backend_name == "numpy" else input_tensor.exp()
        elif cmd.op == "log":
            result = np.log(input_tensor) if self.backend_name == "numpy" else input_tensor.log()
        elif cmd.op == "sum":
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

        # If engine is buffering (hot sequence), queue the operation
        # Don't validate inputs - they might be results from earlier buffered ops
        if hasattr(self.engine, 'is_buffering') and self.engine.is_buffering():
            op = Operation(
                op_type='reshape',
                op_name=cmd.op,
                result_id=cmd.result_id,
                input_ids=[cmd.input_id],
                params=cmd.params
            )
            self.engine.queue_operation(op, self.tensors)
            return WorkerResponse(success=True)

        # Otherwise execute eagerly - validate inputs
        if cmd.input_id not in self.tensors:
            return WorkerResponse(success=False, error="Input tensor not found")

        input_tensor = self.tensors[cmd.input_id]

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
                if self.backend_name == "pytorch":
                    result = input_tensor.squeeze()
                else:
                    result = np.squeeze(input_tensor)
            else:
                dim = cmd.params[0]
                if self.backend_name == "pytorch":
                    result = input_tensor.squeeze(dim)
                else:
                    result = np.squeeze(input_tensor, axis=dim)
        else:
            return WorkerResponse(success=False, error=f"Unknown reshape op: {cmd.op}")

        self.tensors[cmd.result_id] = result
        return WorkerResponse(success=True)

    def _handle_slice_op(self, cmd: WorkerSliceOp) -> WorkerResponse:
        """Execute slice operation (tensor subscripting)."""
        self.stats['operations']['total'] += 1

        # If engine is buffering (hot sequence), queue the operation
        # Don't validate inputs - they might be results from earlier buffered ops
        if hasattr(self.engine, 'is_buffering') and self.engine.is_buffering():
            op = Operation(
                op_type='slice',
                op_name='subscript',
                result_id=cmd.result_id,
                input_ids=[cmd.input_id],
                params=cmd.key
            )
            self.engine.queue_operation(op, self.tensors)
            return WorkerResponse(success=True)

        # Otherwise execute eagerly - validate inputs
        if cmd.input_id not in self.tensors:
            return WorkerResponse(success=False, error="Input tensor not found")

        input_tensor = self.tensors[cmd.input_id]

        try:
            result = input_tensor[cmd.key]
        except Exception as e:
            return WorkerResponse(success=False, error=f"Slice operation failed: {e}")

        self.tensors[cmd.result_id] = result
        return WorkerResponse(success=True)

    def _handle_get_data(self, cmd: WorkerGetData) -> WorkerResponse:
        """Get tensor data."""
        if cmd.tensor_id not in self.tensors:
            return WorkerResponse(success=False, error="Tensor not found")

        tensor = self.tensors[cmd.tensor_id]
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

        # Add hot path detection stats if enabled
        if self.hotpath_detector:
            stats['hotpath'] = self.hotpath_detector.get_stats()

        # Add compilation stats from engine
        if hasattr(self.engine, 'get_compilation_stats'):
            stats['compilation'] = self.engine.get_compilation_stats()

        return WorkerResponse(success=True, data=stats)
