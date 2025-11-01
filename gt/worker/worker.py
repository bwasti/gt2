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
    WorkerCommand, WorkerCreateTensor, WorkerBinaryOp, WorkerUnaryOp, WorkerReshapeOp, WorkerSliceOp,
    WorkerGetData, WorkerFreeTensor, WorkerCompileStart, WorkerCompileEnd, WorkerGetStats, WorkerResponse
)
from gt.worker.engine import create_engine, Engine, Operation
from gt.worker.hotpath_detector import HotPathDetector
from gt.debug import debug_print_worker, debug_print_compile


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

        # Hot path detection configuration
        auto_compile = os.environ.get('GT_AUTO_COMPILE', '0') == '1'
        enable_compilation = auto_compile  # Enable compilation if auto-compile is on

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
        self.hot_sequence_buffer = []  # Buffer operations during hot sequence
        self.in_hot_sequence = False
        self.hot_sequence_remaining = 0

        if auto_compile:
            window_size = int(os.environ.get('GT_HOTPATH_WINDOW', '20'))
            hot_threshold = int(os.environ.get('GT_HOTPATH_THRESHOLD', '10'))
            min_seq_length = int(os.environ.get('GT_HOTPATH_MIN_SEQ', '3'))

            self.hotpath_detector = HotPathDetector(
                window_size=window_size,
                hot_threshold=hot_threshold,
                min_sequence_length=min_seq_length,
            )
            debug_print_compile(f"Worker {self.worker_id}: Hot path compilation enabled (threshold={hot_threshold})")
        else:
            debug_print_worker(f"Worker {self.worker_id}: Stream processing mode")

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
            print(f"Worker {self.worker_id} failed to register: {reg_response.error}")  # Keep error visible
            return

        debug_print_worker(f"Worker {self.worker_id} registered successfully")

        # Process commands
        while True:
            try:
                cmd = self.conn.recv()
                response = self._process_command(cmd)
                self.conn.send(response)
            except Exception as e:
                print(f"Worker {self.worker_id} error: {e}")  # Keep errors visible
                break

    def _process_command(self, cmd: WorkerCommand) -> WorkerResponse:
        """Process a command from dispatcher (one at a time, no batching)."""
        try:
            # Record instruction in hot path detector (before execution)
            is_hot = False
            if self.hotpath_detector and self._is_compute_op(cmd):
                is_hot, seq_length = self._record_in_hotpath_detector(cmd)

                # Start buffering if hot sequence detected
                if is_hot and not self.in_hot_sequence:
                    self.in_hot_sequence = True
                    self.hot_sequence_remaining = seq_length
                    self.hot_sequence_buffer = []
                    debug_print_compile(f"Starting hot sequence (length={seq_length})")

            # Sync operations always flush and execute immediately
            if isinstance(cmd, WorkerGetData):
                # Flush any buffered operations
                if self.in_hot_sequence:
                    self._execute_hot_sequence_buffer()
                if self.hotpath_detector:
                    self.hotpath_detector.reset_sequence()
                return self._handle_get_data(cmd)
            elif isinstance(cmd, WorkerFreeTensor):
                # CRITICAL: Don't free tensors while buffering!
                # The buffer might reference this tensor. Flush first.
                if self.in_hot_sequence:
                    self._execute_hot_sequence_buffer()
                return self._handle_free_tensor(cmd)
            elif isinstance(cmd, WorkerGetStats):
                return self._handle_get_stats(cmd)
            elif isinstance(cmd, WorkerCreateTensor):
                return self._handle_create_tensor(cmd)

            # Compute operations: buffer if in hot sequence, otherwise execute
            if self.in_hot_sequence and self._is_compute_op(cmd):
                # Check if all inputs are available
                input_ids = self._get_input_ids(cmd)
                all_inputs_available = all(
                    tid in self.tensors or any(
                        bcmd for bcmd in self.hot_sequence_buffer
                        if self._get_result_id(bcmd) == tid
                    )
                    for tid in input_ids
                )

                # If inputs not available, flush buffer and execute eagerly
                if not all_inputs_available:
                    debug_print_compile("Input missing, flushing buffer and executing eagerly")
                    self._execute_hot_sequence_buffer()
                    # Execute this operation eagerly
                    return self._execute_op_eagerly(cmd)

                # Buffer the operation
                self.hot_sequence_buffer.append(cmd)
                self.hot_sequence_remaining -= 1

                # Execute batch when sequence completes
                if self.hot_sequence_remaining <= 0:
                    self._execute_hot_sequence_buffer()

                return WorkerResponse(success=True)

            # Not in hot sequence - execute immediately (eager mode)
            if isinstance(cmd, WorkerBinaryOp):
                return self._handle_binary_op(cmd)
            elif isinstance(cmd, WorkerUnaryOp):
                return self._handle_unary_op(cmd)
            elif isinstance(cmd, WorkerReshapeOp):
                return self._handle_reshape_op(cmd)
            elif isinstance(cmd, WorkerSliceOp):
                return self._handle_slice_op(cmd)
            else:
                return WorkerResponse(success=False, error=f"Unknown command: {type(cmd)}")

        except Exception as e:
            # Reset hot sequence on error
            self.in_hot_sequence = False
            self.hot_sequence_buffer = []
            return WorkerResponse(success=False, error=str(e))

    def _is_compute_op(self, cmd: WorkerCommand) -> bool:
        """Check if command is a compute operation (vs control/data ops)."""
        return isinstance(cmd, (WorkerBinaryOp, WorkerUnaryOp, WorkerReshapeOp, WorkerSliceOp))

    def _get_input_ids(self, cmd: WorkerCommand) -> list:
        """Get input tensor IDs from a command."""
        if isinstance(cmd, WorkerBinaryOp):
            return [cmd.left_id, cmd.right_id]
        elif isinstance(cmd, WorkerUnaryOp):
            return [cmd.input_id] if cmd.input_id else []
        elif isinstance(cmd, WorkerReshapeOp):
            return [cmd.input_id]
        elif isinstance(cmd, WorkerSliceOp):
            return [cmd.input_id]
        return []

    def _get_result_id(self, cmd: WorkerCommand) -> str:
        """Get result tensor ID from a command."""
        if hasattr(cmd, 'result_id'):
            return cmd.result_id
        return None

    def _execute_op_eagerly(self, cmd: WorkerCommand):
        """Execute a single operation eagerly (no buffering)."""
        if isinstance(cmd, WorkerBinaryOp):
            return self._handle_binary_op(cmd)
        elif isinstance(cmd, WorkerUnaryOp):
            return self._handle_unary_op(cmd)
        elif isinstance(cmd, WorkerReshapeOp):
            return self._handle_reshape_op(cmd)
        elif isinstance(cmd, WorkerSliceOp):
            return self._handle_slice_op(cmd)
        return WorkerResponse(success=False, error=f"Unknown command: {type(cmd)}")

    def _record_in_hotpath_detector(self, cmd: WorkerCommand):
        """Record operation in hot path detector."""
        if isinstance(cmd, WorkerBinaryOp):
            return self.hotpath_detector.record_instruction(
                'binary',
                cmd.op,
                result_id=cmd.result_id,
                input_ids=[cmd.left_id, cmd.right_id]
            )
        elif isinstance(cmd, WorkerUnaryOp):
            if cmd.input_id is not None:  # Only record compute ops
                return self.hotpath_detector.record_instruction(
                    'unary',
                    cmd.op,
                    result_id=cmd.result_id,
                    input_ids=[cmd.input_id]
                )
        elif isinstance(cmd, WorkerReshapeOp):
            return self.hotpath_detector.record_instruction(
                'reshape',
                cmd.op,
                result_id=cmd.result_id,
                input_ids=[cmd.input_id]
            )
        elif isinstance(cmd, WorkerSliceOp):
            return self.hotpath_detector.record_instruction(
                'slice',
                'subscript',  # Generic name for slice operations
                result_id=cmd.result_id,
                input_ids=[cmd.input_id]
            )
        return False, None

    def _execute_hot_sequence_buffer(self):
        """Execute buffered operations as a compiled batch."""
        if not self.hot_sequence_buffer:
            return

        debug_print_compile(f"Compiling and executing {len(self.hot_sequence_buffer)} operations")

        # Convert commands to Operation objects
        operations = []
        for cmd in self.hot_sequence_buffer:
            if isinstance(cmd, WorkerBinaryOp):
                operations.append(Operation(
                    op_type='binary',
                    op_name=cmd.op,
                    result_id=cmd.result_id,
                    input_ids=[cmd.left_id, cmd.right_id]
                ))
            elif isinstance(cmd, WorkerUnaryOp):
                params = {}
                if hasattr(cmd, 'axis') and cmd.axis is not None:
                    params['axis'] = cmd.axis
                if hasattr(cmd, 'keepdims'):
                    params['keepdims'] = cmd.keepdims
                if hasattr(cmd, 'shape') and cmd.shape is not None:
                    params['shape'] = cmd.shape
                if hasattr(cmd, 'dtype') and cmd.dtype is not None:
                    params['dtype'] = cmd.dtype

                operations.append(Operation(
                    op_type='unary',
                    op_name=cmd.op,
                    result_id=cmd.result_id,
                    input_ids=[cmd.input_id] if cmd.input_id else [],
                    params=params if params else None
                ))
            elif isinstance(cmd, WorkerReshapeOp):
                operations.append(Operation(
                    op_type='reshape',
                    op_name=cmd.op,
                    result_id=cmd.result_id,
                    input_ids=[cmd.input_id],
                    params=cmd.params
                ))
            elif isinstance(cmd, WorkerSliceOp):
                operations.append(Operation(
                    op_type='slice',
                    op_name='subscript',
                    result_id=cmd.result_id,
                    input_ids=[cmd.input_id],
                    params=cmd.key
                ))

        # Execute as compiled batch
        try:
            results = self.engine.execute_batch(operations, self.tensors)

            # Store results
            for tensor_id, tensor_value in results.items():
                self.tensors[tensor_id] = tensor_value

            self.stats['operations']['compiled'] += len(operations)
            debug_print_compile(f"Successfully compiled {len(operations)} ops")

        except Exception as e:
            debug_print_compile(f"Compilation failed, falling back to eager: {e}")
            # Fall back to eager execution
            for cmd in self.hot_sequence_buffer:
                if isinstance(cmd, WorkerBinaryOp):
                    self._handle_binary_op(cmd)
                elif isinstance(cmd, WorkerUnaryOp):
                    self._handle_unary_op(cmd)
                elif isinstance(cmd, WorkerReshapeOp):
                    self._handle_reshape_op(cmd)
                elif isinstance(cmd, WorkerSliceOp):
                    self._handle_slice_op(cmd)

            self.stats['operations']['eager'] += len(operations)

        # Reset buffer
        self.in_hot_sequence = False
        self.hot_sequence_buffer = []
        self.hot_sequence_remaining = 0

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

    def _handle_slice_op(self, cmd: WorkerSliceOp) -> WorkerResponse:
        """Execute slice operation (tensor subscripting)."""
        self.stats['operations']['total'] += 1

        if cmd.input_id not in self.tensors:
            return WorkerResponse(success=False, error="Input tensor not found")

        input_tensor = self.tensors[cmd.input_id]

        # Execute slice operation - Python slicing works the same for PyTorch/NumPy
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

        # Add hot path detection stats if enabled
        if self.hotpath_detector:
            stats['hotpath'] = self.hotpath_detector.get_stats()

        return WorkerResponse(success=True, data=stats)
