"""
Dispatcher coordinates between clients and workers.

Handles multiple clients and schedules operations to workers.

Keep this SIMPLE and READABLE.
"""

import threading
import time
from typing import Optional
from gt.transport.connection import create_server, Connection
from gt.debug import debug_print_dispatcher
from gt.errors import no_workers_error
from gt.transport.protocol import (
    ClientCommand, CreateTensor, BinaryOp, UnaryOp, ReshapeOp, SliceOp, GetData, FreeTensor, CopyTensor,
    CompileStart, CompileEnd, GetWorkerStats, RegisterWorker,
    ClientResponse, WorkerCreateTensor, WorkerBinaryOp, WorkerUnaryOp, WorkerReshapeOp, WorkerSliceOp,
    WorkerGetData, WorkerFreeTensor, WorkerCompileStart, WorkerCompileEnd, WorkerGetStats, WorkerResponse,
    serialize, deserialize
)
from gt.dispatcher.tensor_handle import TensorHandle


class InstructionStream:
    """Records all client-dispatcher-worker instructions with timestamps for debugging and monitoring."""

    def __init__(self, log_file: str = None, console: bool = True, monitor_socket=None):
        self.entries = []
        self.lock = threading.Lock()
        self.start_time = time.time()
        self.sequence = 0
        self.console = console
        self.log_file = log_file
        self.file_handle = None
        self.monitor_socket = monitor_socket  # ZMQ PUB socket for live monitoring

        # Open log file if specified
        if self.log_file:
            try:
                self.file_handle = open(self.log_file, 'w', buffering=1)  # Line buffered
                self._write_header()
            except Exception as e:
                print(f"Warning: Could not open instruction log file {self.log_file}: {e}")
                self.file_handle = None

    def _write_header(self):
        """Write header to log file."""
        if self.file_handle:
            import datetime
            header = f"""
{'='*100}
GT2 Dispatcher Instruction Stream
Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*100}

This log records all instructions flowing through the dispatcher:
  - Client commands (RECV/SEND)
  - Worker operations (WORKER_SEND/WORKER_RECV)
  - Connection events (CONNECT/DISCONNECT)

Format: <elapsed> | #<seq> | <event_type> | <source> | <command> | <details>

Column Details:
  elapsed  = Seconds since dispatcher start (with millisecond precision)
  seq      = Sequence number (increments for each instruction)
  event    = Event type (CONNECT, RECV, SEND, WORKER_SEND, WORKER_RECV, DISCONNECT, ERROR)
  source   = Source identifier (CLIENT/WORKER with ID)
  command  = Command type (e.g., CreateTensor, BinaryOp, UnaryOp)
  details  = Additional context (tensor IDs, operation names, etc.)

""".lstrip()
            self.file_handle.write(header)
            self.file_handle.flush()

    def log(self, event_type: str, client_id: str, command_type: str, details: str = "", size_bytes: int = 0):
        """Log an operation event with message size."""
        with self.lock:
            elapsed = time.time() - self.start_time
            self.sequence += 1

            entry = {
                "seq": self.sequence,
                "timestamp": elapsed,
                "event": event_type,
                "client": client_id,
                "command": command_type,
                "details": details,
                "size_bytes": size_bytes
            }
            self.entries.append(entry)

            # Format the log line
            log_line = self._format_log_entry(entry)

            # Print to console if enabled
            if self.console:
                print(log_line)

            # Write to file if enabled
            if self.file_handle:
                self.file_handle.write(log_line + "\n")
                self.file_handle.flush()

            # Broadcast to monitoring clients (non-blocking)
            if self.monitor_socket:
                try:
                    import zmq
                    import json
                    msg = json.dumps(entry).encode('utf-8')
                    self.monitor_socket.send(msg, flags=zmq.NOBLOCK)
                except Exception as e:
                    # Debug: print first error only to avoid spam
                    if not hasattr(self, '_broadcast_error_shown'):
                        print(f"Warning: Monitor broadcast failed: {e}")
                        self._broadcast_error_shown = True

    def _format_log_entry(self, entry: dict) -> str:
        """Format a log entry for output."""
        elapsed = entry["timestamp"]
        seq = entry["seq"]
        event = entry["event"]
        client = entry["client"]
        command = entry["command"]
        details = entry["details"]
        size_bytes = entry.get("size_bytes", 0)

        # Determine source type
        if "WORKER" in event:
            source_type = "WORKER"
        elif event in ["CONNECT", "DISCONNECT"]:
            source_type = "CLIENT"
        else:
            source_type = "CLIENT"

        # Format size nicely
        if size_bytes == 0:
            size_str = ""
        elif size_bytes < 1024:
            size_str = f"{size_bytes}B"
        elif size_bytes < 1024 * 1024:
            size_str = f"{size_bytes / 1024:.1f}KB"
        else:
            size_str = f"{size_bytes / (1024 * 1024):.2f}MB"

        # Format with nice alignment
        # Format: "  0.123s | #0042 | RECV         | CLIENT 127.0.0.1:12345 | BinaryOp        | 123KB | result=42 op=add"
        time_str = f"{elapsed:7.3f}s"
        seq_str = f"#{seq:04d}"
        event_str = f"{event:12s}"
        source_str = f"{source_type} {client:20s}"
        command_str = f"{command:15s}"

        # Build the line with size
        if size_str:
            size_field = f"{size_str:>10s}"
            if details:
                line = f"{time_str} | {seq_str} | {event_str} | {source_str} | {command_str} | {size_field} | {details}"
            else:
                line = f"{time_str} | {seq_str} | {event_str} | {source_str} | {command_str} | {size_field}"
        else:
            if details:
                line = f"{time_str} | {seq_str} | {event_str} | {source_str} | {command_str} | {details}"
            else:
                line = f"{time_str} | {seq_str} | {event_str} | {source_str} | {command_str}"

        return line

    def dump(self):
        """Dump all recorded entries."""
        with self.lock:
            return list(self.entries)

    def close(self):
        """Close the log file."""
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None


class Dispatcher:
    """
    Dispatcher coordinates multiple clients and workers.

    Takes commands from clients and schedules them to workers.
    """

    def __init__(self, host="localhost", port=9000, log_file: str = None, console_log: bool = True, enable_sharding: bool = False, enable_monitoring: bool = True):
        self.host = host
        self.port = port
        self.tensor_handles = TensorHandle()
        self.workers = []  # List of worker connections
        self.worker_identities = set()  # Track ZMQ identities of workers
        self.next_worker_idx = 0  # Simple round-robin scheduling
        self.running = False
        self.server_socket = None
        self.monitor_socket = None

        # Create monitoring PUB socket (non-blocking broadcasts)
        if enable_monitoring:
            try:
                import zmq
                context = zmq.Context()
                self.monitor_socket = context.socket(zmq.PUB)
                # Bind to any available port (0 means OS will choose)
                if host in ('localhost', '127.0.0.1', '0.0.0.0'):
                    # Use IPC for localhost
                    monitor_addr = f"ipc:///tmp/gt_monitor_{port}.ipc"
                else:
                    # Use TCP with auto port selection
                    monitor_addr = f"tcp://{host}:0"
                self.monitor_socket.bind(monitor_addr)
            except Exception as e:
                print(f"Warning: Could not create monitoring socket: {e}")
                self.monitor_socket = None

        self.instruction_stream = InstructionStream(log_file=log_file, console=console_log, monitor_socket=self.monitor_socket)  # Record all instructions for debugging/monitoring

        # Per-client signal stacks (track current signal scope for each client)
        self.client_signal_stacks = {}  # client_id -> list of signal names (stack)
        self.signal_stack_lock = threading.Lock()

        # Per-worker command buffers for batching
        self.worker_buffers = {}  # worker_id -> list of pending commands
        self.buffer_lock = threading.Lock()
        self.max_batch_size = 50  # Flush buffer when it reaches this size

        # Stream modifiers (signal-driven sharding)
        from gt.dispatcher.sharding_modifier import ShardingStreamModifier
        self.sharding_modifier = ShardingStreamModifier()

    def register_worker(self, worker_identity: bytes, worker_id: str):
        """Register a worker by its ZMQ identity."""
        self.workers.append({
            "id": worker_id,
            "identity": worker_identity
        })
        self.worker_identities.add(worker_identity)  # Track for message routing
        debug_print_dispatcher(f"Worker {worker_id} registered")

    def _get_current_signal(self, client_id: str) -> Optional[str]:
        """Get the current signal scope for a client (top of stack)."""
        with self.signal_stack_lock:
            if client_id not in self.client_signal_stacks:
                return None
            stack = self.client_signal_stacks[client_id]
            if not stack:
                return None
            return stack[-1]

    def _push_signal(self, client_id: str, signal_name: str):
        """Push a signal scope onto the client's stack."""
        with self.signal_stack_lock:
            if client_id not in self.client_signal_stacks:
                self.client_signal_stacks[client_id] = []
            self.client_signal_stacks[client_id].append(signal_name)
            debug_print_dispatcher(f"Client {client_id}: pushed signal '{signal_name}', stack depth={len(self.client_signal_stacks[client_id])}")

    def _pop_signal(self, client_id: str) -> Optional[str]:
        """Pop a signal scope from the client's stack."""
        with self.signal_stack_lock:
            if client_id not in self.client_signal_stacks:
                debug_print_dispatcher(f"WARNING: Client {client_id} tried to pop signal but has no stack")
                return None
            stack = self.client_signal_stacks[client_id]
            if not stack:
                debug_print_dispatcher(f"WARNING: Client {client_id} tried to pop signal but stack is empty")
                return None
            signal_name = stack.pop()
            debug_print_dispatcher(f"Client {client_id}: popped signal '{signal_name}', stack depth={len(stack)}")
            return signal_name

    def _send_to_worker(self, worker, cmd):
        """Send a command to a worker via ROUTER socket."""
        import zmq

        identity = worker["identity"]
        data = serialize(cmd)

        self.server_socket.send(identity, zmq.SNDMORE)
        self.server_socket.send(b'', zmq.SNDMORE)
        self.server_socket.send(data)

        # Return size for logging
        return len(data)

    def _send_to_worker_async(self, worker, cmd, cmd_type_str: str):
        """
        Buffer a command to be sent to worker in a batch.

        Instead of sending immediately, commands are buffered and sent together
        in batches, reducing network overhead. Buffer is flushed when:
        - Reaching max_batch_size
        - Before GetData (need to ensure previous ops are done)
        - End of client command processing

        Use this for operations that don't return data (CreateTensor, BinaryOp, etc.).
        For operations that need data (GetData), use _send_to_worker + _recv_from_worker.
        """
        worker_id = worker["id"]

        # Initialize buffer for this worker if needed
        with self.buffer_lock:
            if worker_id not in self.worker_buffers:
                self.worker_buffers[worker_id] = []

            # Add command to buffer
            self.worker_buffers[worker_id].append((cmd, cmd_type_str))
            buffer_size = len(self.worker_buffers[worker_id])

        debug_print_dispatcher(f"[BATCH] Buffered {cmd_type_str} for worker {worker_id} (buffer size: {buffer_size})")

        # Flush if buffer is full
        if buffer_size >= self.max_batch_size:
            self._flush_worker_buffer(worker)

    def _flush_worker_buffer(self, worker):
        """
        Send all buffered commands to a worker in a single batch.

        This reduces network overhead by sending multiple operations together.
        """
        from gt.transport.protocol import WorkerBatch

        worker_id = worker["id"]

        with self.buffer_lock:
            if worker_id not in self.worker_buffers or not self.worker_buffers[worker_id]:
                return  # Nothing to flush

            # Get all buffered commands
            buffered = self.worker_buffers[worker_id]
            self.worker_buffers[worker_id] = []  # Clear buffer

        # Always send as batch (even if only 1 command)
        commands = [cmd for cmd, _ in buffered]
        batch = WorkerBatch(commands=commands)
        send_size = self._send_to_worker(worker, batch)

        # Log the batch
        cmd_types = ", ".join(cmd_type for _, cmd_type in buffered[:5])  # Show first 5
        if len(buffered) > 5:
            cmd_types += f", ... (+{len(buffered)-5} more)"
        self._log_worker_cmd(worker_id, "WorkerBatch", f"[{len(commands)} cmds: {cmd_types}]", size_bytes=send_size)
        debug_print_dispatcher(f"[BATCH] Flushed {len(commands)} commands to worker {worker_id}")

    def _flush_all_worker_buffers(self):
        """Flush buffers for all workers."""
        for worker in self.workers:
            self._flush_worker_buffer(worker)

    def _recv_from_worker(self, worker):
        """Receive a response from a worker (blocking).

        WARNING: This shares the same ROUTER socket with the main loop,
        so we might receive client messages instead. We need to handle both.
        """
        import zmq
        from gt.transport.protocol import WorkerResponse

        # In ROUTER mode, we need to receive from the socket and match by identity
        # Keep trying until we get a response from the expected worker
        while True:
            identity = self.server_socket.recv()
            empty = self.server_socket.recv()
            data = self.server_socket.recv()

            response = deserialize(data)

            # If this is a WorkerResponse (expected), return it
            if isinstance(response, WorkerResponse):
                return response, len(data)

            # Otherwise, this is a client message that arrived while we were waiting
            # Put it back by processing it inline
            cmd_type = type(response).__name__
            source_id = identity.hex()

            # Determine if this is a worker or client
            is_worker_identity = identity in self.worker_identities
            source_type = "WORKER" if is_worker_identity else "CLIENT"

            self.instruction_stream.log("RECV", f"{source_type} {source_id}", cmd_type, self._get_cmd_details(response), size_bytes=len(data))

            # Process the command
            if isinstance(response, RegisterWorker):
                self.register_worker(identity, response.worker_id)
                reply = ClientResponse(success=True)
            elif is_worker_identity:
                from gt.debug import debug_print_dispatcher
                debug_print_dispatcher(f"WARNING: Unexpected worker message during recv: {cmd_type}")
                continue  # Skip replying, keep waiting for WorkerResponse
            else:
                # Client command - process it
                reply = None
                for modified_cmd in self.sharding_modifier.process(response):
                    reply = self._process_command(modified_cmd, source_id)
                    if not isinstance(reply, ClientResponse):
                        reply = ClientResponse(success=False, error=f"Internal error: handler returned {type(reply).__name__}")
                        break
                    if not reply.success:
                        break
                if reply is None:
                    reply = ClientResponse(success=True)

            # Send reply back
            reply_data = serialize(reply)
            self.server_socket.send(identity, zmq.SNDMORE)
            self.server_socket.send(b'', zmq.SNDMORE)
            self.server_socket.send(reply_data)

            self.instruction_stream.log("SEND", f"{source_type} {source_id}", cmd_type, f"success={reply.success}", size_bytes=len(reply_data))

            # Continue loop to wait for actual WorkerResponse

    def start(self):
        """Start the dispatcher server using ZMQ ROUTER."""
        import zmq

        self.running = True

        # Create ZMQ ROUTER socket (uses IPC for localhost, TCP for remote)
        self.server_socket = create_server(self.host, self.port)

        # Determine endpoint type for logging
        transport = "IPC" if self.host in ('localhost', '127.0.0.1', '0.0.0.0') else "TCP"
        from gt.debug import verbose_print
        verbose_print(f"Dispatcher listening on {self.host}:{self.port} ({transport})")

        while self.running:
            try:
                # Receive message: [identity, empty, data]
                identity = self.server_socket.recv()
                empty = self.server_socket.recv()
                data = self.server_socket.recv()

                # Track message size
                recv_size = len(data)

                # Deserialize command
                cmd = deserialize(data)

                # Determine source type
                is_worker = identity in self.worker_identities
                source_type = "WORKER" if is_worker else "CLIENT"
                source_id = identity.hex()

                # Log received command with size
                cmd_type = type(cmd).__name__
                self.instruction_stream.log("RECV", f"{source_type} {source_id}", cmd_type, self._get_cmd_details(cmd), size_bytes=recv_size)

                # Handle worker registration
                if isinstance(cmd, RegisterWorker):
                    self.register_worker(identity, cmd.worker_id)
                    response = ClientResponse(success=True)
                elif is_worker:
                    # Workers shouldn't send unsolicited messages (except RegisterWorker)
                    # This is likely a protocol error - ignore it
                    from gt.debug import debug_print_dispatcher
                    debug_print_dispatcher(f"WARNING: Received unexpected message from worker: {cmd_type}")
                    continue  # Don't process or respond
                else:
                    # Run command through sharding modifier (may yield multiple commands)
                    response = None
                    for modified_cmd in self.sharding_modifier.process(cmd):
                        response = self._process_command(modified_cmd, source_id)
                        # Verify we got a proper response
                        if not isinstance(response, ClientResponse):
                            response = ClientResponse(success=False, error=f"Internal error: handler returned {type(response).__name__} instead of ClientResponse")
                            break
                        # If any command fails, return that failure
                        if not response.success:
                            break

                    # If no commands were yielded, return success
                    if response is None:
                        response = ClientResponse(success=True)

                # Flush all worker buffers before sending response to client
                # This ensures all buffered operations are sent to workers
                self._flush_all_worker_buffers()

                # Serialize response
                response_data = serialize(response)
                send_size = len(response_data)

                # Send response: [identity, empty, data]
                self.server_socket.send(identity, zmq.SNDMORE)
                self.server_socket.send(b'', zmq.SNDMORE)
                self.server_socket.send(response_data)

                # Log sent response with size
                self.instruction_stream.log("SEND", f"{source_type} {source_id}", cmd_type, f"success={response.success}", size_bytes=send_size)

            except Exception as e:
                if self.running:
                    print(f"Error processing message: {e}")  # Keep errors visible
                    import traceback
                    traceback.print_exc()

    def stop(self):
        """Stop the dispatcher."""
        self.running = False
        if self.server_socket:
            self.server_socket.close()
        # Close the instruction stream log file
        self.instruction_stream.close()

    def _handle_client(self, conn: Connection, client_id: str):
        """Handle a client connection."""
        debug_print_dispatcher(f"Handling client {client_id}")
        self.instruction_stream.log("CONNECT", client_id, "CLIENT", "")

        try:
            while self.running:
                cmd = conn.recv()
                cmd_type = type(cmd).__name__
                self.instruction_stream.log("RECV", client_id, cmd_type, self._get_cmd_details(cmd))

                # Run command through sharding modifier (may yield multiple commands)
                response = None
                for modified_cmd in self.sharding_modifier.process(cmd):
                    response = self._process_command(modified_cmd, client_id)
                    # If any command fails, return that failure
                    if not response.success:
                        break

                # If no commands were yielded, return success
                if response is None:
                    response = ClientResponse(success=True)

                self.instruction_stream.log("SEND", client_id, cmd_type, f"success={response.success}")
                conn.send(response)
        except Exception as e:
            self.instruction_stream.log("ERROR", client_id, "EXCEPTION", str(e))
            print(f"Client {client_id} error: {e}")  # Keep errors visible
        finally:
            conn.close()
            self.instruction_stream.log("DISCONNECT", client_id, "CLIENT", "")
            debug_print_dispatcher(f"Client {client_id} disconnected")

    def _process_command(self, cmd: ClientCommand, client_id: str) -> ClientResponse:
        """Process a command from a client."""
        try:
            if isinstance(cmd, CreateTensor):
                return self._handle_create_tensor(cmd, client_id)
            elif isinstance(cmd, BinaryOp):
                return self._handle_binary_op(cmd, client_id)
            elif isinstance(cmd, UnaryOp):
                return self._handle_unary_op(cmd, client_id)
            elif isinstance(cmd, ReshapeOp):
                return self._handle_reshape_op(cmd, client_id)
            elif isinstance(cmd, SliceOp):
                return self._handle_slice_op(cmd, client_id)
            elif isinstance(cmd, GetData):
                return self._handle_get_data(cmd, client_id)
            elif isinstance(cmd, FreeTensor):
                return self._handle_free_tensor(cmd, client_id)
            elif isinstance(cmd, CopyTensor):
                return self._handle_copy_tensor(cmd, client_id)
            elif isinstance(cmd, CompileStart):
                return self._handle_compile_start(cmd, client_id)
            elif isinstance(cmd, CompileEnd):
                return self._handle_compile_end(cmd, client_id)
            elif isinstance(cmd, GetWorkerStats):
                return self._handle_get_worker_stats(cmd, client_id)
            else:
                return ClientResponse(success=False, error=f"Unknown command: {type(cmd)}")
        except Exception as e:
            return ClientResponse(success=False, error=str(e))

    def _handle_create_tensor(self, cmd: CreateTensor, client_id: str) -> ClientResponse:
        """Handle tensor creation (async - immediate ack)."""
        # If sharding modifier specified a worker, use that. Otherwise use round-robin.
        if cmd.worker_id is not None:
            worker = self._get_worker(cmd.worker_id)
            if not worker:
                return ClientResponse(success=False, error=f"Worker {cmd.worker_id} not found")
        else:
            worker = self._pick_worker()
            if not worker:
                return ClientResponse(success=False, error=no_workers_error(len(self.workers)))

        # Create worker-local tensor ID
        worker_tensor_id = f"{client_id}_{cmd.tensor_id}"

        # Register tensor location FIRST (optimistic - assume worker will succeed)
        # If worker fails, error will be discovered later when trying to use the tensor
        from gt.dispatcher.tensor_handle import ShardInfo
        shard_info = None
        if cmd.shard_info:
            shard_info = ShardInfo(
                axis=cmd.shard_info['axis'],
                shard_index=cmd.shard_info['shard_index'],
                num_shards=cmd.shard_info['num_shards']
            )

        self.tensor_handles.register(
            client_id=client_id,
            tensor_id=cmd.tensor_id,
            worker_id=worker["id"],
            worker_tensor_id=worker_tensor_id,
            shape=cmd.shape,
            dtype=cmd.dtype,
            shard_info=shard_info
        )

        # Send command to worker asynchronously (don't wait for response)
        worker_cmd = WorkerCreateTensor(
            tensor_id=worker_tensor_id,
            data=cmd.data,
            dtype=cmd.dtype,
            shape=cmd.shape
        )
        self._send_to_worker_async(worker, worker_cmd, "WorkerCreateTensor")

        # Return immediate ack to client (don't wait for worker)
        return ClientResponse(success=True)

    def _handle_binary_op(self, cmd: BinaryOp, client_id: str) -> ClientResponse:
        """Handle binary operation."""
        # Get locations of input tensors
        left_locs = self.tensor_handles.get_locations(client_id, cmd.left_id)
        right_locs = self.tensor_handles.get_locations(client_id, cmd.right_id)

        if not left_locs or not right_locs:
            return ClientResponse(success=False, error="Input tensor not found")

        # Check if tensors are sharded
        left_sharded = len(left_locs) > 1
        right_sharded = len(right_locs) > 1

        # DISTRIBUTED MATMUL: A @ B where A is sharded
        if cmd.op == "matmul" and left_sharded:
            if right_sharded:
                # Both sharded: need all-gather for B
                return self._handle_distributed_matmul_both_sharded(cmd, client_id, left_locs, right_locs)
            else:
                # Only A sharded: embarrassingly parallel
                return self._handle_distributed_matmul(cmd, client_id, left_locs, right_locs)

        # Non-sharded case (or both on same worker)
        left_loc = left_locs[0]
        right_loc = right_locs[0]

        # Handle cross-worker operations by moving one tensor to the other
        if left_loc.worker_id != right_loc.worker_id:
            # Move right tensor to left worker
            left_worker = self._get_worker(left_loc.worker_id)
            right_worker = self._get_worker(right_loc.worker_id)

            if not left_worker or not right_worker:
                return ClientResponse(success=False, error="Worker not found")

            # Fetch right tensor data
            get_cmd = WorkerGetData(tensor_id=right_loc.worker_tensor_id)
            self._send_to_worker(right_worker, get_cmd)
            right_response, _ = self._recv_from_worker(right_worker)

            if not right_response.success:
                return ClientResponse(success=False, error=f"Failed to fetch right tensor: {right_response.error}")

            # Create right tensor on left worker (async - no response expected)
            right_copy_id = f"{client_id}_{cmd.right_id}_copy"
            create_cmd = WorkerCreateTensor(
                tensor_id=right_copy_id,
                data=right_response.data,
                dtype=right_loc.dtype,
                shape=right_loc.shape if right_loc.shape else right_response.data.shape
            )
            self._send_to_worker_async(left_worker, create_cmd, "WorkerCreateTensor")
            # No response expected - optimistic execution

            # Update right_loc to point to the copy on left worker
            right_loc = type('obj', (object,), {
                'worker_id': left_loc.worker_id,
                'worker_tensor_id': right_copy_id,
                'shape': right_loc.shape,
                'dtype': right_loc.dtype
            })()

        worker = self._get_worker(left_loc.worker_id)
        if not worker:
            return ClientResponse(success=False, error="Worker not found")

        # Create result tensor ID
        result_tensor_id = f"{client_id}_{cmd.result_id}"

        # Register result location FIRST (optimistic)
        self.tensor_handles.register(
            client_id=client_id,
            tensor_id=cmd.result_id,
            worker_id=worker["id"],
            worker_tensor_id=result_tensor_id,
            shape=left_loc.shape,  # Placeholder (TODO: get actual shape from op)
            dtype=left_loc.dtype
        )

        # Send command to worker asynchronously (don't wait for response)
        worker_cmd = WorkerBinaryOp(
            result_id=result_tensor_id,
            op=cmd.op,
            left_id=left_loc.worker_tensor_id,
            right_id=right_loc.worker_tensor_id
        )
        self._send_to_worker_async(worker, worker_cmd, "WorkerBinaryOp")

        # Return immediate ack to client
        return ClientResponse(success=True)

    def _handle_unary_op(self, cmd: UnaryOp, client_id: str) -> ClientResponse:
        """Handle unary operation."""
        # For ops like randn that don't have inputs
        if cmd.input_id is None:
            # If sharding modifier specified a worker, use that. Otherwise use round-robin.
            if cmd.worker_id is not None:
                worker = self._get_worker(cmd.worker_id)
                if not worker:
                    return ClientResponse(success=False, error=f"Worker {cmd.worker_id} not found")
            else:
                worker = self._pick_worker()
                if not worker:
                    error_msg = f"""

======================================================================
GT DISPATCHER ERROR: No workers available
======================================================================
The dispatcher has no connected workers to execute operations.

Solutions:
  1. If using auto-start: This is a bug - workers should be
     automatically started. Please report this issue.

  2. If using manual setup:
     - Start at least one worker with:
       python -m gt.worker --host <dispatcher_host> -p <port>

  3. Check if workers disconnected (check worker logs)

Current registered workers: {len(self.workers)}
======================================================================
"""
                    return ClientResponse(success=False, error=error_msg)

            result_tensor_id = f"{client_id}_{cmd.result_id}"

            # Register tensor location FIRST (optimistic)
            from gt.dispatcher.tensor_handle import ShardInfo
            shard_info = None
            if cmd.shard_info:
                shard_info = ShardInfo(
                    axis=cmd.shard_info['axis'],
                    shard_index=cmd.shard_info['shard_index'],
                    num_shards=cmd.shard_info['num_shards']
                )

            self.tensor_handles.register(
                client_id=client_id,
                tensor_id=cmd.result_id,
                worker_id=worker["id"],
                worker_tensor_id=result_tensor_id,
                shape=cmd.shape,
                dtype=cmd.dtype,
                shard_info=shard_info
            )

            # Send command to worker asynchronously
            worker_cmd = WorkerUnaryOp(
                result_id=result_tensor_id,
                op=cmd.op,
                input_id=None,
                shape=cmd.shape,
                dtype=cmd.dtype,
                axis=getattr(cmd, 'axis', None),
                keepdims=getattr(cmd, 'keepdims', False)
            )
            self._send_to_worker_async(worker, worker_cmd, "WorkerUnaryOp")

            # Return immediate ack to client
            return ClientResponse(success=True)

        # For ops with inputs
        input_locs = self.tensor_handles.get_locations(client_id, cmd.input_id)
        if not input_locs:
            return ClientResponse(success=False, error="Input tensor not found")

        # Check if tensor is sharded and operation is a reduction
        is_sharded = len(input_locs) > 1
        is_reduction = cmd.op in ["sum", "mean"]

        if is_sharded and is_reduction:
            # Distributed reduction - needs all-reduce
            return self._handle_distributed_reduction(cmd, client_id, input_locs)

        # Non-sharded case or non-reduction operation
        input_loc = input_locs[0]
        worker = self._get_worker(input_loc.worker_id)
        if not worker:
            return ClientResponse(success=False, error="Worker not found")

        result_tensor_id = f"{client_id}_{cmd.result_id}"

        # Register result location FIRST (optimistic)
        self.tensor_handles.register(
            client_id=client_id,
            tensor_id=cmd.result_id,
            worker_id=worker["id"],
            worker_tensor_id=result_tensor_id,
            shape=input_loc.shape,  # Placeholder (TODO: compute actual shape)
            dtype=input_loc.dtype
        )

        # Send command to worker asynchronously
        worker_cmd = WorkerUnaryOp(
            result_id=result_tensor_id,
            op=cmd.op,
            input_id=input_loc.worker_tensor_id,
            axis=getattr(cmd, 'axis', None),
            keepdims=getattr(cmd, 'keepdims', False)
        )
        self._send_to_worker_async(worker, worker_cmd, "WorkerUnaryOp")

        # Return immediate ack to client
        return ClientResponse(success=True)

    def _handle_slice_op(self, cmd: SliceOp, client_id: str) -> ClientResponse:
        """Handle slice operation (subscripting)."""
        # Get location of input tensor
        input_locs = self.tensor_handles.get_locations(client_id, cmd.input_id)
        if not input_locs:
            return ClientResponse(success=False, error="Input tensor not found")

        # For now, only support non-sharded tensors
        # TODO: Handle sharded tensors (would need to slice each shard appropriately)
        if len(input_locs) > 1:
            return ClientResponse(success=False, error="Slice on sharded tensors not yet supported")

        input_loc = input_locs[0]
        worker = self._get_worker(input_loc.worker_id)
        if not worker:
            return ClientResponse(success=False, error="Worker not found")

        result_tensor_id = f"{client_id}_{cmd.result_id}"

        # Register result location FIRST (optimistic)
        # TODO: Compute actual result shape based on slice key
        self.tensor_handles.register(
            client_id=client_id,
            tensor_id=cmd.result_id,
            worker_id=worker["id"],
            worker_tensor_id=result_tensor_id,
            shape=None,  # Could compute from input_loc.shape and cmd.key
            dtype=input_loc.dtype
        )

        # Send command to worker asynchronously
        worker_cmd = WorkerSliceOp(
            result_id=result_tensor_id,
            input_id=input_loc.worker_tensor_id,
            key=cmd.key
        )
        self._send_to_worker_async(worker, worker_cmd, "WorkerSliceOp")

        # Return immediate ack to client
        return ClientResponse(success=True)

    def _handle_reshape_op(self, cmd: ReshapeOp, client_id: str) -> ClientResponse:
        """Handle reshape operation (reshape, unsqueeze, squeeze)."""
        # Get location of input tensor
        input_locs = self.tensor_handles.get_locations(client_id, cmd.input_id)
        if not input_locs:
            return ClientResponse(success=False, error="Input tensor not found")

        # For now, only support non-sharded tensors
        # TODO: Handle sharded tensors (would need to reshape each shard)
        if len(input_locs) > 1:
            return ClientResponse(success=False, error="Reshape on sharded tensors not yet supported")

        input_loc = input_locs[0]
        worker = self._get_worker(input_loc.worker_id)
        if not worker:
            return ClientResponse(success=False, error="Worker not found")

        result_tensor_id = f"{client_id}_{cmd.result_id}"

        # Compute result shape based on operation
        import numpy as np
        result_shape = input_loc.shape
        if cmd.op == "reshape":
            result_shape = cmd.params
        elif cmd.op == "unsqueeze":
            dim = cmd.params[0]
            if input_loc.shape:
                shape_list = list(input_loc.shape)
                # Handle negative dimensions
                if dim < 0:
                    dim = len(shape_list) + dim + 1
                shape_list.insert(dim, 1)
                result_shape = tuple(shape_list)
            else:
                result_shape = (1,)
        elif cmd.op == "squeeze":
            if input_loc.shape:
                if len(cmd.params) == 0:
                    # Squeeze all dimensions of size 1
                    result_shape = tuple(d for d in input_loc.shape if d != 1)
                else:
                    # Squeeze specific dimension
                    dim = cmd.params[0]
                    shape_list = list(input_loc.shape)
                    if shape_list[dim] == 1:
                        shape_list.pop(dim)
                    result_shape = tuple(shape_list)
            else:
                result_shape = ()

        # Register result location FIRST (optimistic)
        self.tensor_handles.register(
            client_id=client_id,
            tensor_id=cmd.result_id,
            worker_id=worker["id"],
            worker_tensor_id=result_tensor_id,
            shape=result_shape,
            dtype=input_loc.dtype
        )

        # Send command to worker asynchronously
        worker_cmd = WorkerReshapeOp(
            result_id=result_tensor_id,
            op=cmd.op,
            input_id=input_loc.worker_tensor_id,
            params=cmd.params
        )
        self._send_to_worker_async(worker, worker_cmd, "WorkerReshapeOp")

        # Return immediate ack to client
        return ClientResponse(success=True)

    def _handle_get_data(self, cmd: GetData, client_id: str) -> ClientResponse:
        """Handle data request."""
        # Flush all buffers before fetching data to ensure operations are complete
        self._flush_all_worker_buffers()

        locs = self.tensor_handles.get_locations(client_id, cmd.tensor_id)
        if not locs:
            return ClientResponse(success=False, error="Tensor not found")

        # If sharded, gather all shards and concatenate
        if len(locs) > 1:
            import numpy as np

            shards = []
            for loc in sorted(locs, key=lambda l: l.shard_info.shard_index if l.shard_info else 0):
                worker = self._get_worker(loc.worker_id)
                if not worker:
                    return ClientResponse(success=False, error=f"Worker {loc.worker_id} not found")

                worker_cmd = WorkerGetData(tensor_id=loc.worker_tensor_id)
                self._send_to_worker(worker, worker_cmd)
                worker_response, _ = self._recv_from_worker(worker)

                if not worker_response.success:
                    return ClientResponse(success=False, error=worker_response.error)

                shards.append(worker_response.data)

            # Concatenate along the sharded axis
            shard_axis = locs[0].shard_info.axis if locs[0].shard_info else 0
            combined_data = np.concatenate(shards, axis=shard_axis)
            return ClientResponse(success=True, data=combined_data)

        # Non-sharded case
        loc = locs[0]
        worker = self._get_worker(loc.worker_id)
        if not worker:
            return ClientResponse(success=False, error="Worker not found")

        worker_cmd = WorkerGetData(tensor_id=loc.worker_tensor_id)
        send_size = self._send_to_worker(worker, worker_cmd)
        self._log_worker_cmd(worker["id"], "WorkerGetData", f"tensor={loc.worker_tensor_id}", size_bytes=send_size)
        worker_response, recv_size = self._recv_from_worker(worker)
        self._log_worker_response(worker["id"], "WorkerGetData", worker_response.success, size_bytes=recv_size)

        if not worker_response.success:
            return ClientResponse(success=False, error=worker_response.error)

        return ClientResponse(success=True, data=worker_response.data)

    def _handle_free_tensor(self, cmd: FreeTensor, client_id: str) -> ClientResponse:
        """Handle tensor free (async - immediate ack)."""
        loc = self.tensor_handles.get_location(client_id, cmd.tensor_id)
        if not loc:
            return ClientResponse(success=True)  # Already freed

        # Free from dispatcher's registry immediately
        self.tensor_handles.free(client_id, cmd.tensor_id)

        # Send free command to worker asynchronously (don't wait)
        worker = self._get_worker(loc.worker_id)
        if worker:
            worker_cmd = WorkerFreeTensor(tensor_id=loc.worker_tensor_id)
            try:
                self._send_to_worker_async(worker, worker_cmd, "WorkerFreeTensor")
            except:
                pass  # Worker might be dead

        # Return immediate ack
        return ClientResponse(success=True)

    def _handle_copy_tensor(self, cmd: CopyTensor, client_id: str) -> ClientResponse:
        """Handle in-place tensor copy (dest_tensor = src_tensor)."""
        # Get locations of both tensors
        dest_loc = self.tensor_handles.get_location(client_id, cmd.dest_id)
        src_loc = self.tensor_handles.get_location(client_id, cmd.src_id)

        if not dest_loc:
            return ClientResponse(success=False, error="Destination tensor not found")
        if not src_loc:
            return ClientResponse(success=False, error="Source tensor not found")

        # For simplicity, assume they're on the same worker (common case for parameter updates)
        # In a full implementation, would need to handle cross-worker copies
        if dest_loc.worker_id != src_loc.worker_id:
            return ClientResponse(success=False, error="Cross-worker copy not yet supported")

        # Send copy command to worker (reuse CreateTensor to overwrite)
        worker = self._get_worker(dest_loc.worker_id)
        if not worker:
            return ClientResponse(success=False, error="Worker not found")

        # Get the source data
        get_cmd = WorkerGetData(tensor_id=src_loc.worker_tensor_id)
        self._send_to_worker(worker, get_cmd)
        get_response, recv_size = self._recv_from_worker(worker)

        if not get_response.success:
            return ClientResponse(success=False, error=f"Failed to get source data: {get_response.error}")

        # Overwrite the destination tensor
        create_cmd = WorkerCreateTensor(
            tensor_id=dest_loc.worker_tensor_id,
            data=get_response.data,
            dtype=dest_loc.dtype,
            shape=get_response.data.shape
        )
        self._send_to_worker(worker, create_cmd)
        create_response, recv_size = self._recv_from_worker(worker)

        if not create_response.success:
            return ClientResponse(success=False, error=f"Failed to copy data: {create_response.error}")

        # Update the destination tensor's metadata to match source
        dest_loc.shape = src_loc.shape
        dest_loc.dtype = src_loc.dtype

        return ClientResponse(success=True)

    def _handle_distributed_matmul_both_sharded(self, cmd: BinaryOp, client_id: str, left_locs, right_locs) -> ClientResponse:
        """
        Handle distributed matmul: A @ B where both A and B are sharded on axis 0.

        Strategy: All-gather B, then compute A_shard @ B_full on each worker.
        Result is sharded on axis 0 (same as A).
        """
        from gt.dispatcher.tensor_handle import ShardInfo
        from gt.transport.protocol import WorkerGetData, WorkerCreateTensor, WorkerBinaryOp
        import numpy as np

        # Step 1: All-gather B - collect all shards from workers
        b_shards = []
        for loc in right_locs:
            worker = self._get_worker(loc.worker_id)
            if not worker:
                return ClientResponse(success=False, error=f"Worker {loc.worker_id} not found")

            get_cmd = WorkerGetData(tensor_id=loc.worker_tensor_id)
            self._send_to_worker(worker, get_cmd)
            response, _ = self._recv_from_worker(worker)

            if not response.success:
                return ClientResponse(success=False, error=f"Failed to get B shard from {loc.worker_id}")

            b_shards.append(response.data)

        # Concatenate B shards to get full B (shards are on axis 0)
        b_full = np.concatenate(b_shards, axis=0)

        # Step 2: Distribute full B to each worker and compute A_shard @ B_full
        for shard_idx, left_loc in enumerate(left_locs):
            worker = self._get_worker(left_loc.worker_id)
            if not worker:
                return ClientResponse(success=False, error=f"Worker {left_loc.worker_id} not found")

            # Create full B on this worker
            b_full_id = f"{client_id}_{cmd.right_id}_full_worker{shard_idx}"
            create_cmd = WorkerCreateTensor(
                tensor_id=b_full_id,
                data=b_full,
                dtype=right_locs[0].dtype,
                shape=b_full.shape
            )
            self._send_to_worker(worker, create_cmd)
            create_response, _ = self._recv_from_worker(worker)

            if not create_response.success:
                return ClientResponse(success=False, error="Failed to create full B")

            # Compute A_shard @ B_full on this worker
            result_tensor_id = f"{client_id}_{cmd.result_id}_shard{shard_idx}"
            matmul_cmd = WorkerBinaryOp(
                result_id=result_tensor_id,
                op="matmul",
                left_id=left_loc.worker_tensor_id,
                right_id=b_full_id
            )
            self._send_to_worker(worker, matmul_cmd)
            matmul_response, _ = self._recv_from_worker(worker)

            if not matmul_response.success:
                return ClientResponse(success=False, error=f"Matmul failed on worker {left_loc.worker_id}")

            # Register result shard
            result_shape = list(left_loc.shape)
            result_shape[-1] = b_full.shape[-1]  # Output columns from B

            shard_info = ShardInfo(
                axis=0,  # Result sharded on axis 0 (same as A)
                num_shards=len(left_locs),
                shard_index=shard_idx
            )

            self.tensor_handles.register(
                client_id=client_id,
                tensor_id=cmd.result_id,
                worker_id=worker["id"],
                worker_tensor_id=result_tensor_id,
                shape=tuple(result_shape),
                dtype=left_loc.dtype,
                shard_info=shard_info
            )

        return ClientResponse(success=True)

    def _handle_distributed_matmul(self, cmd: BinaryOp, client_id: str, left_locs, right_locs) -> ClientResponse:
        """
        Handle distributed matmul: A @ B where A is sharded on axis 0.

        Each worker computes: A_shard @ B (locally)
        Then we need to gather all results (no all-reduce needed for row-sharded A).
        Result is sharded on axis 0 (same as A).
        """
        from gt.dispatcher.tensor_handle import ShardInfo
        import numpy as np

        # Broadcast B to all workers (if not already there)
        right_loc = right_locs[0]

        # For each shard of A, compute A_shard @ B
        result_locs = []
        for shard_idx, left_loc in enumerate(left_locs):
            worker = self._get_worker(left_loc.worker_id)
            if not worker:
                return ClientResponse(success=False, error=f"Worker {left_loc.worker_id} not found")

            # First, ensure B is on this worker
            # For simplicity, create a copy of B on each worker
            b_copy_id = f"{client_id}_{cmd.right_id}_copy_worker{shard_idx}"

            # Get B from original worker
            right_worker = self._get_worker(right_loc.worker_id)
            if not right_worker:
                return ClientResponse(success=False, error="Right worker not found")

            # Fetch B data
            get_cmd = WorkerGetData(tensor_id=right_loc.worker_tensor_id)
            self._send_to_worker(right_worker, get_cmd)
            right_response, _ = self._recv_from_worker(right_worker)
            if not right_response.success:
                return ClientResponse(success=False, error="Failed to get B data")

            # Create B on this worker
            create_b_cmd = WorkerCreateTensor(
                tensor_id=b_copy_id,
                data=right_response.data,
                dtype=right_loc.dtype,
                shape=right_loc.shape
            )
            self._send_to_worker(worker, create_b_cmd)
            create_response, _ = self._recv_from_worker(worker)
            if not create_response.success:
                return ClientResponse(success=False, error="Failed to create B copy")

            # Now compute A_shard @ B on this worker
            result_tensor_id = f"{client_id}_{cmd.result_id}_shard{shard_idx}"

            matmul_cmd = WorkerBinaryOp(
                result_id=result_tensor_id,
                op="matmul",
                left_id=left_loc.worker_tensor_id,
                right_id=b_copy_id
            )
            self._send_to_worker(worker, matmul_cmd)
            matmul_response, _ = self._recv_from_worker(worker)

            if not matmul_response.success:
                return ClientResponse(success=False, error=f"Matmul failed on worker {left_loc.worker_id}")

            # Register result shard
            result_shape = list(left_loc.shape)
            result_shape[-1] = right_loc.shape[-1]  # Output columns from B

            shard_info = ShardInfo(
                axis=0,  # Result sharded on axis 0
                num_shards=len(left_locs),
                shard_index=shard_idx
            )

            self.tensor_handles.register(
                client_id=client_id,
                tensor_id=cmd.result_id,
                worker_id=worker["id"],
                worker_tensor_id=result_tensor_id,
                shape=tuple(result_shape),
                dtype=left_loc.dtype,
                shard_info=shard_info
            )

        return ClientResponse(success=True)

    def _handle_distributed_reduction(self, cmd: UnaryOp, client_id: str, input_locs) -> ClientResponse:
        """
        Handle distributed reduction (sum/mean) on sharded tensor.

        Each worker computes partial result, then we combine:
        - sum: add all partial sums
        - mean: weighted average based on shard sizes
        """
        import numpy as np

        partial_results = []
        shard_sizes = []

        # Collect partial results from all workers
        for loc in input_locs:
            worker = self._get_worker(loc.worker_id)
            if not worker:
                return ClientResponse(success=False, error=f"Worker {loc.worker_id} not found")

            # Ask worker to compute local reduction
            result_tensor_id = f"{client_id}_{cmd.result_id}_partial_{loc.worker_id}"
            worker_cmd = WorkerUnaryOp(
                result_id=result_tensor_id,
                op=cmd.op,  # sum or mean
                input_id=loc.worker_tensor_id
            )
            self._send_to_worker(worker, worker_cmd)
            worker_response, _ = self._recv_from_worker(worker)

            if not worker_response.success:
                return ClientResponse(success=False, error=worker_response.error)

            # Now get the data from the computed result
            from gt.transport.protocol import WorkerGetData
            get_cmd = WorkerGetData(tensor_id=result_tensor_id)
            self._send_to_worker(worker, get_cmd)
            get_response, _ = self._recv_from_worker(worker)

            if not get_response.success:
                return ClientResponse(success=False, error=get_response.error)

            partial_results.append(get_response.data)

            # Track number of elements for mean calculation
            if cmd.op == "mean" and loc.shard_info:
                # Get total number of elements in this shard
                num_elements = np.prod(loc.shape)
                shard_sizes.append(num_elements)

        # Combine partial results
        if cmd.op == "sum":
            # All-reduce: sum all partial sums
            final_result = np.sum(partial_results)
        elif cmd.op == "mean":
            # Weighted average: sum of (partial_mean * num_elements) / total_elements
            total_elements = sum(shard_sizes)
            weighted_sum = sum(float(partial) * size for partial, size in zip(partial_results, shard_sizes))
            final_result = weighted_sum / total_elements
        else:
            return ClientResponse(success=False, error=f"Unknown reduction op: {cmd.op}")

        # Store result on first worker (could be any worker since it's a scalar)
        first_worker = self._get_worker(input_locs[0].worker_id)
        result_tensor_id = f"{client_id}_{cmd.result_id}"

        # Create tensor from the scalar result
        from gt.transport.protocol import WorkerCreateTensor
        result_data = np.array(final_result, dtype=input_locs[0].dtype)
        create_cmd = WorkerCreateTensor(
            tensor_id=result_tensor_id,
            data=result_data,
            dtype=input_locs[0].dtype,
            shape=result_data.shape
        )
        self._send_to_worker(first_worker, create_cmd)
        worker_response, _ = self._recv_from_worker(first_worker)

        if not worker_response.success:
            return ClientResponse(success=False, error=worker_response.error)

        # Register result tensor (scalar shape)
        self.tensor_handles.register(
            client_id=client_id,
            tensor_id=cmd.result_id,
            worker_id=first_worker["id"],
            worker_tensor_id=result_tensor_id,
            shape=result_data.shape,
            dtype=input_locs[0].dtype
        )

        return ClientResponse(success=True)

    def _handle_sharded_creation(self, cmd: UnaryOp, client_id: str, shard_axis: int) -> ClientResponse:
        """Create a sharded tensor across all workers."""
        from gt.dispatcher.tensor_handle import ShardInfo
        import numpy as np

        num_workers = len(self.workers)
        full_shape = list(cmd.shape)

        # Calculate shard size (divide axis by num_workers)
        if full_shape[shard_axis] % num_workers != 0:
            return ClientResponse(success=False,
                error=f"Cannot shard axis {shard_axis} of size {full_shape[shard_axis]} across {num_workers} workers evenly")

        shard_size = full_shape[shard_axis] // num_workers
        shard_shape = list(full_shape)
        shard_shape[shard_axis] = shard_size

        # Create shard on each worker
        for shard_idx, worker in enumerate(self.workers):
            result_tensor_id = f"{client_id}_{cmd.result_id}_shard{shard_idx}"

            worker_cmd = WorkerUnaryOp(
                result_id=result_tensor_id,
                op=cmd.op,
                input_id=None,
                shape=tuple(shard_shape),
                dtype=cmd.dtype
            )
            self._send_to_worker(worker, worker_cmd)
            worker_response, _ = self._recv_from_worker(worker)

            if not worker_response.success:
                return ClientResponse(success=False, error=worker_response.error)

            # Register this shard
            shard_info = ShardInfo(
                axis=shard_axis,
                num_shards=num_workers,
                shard_index=shard_idx
            )

            self.tensor_handles.register(
                client_id=client_id,
                tensor_id=cmd.result_id,
                worker_id=worker["id"],
                worker_tensor_id=result_tensor_id,
                shape=tuple(shard_shape),
                dtype=cmd.dtype,
                shard_info=shard_info
            )

        return ClientResponse(success=True)

    def _pick_worker(self):
        """Pick a worker using round-robin."""
        if not self.workers:
            return None
        worker = self.workers[self.next_worker_idx]
        self.next_worker_idx = (self.next_worker_idx + 1) % len(self.workers)
        return worker

    def _get_worker(self, worker_id):
        """Get a worker by ID or index."""
        # Handle integer indices (e.g., 0, 1, 2 from shard config)
        if isinstance(worker_id, int):
            if 0 <= worker_id < len(self.workers):
                return self.workers[worker_id]
            return None

        # Handle string IDs (e.g., "auto_worker_0", "worker_foo")
        for worker in self.workers:
            if worker["id"] == str(worker_id):
                return worker
        return None

    def _handle_compile_start(self, cmd: CompileStart, client_id: str) -> ClientResponse:
        """Handle compile start - push signal onto client's stack and forward to workers if needed."""
        from gt.config import get_signal_config

        # Push signal onto client's stack (always track, regardless of compilation)
        self._push_signal(client_id, cmd.signal_name)

        signal_config = get_signal_config(cmd.signal_name)
        should_compile = signal_config and signal_config.compile

        if not should_compile:
            return ClientResponse(success=True)

        for worker in self.workers:
            worker_cmd = WorkerCompileStart(signal_name=cmd.signal_name)
            send_size = self._send_to_worker(worker, worker_cmd)
            self._log_worker_cmd(worker["id"], "WorkerCompileStart", f"signal={cmd.signal_name}", size_bytes=send_size)
            worker_response, recv_size = self._recv_from_worker(worker)
            self._log_worker_response(worker["id"], "WorkerCompileStart", worker_response.success, size_bytes=recv_size)

            if not worker_response.success:
                return ClientResponse(success=False, error=f"Worker {worker['id']} failed: {worker_response.error}")

        return ClientResponse(success=True)

    def _handle_compile_end(self, cmd: CompileEnd, client_id: str) -> ClientResponse:
        """Handle compile end - pop signal from client's stack and forward to workers if needed."""
        from gt.config import get_signal_config

        # Pop signal from client's stack (always track, regardless of compilation)
        popped_signal = self._pop_signal(client_id)
        if popped_signal != cmd.signal_name:
            debug_print_dispatcher(f"WARNING: Client {client_id} popped signal '{popped_signal}' but expected '{cmd.signal_name}' (mismatched push/pop)")

        signal_config = get_signal_config(cmd.signal_name)
        should_compile = signal_config and signal_config.compile

        if not should_compile:
            return ClientResponse(success=True)

        for worker in self.workers:
            worker_cmd = WorkerCompileEnd(signal_name=cmd.signal_name)
            send_size = self._send_to_worker(worker, worker_cmd)
            self._log_worker_cmd(worker["id"], "WorkerCompileEnd", f"signal={cmd.signal_name}", size_bytes=send_size)
            worker_response, recv_size = self._recv_from_worker(worker)
            self._log_worker_response(worker["id"], "WorkerCompileEnd", worker_response.success, size_bytes=recv_size)

            if not worker_response.success:
                return ClientResponse(success=False, error=f"Worker {worker['id']} failed: {worker_response.error}")

        return ClientResponse(success=True)

    def _handle_get_worker_stats(self, cmd: GetWorkerStats, client_id: str) -> ClientResponse:
        """Handle worker stats request - query all workers and aggregate."""
        # Flush all buffers before getting stats
        self._flush_all_worker_buffers()

        all_stats = {}

        for worker in self.workers:
            worker_cmd = WorkerGetStats()
            send_size = self._send_to_worker(worker, worker_cmd)
            self._log_worker_cmd(worker["id"], "WorkerGetStats", "", size_bytes=send_size)
            worker_response, recv_size = self._recv_from_worker(worker)
            self._log_worker_response(worker["id"], "WorkerGetStats", worker_response.success, size_bytes=recv_size)

            if worker_response.success:
                all_stats[worker["id"]] = worker_response.data
            else:
                all_stats[worker["id"]] = {'error': worker_response.error}

        return ClientResponse(success=True, data=all_stats)

    def _get_cmd_details(self, cmd) -> str:
        """Extract useful details from command for logging."""
        if isinstance(cmd, CreateTensor):
            return f"tensor_id={cmd.tensor_id} shape={cmd.shape}"
        elif isinstance(cmd, BinaryOp):
            return f"result={cmd.result_id} op={cmd.op} left={cmd.left_id} right={cmd.right_id}"
        elif isinstance(cmd, UnaryOp):
            if cmd.input_id is None:
                return f"result={cmd.result_id} op={cmd.op} shape={cmd.shape}"
            return f"result={cmd.result_id} op={cmd.op} input={cmd.input_id}"
        elif isinstance(cmd, GetData):
            return f"tensor_id={cmd.tensor_id}"
        elif isinstance(cmd, FreeTensor):
            return f"tensor_id={cmd.tensor_id}"
        elif isinstance(cmd, CopyTensor):
            return f"dest={cmd.dest_id} src={cmd.src_id}"
        elif isinstance(cmd, RegisterWorker):
            return f"worker_id={cmd.worker_id}"
        elif isinstance(cmd, CompileStart):
            return f"signal={cmd.signal_name}"
        elif isinstance(cmd, CompileEnd):
            return f"signal={cmd.signal_name}"
        elif isinstance(cmd, GetWorkerStats):
            return ""
        return ""

    def _log_worker_cmd(self, worker_id: str, cmd_type: str, details: str = "", size_bytes: int = 0):
        """Log a command being sent to a worker."""
        self.instruction_stream.log("WORKER_SEND", worker_id, cmd_type, details, size_bytes=size_bytes)

    def _log_worker_response(self, worker_id: str, cmd_type: str, success: bool, size_bytes: int = 0):
        """Log a response received from a worker."""
        self.instruction_stream.log("WORKER_RECV", worker_id, cmd_type, f"success={success}", size_bytes=size_bytes)
