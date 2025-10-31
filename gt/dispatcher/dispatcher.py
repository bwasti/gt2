"""
Dispatcher coordinates between clients and workers.

Handles multiple clients and schedules operations to workers.

Keep this SIMPLE and READABLE.
"""

import threading
import time
from gt.transport.connection import create_server, Connection
from gt.transport.protocol import (
    ClientCommand, CreateTensor, BinaryOp, UnaryOp, GetData, FreeTensor, CopyTensor,
    CompileStart, CompileEnd, GetWorkerStats, RegisterWorker,
    ClientResponse, WorkerCreateTensor, WorkerBinaryOp, WorkerUnaryOp,
    WorkerGetData, WorkerFreeTensor, WorkerCompileStart, WorkerCompileEnd, WorkerGetStats, WorkerResponse
)
from gt.dispatcher.tensor_handle import TensorHandle


class InstructionStream:
    """Records all client-dispatcher-worker instructions with timestamps for debugging and monitoring."""

    def __init__(self, log_file: str = None, console: bool = True):
        self.entries = []
        self.lock = threading.Lock()
        self.start_time = time.time()
        self.sequence = 0
        self.console = console
        self.log_file = log_file
        self.file_handle = None

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

    def __init__(self, host="localhost", port=9000, log_file: str = None, console_log: bool = True):
        self.host = host
        self.port = port
        self.tensor_handles = TensorHandle()
        self.workers = []  # List of worker connections
        self.next_worker_idx = 0  # Simple round-robin scheduling
        self.running = False
        self.server_socket = None
        self.instruction_stream = InstructionStream(log_file=log_file, console=console_log)  # Record all instructions for debugging/monitoring

    def register_worker(self, worker_identity: bytes, worker_id: str):
        """Register a worker by its ZMQ identity."""
        self.workers.append({
            "id": worker_id,
            "identity": worker_identity
        })
        print(f"Worker {worker_id} registered")

    def _send_to_worker(self, worker, cmd):
        """Send a command to a worker via ROUTER socket."""
        import zmq
        from gt.transport.protocol import serialize

        identity = worker["identity"]
        data = serialize(cmd)

        self.server_socket.send(identity, zmq.SNDMORE)
        self.server_socket.send(b'', zmq.SNDMORE)
        self.server_socket.send(data)

        # Return size for logging
        return len(data)

    def _recv_from_worker(self, worker):
        """Receive a response from a worker (blocking)."""
        import zmq
        from gt.transport.protocol import deserialize

        # In ROUTER mode, we need to receive from the socket and match by identity
        # This is a simplified version - in production would need proper matching
        identity = self.server_socket.recv()
        empty = self.server_socket.recv()
        data = self.server_socket.recv()

        # Return both response and size
        return deserialize(data), len(data)

    def start(self):
        """Start the dispatcher server using ZMQ ROUTER."""
        import zmq

        self.running = True

        # Create ZMQ ROUTER socket (uses IPC for localhost, TCP for remote)
        self.server_socket = create_server(self.host, self.port)

        # Determine endpoint type for logging
        transport = "IPC" if self.host in ('localhost', '127.0.0.1', '0.0.0.0') else "TCP"
        print(f"Dispatcher listening on {self.host}:{self.port} ({transport})")

        while self.running:
            try:
                # Receive message: [identity, empty, data]
                identity = self.server_socket.recv()
                empty = self.server_socket.recv()
                data = self.server_socket.recv()

                # Track message size
                recv_size = len(data)

                # Deserialize command
                from gt.transport.protocol import deserialize, serialize
                cmd = deserialize(data)

                # Use identity as client ID
                client_id = identity.hex()

                # Log received command with size
                cmd_type = type(cmd).__name__
                self.instruction_stream.log("RECV", client_id, cmd_type, self._get_cmd_details(cmd), size_bytes=recv_size)

                # Handle worker registration
                if isinstance(cmd, RegisterWorker):
                    self.register_worker(identity, cmd.worker_id)
                    response = ClientResponse(success=True)
                else:
                    # Process regular command
                    response = self._process_command(cmd, client_id)

                # Serialize response
                response_data = serialize(response)
                send_size = len(response_data)

                # Send response: [identity, empty, data]
                self.server_socket.send(identity, zmq.SNDMORE)
                self.server_socket.send(b'', zmq.SNDMORE)
                self.server_socket.send(response_data)

                # Log sent response with size
                self.instruction_stream.log("SEND", client_id, cmd_type, f"success={response.success}", size_bytes=send_size)

            except Exception as e:
                if self.running:
                    print(f"Error processing message: {e}")
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
        print(f"Handling client {client_id}")
        self.instruction_stream.log("CONNECT", client_id, "CLIENT", "")

        try:
            while self.running:
                cmd = conn.recv()
                cmd_type = type(cmd).__name__
                self.instruction_stream.log("RECV", client_id, cmd_type, self._get_cmd_details(cmd))

                response = self._process_command(cmd, client_id)

                self.instruction_stream.log("SEND", client_id, cmd_type, f"success={response.success}")
                conn.send(response)
        except Exception as e:
            self.instruction_stream.log("ERROR", client_id, "EXCEPTION", str(e))
            print(f"Client {client_id} error: {e}")
        finally:
            conn.close()
            self.instruction_stream.log("DISCONNECT", client_id, "CLIENT", "")
            print(f"Client {client_id} disconnected")

    def _process_command(self, cmd: ClientCommand, client_id: str) -> ClientResponse:
        """Process a command from a client."""
        try:
            if isinstance(cmd, CreateTensor):
                return self._handle_create_tensor(cmd, client_id)
            elif isinstance(cmd, BinaryOp):
                return self._handle_binary_op(cmd, client_id)
            elif isinstance(cmd, UnaryOp):
                return self._handle_unary_op(cmd, client_id)
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
        """Handle tensor creation."""
        # Pick a worker (simple round-robin)
        worker = self._pick_worker()
        if not worker:
            return ClientResponse(success=False, error="No workers available")

        # Create worker-local tensor ID
        worker_tensor_id = f"{client_id}_{cmd.tensor_id}"

        # Send command to worker
        worker_cmd = WorkerCreateTensor(
            tensor_id=worker_tensor_id,
            data=cmd.data,
            dtype=cmd.dtype,
            shape=cmd.shape
        )
        send_size = self._send_to_worker(worker, worker_cmd)
        self._log_worker_cmd(worker["id"], "WorkerCreateTensor", f"tensor={worker_tensor_id}", size_bytes=send_size)
        worker_response, recv_size = self._recv_from_worker(worker)
        self._log_worker_response(worker["id"], "WorkerCreateTensor", worker_response.success, size_bytes=recv_size)

        if not worker_response.success:
            return ClientResponse(success=False, error=worker_response.error)

        # Register tensor location
        self.tensor_handles.register(
            client_id=client_id,
            tensor_id=cmd.tensor_id,
            worker_id=worker["id"],
            worker_tensor_id=worker_tensor_id,
            shape=cmd.shape,
            dtype=cmd.dtype
        )

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

        # DISTRIBUTED MATMUL: A @ B where A is sharded on axis 0
        if cmd.op == "matmul" and left_sharded and not right_sharded:
            return self._handle_distributed_matmul(cmd, client_id, left_locs, right_locs)

        # Non-sharded case (or both on same worker)
        left_loc = left_locs[0]
        right_loc = right_locs[0]

        # For now, assume they're on the same worker
        if left_loc.worker_id != right_loc.worker_id:
            return ClientResponse(success=False, error="Cross-worker ops not yet supported")

        worker = self._get_worker(left_loc.worker_id)
        if not worker:
            return ClientResponse(success=False, error="Worker not found")

        # Create result tensor ID
        result_tensor_id = f"{client_id}_{cmd.result_id}"

        # Send command to worker
        worker_cmd = WorkerBinaryOp(
            result_id=result_tensor_id,
            op=cmd.op,
            left_id=left_loc.worker_tensor_id,
            right_id=right_loc.worker_tensor_id
        )
        send_size = self._send_to_worker(worker, worker_cmd)
        self._log_worker_cmd(worker["id"], "WorkerBinaryOp", f"op={cmd.op} result={result_tensor_id}", size_bytes=send_size)
        worker_response, recv_size = self._recv_from_worker(worker)
        self._log_worker_response(worker["id"], "WorkerBinaryOp", worker_response.success, size_bytes=recv_size)

        if not worker_response.success:
            return ClientResponse(success=False, error=worker_response.error)

        # Register result location
        # TODO: get actual shape/dtype from worker
        self.tensor_handles.register(
            client_id=client_id,
            tensor_id=cmd.result_id,
            worker_id=worker["id"],
            worker_tensor_id=result_tensor_id,
            shape=left_loc.shape,  # Placeholder
            dtype=left_loc.dtype
        )

        return ClientResponse(success=True)

    def _handle_unary_op(self, cmd: UnaryOp, client_id: str) -> ClientResponse:
        """Handle unary operation."""
        # For ops like randn that don't have inputs
        if cmd.input_id is None:
            # SHARDING LOGIC: If we have multiple workers, shard across workers
            num_workers = len(self.workers)
            if num_workers > 1 and cmd.shape and len(cmd.shape) >= 2:
                # Shard along axis 0 (rows)
                return self._handle_sharded_creation(cmd, client_id, shard_axis=0)

            # Single worker or non-shardable - use old logic
            worker = self._pick_worker()
            if not worker:
                return ClientResponse(success=False, error="No workers available")

            result_tensor_id = f"{client_id}_{cmd.result_id}"

            worker_cmd = WorkerUnaryOp(
                result_id=result_tensor_id,
                op=cmd.op,
                input_id=None,
                shape=cmd.shape,
                dtype=cmd.dtype
            )
            send_size = self._send_to_worker(worker, worker_cmd)
            self._log_worker_cmd(worker["id"], "WorkerUnaryOp", f"op={cmd.op} shape={cmd.shape}", size_bytes=send_size)
            worker_response, recv_size = self._recv_from_worker(worker)
            self._log_worker_response(worker["id"], "WorkerUnaryOp", worker_response.success, size_bytes=recv_size)

            if not worker_response.success:
                return ClientResponse(success=False, error=worker_response.error)

            self.tensor_handles.register(
                client_id=client_id,
                tensor_id=cmd.result_id,
                worker_id=worker["id"],
                worker_tensor_id=result_tensor_id,
                shape=cmd.shape,
                dtype=cmd.dtype
            )

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

        worker_cmd = WorkerUnaryOp(
            result_id=result_tensor_id,
            op=cmd.op,
            input_id=input_loc.worker_tensor_id
        )
        send_size = self._send_to_worker(worker, worker_cmd)
        self._log_worker_cmd(worker["id"], "WorkerUnaryOp", f"op={cmd.op} input={input_loc.worker_tensor_id}", size_bytes=send_size)
        worker_response, recv_size = self._recv_from_worker(worker)
        self._log_worker_response(worker["id"], "WorkerUnaryOp", worker_response.success, size_bytes=recv_size)

        if not worker_response.success:
            return ClientResponse(success=False, error=worker_response.error)

        self.tensor_handles.register(
            client_id=client_id,
            tensor_id=cmd.result_id,
            worker_id=worker["id"],
            worker_tensor_id=result_tensor_id,
            shape=input_loc.shape,  # Placeholder
            dtype=input_loc.dtype
        )

        return ClientResponse(success=True)

    def _handle_get_data(self, cmd: GetData, client_id: str) -> ClientResponse:
        """Handle data request."""
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
                worker_response: WorkerResponse = self._recv_from_worker(worker)

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
        """Handle tensor free."""
        loc = self.tensor_handles.get_location(client_id, cmd.tensor_id)
        if not loc:
            return ClientResponse(success=True)  # Already freed

        worker = self._get_worker(loc.worker_id)
        if worker:
            worker_cmd = WorkerFreeTensor(tensor_id=loc.worker_tensor_id)
            try:
                send_size = self._send_to_worker(worker, worker_cmd)
                self._log_worker_cmd(worker["id"], "WorkerFreeTensor", f"tensor={loc.worker_tensor_id}", size_bytes=send_size)
                response, recv_size = self._recv_from_worker(worker)
                self._log_worker_response(worker["id"], "WorkerFreeTensor", response.success if response else False, size_bytes=recv_size)
            except:
                pass  # Worker might be dead

        self.tensor_handles.free(client_id, cmd.tensor_id)
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
            right_self._send_to_worker(worker, get_cmd)
            right_response: WorkerResponse = right_self._recv_from_worker(worker)
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
            create_response: WorkerResponse = self._recv_from_worker(worker)
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
            matmul_response: WorkerResponse = self._recv_from_worker(worker)

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
            worker_response: WorkerResponse = self._recv_from_worker(worker)

            if not worker_response.success:
                return ClientResponse(success=False, error=worker_response.error)

            # Now get the data from the computed result
            from gt.transport.protocol import WorkerGetData
            get_cmd = WorkerGetData(tensor_id=result_tensor_id)
            self._send_to_worker(worker, get_cmd)
            get_response: WorkerResponse = self._recv_from_worker(worker)

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
        first_self._send_to_worker(worker, create_cmd)
        worker_response: WorkerResponse = first_self._recv_from_worker(worker)

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
            worker_response: WorkerResponse = self._recv_from_worker(worker)

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

    def _get_worker(self, worker_id: str):
        """Get a worker by ID."""
        for worker in self.workers:
            if worker["id"] == worker_id:
                return worker
        return None

    def _handle_compile_start(self, cmd: CompileStart, client_id: str) -> ClientResponse:
        """Handle compile start - check config and forward to workers if needed."""
        from gt.config import get_signal_config

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
        """Handle compile end - check config and forward to workers if needed."""
        from gt.config import get_signal_config

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
