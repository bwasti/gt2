"""
Dispatcher coordinates between clients and workers.

Handles multiple clients and schedules operations to workers.

Keep this SIMPLE and READABLE.
"""

import threading
from gt.transport.connection import create_server, Connection
from gt.transport.protocol import (
    ClientCommand, CreateTensor, BinaryOp, UnaryOp, GetData, FreeTensor,
    ClientResponse, WorkerCreateTensor, WorkerBinaryOp, WorkerUnaryOp,
    WorkerGetData, WorkerFreeTensor, WorkerResponse
)
from gt.dispatcher.tensor_handle import TensorHandle


class Dispatcher:
    """
    Dispatcher coordinates multiple clients and workers.

    Takes commands from clients and schedules them to workers.
    """

    def __init__(self, host="localhost", port=9000):
        self.host = host
        self.port = port
        self.tensor_handles = TensorHandle()
        self.workers = []  # List of worker connections
        self.next_worker_idx = 0  # Simple round-robin scheduling
        self.running = False
        self.server_socket = None

    def register_worker(self, worker_conn: Connection, worker_id: str):
        """Register a worker."""
        self.workers.append({
            "id": worker_id,
            "conn": worker_conn
        })
        print(f"Worker {worker_id} registered")

    def start(self):
        """Start the dispatcher server."""
        self.running = True
        self.server_socket = create_server(self.host, self.port)
        print(f"Dispatcher listening on {self.host}:{self.port}")

        while self.running:
            try:
                sock, addr = self.server_socket.accept()
                conn = Connection(sock)
                print(f"New connection from {addr}")

                # For now, assume it's a client connection
                # In a real system, we'd do a handshake to determine client vs worker
                client_id = f"{addr[0]}:{addr[1]}"
                thread = threading.Thread(
                    target=self._handle_client,
                    args=(conn, client_id),
                    daemon=True
                )
                thread.start()
            except Exception as e:
                if self.running:
                    print(f"Error accepting connection: {e}")

    def stop(self):
        """Stop the dispatcher."""
        self.running = False
        if self.server_socket:
            self.server_socket.close()

    def _handle_client(self, conn: Connection, client_id: str):
        """Handle a client connection."""
        print(f"Handling client {client_id}")

        try:
            while self.running:
                cmd = conn.recv()
                response = self._process_command(cmd, client_id)
                conn.send(response)
        except Exception as e:
            print(f"Client {client_id} error: {e}")
        finally:
            conn.close()
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
        worker["conn"].send(worker_cmd)
        worker_response: WorkerResponse = worker["conn"].recv()

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
        left_loc = self.tensor_handles.get_location(client_id, cmd.left_id)
        right_loc = self.tensor_handles.get_location(client_id, cmd.right_id)

        if not left_loc or not right_loc:
            return ClientResponse(success=False, error="Input tensor not found")

        # For now, assume they're on the same worker
        # TODO: handle cross-worker operations
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
        worker["conn"].send(worker_cmd)
        worker_response: WorkerResponse = worker["conn"].recv()

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
            worker["conn"].send(worker_cmd)
            worker_response: WorkerResponse = worker["conn"].recv()

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
        input_loc = self.tensor_handles.get_location(client_id, cmd.input_id)
        if not input_loc:
            return ClientResponse(success=False, error="Input tensor not found")

        worker = self._get_worker(input_loc.worker_id)
        if not worker:
            return ClientResponse(success=False, error="Worker not found")

        result_tensor_id = f"{client_id}_{cmd.result_id}"

        worker_cmd = WorkerUnaryOp(
            result_id=result_tensor_id,
            op=cmd.op,
            input_id=input_loc.worker_tensor_id
        )
        worker["conn"].send(worker_cmd)
        worker_response: WorkerResponse = worker["conn"].recv()

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
        loc = self.tensor_handles.get_location(client_id, cmd.tensor_id)
        if not loc:
            return ClientResponse(success=False, error="Tensor not found")

        worker = self._get_worker(loc.worker_id)
        if not worker:
            return ClientResponse(success=False, error="Worker not found")

        worker_cmd = WorkerGetData(tensor_id=loc.worker_tensor_id)
        worker["conn"].send(worker_cmd)
        worker_response: WorkerResponse = worker["conn"].recv()

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
                worker["conn"].send(worker_cmd)
                worker["conn"].recv()  # Ignore response
            except:
                pass  # Worker might be dead

        self.tensor_handles.free(client_id, cmd.tensor_id)
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
