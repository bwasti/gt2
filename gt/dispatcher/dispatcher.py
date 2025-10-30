"""
Dispatcher coordinates between clients and workers.

Handles multiple clients and schedules operations to workers.

Keep this SIMPLE and READABLE.
"""

import threading
from gt.transport.connection import create_server, Connection
from gt.transport.protocol import (
    ClientCommand, CreateTensor, BinaryOp, UnaryOp, GetData, FreeTensor, CopyTensor,
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
            elif isinstance(cmd, CopyTensor):
                return self._handle_copy_tensor(cmd, client_id)
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
                worker["conn"].send(worker_cmd)
                worker_response: WorkerResponse = worker["conn"].recv()

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
        worker["conn"].send(get_cmd)
        get_response = worker["conn"].recv()

        if not get_response.success:
            return ClientResponse(success=False, error=f"Failed to get source data: {get_response.error}")

        # Overwrite the destination tensor
        create_cmd = WorkerCreateTensor(
            tensor_id=dest_loc.worker_tensor_id,
            data=get_response.data,
            dtype=dest_loc.dtype,
            shape=get_response.data.shape
        )
        worker["conn"].send(create_cmd)
        create_response = worker["conn"].recv()

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
            right_worker["conn"].send(get_cmd)
            right_response: WorkerResponse = right_worker["conn"].recv()
            if not right_response.success:
                return ClientResponse(success=False, error="Failed to get B data")

            # Create B on this worker
            create_b_cmd = WorkerCreateTensor(
                tensor_id=b_copy_id,
                data=right_response.data,
                dtype=right_loc.dtype,
                shape=right_loc.shape
            )
            worker["conn"].send(create_b_cmd)
            create_response: WorkerResponse = worker["conn"].recv()
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
            worker["conn"].send(matmul_cmd)
            matmul_response: WorkerResponse = worker["conn"].recv()

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
            worker["conn"].send(worker_cmd)
            worker_response: WorkerResponse = worker["conn"].recv()

            if not worker_response.success:
                return ClientResponse(success=False, error=worker_response.error)

            # Now get the data from the computed result
            from gt.transport.protocol import WorkerGetData
            get_cmd = WorkerGetData(tensor_id=result_tensor_id)
            worker["conn"].send(get_cmd)
            get_response: WorkerResponse = worker["conn"].recv()

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
        first_worker["conn"].send(create_cmd)
        worker_response: WorkerResponse = first_worker["conn"].recv()

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
            worker["conn"].send(worker_cmd)
            worker_response: WorkerResponse = worker["conn"].recv()

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
