"""
GT - Distributed GPU ML Operations

A distributed frontend for GPU ML operations.

Simple PyTorch-like API:
    import gt

    # Optionally connect to remote server
    # gt.connect('localhost:12345')

    # Create tensors and compute
    a = gt.tensor([1, 2, 3])
    b = gt.tensor([4, 5, 6])
    c = a + b

    # Get data back
    data = c.data.numpy()
"""

__version__ = "0.1.0"

import threading
import atexit
import numpy as np
from typing import Optional, List, Union

# Global state
_connected = False
_auto_server = None
_client = None
_num_gpu_workers = 1  # Default to 1 worker in auto-connect mode


def connect(address: str):
    """
    Connect to a remote GT server.

    Args:
        address: Server address in format "host:port" (e.g., "localhost:12345")

    Example:
        gt.connect('localhost:12345')
    """
    global _connected, _client

    if _connected:
        raise RuntimeError("Already connected to a server")

    host, port = address.split(':')
    port = int(port)

    from gt.client.client import Client
    _client = Client(dispatcher_host=host, dispatcher_port=port)
    _client.connect()
    _connected = True


def gpu_workers(n: int):
    """
    Configure number of GPU workers for auto-connect mode.

    Must be called BEFORE any tensor operations (before auto-connect happens).

    Args:
        n: Number of GPU workers to spawn (default: 1)

    Example:
        import gt
        gt.gpu_workers(4)  # Use 4 GPUs
        a = gt.randn(128, 64)  # Will auto-shard across 4 workers
    """
    global _num_gpu_workers, _connected

    if _connected:
        raise RuntimeError("gpu_workers() must be called before any tensor operations")

    _num_gpu_workers = n


def _ensure_connected():
    """Ensure we're connected to a server, starting one if needed."""
    global _connected, _auto_server, _client, _num_gpu_workers

    if _connected:
        return

    # Auto-start local server
    import time
    start_time = time.time()
    num_workers = _num_gpu_workers
    print(f"GT: Auto-starting local server with {num_workers} worker(s)...")
    from gt.dispatcher.dispatcher import Dispatcher
    from gt.worker.worker import Worker
    from gt.transport.connection import create_server, Connection

    dispatcher = Dispatcher(host='localhost', port=0)  # Use port 0 for auto-assign
    dispatcher.running = True

    # Create server socket
    server_sock = create_server('localhost', 0)
    actual_port = server_sock.getsockname()[1]
    t1 = time.time()
    print(f"GT: Server socket created ({(t1-start_time)*1000:.1f}ms)")

    # Event to signal when all workers are connected
    workers_ready = threading.Event()

    # Start dispatcher in thread
    def run_dispatcher():
        # Accept N worker connections
        for i in range(num_workers):
            sock, addr = server_sock.accept()
            conn = Connection(sock)
            worker_id = f"auto_worker_{i}" if num_workers > 1 else "auto_worker"
            dispatcher.register_worker(conn, worker_id)

        # Signal that all workers are connected
        workers_ready.set()

        # Accept client connections
        while dispatcher.running:
            try:
                sock, addr = server_sock.accept()
                conn = Connection(sock)
                client_id = f"{addr[0]}:{addr[1]}"
                thread = threading.Thread(
                    target=dispatcher._handle_client,
                    args=(conn, client_id),
                    daemon=True
                )
                thread.start()
            except:
                break

    dispatcher_thread = threading.Thread(target=run_dispatcher, daemon=True)
    dispatcher_thread.start()

    # Start N workers in threads immediately
    t2 = time.time()
    print(f"GT: Spawning {num_workers} worker(s)... ({(t2-t1)*1000:.1f}ms)")
    worker_threads = []
    for i in range(num_workers):
        def make_run_worker(worker_id, gpu_id):
            def run_worker():
                # Set CUDA_VISIBLE_DEVICES to assign this worker to a specific GPU
                import os
                os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

                # Use PyTorch backend for multi-worker (sharding), numpy for single worker
                backend = 'pytorch' if num_workers > 1 else 'numpy'
                worker = Worker(worker_id=worker_id, backend=backend)
                worker.connect_to_dispatcher(dispatcher_host='localhost', dispatcher_port=actual_port)
            return run_worker

        worker_id = f"auto_worker_{i}" if num_workers > 1 else "auto_worker"
        worker_thread = threading.Thread(
            target=make_run_worker(worker_id, i),
            daemon=True
        )
        worker_thread.start()
        worker_threads.append(worker_thread)

    # Wait for all workers to connect
    workers_ready.wait()
    t3 = time.time()
    print(f"GT: All workers connected ({(t3-t2)*1000:.1f}ms)")

    # Connect client
    from gt.client.client import Client
    _client = Client(dispatcher_host='localhost', dispatcher_port=actual_port)
    _client.connect()
    _connected = True
    _auto_server = (dispatcher, server_sock)

    t4 = time.time()
    print(f"GT: Ready! Total startup time: {(t4-start_time)*1000:.1f}ms")


def tensor(data: Union[List, np.ndarray], dtype: str = 'float32', requires_grad: bool = False):
    """
    Create a tensor from data.

    Args:
        data: List or numpy array
        dtype: Data type (e.g., 'float32', 'int32')
        requires_grad: Whether to track gradients (default: False)

    Returns:
        Tensor object

    Example:
        a = gt.tensor([1, 2, 3, 4])
        b = gt.tensor(np.array([[1, 2], [3, 4]]), requires_grad=True)
    """
    _ensure_connected()

    from gt.client.tensor import from_numpy

    if not isinstance(data, np.ndarray):
        data = np.array(data, dtype=dtype)

    return from_numpy(data, requires_grad=requires_grad)


def randn(*shape, dtype: str = 'float32'):
    """
    Create a tensor with random normal values.

    Args:
        *shape: Shape of the tensor
        dtype: Data type

    Returns:
        Tensor object

    Example:
        a = gt.randn(3, 4)
    """
    _ensure_connected()

    from gt.client.tensor import randn as _randn
    return _randn(*shape, dtype=dtype)


def zeros(*shape, dtype: str = 'float32'):
    """
    Create a tensor filled with zeros.

    Args:
        *shape: Shape of the tensor
        dtype: Data type

    Returns:
        Tensor object

    Example:
        a = gt.zeros(3, 4)
    """
    _ensure_connected()

    from gt.client.tensor import zeros as _zeros
    return _zeros(*shape, dtype=dtype)


def from_numpy(array: np.ndarray, requires_grad: bool = False):
    """
    Create a tensor from a numpy array (PyTorch-compatible alias).

    Args:
        array: Numpy array
        requires_grad: Whether to track gradients (default: False)

    Returns:
        Tensor object

    Example:
        import numpy as np
        a = gt.from_numpy(np.array([1, 2, 3]))
    """
    _ensure_connected()

    from gt.client.tensor import from_numpy as _from_numpy
    return _from_numpy(array, requires_grad=requires_grad)


class no_grad:
    """
    Context manager to disable gradient tracking (PyTorch-compatible).

    Example:
        with gt.no_grad():
            # Operations here won't track gradients
            output = model(input)
    """

    def __init__(self):
        self.prev_requires_grad = []

    def __enter__(self):
        # For now, this is a no-op since we don't have a global grad mode
        # In PyTorch, this would set torch.is_grad_enabled() = False
        return self

    def __exit__(self, *args):
        # Restore gradient tracking
        pass


# Cleanup on exit
def _cleanup():
    global _client, _auto_server
    if _client:
        try:
            _client.disconnect()
        except:
            pass
    if _auto_server:
        dispatcher, server_sock = _auto_server
        dispatcher.running = False
        try:
            server_sock.close()
        except:
            pass

atexit.register(_cleanup)
