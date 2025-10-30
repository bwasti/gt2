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


def _ensure_connected():
    """Ensure we're connected to a server, starting one if needed."""
    global _connected, _auto_server, _client

    if _connected:
        return

    # Auto-start local server
    print("GT: Auto-starting local server...")
    from gt.dispatcher.dispatcher import Dispatcher
    from gt.worker.worker import Worker
    from gt.transport.connection import create_server, Connection
    import time

    dispatcher = Dispatcher(host='localhost', port=0)  # Use port 0 for auto-assign
    dispatcher.running = True

    # Create server socket
    server_sock = create_server('localhost', 0)
    actual_port = server_sock.getsockname()[1]
    print(f"GT: Server listening on localhost:{actual_port}")

    # Start dispatcher in thread
    def run_dispatcher():
        # Accept worker connection
        sock, addr = server_sock.accept()
        conn = Connection(sock)
        dispatcher.register_worker(conn, "auto_worker")

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
    time.sleep(0.1)

    # Start worker in thread
    def run_worker():
        time.sleep(0.1)
        worker = Worker(worker_id="auto_worker", backend='numpy')
        worker.connect_to_dispatcher(dispatcher_host='localhost', dispatcher_port=actual_port)

    worker_thread = threading.Thread(target=run_worker, daemon=True)
    worker_thread.start()
    time.sleep(0.2)

    # Connect client
    from gt.client.client import Client
    _client = Client(dispatcher_host='localhost', dispatcher_port=actual_port)
    _client.connect()
    _connected = True
    _auto_server = (dispatcher, server_sock)

    print("GT: Ready!")


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
