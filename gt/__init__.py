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

# Dtype constants (PyTorch-compatible)
# These are string constants that work with GT's dtype system
float32 = 'float32'
float64 = 'float64'
int32 = 'int32'
int64 = 'int64'
bool = 'bool'

# Global state
_connected = False
_auto_server = None
_client = None
_num_gpu_workers = 1  # Default to 1 worker in auto-connect mode
_auto_workers = []  # Store worker objects for proper cleanup


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


def _detect_num_gpus():
    """
    Detect number of available GPUs.

    Returns:
        int: Number of GPUs detected, or 1 if detection fails
    """
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.device_count()
    except ImportError:
        pass

    # Fallback: try nvidia-smi
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0:
            gpu_count = len(result.stdout.strip().split('\n'))
            return gpu_count if gpu_count > 0 else 1
    except (subprocess.SubprocessError, FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # No GPUs detected
    return 1


def _ensure_connected():
    """Ensure we're connected to a server, starting one if needed."""
    global _connected, _auto_server, _client, _num_gpu_workers

    if _connected:
        return

    # Auto-start local server
    import time
    import os
    from gt.debug import verbose_print
    start_time = time.time()

    # Check if GT_AUTO_SHARD is enabled and auto-detect GPUs
    auto_shard = os.environ.get('GT_AUTO_SHARD', '0') == '1'
    if auto_shard and _num_gpu_workers == 1:
        # GT_AUTO_SHARD is enabled and user hasn't explicitly set gpu_workers()
        detected_gpus = _detect_num_gpus()
        if detected_gpus > 1:
            verbose_print(f"GT_AUTO_SHARD: Detected {detected_gpus} GPU(s), will use all for auto-sharding")
            _num_gpu_workers = detected_gpus
        else:
            verbose_print(f"GT_AUTO_SHARD: Only 1 GPU detected, auto-sharding disabled")

    num_workers = _num_gpu_workers
    if auto_shard and num_workers > 1:
        verbose_print(f"GT: Auto-starting local server with {num_workers} worker(s) (GT_AUTO_SHARD=1)...")
    else:
        verbose_print(f"GT: Auto-starting local server with {num_workers} worker(s)...")
    from gt.dispatcher.dispatcher import Dispatcher
    from gt.worker.worker import Worker
    from gt.transport.connection import create_server, Connection

    # Check for instruction log file from environment variable
    log_file = os.environ.get('GT_INSTRUCTION_LOG', None)
    if log_file:
        verbose_print(f"GT: Instruction stream logging to: {log_file}")

    # Check for config file from environment variable
    config_file = os.environ.get('GT_CONFIG', None)
    if config_file:
        verbose_print(f"GT: Loading sharding config from: {config_file}")
        from gt.config import load_config as _load_config
        _load_config(config_file)

    # Choose a port (use 59000 for auto-start to avoid conflicts)
    actual_port = 59000

    # Create dispatcher with ZMQ ROUTER
    dispatcher = Dispatcher(host='localhost', port=actual_port, log_file=log_file, console_log=False)
    t1 = time.time()
    verbose_print(f"GT: Dispatcher created on port {actual_port} ({(t1-start_time)*1000:.1f}ms)")

    # Start dispatcher in background thread
    dispatcher_thread = threading.Thread(target=dispatcher.start, daemon=True)
    dispatcher_thread.start()

    # Give dispatcher time to bind
    time.sleep(0.1)

    # Start N workers in threads immediately
    t2 = time.time()
    verbose_print(f"GT: Spawning {num_workers} worker(s)... ({(t2-t1)*1000:.1f}ms)")
    worker_threads = []
    global _auto_workers
    _auto_workers = []  # Reset worker list

    for i in range(num_workers):
        def make_run_worker(worker_id, gpu_id):
            def run_worker():
                # Set CUDA_VISIBLE_DEVICES to assign this worker to a specific GPU
                import os
                os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

                # Backend selection:
                # 1. Use GT_BACKEND env var if set (e.g., GT_BACKEND=numpy)
                # 2. Otherwise default to pytorch if available
                # 3. Fall back to numpy if pytorch not installed
                backend = os.environ.get('GT_BACKEND', None)
                if backend is None:
                    # Try to use pytorch by default
                    try:
                        import torch
                        backend = 'pytorch'
                    except ImportError:
                        backend = 'numpy'

                worker = Worker(worker_id=worker_id, backend=backend)
                _auto_workers.append(worker)  # Store for cleanup
                worker.connect_to_dispatcher(dispatcher_host='localhost', dispatcher_port=actual_port)
            return run_worker

        worker_id = f"auto_worker_{i}" if num_workers > 1 else "auto_worker"
        worker_thread = threading.Thread(
            target=make_run_worker(worker_id, i),
            daemon=True
        )
        worker_thread.start()
        worker_threads.append(worker_thread)

    # Wait for all workers to register (with timeout)
    # This is crucial for auto-start since PyTorch can take a while to load
    timeout = 30.0  # 30 second timeout
    poll_interval = 0.1  # Check every 100ms
    elapsed = 0.0
    while len(dispatcher.workers) < num_workers and elapsed < timeout:
        time.sleep(poll_interval)
        elapsed += poll_interval

    t3 = time.time()

    if len(dispatcher.workers) < num_workers:
        # Timeout - not all workers registered
        verbose_print(f"GT: WARNING - Only {len(dispatcher.workers)}/{num_workers} workers registered after {timeout}s")
        verbose_print(f"GT: This may cause 'No workers available' errors")
    else:
        verbose_print(f"GT: All {num_workers} worker(s) registered ({(t3-t2)*1000:.1f}ms)")

    verbose_print(f"GT: Registered workers: {len(dispatcher.workers)}")

    # Connect client
    from gt.client.client import Client
    _client = Client(dispatcher_host='localhost', dispatcher_port=actual_port)
    _client.connect()
    _connected = True
    _auto_server = dispatcher

    t4 = time.time()
    verbose_print(f"GT: Ready! Total startup time: {(t4-start_time)*1000:.1f}ms")


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


def randn(*shape, dtype: str = 'float32', requires_grad: bool = False):
    """
    Create a tensor with random normal values.

    Args:
        *shape: Shape of the tensor
        dtype: Data type
        requires_grad: Whether to track gradients (default: False)

    Returns:
        Tensor object

    Example:
        a = gt.randn(3, 4)
        b = gt.randn(10, 10, requires_grad=True)
    """
    _ensure_connected()

    from gt.client.tensor import randn as _randn
    return _randn(*shape, dtype=dtype, requires_grad=requires_grad)


def zeros(*shape, dtype: str = 'float32', requires_grad: bool = False):
    """
    Create a tensor filled with zeros.

    Args:
        *shape: Shape of the tensor
        dtype: Data type
        requires_grad: Whether to track gradients (default: False)

    Returns:
        Tensor object

    Example:
        a = gt.zeros(3, 4)
        b = gt.zeros(10, 10, requires_grad=True)
    """
    _ensure_connected()

    from gt.client.tensor import zeros as _zeros
    return _zeros(*shape, dtype=dtype, requires_grad=requires_grad)


def ones(*shape, dtype: str = 'float32', requires_grad: bool = False):
    """
    Create a tensor filled with ones.

    Args:
        *shape: Shape of the tensor
        dtype: Data type
        requires_grad: Whether to track gradients (default: False)

    Returns:
        Tensor object

    Example:
        a = gt.ones(3, 4)
        b = gt.ones(10, 10, requires_grad=True)
    """
    _ensure_connected()

    from gt.client.tensor import ones as _ones
    return _ones(*shape, dtype=dtype, requires_grad=requires_grad)


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


# Signal API for configuration-based sharding
from gt import signals as signal  # Import module for signal.context() and signal.tensor(), renamed to avoid shadowing stdlib

# Config loading
from gt.config import load_config, get_config, get_signal_config, register_config, SignalConfig, ShardConfig

# Debug utilities
from gt import debug  # Import module for debug.get_tape(), debug.get_worker_stats()


# Cleanup on exit
def _cleanup():
    """
    Clean shutdown of auto-started components.

    With daemon threads + ZMQ, graceful shutdown is tricky. We suppress
    all errors to avoid spurious failures during Python exit.
    """
    try:
        # Redirect stderr to suppress ZMQ cleanup assertions
        import sys
        import os

        stderr_fd = sys.stderr.fileno()
        null_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(null_fd, stderr_fd)
        os.close(null_fd)
    except:
        pass


atexit.register(_cleanup)
