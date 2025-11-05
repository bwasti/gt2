"""
ZMQ-based connection helpers for high-performance messaging.

ZMQ provides:
- Automatic message batching and queueing
- Higher throughput than raw TCP
- Built-in reconnection handling
- Multiple messaging patterns (DEALER/ROUTER, REQ/REP, PUSH/PULL, etc.)
- IPC transport for localhost (Unix domain sockets - bypasses TCP/IP stack)

Performance optimization:
- localhost connections use ipc:// (Unix domain sockets)
- Remote connections use tcp://
- IPC provides ~30-50% lower latency than TCP for local communication

Keep this SIMPLE and READABLE.
"""

import zmq
from .protocol import serialize, deserialize


class Connection:
    """ZMQ connection wrapper for client-dispatcher or dispatcher-worker communication."""

    def __init__(self, socket):
        self.socket = socket

        # Performance optimizations for low-latency, high-throughput
        # These apply to both IPC and TCP

        # Send immediately without waiting for more messages
        # (Similar to TCP_NODELAY, but for ZMQ's internal batching)
        self.socket.setsockopt(zmq.IMMEDIATE, 1)

        # Increase high water marks for better throughput
        # (Allow more messages to queue before blocking)
        self.socket.setsockopt(zmq.SNDHWM, 10000)  # Send high water mark
        self.socket.setsockopt(zmq.RCVHWM, 10000)  # Receive high water mark

        # Set linger to 0 for faster shutdown (don't wait for unsent messages)
        self.socket.setsockopt(zmq.LINGER, 0)

        # Increase kernel socket buffer sizes (helps with burst traffic)
        self.socket.setsockopt(zmq.SNDBUF, 1048576)  # 1MB send buffer
        self.socket.setsockopt(zmq.RCVBUF, 1048576)  # 1MB receive buffer

    def send(self, obj):
        """Send an object over the connection.

        DEALER sockets send: [empty, data]
        """
        data = serialize(obj)
        self.socket.send(b'', zmq.SNDMORE)
        self.socket.send(data)

    def recv(self):
        """Receive an object from the connection.

        DEALER sockets receive: [empty, data]
        """
        empty = self.socket.recv()  # Empty delimiter
        data = self.socket.recv()
        return deserialize(data)

    def close(self):
        """Close the connection."""
        self.socket.close()


def create_server(host, port):
    """Create a ZMQ ROUTER server socket.

    For localhost, uses IPC (Unix domain sockets) for lower latency.
    For remote hosts, uses TCP.
    """
    context = zmq.Context()
    socket = context.socket(zmq.ROUTER)

    # Performance optimizations
    socket.setsockopt(zmq.IMMEDIATE, 1)      # No message queueing to disconnected peers
    socket.setsockopt(zmq.SNDHWM, 10000)     # High water mark for send queue
    socket.setsockopt(zmq.RCVHWM, 10000)     # High water mark for receive queue
    socket.setsockopt(zmq.LINGER, 0)         # Don't wait for unsent messages on close
    socket.setsockopt(zmq.SNDBUF, 1048576)   # 1MB send buffer
    socket.setsockopt(zmq.RCVBUF, 1048576)   # 1MB receive buffer

    # Optimization: Use IPC for localhost connections
    if host in ('localhost', '127.0.0.1', '0.0.0.0'):
        # IPC (Unix domain socket) - faster than TCP for local communication
        endpoint = f"ipc:///tmp/gt_dispatcher_{port}.ipc"
        socket.bind(endpoint)
    else:
        # TCP for remote connections
        socket.bind(f"tcp://{host}:{port}")

    return socket


def connect(host, port):
    """Connect to a ZMQ ROUTER server using DEALER socket.

    For localhost, uses IPC (Unix domain sockets) for lower latency.
    For remote hosts, uses TCP.
    """
    context = zmq.Context()
    socket = context.socket(zmq.DEALER)

    # Optimization: Use IPC for localhost connections
    if host in ('localhost', '127.0.0.1', '0.0.0.0'):
        # IPC (Unix domain socket) - faster than TCP for local communication
        endpoint = f"ipc:///tmp/gt_dispatcher_{port}.ipc"
        socket.connect(endpoint)
    else:
        # TCP for remote connections
        socket.connect(f"tcp://{host}:{port}")

    return Connection(socket)
