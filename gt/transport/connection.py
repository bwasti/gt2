"""
Simple TCP connection helpers.

Keep this SIMPLE and READABLE.
"""

import socket
import struct
from .protocol import serialize, deserialize


class Connection:
    """Simple TCP connection wrapper with optimizations."""

    def __init__(self, sock):
        self.sock = sock
        # Enable TCP_NODELAY for lower latency (disable Nagle's algorithm)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        # Increase socket buffer sizes
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)  # 1MB
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024 * 1024)  # 1MB

    def send(self, obj):
        """Send an object over the connection."""
        data = serialize(obj)
        # Send length prefix (4 bytes) then data
        length = len(data)
        self.sock.sendall(struct.pack('!I', length))
        self.sock.sendall(data)

    def recv(self):
        """Receive an object from the connection."""
        # Read length prefix
        length_data = self._recv_exactly(4)
        length = struct.unpack('!I', length_data)[0]

        # Read data
        data = self._recv_exactly(length)
        return deserialize(data)

    def _recv_exactly(self, n):
        """Receive exactly n bytes (optimized)."""
        # Use a larger buffer to reduce number of recv() calls
        data = bytearray(n)
        view = memoryview(data)
        pos = 0

        while pos < n:
            # Request remaining bytes, socket will return what's available
            nread = self.sock.recv_into(view[pos:])
            if nread == 0:
                raise ConnectionError("Connection closed")
            pos += nread

        return bytes(data)

    def close(self):
        """Close the connection."""
        self.sock.close()


def create_server(host, port):
    """Create a TCP server socket."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.listen(10)
    return sock


def connect(host, port):
    """Connect to a TCP server."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))
    return Connection(sock)
