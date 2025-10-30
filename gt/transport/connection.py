"""
Simple TCP connection helpers.

Keep this SIMPLE and READABLE.
"""

import socket
import struct
from .protocol import serialize, deserialize


class Connection:
    """Simple TCP connection wrapper."""

    def __init__(self, sock):
        self.sock = sock

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
        """Receive exactly n bytes."""
        data = b''
        while len(data) < n:
            chunk = self.sock.recv(n - len(data))
            if not chunk:
                raise ConnectionError("Connection closed")
            data += chunk
        return data

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
