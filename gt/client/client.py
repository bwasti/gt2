"""
Client connection to dispatcher.

Keep this SIMPLE and READABLE.
"""

from gt.transport.connection import connect
from gt.debug import verbose_print
import gt.client.tensor as tensor_module


class Client:
    """Client connection to dispatcher."""

    def __init__(self, dispatcher_host="localhost", dispatcher_port=9000):
        self.host = dispatcher_host
        self.port = dispatcher_port
        self.connection = None

    def connect(self):
        """Connect to the dispatcher."""
        self.connection = connect(self.host, self.port)
        # Set global connection in tensor module
        tensor_module._client_connection = self.connection
        verbose_print(f"Connected to dispatcher at {self.host}:{self.port}")

    def disconnect(self):
        """Disconnect from the dispatcher."""
        if self.connection:
            self.connection.close()
            self.connection = None
            tensor_module._client_connection = None
        verbose_print("Disconnected from dispatcher")
