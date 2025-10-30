"""
Pytest configuration and fixtures.

Keep this SIMPLE and READABLE.
"""

import pytest
import threading
import time
import sys

sys.path.insert(0, '/home/bwasti/oss/gt2')

from gt.dispatcher.dispatcher import Dispatcher
from gt.worker.worker import Worker
from gt.client.client import Client


@pytest.fixture(scope="session")
def gt_system():
    """
    Start GT system (dispatcher + worker) for testing.

    This fixture runs once per test session and tears down after all tests.
    """
    print("\n=== Starting GT system for testing ===")

    # Start dispatcher in background thread
    from gt.transport.connection import create_server, Connection

    dispatcher = Dispatcher(host='localhost', port=9001)  # Use different port for tests

    def run_dispatcher():
        server_sock = create_server('localhost', 9001)
        print("Test dispatcher listening on localhost:9001")

        # Accept worker connection
        sock, addr = server_sock.accept()
        conn = Connection(sock)
        dispatcher.register_worker(conn, "test_worker")
        print(f"Test worker connected")

        # Accept client connections
        while True:
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
    time.sleep(0.5)

    # Start worker in background thread
    def run_worker():
        time.sleep(0.5)
        worker = Worker(worker_id="test_worker", backend='numpy')
        worker.connect_to_dispatcher(dispatcher_host='localhost', dispatcher_port=9001)

    worker_thread = threading.Thread(target=run_worker, daemon=True)
    worker_thread.start()
    time.sleep(1)  # Give system time to start

    print("GT system ready for testing\n")

    yield  # Tests run here

    print("\n=== Shutting down GT system ===")


@pytest.fixture
def client(gt_system):
    """
    Create a client connected to the test GT system.

    Each test gets a fresh client.
    """
    client = Client(dispatcher_host="localhost", dispatcher_port=9001)
    client.connect()
    yield client
    client.disconnect()
