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


@pytest.fixture(scope="session")
def gt_system_4workers():
    """
    Start GT system with 4 workers for sharding tests.

    This fixture runs once per test session and tears down after all tests.
    """
    print("\n=== Starting GT system with 4 workers for testing ===")

    # Start dispatcher in background thread
    from gt.transport.connection import create_server, Connection

    dispatcher = Dispatcher(host='localhost', port=9002)  # Different port for 4-worker tests
    dispatcher.running = True  # IMPORTANT!

    def run_dispatcher():
        server_sock = create_server('localhost', 9002)
        print("Test dispatcher (4 workers) listening on localhost:9002")

        # Accept 4 worker connections
        for i in range(4):
            sock, addr = server_sock.accept()
            conn = Connection(sock)
            worker_id = f"test_worker_{i}"
            dispatcher.register_worker(conn, worker_id)
            print(f"Worker {worker_id} connected")

        print("All 4 workers connected")

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

    # Start 4 workers in background threads
    worker_threads = []
    for i in range(4):
        def make_run_worker(worker_id, delay):
            def run_worker():
                time.sleep(delay)
                worker = Worker(worker_id=worker_id, backend='pytorch')
                worker.connect_to_dispatcher(dispatcher_host='localhost', dispatcher_port=9002)
            return run_worker

        worker_thread = threading.Thread(target=make_run_worker(f"test_worker_{i}", 0.5 + i * 0.1), daemon=True)
        worker_thread.start()
        worker_threads.append(worker_thread)

    time.sleep(2)  # Give system time to start

    print("GT system with 4 workers ready for testing\n")

    yield  # Tests run here

    print("\n=== Shutting down GT system with 4 workers ===")


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


@pytest.fixture
def client_4workers(gt_system_4workers):
    """
    Create a client connected to the 4-worker test GT system.

    Each test gets a fresh client.
    """
    client = Client(dispatcher_host="localhost", dispatcher_port=9002)
    client.connect()
    yield client
    client.disconnect()
