"""
Simplest possible launch script.

Runs everything in one process for now to keep it SIMPLE.

Keep this SIMPLE and READABLE.
"""

import threading
import time


def run_dispatcher(num_workers=1, run_test=False):
    """Run dispatcher."""
    from gt.dispatcher.dispatcher import Dispatcher
    dispatcher = Dispatcher(host='localhost', port=9000)
    dispatcher.running = True  # IMPORTANT: Set running flag!

    # Start workers manually
    from gt.worker.worker import Worker
    from gt.transport.connection import Connection

    print("Starting dispatcher server...")
    import socket
    from gt.transport.connection import create_server

    server_sock = create_server('localhost', 9000)
    print("Dispatcher listening on localhost:9000")

    # Accept worker connections
    print(f"Waiting for {num_workers} worker connection(s)...")
    worker_conns = []
    for i in range(num_workers):
        sock, addr = server_sock.accept()
        conn = Connection(sock)
        worker_id = f"worker_{i}"
        dispatcher.register_worker(conn, worker_id)
        print(f"Worker {worker_id} connected from {addr}")
        worker_conns.append(conn)

    print("\nAll workers connected. Ready for client connections.")

    # If we should run test, start it in a thread
    if run_test:
        time.sleep(0.5)  # Give everything time to settle
        test_thread = threading.Thread(target=run_test_client, daemon=True)
        test_thread.start()

    # Now accept client connections
    while True:
        try:
            sock, addr = server_sock.accept()
            conn = Connection(sock)
            client_id = f"{addr[0]}:{addr[1]}"
            print(f"Client {client_id} connected")

            # Handle client in thread
            thread = threading.Thread(
                target=dispatcher._handle_client,
                args=(conn, client_id),
                daemon=True
            )
            thread.start()
        except Exception as e:
            print(f"Error: {e}")
            break


def run_worker(worker_id):
    """Run a worker."""
    from gt.worker.worker import Worker
    time.sleep(0.5)  # Let dispatcher start
    worker = Worker(worker_id=worker_id, backend='pytorch')
    print(f"Worker {worker_id} connecting to dispatcher...")
    worker.connect_to_dispatcher(dispatcher_host='localhost', dispatcher_port=9000)


def run_test_client():
    """Run the test client."""
    print("\n" + "="*60)
    print("Running test client...")
    print("="*60 + "\n")
    time.sleep(0.5)  # Give system time to be ready

    from gt.client.client import Client
    from gt.client.tensor import randn, from_numpy, zeros
    import numpy as np

    print("Connecting to dispatcher...")
    client = Client(dispatcher_host="localhost", dispatcher_port=9000)
    client.connect()

    print("\n=== Test 1: Create random tensors ===")
    a = randn(3, 3)
    print(f"Created tensor a: {a}")
    print(f"Data:\n{a.data}")

    b = randn(3, 3)
    print(f"\nCreated tensor b: {b}")
    print(f"Data:\n{b.data}")

    print("\n=== Test 2: Matrix multiplication ===")
    c = a @ b
    print(f"Created tensor c = a @ b: {c}")
    print(f"Data:\n{c.data}")

    print("\n=== Test 3: Addition ===")
    d = a + b
    print(f"Created tensor d = a + b: {d}")
    print(f"Data:\n{d.data}")

    print("\n=== Test 4: Create from numpy ===")
    np_array = np.array([[1, 2], [3, 4]], dtype=np.float32)
    e = from_numpy(np_array)
    print(f"Created tensor e from numpy: {e}")
    print(f"Data:\n{e.data}")

    print("\n=== Test 5: Zeros ===")
    f = zeros(2, 2)
    print(f"Created tensor f (zeros): {f}")
    print(f"Data:\n{f.data}")

    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60 + "\n")

    client.disconnect()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Launch GT2 system")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of workers to start (default: 1)")
    parser.add_argument("--test", action="store_true",
                        help="Run test automatically (default: True)")
    parser.add_argument("--no-test", action="store_true",
                        help="Don't run test automatically")
    args = parser.parse_args()

    # By default, run test
    run_test = not args.no_test

    print("Starting GT2 system...")
    if run_test:
        print("Test will run automatically after startup")

    # Start workers in threads
    for i in range(args.workers):
        worker_thread = threading.Thread(target=run_worker, args=(f"worker_{i}",), daemon=True)
        worker_thread.start()

    # Run dispatcher in main thread
    run_dispatcher(num_workers=args.workers, run_test=run_test)
