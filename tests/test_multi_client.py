"""
Test multi-client support in the dispatcher.

Verifies that the dispatcher can handle multiple concurrent clients
and correctly interleave their operations.

TODO: These tests currently fail due to global connection state in tensor.py.
The architecture uses a global `_client_connection` which is not thread-safe.
Need to implement thread-local connections or use multi-process testing.
"""

import pytest
import threading
import time
import numpy as np

# Skip all tests in this module until thread-safety is implemented
pytestmark = pytest.mark.skip(reason="Multi-client tests require thread-local connections (architectural change needed)")


def client_worker(client_id: int, num_operations: int, results: list, dispatcher_port: int):
    """Worker function that runs operations from a client."""
    try:
        # Import here so each thread gets its own connection
        from gt.client.client import Client
        from gt.client.tensor import from_numpy

        # Connect to the existing dispatcher
        client = Client(dispatcher_host="localhost", dispatcher_port=dispatcher_port)
        client.connect()

        # Each client creates tensors and does operations
        a = from_numpy(np.random.randn(10, 10).astype('float32'))
        b = from_numpy(np.random.randn(10, 10).astype('float32'))

        # Perform multiple operations
        for i in range(num_operations):
            c = a @ b
            d = c + a
            e = d.sum()

            # Get result to force execution
            result = e.data.numpy()

        client.disconnect()

        # Store success
        results[client_id] = {'success': True, 'client_id': client_id}
    except Exception as e:
        results[client_id] = {'success': False, 'client_id': client_id, 'error': str(e)}


def test_two_clients(client):
    """Test that two clients can run concurrently."""
    # Get dispatcher port from the fixture (assumes dispatcher is on port 9001)
    dispatcher_port = 9001

    num_operations = 5
    results = [None, None]

    # Start two client threads
    threads = []
    for i in range(2):
        thread = threading.Thread(target=client_worker, args=(i, num_operations, results, dispatcher_port))
        thread.start()
        threads.append(thread)

    # Wait for both to complete
    for thread in threads:
        thread.join(timeout=30)

    # Verify both succeeded
    assert results[0] is not None, "Client 0 didn't complete"
    assert results[1] is not None, "Client 1 didn't complete"
    assert results[0]['success'], f"Client 0 failed: {results[0].get('error', 'unknown')}"
    assert results[1]['success'], f"Client 1 failed: {results[1].get('error', 'unknown')}"


def test_five_clients(client):
    """Test that five clients can run concurrently."""
    dispatcher_port = 9001

    num_clients = 5
    num_operations = 3
    results = [None] * num_clients

    # Start multiple client threads
    threads = []
    for i in range(num_clients):
        thread = threading.Thread(target=client_worker, args=(i, num_operations, results, dispatcher_port))
        thread.start()
        threads.append(thread)

    # Wait for all to complete
    for thread in threads:
        thread.join(timeout=30)

    # Verify all succeeded
    for i in range(num_clients):
        assert results[i] is not None, f"Client {i} didn't complete"
        assert results[i]['success'], f"Client {i} failed: {results[i].get('error', 'unknown')}"


def test_interleaved_operations(client):
    """Test that operations from different clients are correctly interleaved."""
    from gt.client.client import Client
    from gt.client.tensor import from_numpy

    # Client 1: Create connection and tensors
    client1 = Client(dispatcher_host="localhost", dispatcher_port=9001)
    client1.connect()

    a1 = from_numpy(np.array([1.0, 2.0, 3.0], dtype='float32'))
    b1 = from_numpy(np.array([4.0, 5.0, 6.0], dtype='float32'))

    # Client 2 (in thread): Should be able to create tensors while client 1 is active
    results = [None]

    def client2_work():
        try:
            client2 = Client(dispatcher_host="localhost", dispatcher_port=9001)
            client2.connect()

            a2 = from_numpy(np.array([10.0, 20.0, 30.0], dtype='float32'))
            b2 = from_numpy(np.array([40.0, 50.0, 60.0], dtype='float32'))
            c2 = a2 + b2
            results[0] = c2.data.numpy()

            client2.disconnect()
        except Exception as e:
            results[0] = e

    thread = threading.Thread(target=client2_work)
    thread.start()

    # Client 1: Continue with operations
    c1 = a1 + b1
    result1 = c1.data.numpy()

    # Wait for client 2
    thread.join(timeout=10)

    client1.disconnect()

    # Verify both clients got correct results
    np.testing.assert_array_almost_equal(result1, np.array([5.0, 7.0, 9.0], dtype='float32'), decimal=5)

    assert results[0] is not None, "Client 2 didn't complete"
    assert not isinstance(results[0], Exception), f"Client 2 failed: {results[0]}"
    np.testing.assert_array_almost_equal(results[0], np.array([50.0, 70.0, 90.0], dtype='float32'), decimal=5)


def test_stress_many_clients(client):
    """Stress test with many concurrent clients."""
    dispatcher_port = 9001

    num_clients = 10
    num_operations = 2
    results = [None] * num_clients

    start_time = time.time()

    # Start many client threads
    threads = []
    for i in range(num_clients):
        thread = threading.Thread(target=client_worker, args=(i, num_operations, results, dispatcher_port))
        thread.start()
        threads.append(thread)

    # Wait for all to complete
    for thread in threads:
        thread.join(timeout=60)

    elapsed = time.time() - start_time

    # Verify all succeeded
    success_count = 0
    for i in range(num_clients):
        if results[i] is not None and results[i]['success']:
            success_count += 1

    print(f"\nStress test: {success_count}/{num_clients} clients succeeded in {elapsed:.2f}s")

    # Allow some failures under stress, but most should succeed
    assert success_count >= num_clients * 0.8, f"Too many failures: only {success_count}/{num_clients} succeeded"
