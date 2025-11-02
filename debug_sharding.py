"""
Simple script to debug sharded tensor + reduction.

This replicates the failing test_distributed_sum_4_workers.
"""
import sys
import time
import threading
from gt.dispatcher.dispatcher import Dispatcher
from gt.worker.worker import Worker
from gt.client.client import Client
from gt.client.tensor import randn

print("=== Starting GT system with 4 workers ===")

# Start dispatcher
dispatcher = Dispatcher(host='localhost', port=9003, console_log=True)

def run_dispatcher():
    print("Dispatcher starting on localhost:9003")
    dispatcher.start()

dispatcher_thread = threading.Thread(target=run_dispatcher, daemon=True)
dispatcher_thread.start()
time.sleep(0.5)

# Start 4 workers
worker_threads = []
for i in range(4):
    def make_run_worker(worker_id):
        def run_worker():
            time.sleep(0.5 + i * 0.1)
            worker = Worker(worker_id=worker_id, backend='pytorch')
            worker.connect_to_dispatcher(dispatcher_host='localhost', dispatcher_port=9003)
        return run_worker

    worker_thread = threading.Thread(target=make_run_worker(f"worker_{i}"), daemon=True)
    worker_thread.start()
    worker_threads.append(worker_thread)

time.sleep(2)  # Give system time to start
print("System ready\n")

# Connect client
client = Client(dispatcher_host="localhost", dispatcher_port=9003)
client.connect()

print("=== Creating sharded tensor ===")
# Create tensor that will be sharded (128 = 32 per worker with 4 workers)
a = randn(128, 64)
print(f"Created tensor a with shape (128, 64)")
print(f"Tensor ID: {a.id}")

print("\n=== Computing sum (this triggers distributed reduction) ===")
try:
    result = a.sum()
    print(f"Sum computed successfully!")
    print(f"Result tensor ID: {result.id}")

    print("\n=== Getting result data ===")
    data = result.data.numpy()
    print(f"Result: {data}")
    print(f"Result shape: {data.shape}")
    print("✓ SUCCESS")
except Exception as e:
    print(f"✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

client.disconnect()
dispatcher.stop()
