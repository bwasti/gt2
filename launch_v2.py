"""
Better launch script for GT2 system.

Starts dispatcher in main thread, workers in subprocesses.
Workers connect to dispatcher.

Keep this SIMPLE and READABLE.
"""

import subprocess
import time
import sys
import threading


def launch_system(num_workers=1):
    """
    Launch the GT2 system.

    Args:
        num_workers: Number of worker processes to start (1 per GPU)
    """
    worker_processes = []

    # Start dispatcher in a thread
    print("Starting dispatcher...")
    from dispatcher.dispatcher import Dispatcher
    dispatcher = Dispatcher(host='localhost', port=9000)

    # Start dispatcher in background thread
    dispatcher_thread = threading.Thread(target=dispatcher.start, daemon=True)
    dispatcher_thread.start()
    time.sleep(1)  # Give dispatcher time to start

    # Start workers in subprocesses
    for i in range(num_workers):
        print(f"Starting worker {i}...")
        worker_proc = subprocess.Popen(
            [sys.executable, "-c", f"""
import sys
sys.path.insert(0, '/home/bwasti/oss/gt2')
from worker.worker import Worker
import time
time.sleep(0.5)  # Let dispatcher fully start
worker = Worker(worker_id='worker_{i}', backend='numpy')
worker.connect_to_dispatcher(dispatcher_host='localhost', dispatcher_port=9000)
"""],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        worker_processes.append((f"worker_{i}", worker_proc))
        time.sleep(0.5)

    # Register workers with dispatcher
    # For now, the workers connect as clients, but we can mark them as workers
    time.sleep(1)

    print(f"\nGT2 system started with {num_workers} worker(s)")
    print("Dispatcher: localhost:9000")
    print("\nYou can now run test_simple.py in another terminal")
    print("\nPress Ctrl+C to shutdown")

    # Wait for interrupt
    try:
        while True:
            # Check if any worker process died
            for name, proc in worker_processes:
                if proc.poll() is not None:
                    print(f"\n{name} process died!")
                    output = proc.stdout.read()
                    if output:
                        print(f"{name} output:\n{output}")

            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        dispatcher.stop()
        for name, proc in worker_processes:
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
            print(f"Stopped {name}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Launch GT2 system")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of workers to start (default: 1)")
    args = parser.parse_args()

    launch_system(num_workers=args.workers)
