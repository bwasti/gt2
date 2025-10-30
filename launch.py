"""
Launch script for GT2 system.

Starts dispatcher and workers, connects them together.

Keep this SIMPLE and READABLE.
"""

import subprocess
import time
import sys
import signal
import os


def launch_system(num_workers=1):
    """
    Launch the GT2 system.

    Args:
        num_workers: Number of worker processes to start (1 per GPU)
    """
    processes = []

    # Start dispatcher
    print("Starting dispatcher...")
    dispatcher_proc = subprocess.Popen(
        [sys.executable, "-c", """
import sys
sys.path.insert(0, '/home/bwasti/oss/gt2')
from dispatcher.dispatcher import Dispatcher
dispatcher = Dispatcher(host='localhost', port=9000)
dispatcher.start()
"""],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    processes.append(("dispatcher", dispatcher_proc))
    time.sleep(1)  # Give dispatcher time to start

    # Start workers
    for i in range(num_workers):
        print(f"Starting worker {i}...")
        worker_proc = subprocess.Popen(
            [sys.executable, "-c", f"""
import sys
sys.path.insert(0, '/home/bwasti/oss/gt2')
from worker.worker import Worker
worker = Worker(worker_id='worker_{i}', backend='numpy')
worker.connect_to_dispatcher(dispatcher_host='localhost', dispatcher_port=9000)
"""],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        processes.append((f"worker_{i}", worker_proc))
        time.sleep(0.5)

    print(f"\nGT2 system started with {num_workers} worker(s)")
    print("Dispatcher: localhost:9000")
    print("\nPress Ctrl+C to shutdown")

    # Wait for interrupt
    try:
        while True:
            # Check if any process died
            for name, proc in processes:
                if proc.poll() is not None:
                    print(f"\n{name} process died!")
                    # Print output
                    output = proc.stdout.read()
                    if output:
                        print(f"{name} output:\n{output}")
                    shutdown_system(processes)
                    return

            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        shutdown_system(processes)


def shutdown_system(processes):
    """Shutdown all processes."""
    for name, proc in processes:
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
