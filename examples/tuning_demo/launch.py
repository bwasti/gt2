"""
Launch script for 8-GPU distributed training demo.

Starts:
- 1 dispatcher/server
- 8 workers (simulating 8 GPUs)
- Training script with 1F1B pipeline schedule
"""

import subprocess
import time
import sys
import signal
import os

# Process management
processes = []


def cleanup(signum=None, frame=None):
    """Clean up all spawned processes."""
    print("\n\nShutting down...")
    for name, proc in processes:
        if proc.poll() is None:  # Still running
            print(f"  Stopping {name}...")
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
    print("✓ All processes stopped")
    sys.exit(0)


# Register cleanup handlers
signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)


def start_server(port=12345):
    """Start the GT server/dispatcher."""
    print("Starting GT server...")
    proc = subprocess.Popen(
        ['python', '-m', 'gt.server', '-p', str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, 'GT_VERBOSE': '1'}
    )
    processes.append(('Server', proc))
    time.sleep(2)  # Give server time to start
    print("✓ Server started")
    return proc


def start_worker(worker_id, host='localhost', port=12345):
    """Start a GT worker."""
    print(f"Starting worker {worker_id}...")
    proc = subprocess.Popen(
        ['python', '-m', 'gt.worker', '--host', host, '-p', str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, 'GT_VERBOSE': '1', 'WORKER_ID': str(worker_id)}
    )
    processes.append((f'Worker {worker_id}', proc))
    time.sleep(0.5)  # Stagger worker starts
    return proc


def run_training(script='examples/tuning_demo/train.py', args=None):
    """Run the training script."""
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80 + "\n")

    cmd = ['python', script]
    if args:
        cmd.extend(args)

    proc = subprocess.Popen(
        cmd,
        stdout=sys.stdout,
        stderr=sys.stderr,
        env={**os.environ, 'GT_VERBOSE': '1'}
    )
    processes.append(('Training', proc))

    try:
        proc.wait()
    except KeyboardInterrupt:
        cleanup()

    return proc.returncode


def main():
    """Main launch sequence."""
    print("=" * 80)
    print("8-GPU Distributed Training Demo Launcher")
    print("=" * 80)
    print("\nThis will start:")
    print("  - 1 dispatcher/server")
    print("  - 8 workers (simulating 8 GPUs)")
    print("  - Training with 1F1B pipeline schedule")
    print("\nPress Ctrl+C to stop all processes\n")
    print("-" * 80)

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description="Launch 8-GPU training demo")
    parser.add_argument('--port', type=int, default=12345, help='Server port')
    parser.add_argument('--workers', type=int, default=8, help='Number of workers')
    parser.add_argument('--steps', type=int, default=10, help='Training steps')
    parser.add_argument('--batch-size', type=int, default=32, help='Global batch size')
    parser.add_argument('--microbatches', type=int, default=8, help='Number of microbatches')
    parser.add_argument('--no-train', action='store_true', help='Skip training (just start system)')

    args = parser.parse_args()

    try:
        # Start server
        start_server(port=args.port)

        # Start workers
        print(f"\nStarting {args.workers} workers...")
        for worker_id in range(args.workers):
            start_worker(worker_id, port=args.port)
        print(f"✓ All {args.workers} workers started")

        # Wait for system to stabilize
        print("\nWaiting for system to stabilize...")
        time.sleep(3)

        if args.no_train:
            print("\n✓ System ready! (--no-train specified)")
            print("\nYou can now run training manually:")
            print(f"  python examples/tuning_demo/train.py --steps {args.steps}")
            print("\nPress Ctrl+C to shut down the system")
            # Keep processes running
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                cleanup()
        else:
            # Run training
            train_args = [
                '--steps', str(args.steps),
                '--batch-size', str(args.batch_size),
                '--microbatches', str(args.microbatches),
            ]
            returncode = run_training(args=train_args)

            # Training finished
            print("\n" + "=" * 80)
            if returncode == 0:
                print("✓ Training completed successfully!")
            else:
                print(f"✗ Training failed with exit code {returncode}")
            print("=" * 80)

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup()


if __name__ == "__main__":
    main()
