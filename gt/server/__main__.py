"""
GT Server CLI.

Usage:
    python -m gt.server -p 12345
    python -m gt.server -p 12345 --spawn_gpu_workers 4
"""

import argparse
import time
import os
import subprocess
from gt.dispatcher.dispatcher import Dispatcher
from gt.transport.connection import create_server, Connection
import threading


def _spawn_workers(num_workers: int, host: str, port: int):
    """Spawn N GPU workers in background processes."""
    from gt.worker.worker import Worker

    worker_threads = []
    for i in range(num_workers):
        def make_run_worker(worker_id, gpu_id, delay):
            def run_worker():
                # Set CUDA_VISIBLE_DEVICES to assign this worker to a specific GPU
                # This ensures each worker only sees one GPU and uses it exclusively
                os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

                time.sleep(delay)
                # Use PyTorch backend for multi-worker setups (enables sharding)
                backend = 'pytorch' if num_workers > 1 else 'numpy'
                worker = Worker(worker_id=worker_id, backend=backend)
                worker.connect_to_dispatcher(dispatcher_host=host, dispatcher_port=port)
            return run_worker

        worker_id = f"spawned_worker_{i}"
        worker_thread = threading.Thread(
            target=make_run_worker(worker_id, i, 0.5 + i * 0.1),
            daemon=True
        )
        worker_thread.start()
        worker_threads.append(worker_thread)
        print(f"  Spawned worker {i+1}/{num_workers} (GPU {i})")

    # Give workers time to start before dispatcher begins accepting
    time.sleep(0.5 + num_workers * 0.1)


def main():
    parser = argparse.ArgumentParser(description='GT Server (Dispatcher)')
    parser.add_argument('-p', '--port', type=int, required=True,
                        help='Port to listen on')
    parser.add_argument('-H', '--host', type=str, default='0.0.0.0',
                        help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--spawn_gpu_workers', type=int, default=0,
                        help='Number of GPU workers to spawn automatically (default: 0 - manual worker setup)')
    parser.add_argument('--log-file', type=str, default=None,
                        help='Write operation log to file (default: None)')
    parser.add_argument('--no-console-log', action='store_true',
                        help='Disable console logging (default: enabled)')
    args = parser.parse_args()

    print(f"Starting GT server on {args.host}:{args.port}")
    if args.log_file:
        print(f"Writing operation log to: {args.log_file}")

    # Spawn workers if requested
    if args.spawn_gpu_workers > 0:
        print(f"Spawning {args.spawn_gpu_workers} GPU worker(s)...")
        _spawn_workers(args.spawn_gpu_workers, args.host, args.port)

    dispatcher = Dispatcher(
        host=args.host,
        port=args.port,
        log_file=args.log_file,
        console_log=not args.no_console_log
    )
    dispatcher.running = True

    server_sock = create_server(args.host, args.port)
    print(f"GT server listening on {args.host}:{args.port}")
    print("Waiting for workers to connect...")

    # Accept connections (workers and clients)
    while dispatcher.running:
        try:
            sock, addr = server_sock.accept()
            conn = Connection(sock)
            print(f"Connection from {addr}")

            # For now, assume first connection is a worker, rest are clients
            # In a real system, we'd do a handshake
            if not dispatcher.workers:
                worker_id = f"worker_{len(dispatcher.workers)}"
                dispatcher.register_worker(conn, worker_id)
                print(f"Registered worker: {worker_id}")
            else:
                # Handle as client
                client_id = f"{addr[0]}:{addr[1]}"
                thread = threading.Thread(
                    target=dispatcher._handle_client,
                    args=(conn, client_id),
                    daemon=True
                )
                thread.start()
        except KeyboardInterrupt:
            print("\nShutting down server...")
            break
        except Exception as e:
            print(f"Error: {e}")
            if dispatcher.running:
                continue
            else:
                break

    server_sock.close()
    print("Server stopped")


if __name__ == '__main__':
    main()
