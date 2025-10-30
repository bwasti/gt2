"""
GT Worker CLI.

Usage:
    python -m gt.worker -h localhost -p 12345
"""

import argparse
from gt.worker.worker import Worker


def main():
    parser = argparse.ArgumentParser(description='GT Worker')
    parser.add_argument('--host', type=str, default='localhost',
                        help='Dispatcher host (default: localhost)')
    parser.add_argument('-p', '--port', type=int, required=True,
                        help='Dispatcher port')
    parser.add_argument('--id', type=str, default=None,
                        help='Worker ID (default: auto-generated)')
    parser.add_argument('--backend', type=str, default='numpy',
                        choices=['numpy', 'pytorch', 'jax'],
                        help='Backend to use (default: numpy)')
    args = parser.parse_args()

    worker_id = args.id or f"worker_{args.host}_{args.port}"

    print(f"Starting GT worker: {worker_id}")
    print(f"Connecting to dispatcher at {args.host}:{args.port}")

    worker = Worker(worker_id=worker_id, backend=args.backend)

    try:
        worker.connect_to_dispatcher(
            dispatcher_host=args.host,
            dispatcher_port=args.port
        )
    except KeyboardInterrupt:
        print("\nWorker stopped")
    except Exception as e:
        print(f"Worker error: {e}")


if __name__ == '__main__':
    main()
