"""
GT Server CLI.

Usage:
    python -m gt.server -p 12345
"""

import argparse
from gt.dispatcher.dispatcher import Dispatcher
from gt.transport.connection import create_server, Connection
import threading


def main():
    parser = argparse.ArgumentParser(description='GT Server (Dispatcher)')
    parser.add_argument('-p', '--port', type=int, required=True,
                        help='Port to listen on')
    parser.add_argument('-H', '--host', type=str, default='0.0.0.0',
                        help='Host to bind to (default: 0.0.0.0)')
    args = parser.parse_args()

    print(f"Starting GT server on {args.host}:{args.port}")

    dispatcher = Dispatcher(host=args.host, port=args.port)
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
