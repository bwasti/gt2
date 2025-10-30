"""
GT command-line interface.

Usage:
    python -m gt.server -p 12345
    python -m gt.worker -h localhost -p 12345
"""

import sys

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python -m gt.server -p <port>")
        print("  python -m gt.worker -h <host> -p <port>")
        sys.exit(1)

    command = sys.argv[1]
    if command == 'server':
        from gt.server.__main__ import main as server_main
        server_main()
    elif command == 'worker':
        from gt.worker.__main__ import main as worker_main
        worker_main()
    else:
        print(f"Unknown command: {command}")
        print("Available commands: server, worker")
        sys.exit(1)

if __name__ == '__main__':
    main()
