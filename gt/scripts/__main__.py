"""
GT Scripts module entry point.

Usage:
    python -m gt.scripts.visualize <tape_log_path>
    python -m gt.scripts.top
"""

import sys
import os

if __name__ == '__main__':
    # Determine which script to run based on command line
    if len(sys.argv) < 2:
        print("GT Scripts")
        print("\nAvailable commands:")
        print("  visualize <tape_log_path>  - Visualize instruction tape timeline")
        print("  top [--pid PID]            - Real-time worker monitoring (htop-style)")
        print("\nUsage:")
        print("  python -m gt.scripts.visualize <tape_log_path> [--output <path>] [--dpi <dpi>]")
        print("  python -m gt.scripts.top [log_path] [--pid PID]")
        sys.exit(1)

    command = sys.argv[1]

    if command == 'visualize' or command.endswith('.log'):
        # If first arg is a log file, run visualize
        from gt.scripts.visualize import main
        if command.endswith('.log'):
            # Shift args so visualize gets the log path as first arg
            sys.argv = ['visualize'] + sys.argv[1:]
        else:
            sys.argv = sys.argv[1:]  # Remove 'visualize' from args
        main()
    elif command == 'top':
        from gt.scripts.top import main
        sys.argv = sys.argv[1:]  # Remove 'top' from args
        main()
    else:
        print(f"Unknown command: {command}")
        print("Available commands: visualize, top")
        sys.exit(1)
