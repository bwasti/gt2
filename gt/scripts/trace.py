#!/usr/bin/env python3
"""
GT Trace Capture - Record dispatcher event stream to log file

Connects to running dispatcher and captures instruction stream for analysis.
Output is compatible with gt.scripts.visualize for timeline visualization.

Usage:
    # Capture 2 seconds of trace
    python -m gt.scripts.trace -s 2 --dir traces/

    # Capture first 100 events (with 10 second timeout)
    python -m gt.scripts.trace -s 10 -n 100 --dir traces/

    # Capture from specific dispatcher port
    python -m gt.scripts.trace -s 5 --port 9000 --dir traces/

    # Then visualize the captured trace
    python -m gt.scripts.visualize traces/trace_*.log
"""

import sys
import time
import os
import json
import argparse
from datetime import datetime
from pathlib import Path

try:
    import zmq
except ImportError:
    print("Error: 'pyzmq' library required")
    print("Install with: pip install pyzmq")
    sys.exit(1)


def find_dispatcher_port():
    """Try to find dispatcher port from running processes or auto-start."""
    import socket

    # Check auto-start port first (59000)
    if _check_port_open('localhost', 59001):  # Monitor port is +1
        return 59000

    # Check common ports
    for port in [9000, 9001, 9002]:
        if _check_port_open('localhost', port + 1):  # Monitor port is +1
            return port

    # Try to find from process list
    try:
        import psutil
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info.get('cmdline', [])
                if cmdline and any('dispatcher' in arg.lower() or 'gt.server' in arg for arg in cmdline):
                    # Look for -p or --port flag
                    for i, arg in enumerate(cmdline):
                        if arg in ['-p', '--port'] and i + 1 < len(cmdline):
                            try:
                                return int(cmdline[i + 1])
                            except ValueError:
                                pass
                    # Default port
                    return 9000
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except ImportError:
        pass

    return None


def _check_port_open(host: str, port: int) -> bool:
    """Check if a port is open and accepting connections."""
    import socket
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.1)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False


def format_log_entry(entry: dict, start_time: float) -> str:
    """Format a log entry in the same format as InstructionStream."""
    elapsed = entry["timestamp"]
    seq = entry["seq"]
    event = entry["event"]
    client = entry["client"]
    command = entry["command"]
    details = entry["details"]
    size_bytes = entry.get("size_bytes", 0)

    # Determine source type
    if "WORKER" in event:
        source_type = "WORKER"
    elif event in ["CONNECT", "DISCONNECT"]:
        source_type = "CLIENT"
    else:
        source_type = "CLIENT"

    # Format size nicely
    if size_bytes == 0:
        size_str = ""
    elif size_bytes < 1024:
        size_str = f"{size_bytes}B"
    elif size_bytes < 1024 * 1024:
        size_str = f"{size_bytes / 1024:.1f}KB"
    else:
        size_str = f"{size_bytes / (1024 * 1024):.2f}MB"

    # Format with nice alignment
    time_str = f"{elapsed:7.3f}s"
    seq_str = f"#{seq:04d}"
    event_str = f"{event:12s}"
    source_str = f"{source_type} {client:20s}"
    command_str = f"{command:15s}"

    # Build the line with size
    if size_str:
        size_field = f"{size_str:>10s}"
        if details:
            line = f"{time_str} | {seq_str} | {event_str} | {source_str} | {command_str} | {size_field} | {details}"
        else:
            line = f"{time_str} | {seq_str} | {event_str} | {source_str} | {command_str} | {size_field}"
    else:
        if details:
            line = f"{time_str} | {seq_str} | {event_str} | {source_str} | {command_str} | {details}"
        else:
            line = f"{time_str} | {seq_str} | {event_str} | {source_str} | {command_str}"

    return line


def capture_trace(host: str, port: int, duration: float, output_file: str, max_events: int = None):
    """Capture trace from dispatcher for specified duration or event limit."""
    monitor_port = port + 1

    # Connect to dispatcher monitor socket
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.SUBSCRIBE, b'')  # Subscribe to all messages
    socket.setsockopt(zmq.RCVTIMEO, 100)  # 100ms timeout

    monitor_url = f"tcp://{host}:{monitor_port}"
    socket.connect(monitor_url)

    print(f"GT Trace Capture")
    print(f"Connected to: {monitor_url}")
    if max_events:
        print(f"Limit: {max_events} events (max {duration}s)")
    else:
        print(f"Duration: {duration}s")
    print(f"Output: {output_file}")
    print(f"Capturing...\n")

    start_time = time.time()
    end_time = start_time + duration
    events_captured = 0

    # Open output file and write header
    with open(output_file, 'w', buffering=1) as f:
        # Write header
        limit_desc = f"{max_events} events" if max_events else f"{duration}s"
        header = f"""
{'='*100}
GT2 Dispatcher Instruction Stream (Trace Capture)
Captured: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Limit: {limit_desc}
{'='*100}

This log records all instructions flowing through the dispatcher:
  - Client commands (RECV/SEND)
  - Worker operations (WORKER_SEND/WORKER_RECV)
  - Connection events (CONNECT/DISCONNECT)

Format: <elapsed> | #<seq> | <event_type> | <source> | <command> | <details>

Column Details:
  elapsed  = Seconds since capture start (with millisecond precision)
  seq      = Sequence number (increments for each instruction)
  event    = Event type (CONNECT, RECV, SEND, WORKER_SEND, WORKER_RECV, DISCONNECT, ERROR)
  source   = Source identifier (CLIENT/WORKER with ID)
  command  = Command type (e.g., CreateTensor, BinaryOp, UnaryOp)
  details  = Additional context (tensor IDs, operation names, etc.)

""".lstrip()
        f.write(header)

        # Capture events for specified duration or until max_events reached
        while time.time() < end_time:
            # Check if we've hit the event limit
            if max_events and events_captured >= max_events:
                break

            try:
                # Receive event (non-blocking with timeout)
                msg = socket.recv()
                event = json.loads(msg.decode('utf-8'))

                # Format and write to file
                log_line = format_log_entry(event, start_time)
                f.write(log_line + "\n")
                events_captured += 1

            except zmq.Again:
                # Timeout - no message received
                pass
            except Exception as e:
                # Parsing error, skip
                print(f"Warning: Failed to parse event: {e}")
                pass

    socket.close()
    context.term()

    elapsed = time.time() - start_time
    print(f"Capture complete!")
    print(f"Events captured: {events_captured}")
    if max_events and events_captured >= max_events:
        print(f"Stopped: Event limit reached")
    print(f"Actual duration: {elapsed:.2f}s")
    print(f"Output file: {output_file}")
    print(f"\nVisualize with:")
    print(f"  python -m gt.scripts.visualize {output_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="GT Trace Capture - Record dispatcher event stream to log file"
    )
    parser.add_argument(
        "-s", "--seconds",
        type=float,
        default=5.0,
        help="Duration to capture in seconds (default: 5.0)"
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Dispatcher host (default: localhost)"
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Dispatcher port (default: auto-detect)"
    )
    parser.add_argument(
        "--dir",
        default=".",
        help="Output directory for trace file (default: current directory)"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output filename (default: trace_YYYYMMDD_HHMMSS.log)"
    )
    parser.add_argument(
        "-n", "--max-events",
        type=int,
        help="Maximum number of events to capture (default: unlimited)"
    )

    args = parser.parse_args()

    # Determine port
    port = args.port
    if not port:
        port = find_dispatcher_port()
        if port:
            print(f"Auto-detected dispatcher on port {port}")
        else:
            print("Error: Could not auto-detect dispatcher port")
            print("Please specify port with --port")
            print("\nMake sure dispatcher is running:")
            print("  python -m gt.server -p 9000")
            sys.exit(1)

    # Determine output filename
    if args.output:
        output_file = args.output
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"trace_{timestamp}.log"

    # Create output directory if needed
    output_dir = Path(args.dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_file

    # Capture trace
    try:
        capture_trace(args.host, port, args.seconds, str(output_path), max_events=args.max_events)
    except zmq.ZMQError as e:
        print(f"\nError connecting to dispatcher: {e}")
        print(f"\nMake sure dispatcher is running:")
        print(f"  python -m gt.server -p {port}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nCapture interrupted by user")
        sys.exit(0)


if __name__ == '__main__':
    main()
