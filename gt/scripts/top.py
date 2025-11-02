#!/usr/bin/env python3
"""
GT Worker Monitor - Real-time htop-style worker activity visualization

Monitors worker activity by subscribing to dispatcher's event stream.
No log files needed - connects directly to running dispatcher.

Usage:
    # Auto-attach to running dispatcher
    python -m gt.scripts.top

    # Attach to specific port
    python -m gt.scripts.top --port 9000
"""

import sys
import time
import re
import os
import json
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import argparse

try:
    import zmq
except ImportError:
    print("Error: 'pyzmq' library required")
    print("Install with: pip install pyzmq")
    sys.exit(1)

try:
    from rich.live import Live
    from rich.console import Console
    from rich.table import Table
    from rich import box
except ImportError:
    print("Error: 'rich' library required for monitor")
    print("Install with: pip install rich")
    sys.exit(1)


@dataclass
class WorkerStats:
    """Statistics for a single worker."""
    name: str
    total_time: float = 0.0
    idle_time: float = 0.0
    op_times: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    last_event_time: Optional[float] = None
    last_event_type: Optional[str] = None
    pending_op_start: Optional[float] = None  # When WORKER_SEND happened

    # EMA smoothed percentages
    ema_percentages: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    ema_idle: float = 0.0


class RealtimeMonitor:
    """Real-time monitor that subscribes to dispatcher events."""

    # EMA alpha (0-1, higher = more responsive, lower = smoother)
    EMA_ALPHA = 0.15

    # Operation colors
    OP_COLORS = {
        'matmul': 'red',
        'add': 'blue',
        'sub': 'cyan',
        'mul': 'magenta',
        'div': 'yellow',
        'relu': 'green',
        'sigmoid': 'bright_blue',
        'tanh': 'bright_cyan',
        'sum': 'bright_magenta',
        'mean': 'bright_yellow',
        'transpose': 'bright_green',
        'getdata': 'bright_red',
        'allgather': 'bright_white',
        'idle': 'white',
    }

    def __init__(self, host: str, port: int, window_seconds: float = 2.0):
        self.host = host
        self.port = port
        self.monitor_port = port + 1  # Monitor socket is dispatcher_port + 1
        self.window_seconds = window_seconds
        self.workers: Dict[str, WorkerStats] = {}
        self.event_window = deque()  # Rolling window of events

        # ZMQ SUB socket
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt(zmq.SUBSCRIBE, b'')  # Subscribe to all messages
        self.socket.setsockopt(zmq.RCVTIMEO, 100)  # 100ms timeout

    def connect(self):
        """Connect to dispatcher monitoring socket."""
        monitor_url = f"tcp://{self.host}:{self.monitor_port}"
        self.socket.connect(monitor_url)
        return monitor_url

    def _extract_op_type_from_details(self, details: str) -> Optional[str]:
        """Extract operation type from details field (e.g., 'op=matmul')."""
        if not details:
            return None

        # Look for op= parameter
        match = re.search(r'op=(\w+)', details)
        if match:
            return match.group(1).lower()

        return None

    def _extract_op_type(self, command: str) -> str:
        """Extract operation type from command."""
        cmd_lower = command.lower()

        if 'matmul' in cmd_lower:
            return 'matmul'
        elif 'binaryop' in cmd_lower or 'add' in cmd_lower:
            return 'add'
        elif 'sub' in cmd_lower:
            return 'sub'
        elif 'mul' in cmd_lower:
            return 'mul'
        elif 'div' in cmd_lower:
            return 'div'
        elif 'relu' in cmd_lower:
            return 'relu'
        elif 'sigmoid' in cmd_lower:
            return 'sigmoid'
        elif 'tanh' in cmd_lower:
            return 'tanh'
        elif 'sum' in cmd_lower:
            return 'sum'
        elif 'mean' in cmd_lower:
            return 'mean'
        elif 'transpose' in cmd_lower:
            return 'transpose'
        elif 'getdata' in cmd_lower:
            return 'getdata'
        elif 'allgather' in cmd_lower:
            return 'allgather'
        else:
            return 'other'

    def _extract_worker_name(self, client: str) -> Optional[str]:
        """Extract worker name from client ID."""
        match = re.search(r'(worker_\d+|\w*worker\w*)', client.lower())
        return match.group(0) if match else None

    def update_worker_stats(self, event: dict):
        """Update worker statistics with new event."""
        timestamp = event['timestamp']
        event_type = event['event']
        client = event['client']
        command = event['command']
        details = event.get('details', '')

        # Extract worker name
        worker_name = self._extract_worker_name(client)
        if not worker_name:
            return

        # Extract operation type (check details field for op= parameter)
        op_type = self._extract_op_type_from_details(details)
        if not op_type:
            op_type = self._extract_op_type(command)

        # Initialize worker if needed
        if worker_name not in self.workers:
            self.workers[worker_name] = WorkerStats(name=worker_name)

        worker = self.workers[worker_name]

        # Track operation timing
        if event_type == 'WORKER_SEND':
            # Operation starts
            worker.pending_op_start = timestamp
            worker.last_event_type = op_type
        elif event_type == 'WORKER_RECV':
            # Operation completes
            if worker.pending_op_start and worker.last_event_type:
                duration = timestamp - worker.pending_op_start
                worker.op_times[worker.last_event_type] += duration
                worker.total_time += duration
                worker.pending_op_start = None
            worker.last_event_time = timestamp

        # Track idle time between operations
        if worker.last_event_time and event_type == 'WORKER_SEND':
            idle_duration = timestamp - worker.last_event_time
            if idle_duration > 0:
                worker.idle_time += idle_duration
                worker.total_time += idle_duration

    def compute_percentages(self):
        """Compute and smooth percentages using EMA."""
        for worker in self.workers.values():
            total = worker.total_time
            if total == 0:
                continue

            # Compute raw percentages
            raw_percentages = {}
            for op, op_time in worker.op_times.items():
                raw_percentages[op] = (op_time / total) * 100

            raw_idle = (worker.idle_time / total) * 100

            # Apply EMA smoothing
            for op, raw_pct in raw_percentages.items():
                if op not in worker.ema_percentages:
                    worker.ema_percentages[op] = raw_pct
                else:
                    worker.ema_percentages[op] = (
                        self.EMA_ALPHA * raw_pct +
                        (1 - self.EMA_ALPHA) * worker.ema_percentages[op]
                    )

            # Smooth idle
            if worker.ema_idle == 0:
                worker.ema_idle = raw_idle
            else:
                worker.ema_idle = (
                    self.EMA_ALPHA * raw_idle +
                    (1 - self.EMA_ALPHA) * worker.ema_idle
                )

    def generate_display(self) -> Table:
        """Generate rich table for display."""
        table = Table(
            title=f"GT Worker Monitor - {self.host}:{self.port}",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan"
        )

        table.add_column("Worker", style="cyan", no_wrap=True, width=15)
        table.add_column("Activity", style="white", width=50)
        table.add_column("Details", style="dim", width=40)

        if not self.workers:
            table.add_row("No workers", "", "Waiting for events...")
            return table

        # Sort workers by name
        sorted_workers = sorted(self.workers.values(), key=lambda w: w.name)

        for worker in sorted_workers:
            # Build activity bar
            bar = self._build_activity_bar(worker)

            # Build details string
            details = self._build_details(worker)

            table.add_row(worker.name, bar, details)

        return table

    def _build_activity_bar(self, worker: WorkerStats) -> str:
        """Build colored activity bar for worker."""
        # Get smoothed percentages
        percentages = []
        for op, pct in sorted(worker.ema_percentages.items(), key=lambda x: -x[1]):
            if pct > 0.5:  # Only show ops > 0.5%
                percentages.append((op, pct))

        # Add idle
        if worker.ema_idle > 0.5:
            percentages.append(('idle', worker.ema_idle))

        # Normalize to 100% (handle floating point errors)
        total_pct = sum(pct for _, pct in percentages)
        if total_pct > 0:
            percentages = [(op, (pct / total_pct) * 100) for op, pct in percentages]

        # Build bar with 50 characters width
        bar_width = 50
        bar_chars = []

        for op, pct in percentages:
            chars = int((pct / 100) * bar_width)
            if chars > 0:
                color = self.OP_COLORS.get(op, 'white')
                bar_chars.append(f"[{color}]{'█' * chars}[/{color}]")

        return ''.join(bar_chars)

    def _build_details(self, worker: WorkerStats) -> str:
        """Build details string showing percentages."""
        details = []

        # Get top operations
        top_ops = sorted(worker.ema_percentages.items(), key=lambda x: -x[1])[:3]

        for op, pct in top_ops:
            if pct > 0.5:
                details.append(f"{op}: {pct:.1f}%")

        if worker.ema_idle > 0.5:
            details.append(f"idle: {worker.ema_idle:.1f}%")

        return " | ".join(details)

    def monitor(self):
        """Main monitoring loop."""
        console = Console()
        monitor_url = self.connect()

        console.print(f"[cyan]GT Worker Monitor[/cyan]")
        console.print(f"Connected to: {monitor_url}")
        console.print("Press Ctrl+C to stop\n")

        try:
            with Live(self.generate_display(), refresh_per_second=4, console=console) as live:
                while True:
                    try:
                        # Receive event (non-blocking with timeout)
                        msg = self.socket.recv()
                        event = json.loads(msg.decode('utf-8'))

                        # Update stats
                        self.update_worker_stats(event)

                    except zmq.Again:
                        # Timeout - no message received
                        pass
                    except Exception as e:
                        # Parsing error, skip
                        pass

                    # Update display
                    self.compute_percentages()
                    live.update(self.generate_display())

        except KeyboardInterrupt:
            console.print("\n[yellow]Monitoring stopped[/yellow]")
        finally:
            self.socket.close()
            self.context.term()


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


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="GT Worker Monitor - Real-time worker activity visualization"
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Dispatcher host (default: localhost)"
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Dispatcher port (default: auto-detect or 9000)"
    )
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=0.15,
        help="EMA smoothing factor (0-1, higher = more responsive, default: 0.15)"
    )

    args = parser.parse_args()

    console = Console()

    # Determine port
    port = args.port
    if not port:
        port = find_dispatcher_port()
        if port:
            console.print(f"[green]Auto-detected dispatcher on port {port}[/green]")
        else:
            port = 9000
            console.print(f"[yellow]Using default port {port}[/yellow]")

    # Start monitoring
    monitor = RealtimeMonitor(args.host, port)
    monitor.EMA_ALPHA = args.ema_alpha

    try:
        monitor.monitor()
    except zmq.ZMQError as e:
        console.print(f"\n[red]Error connecting to dispatcher: {e}[/red]")
        console.print(f"\nMake sure dispatcher is running:")
        console.print(f"  python -m gt.server -p {port}")
        sys.exit(1)


if __name__ == '__main__':
    main()
