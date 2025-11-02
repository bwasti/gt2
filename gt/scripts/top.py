#!/usr/bin/env python3
"""
GT Worker Monitor - Real-time htop-style worker activity visualization

Monitors worker activity from instruction tape log and displays:
- Per-worker operation breakdown (matmul, add, etc.)
- Idle time percentage
- EMA-smoothed percentages for stable display
- Color-coded progress bars

Usage:
    # Auto-attach to running dispatcher
    python -m gt.scripts.top

    # Attach to specific process
    python -m gt.scripts.top --pid 12345

    # Monitor specific log file
    python -m gt.scripts.top /path/to/tape.log
"""

import sys
import time
import re
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import argparse

try:
    import psutil
except ImportError:
    psutil = None

try:
    from rich.live import Live
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, BarColumn, TextColumn
    from rich.panel import Panel
    from rich.layout import Layout
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

    # EMA smoothed percentages
    ema_percentages: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    ema_idle: float = 0.0


class TapeMonitor:
    """Real-time tape log monitor."""

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

    LINE_PATTERN = re.compile(
        r'\s*(?P<timestamp>[\d.]+)s\s*\|\s*#(?P<inst_id>\d+)\s*\|\s*'
        r'(?P<event>\S+)\s*\|\s*(?P<source>[^|]+?)\s*\|\s*'
        r'(?P<msg_type>[^|]+?)\s*\|'
    )

    def __init__(self, log_path: str, window_seconds: float = 2.0):
        self.log_path = log_path
        self.window_seconds = window_seconds
        self.workers: Dict[str, WorkerStats] = {}
        self.start_time = None
        self.last_update = None

    def parse_line(self, line: str) -> Optional[tuple]:
        """Parse log line and return (timestamp, worker_name, event_type, msg_type)."""
        match = self.LINE_PATTERN.match(line)
        if not match:
            return None

        timestamp = float(match.group('timestamp'))
        event = match.group('event').strip()
        source = match.group('source').strip()
        msg_type = match.group('msg_type').strip()

        # Extract worker name
        worker_match = re.search(r'(worker_\d+|\w*worker\w*)', source.lower())
        if not worker_match:
            return None

        worker_name = worker_match.group(0)

        # Extract operation type
        op_type = self._extract_op_type(msg_type)

        return timestamp, worker_name, event, op_type

    def _extract_op_type(self, msg_type: str) -> str:
        """Extract operation type from message type."""
        msg_lower = msg_type.lower()

        # Check for specific operations
        if 'matmul' in msg_lower:
            return 'matmul'
        elif 'binaryop' in msg_lower or 'add' in msg_lower:
            return 'add'
        elif 'sub' in msg_lower:
            return 'sub'
        elif 'mul' in msg_lower:
            return 'mul'
        elif 'div' in msg_lower:
            return 'div'
        elif 'relu' in msg_lower:
            return 'relu'
        elif 'sigmoid' in msg_lower:
            return 'sigmoid'
        elif 'tanh' in msg_lower:
            return 'tanh'
        elif 'sum' in msg_lower:
            return 'sum'
        elif 'mean' in msg_lower:
            return 'mean'
        elif 'transpose' in msg_lower:
            return 'transpose'
        elif 'getdata' in msg_lower:
            return 'getdata'
        elif 'allgather' in msg_lower:
            return 'allgather'
        else:
            return 'other'

    def update_worker_stats(self, timestamp: float, worker_name: str, event: str, op_type: str):
        """Update worker statistics with new event."""
        if worker_name not in self.workers:
            self.workers[worker_name] = WorkerStats(name=worker_name)

        worker = self.workers[worker_name]

        if worker.last_event_time is not None:
            # Calculate time since last event
            elapsed = timestamp - worker.last_event_time

            # Attribute time to previous operation or idle
            if worker.last_event_type and event == 'WORKER_RECV':
                # Time from WORKER_SEND to WORKER_RECV is operation time
                worker.op_times[worker.last_event_type] += elapsed
                worker.total_time += elapsed
            else:
                # Other gaps are idle time
                worker.idle_time += elapsed
                worker.total_time += elapsed

        # Update state
        if event == 'WORKER_SEND':
            worker.last_event_type = op_type

        worker.last_event_time = timestamp

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
            title="GT Worker Monitor",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan"
        )

        table.add_column("Worker", style="cyan", no_wrap=True)
        table.add_column("Activity", style="white")
        table.add_column("Details", style="dim")

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

    def follow_log(self):
        """Follow log file like 'tail -f'."""
        console = Console()

        try:
            with open(self.log_path, 'r') as f:
                # Skip header
                for line in f:
                    if line.strip() and not line.startswith('=') and not line.startswith('This log'):
                        if self.LINE_PATTERN.match(line):
                            break

                # Initial parse of existing data
                for line in f:
                    parsed = self.parse_line(line)
                    if parsed:
                        timestamp, worker_name, event, op_type = parsed
                        self.update_worker_stats(timestamp, worker_name, event, op_type)

                self.compute_percentages()

                # Live update loop
                with Live(self.generate_display(), refresh_per_second=4, console=console) as live:
                    while True:
                        line = f.readline()

                        if line:
                            parsed = self.parse_line(line)
                            if parsed:
                                timestamp, worker_name, event, op_type = parsed
                                self.update_worker_stats(timestamp, worker_name, event, op_type)
                        else:
                            time.sleep(0.1)

                        # Update display
                        self.compute_percentages()
                        live.update(self.generate_display())

        except FileNotFoundError:
            console.print(f"[red]Error: Log file not found: {self.log_path}[/red]")
            console.print("\nMake sure to run your workload with:")
            console.print(f"  GT_INSTRUCTION_LOG={self.log_path} python your_script.py")
            sys.exit(1)
        except KeyboardInterrupt:
            console.print("\n[yellow]Monitoring stopped[/yellow]")


def find_dispatcher_processes():
    """Find running dispatcher processes."""
    if psutil is None:
        return []

    dispatchers = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info.get('cmdline', [])
            if cmdline and any('dispatcher' in arg.lower() or 'gt.server' in arg for arg in cmdline):
                dispatchers.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    return dispatchers


def get_log_path_from_process(proc):
    """Get GT_INSTRUCTION_LOG path from process environment."""
    try:
        env = proc.environ()
        return env.get('GT_INSTRUCTION_LOG')
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None


def find_log_path(pid=None):
    """Find log path from running dispatcher."""
    console = Console()

    if psutil is None:
        console.print("[yellow]Warning: psutil not installed, cannot auto-detect dispatcher[/yellow]")
        console.print("Install with: pip install psutil")
        return None

    if pid:
        # Attach to specific PID
        try:
            proc = psutil.Process(pid)
            log_path = get_log_path_from_process(proc)
            if log_path:
                console.print(f"[green]Found log at: {log_path}[/green]")
                return log_path
            else:
                console.print(f"[yellow]Process {pid} has no GT_INSTRUCTION_LOG set[/yellow]")
                return None
        except psutil.NoSuchProcess:
            console.print(f"[red]Process {pid} not found[/red]")
            return None
    else:
        # Find dispatcher automatically
        dispatchers = find_dispatcher_processes()

        if not dispatchers:
            console.print("[yellow]No running dispatcher found[/yellow]")
            console.print("\nTo monitor a dispatcher, start it with instruction logging:")
            console.print("  GT_INSTRUCTION_LOG=/tmp/gt.log python -m gt.server -p 12345")
            return None

        if len(dispatchers) == 1:
            proc = dispatchers[0]
            log_path = get_log_path_from_process(proc)
            if log_path:
                console.print(f"[green]Attached to dispatcher (PID {proc.pid})[/green]")
                console.print(f"[green]Log: {log_path}[/green]")
                return log_path
            else:
                console.print(f"[yellow]Dispatcher (PID {proc.pid}) has no GT_INSTRUCTION_LOG set[/yellow]")
                console.print("\nRestart dispatcher with:")
                console.print("  GT_INSTRUCTION_LOG=/tmp/gt.log python -m gt.server -p 12345")
                return None
        else:
            console.print(f"[yellow]Found {len(dispatchers)} dispatchers:[/yellow]")
            for proc in dispatchers:
                log_path = get_log_path_from_process(proc)
                status = f"log={log_path}" if log_path else "no log"
                console.print(f"  PID {proc.pid}: {status}")
            console.print("\nSpecify which to monitor:")
            console.print("  python -m gt.scripts.top --pid <PID>")
            return None


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="GT Worker Monitor - Real-time worker activity visualization"
    )
    parser.add_argument(
        "log_path",
        nargs='?',
        help="Path to GT instruction log file (optional, auto-detects if omitted)"
    )
    parser.add_argument(
        "--pid",
        type=int,
        help="Attach to specific dispatcher process ID"
    )
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=0.15,
        help="EMA smoothing factor (0-1, higher = more responsive, default: 0.15)"
    )

    args = parser.parse_args()

    console = Console()

    # Determine log path
    log_path = args.log_path
    if not log_path:
        log_path = find_log_path(args.pid)
        if not log_path:
            sys.exit(1)

    # Start monitoring
    monitor = TapeMonitor(log_path)
    monitor.EMA_ALPHA = args.ema_alpha

    console.print("[cyan]GT Worker Monitor[/cyan]")
    console.print(f"Monitoring: {log_path}")
    console.print("Press Ctrl+C to stop\n")

    monitor.follow_log()


if __name__ == '__main__':
    main()
