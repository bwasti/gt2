#!/usr/bin/env python3
"""
GT Instruction Tape Visualizer

Generates high-resolution timeline visualizations showing instruction flow
from client through dispatcher to workers.

Usage:
    python -m gt.scripts.visualize /path/to/debug/tape.log [--output timeline.png]
"""

import sys
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Optional, Set
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np


@dataclass
class TapeEvent:
    """Single event from the instruction tape."""
    timestamp: float
    instruction_id: int
    event_type: str  # RECV, WORKER_SEND, WORKER_RECV, etc.
    source: str  # CLIENT, DISPATCHER, WORKER worker_0, etc.
    message_type: str
    size_bytes: int
    metadata: Dict[str, str]
    raw_line: str


class TapeParser:
    """Parse GT instruction tape logs."""

    # Example: "  0.123s | #0042 | RECV         | CLIENT 127.0.0.1:12345 | BinaryOp        | 123KB | result=42 op=add"
    LINE_PATTERN = re.compile(
        r'\s*(?P<timestamp>[\d.]+)s\s*\|\s*#(?P<inst_id>\d+)\s*\|\s*'
        r'(?P<event>\S+)\s*\|\s*(?P<source>[^|]+?)\s*\|\s*'
        r'(?P<msg_type>[^|]+?)\s*\|\s*(?P<size>[^|]+?)\s*\|?\s*(?P<metadata>.*)?'
    )

    SIGNAL_PATTERN = re.compile(r'SIGNAL:\s*(?P<name>\w+)')

    def __init__(self, log_path: str):
        self.log_path = log_path
        self.events: List[TapeEvent] = []
        self.signals: List[TapeEvent] = []
        self.workers: Set[str] = set()

    def parse(self) -> List[TapeEvent]:
        """Parse the tape log file."""
        with open(self.log_path, 'r') as f:
            for line in f:
                line = line.rstrip()
                if not line or line.startswith('#'):
                    continue

                event = self._parse_line(line)
                if event:
                    self.events.append(event)

                    # Track workers
                    if 'worker' in event.source.lower():
                        worker_name = self._extract_worker_name(event.source)
                        if worker_name:
                            self.workers.add(worker_name)

                    # Track signals
                    if 'signal' in event.message_type.lower() or \
                       'signal' in str(event.metadata).lower():
                        self.signals.append(event)

        return self.events

    def _parse_line(self, line: str) -> Optional[TapeEvent]:
        """Parse a single log line."""
        match = self.LINE_PATTERN.match(line)
        if not match:
            return None

        # Parse size (e.g., "123KB" -> 123000)
        size_str = match.group('size').strip()
        size_bytes = self._parse_size(size_str)

        # Parse metadata (e.g., "result=42 op=add")
        metadata_str = match.group('metadata') or ''
        metadata = self._parse_metadata(metadata_str)

        return TapeEvent(
            timestamp=float(match.group('timestamp')),
            instruction_id=int(match.group('inst_id')),
            event_type=match.group('event').strip(),
            source=match.group('source').strip(),
            message_type=match.group('msg_type').strip(),
            size_bytes=size_bytes,
            metadata=metadata,
            raw_line=line
        )

    def _parse_size(self, size_str: str) -> int:
        """Parse size string like '123KB' into bytes."""
        size_str = size_str.upper().strip()
        if 'KB' in size_str:
            return int(float(size_str.replace('KB', '')) * 1024)
        elif 'MB' in size_str:
            return int(float(size_str.replace('MB', '')) * 1024 * 1024)
        elif 'B' in size_str:
            return int(float(size_str.replace('B', '')))
        else:
            try:
                return int(float(size_str))
            except ValueError:
                return 0

    def _parse_metadata(self, metadata_str: str) -> Dict[str, str]:
        """Parse metadata string like 'result=42 op=add'."""
        metadata = {}
        parts = metadata_str.split()
        for part in parts:
            if '=' in part:
                key, value = part.split('=', 1)
                metadata[key.strip()] = value.strip()
        return metadata

    def _extract_worker_name(self, source: str) -> Optional[str]:
        """Extract worker name from source string."""
        # Try worker_0, worker_1, etc.
        match = re.search(r'worker_\d+', source.lower())
        if match:
            return match.group(0)

        # Try auto_worker, my_worker, etc.
        match = re.search(r'\w*worker\w*', source.lower())
        if match:
            return match.group(0)

        return None


class TapeVisualizer:
    """Create timeline visualization of tape events."""

    # Color scheme for different operation types
    OPERATION_COLORS = {
        'BinaryOp': '#3498db',       # Blue
        'UnaryOp': '#2ecc71',        # Green
        'MatMul': '#e74c3c',         # Red
        'Transpose': '#f39c12',      # Orange
        'Sum': '#9b59b6',            # Purple
        'Mean': '#1abc9c',           # Teal
        'ReLU': '#34495e',           # Dark gray
        'Sigmoid': '#e67e22',        # Dark orange
        'UserTensor': '#95a5a6',     # Gray
        'GetData': '#d35400',        # Dark red
        'AllGather': '#c0392b',      # Darker red
        'Move': '#8e44ad',           # Dark purple
        'FreeTensor': '#7f8c8d',     # Light gray
        'default': '#bdc3c7'         # Light gray
    }

    EVENT_MARKERS = {
        'RECV': 'o',
        'WORKER_SEND': '>',
        'WORKER_RECV': '<',
        'SIGNAL': '*'
    }

    def __init__(self, parser: TapeParser):
        self.parser = parser
        self.events = parser.events
        self.signals = parser.signals
        self.workers = sorted(list(parser.workers))

        # Build timeline structure
        self._build_timeline()

    def _build_timeline(self):
        """Organize events into timeline lanes."""
        self.lanes = ['Client', 'Dispatcher'] + [f'Worker {i}' for i in range(len(self.workers))]
        self.lane_events = {lane: [] for lane in self.lanes}

        # Group events by lane
        for event in self.events:
            lane = self._event_to_lane(event)
            if lane:
                self.lane_events[lane].append(event)

    def _event_to_lane(self, event: TapeEvent) -> Optional[str]:
        """Map event to timeline lane."""
        source_lower = event.source.lower()
        event_type = event.event_type.upper()

        # Dispatcher events: RECV (from client), SEND (to client), or explicit DISPATCHER source
        if event_type in ['RECV', 'SEND'] and 'worker' not in source_lower:
            return 'Dispatcher'
        elif 'dispatcher' in source_lower:
            return 'Dispatcher'
        # Worker events: WORKER_SEND, WORKER_RECV, or WORKER in source
        elif event_type in ['WORKER_SEND', 'WORKER_RECV'] or 'worker' in source_lower:
            worker_name = self.parser._extract_worker_name(event.source)
            if worker_name:
                worker_idx = self.workers.index(worker_name)
                return f'Worker {worker_idx}'
            # Fallback to Worker 0 if we can't extract name but it's clearly a worker event
            elif event_type in ['WORKER_SEND', 'WORKER_RECV']:
                return 'Worker 0'
        # Client events
        elif 'client' in source_lower:
            return 'Client'

        return None

    def _get_operation_color(self, event: TapeEvent) -> str:
        """Get color for operation type."""
        msg_type = event.message_type.strip()

        # Check for specific operation types
        for op_type, color in self.OPERATION_COLORS.items():
            if op_type.lower() in msg_type.lower():
                return color

        # Check metadata for operation
        if 'op' in event.metadata:
            op = event.metadata['op']
            for op_type, color in self.OPERATION_COLORS.items():
                if op_type.lower() in op.lower():
                    return color

        return self.OPERATION_COLORS['default']

    def _get_operation_label(self, event: TapeEvent) -> Optional[str]:
        """Get a short label for the operation."""
        # Extract operation from metadata
        if 'op' in event.metadata:
            return event.metadata['op']

        # Extract from message type
        msg_type = event.message_type.strip()
        if 'UnaryOp' in msg_type and 'op' in event.metadata:
            return event.metadata['op']
        elif 'BinaryOp' in msg_type and 'op' in event.metadata:
            return event.metadata['op']
        elif 'MatMul' in msg_type:
            return 'matmul'
        elif 'ReLU' in msg_type:
            return 'relu'
        elif 'Sum' in msg_type:
            return 'sum'

        return None

    def visualize(self, output_path: str = 'tape_timeline.png', dpi: int = 150):
        """Generate high-resolution timeline visualization."""
        if not self.events:
            print("No events to visualize!")
            return

        # Calculate time range
        min_time = min(e.timestamp for e in self.events)
        max_time = max(e.timestamp for e in self.events)
        duration = max_time - min_time

        # Create figure with high resolution
        fig_width = max(20, duration * 2)  # Scale width with duration
        fig_height = max(12, len(self.lanes) * 1.5)

        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)

        # Draw timeline for each lane
        for lane_idx, lane in enumerate(self.lanes):
            y_pos = len(self.lanes) - lane_idx - 1

            # Draw lane background
            ax.axhline(y=y_pos, color='lightgray', linestyle='-', linewidth=0.5, alpha=0.3)

            # Draw events in this lane
            lane_events = self.lane_events[lane]
            for event in lane_events:
                self._draw_event(ax, event, y_pos)

        # Draw communication arrows
        self._draw_communication_arrows(ax)

        # Draw signal overlays
        self._draw_signals(ax)

        # Styling
        ax.set_ylim(-0.5, len(self.lanes) - 0.5)
        ax.set_xlim(min_time - duration * 0.02, max_time + duration * 0.02)

        ax.set_yticks(range(len(self.lanes)))
        ax.set_yticklabels([self.lanes[len(self.lanes) - i - 1] for i in range(len(self.lanes))])
        ax.set_xlabel('Time (seconds)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Component', fontsize=14, fontweight='bold')

        # Title with stats
        total_ops = len(self.events)
        total_workers = len(self.workers)
        title = f'GT Instruction Tape Timeline\n'
        title += f'{total_ops} operations across {total_workers} workers over {duration:.3f}s'
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

        # Create legend
        legend_elements = self._create_legend()
        ax.legend(handles=legend_elements, loc='upper right',
                 bbox_to_anchor=(1.15, 1), fontsize=10, framealpha=0.9)

        # Grid
        ax.grid(axis='x', alpha=0.3, linestyle='--')

        # Save high-res image
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"✓ Visualization saved to: {output_path}")
        print(f"  Resolution: {fig_width * dpi:.0f} x {fig_height * dpi:.0f} pixels")
        print(f"  Events: {len(self.events)}")
        print(f"  Workers: {len(self.workers)}")
        print(f"  Duration: {duration:.3f}s")

        plt.close()

    def _draw_event(self, ax, event: TapeEvent, y_pos: int):
        """Draw a single event on the timeline."""
        color = self._get_operation_color(event)
        marker = self.EVENT_MARKERS.get(event.event_type, 'o')

        # Size based on data size (logarithmic scale)
        size = 50 + min(200, np.log10(max(1, event.size_bytes)) * 20)

        # Draw marker
        ax.scatter(event.timestamp, y_pos, s=size, c=color,
                  marker=marker, alpha=0.7, edgecolors='black', linewidth=0.5,
                  zorder=10)

        # Add instruction ID annotation for important events
        if event.event_type in ['RECV', 'SIGNAL'] or event.size_bytes > 100000:
            ax.annotate(f'#{event.instruction_id}',
                       xy=(event.timestamp, y_pos),
                       xytext=(0, 10), textcoords='offset points',
                       fontsize=7, ha='center', alpha=0.6)

        # Add operation label for key operations
        if event.event_type in ['RECV', 'WORKER_SEND']:
            op_label = self._get_operation_label(event)
            if op_label:
                ax.annotate(op_label,
                           xy=(event.timestamp, y_pos),
                           xytext=(0, -15), textcoords='offset points',
                           fontsize=6, ha='center', alpha=0.5,
                           style='italic')

    def _draw_communication_arrows(self, ax):
        """Draw arrows showing communication between components."""
        # Group events by instruction ID to show flow
        instruction_events = defaultdict(list)
        for event in self.events:
            instruction_events[event.instruction_id].append(event)

        # Draw arrows for instructions that span multiple components
        for inst_id, events in instruction_events.items():
            if len(events) < 2:
                continue

            # Sort by timestamp
            events = sorted(events, key=lambda e: e.timestamp)

            # Draw arrows between consecutive events
            for i in range(len(events) - 1):
                src_event = events[i]
                dst_event = events[i + 1]

                src_lane = self._event_to_lane(src_event)
                dst_lane = self._event_to_lane(dst_event)

                if src_lane and dst_lane and src_lane != dst_lane:
                    src_y = len(self.lanes) - self.lanes.index(src_lane) - 1
                    dst_y = len(self.lanes) - self.lanes.index(dst_lane) - 1

                    # Draw arrow
                    ax.annotate('',
                               xy=(dst_event.timestamp, dst_y),
                               xytext=(src_event.timestamp, src_y),
                               arrowprops=dict(arrowstyle='->', lw=0.5,
                                             color='gray', alpha=0.3))

    def _draw_signals(self, ax):
        """Draw signal overlays."""
        if not self.signals:
            return

        for signal in self.signals:
            # Draw vertical line for signal
            ax.axvline(x=signal.timestamp, color='red', linestyle='--',
                      linewidth=2, alpha=0.6, zorder=5)

            # Label
            signal_name = signal.metadata.get('signal', 'SIGNAL')
            ax.text(signal.timestamp, len(self.lanes) - 0.3, signal_name,
                   rotation=90, ha='right', va='bottom', fontsize=10,
                   color='red', fontweight='bold', alpha=0.8)

    def _create_legend(self) -> List:
        """Create legend elements."""
        legend_elements = []

        # Operation types
        legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                     label='Operation Types:',
                                     markerfacecolor='white', markersize=0))

        for op_type, color in sorted(self.OPERATION_COLORS.items())[:8]:  # Top 8
            if op_type != 'default':
                legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                             label=f'  {op_type}',
                                             markerfacecolor=color, markersize=8))

        # Event types
        legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                     label='Event Types:',
                                     markerfacecolor='white', markersize=0))

        for event_type, marker in self.EVENT_MARKERS.items():
            legend_elements.append(Line2D([0], [0], marker=marker, color='w',
                                         label=f'  {event_type}',
                                         markerfacecolor='gray', markersize=8))

        return legend_elements


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python -m gt.scripts.visualize <tape_log_path> [--output <output_path>] [--dpi <dpi>]")
        print("\nExample:")
        print("  GT_INSTRUCTION_LOG=/tmp/debug.log python your_script.py")
        print("  python -m gt.scripts.visualize /tmp/debug.log --output timeline.png --dpi 200")
        sys.exit(1)

    tape_path = sys.argv[1]
    output_path = 'tape_timeline.png'
    dpi = 150

    # Parse optional arguments
    for i in range(2, len(sys.argv)):
        if sys.argv[i] == '--output' and i + 1 < len(sys.argv):
            output_path = sys.argv[i + 1]
        elif sys.argv[i] == '--dpi' and i + 1 < len(sys.argv):
            dpi = int(sys.argv[i + 1])

    print(f"Parsing tape log: {tape_path}")
    parser = TapeParser(tape_path)
    parser.parse()

    print(f"Found {len(parser.events)} events, {len(parser.workers)} workers, {len(parser.signals)} signals")

    print(f"Generating visualization...")
    visualizer = TapeVisualizer(parser)
    visualizer.visualize(output_path=output_path, dpi=dpi)


if __name__ == '__main__':
    main()
