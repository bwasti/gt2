#!/usr/bin/env python3
"""
GT Instruction Tape Visualizer - Timeline View (perf-style)

Generates dense timeline visualizations showing operations as duration blocks.
Similar to performance profiler timelines (perf, chrome://tracing, etc).

Usage:
    python -m gt.scripts.visualize /path/to/debug/tape.log [--output timeline.png]
"""

import sys
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


@dataclass
class Operation:
    """An operation with start and end time."""
    start_time: float
    end_time: float
    op_type: str  # matmul, add, relu, etc
    worker: str
    instruction_id: int
    metadata: Dict[str, str]


class TimelineVisualizer:
    """Creates dense timeline visualization (perf-style)."""

    # Operation colors (same as gt.scripts.top)
    OP_COLORS = {
        'matmul': '#e74c3c',    # bright red
        'add': '#3498db',        # bright blue
        'sub': '#16a085',        # teal
        'mul': '#9b59b6',        # purple
        'div': '#f39c12',        # orange
        'relu': '#27ae60',       # bright green
        'sigmoid': '#2980b9',    # blue
        'tanh': '#1abc9c',       # cyan
        'sum': '#8e44ad',        # dark purple
        'mean': '#d35400',       # dark orange
        'transpose': '#229954',  # green
        'getdata': '#c0392b',    # dark red
        'allgather': '#ecf0f1',  # light gray
        'randn': '#e84393',      # pink
        'freetensor': '#95a5a6', # gray
        'other': '#7f8c8d',      # dark gray
    }

    def __init__(self, log_path: str):
        self.log_path = log_path
        self.operations: List[Operation] = []
        self.workers: List[str] = []

    def parse_and_visualize(self, output_path: str = 'timeline.png', dpi: int = 150):
        """Parse log and create timeline visualization."""
        print(f"Parsing {self.log_path}...")
        self._parse_log()

        if not self.operations:
            print("No operations found!")
            return

        print(f"Found {len(self.operations)} operations across {len(self.workers)} workers")
        print(f"Creating timeline visualization...")
        self._create_timeline(output_path, dpi)

    def _parse_log(self):
        """Parse log file to extract operations."""
        # Pattern matches: "  0.123s | #0042 | WORKER_SEND | WORKER worker_0 | ..."
        pattern = re.compile(
            r'\s*(?P<timestamp>[\d.]+)s\s*\|\s*#(?P<inst_id>\d+)\s*\|\s*'
            r'(?P<event>\S+)\s*\|\s*(?P<source>[^|]+?)\s*\|\s*'
            r'(?P<msg_type>[^|]+?)\s*\|.*?(?P<metadata>op=\w+)?'
        )

        # Track pending operations (WORKER_SEND waiting for WORKER_RECV)
        pending = {}  # worker -> (start_time, op_type, inst_id, metadata)

        with open(self.log_path, 'r') as f:
            for line in f:
                match = pattern.match(line)
                if not match:
                    continue

                timestamp = float(match.group('timestamp'))
                inst_id = int(match.group('inst_id'))
                event = match.group('event').strip()
                source = match.group('source').strip()

                # Extract worker name
                worker = self._extract_worker(source)
                if not worker:
                    continue

                # Track workers
                if worker not in self.workers:
                    self.workers.append(worker)

                # Extract operation type
                op_type = self._extract_op_type(line, match.group('msg_type'))

                if event == 'WORKER_SEND':
                    # Operation starts
                    metadata = {}
                    if match.group('metadata'):
                        parts = match.group('metadata').split('=')
                        if len(parts) == 2:
                            metadata[parts[0]] = parts[1]
                    pending[worker] = (timestamp, op_type, inst_id, metadata)

                elif event == 'WORKER_RECV':
                    # Operation ends
                    if worker in pending:
                        start_time, op, start_inst_id, metadata = pending.pop(worker)
                        self.operations.append(Operation(
                            start_time=start_time,
                            end_time=timestamp,
                            op_type=op,
                            worker=worker,
                            instruction_id=start_inst_id,
                            metadata=metadata
                        ))

        # Sort workers for consistent ordering
        self.workers.sort()

    def _extract_worker(self, source: str) -> Optional[str]:
        """Extract worker name from source string."""
        source_lower = source.lower()
        if 'worker' in source_lower:
            # Try to extract worker name/ID
            match = re.search(r'worker[_\s]*(\w+)', source_lower)
            if match:
                return f"worker_{match.group(1)}"
        return None

    def _extract_op_type(self, line: str, msg_type: str) -> str:
        """Extract operation type from line."""
        line_lower = line.lower()

        # Check for op= in line
        match = re.search(r'op=(\w+)', line_lower)
        if match:
            return match.group(1)

        # Check message type
        msg_lower = msg_type.lower()
        if 'matmul' in msg_lower:
            return 'matmul'
        elif 'binaryop' in msg_lower:
            return 'add'
        elif 'unaryop' in msg_lower:
            if 'relu' in line_lower:
                return 'relu'
            elif 'sum' in line_lower:
                return 'sum'
            elif 'mean' in line_lower:
                return 'mean'
            elif 'randn' in line_lower:
                return 'randn'
            return 'other'
        elif 'getdata' in msg_lower:
            return 'getdata'
        elif 'freetensor' in msg_lower:
            return 'freetensor'

        return 'other'

    def _create_timeline(self, output_path: str, dpi: int):
        """Create perf-style timeline visualization."""
        if not self.operations or not self.workers:
            print("No data to visualize")
            return

        # Calculate time range
        min_time = min(op.start_time for op in self.operations)
        max_time = max(op.end_time for op in self.operations)
        duration = max_time - min_time

        # Create wide figure (like perf)
        fig_width = max(24, duration * 10)  # Very wide for timeline
        fig_height = max(6, len(self.workers) * 0.8)  # Compact vertically

        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)

        # Draw operations as horizontal bars
        for op in self.operations:
            worker_idx = self.workers.index(op.worker)
            y_pos = len(self.workers) - worker_idx - 1

            # Calculate bar dimensions
            start_x = op.start_time
            width = op.end_time - op.start_time
            height = 0.7  # Bar height

            # Get color
            color = self.OP_COLORS.get(op.op_type, self.OP_COLORS['other'])

            # Draw rectangle
            rect = mpatches.Rectangle(
                (start_x, y_pos - height/2),
                width,
                height,
                facecolor=color,
                edgecolor='black',
                linewidth=0.3,
                alpha=0.8
            )
            ax.add_patch(rect)

            # Add operation label if bar is wide enough
            if width > duration * 0.01:  # Only label if >1% of timeline
                ax.text(
                    start_x + width/2,
                    y_pos,
                    op.op_type,
                    ha='center',
                    va='center',
                    fontsize=7,
                    fontweight='bold',
                    color='white' if op.op_type in ['matmul', 'relu', 'getdata'] else 'black'
                )

        # Styling
        ax.set_ylim(-0.5, len(self.workers) - 0.5)
        ax.set_xlim(min_time - duration * 0.01, max_time + duration * 0.01)

        # Y-axis: worker labels
        ax.set_yticks(range(len(self.workers)))
        ax.set_yticklabels([
            self.workers[len(self.workers) - i - 1]
            for i in range(len(self.workers))
        ])

        # X-axis: time
        ax.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Worker', fontsize=12, fontweight='bold')

        # Title
        total_ops = len(self.operations)
        title = f'GT Operation Timeline ({total_ops} ops, {len(self.workers)} workers, {duration:.3f}s)'
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)

        # Grid - only vertical lines for time reference
        ax.grid(axis='x', alpha=0.3, linestyle=':', linewidth=0.5)

        # Legend
        legend_elements = []
        # Show only operations that appear in the trace
        ops_in_trace = set(op.op_type for op in self.operations)
        for op_type in sorted(ops_in_trace):
            if op_type in self.OP_COLORS:
                legend_elements.append(
                    mpatches.Patch(
                        facecolor=self.OP_COLORS[op_type],
                        edgecolor='black',
                        label=op_type,
                        linewidth=0.5
                    )
                )

        if legend_elements:
            ax.legend(
                handles=legend_elements,
                loc='upper left',
                bbox_to_anchor=(1.01, 1),
                fontsize=9,
                framealpha=0.9,
                ncol=1 if len(legend_elements) <= 10 else 2
            )

        # Save
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"✓ Timeline saved to: {output_path}")
        print(f"  Resolution: {int(fig_width * dpi)} x {int(fig_height * dpi)} pixels")
        print(f"  Operations: {len(self.operations)}")
        print(f"  Workers: {len(self.workers)}")
        print(f"  Duration: {duration:.3f}s")

        plt.close()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='GT Timeline Visualizer - Generate perf-style operation timelines'
    )
    parser.add_argument(
        'log_file',
        help='Path to GT instruction log file'
    )
    parser.add_argument(
        '--output', '-o',
        default='timeline.png',
        help='Output image path (default: timeline.png)'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=150,
        help='DPI for output image (default: 150, try 200-300 for higher resolution)'
    )

    args = parser.parse_args()

    visualizer = TimelineVisualizer(args.log_file)
    visualizer.parse_and_visualize(args.output, args.dpi)


if __name__ == '__main__':
    main()
