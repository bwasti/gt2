"""
Hot path detection - stream transformer that injects HOTPATH markers.

This is a PURE STREAM TRANSFORMER:
- Takes WorkerCommand stream as input
- Yields WorkerCommand stream as output (with markers injected)
- Does NOT buffer or execute anything
- Engines decide what to do with HOTPATH_START/END markers

The detector identifies repeated sequences and injects markers around them:

Input stream:
  ADD X, Y -> Z
  MUL Z, C -> D
  <sync point>
  ADD X, Y -> Z    (pattern repeats!)
  MUL Z, C -> D

Output stream (with markers):
  ADD X, Y -> Z
  MUL Z, C -> D
  <sync point>
  HOTPATH_START 0xA4E7
  ADD X, Y -> Z
  MUL Z, C -> D
  HOTPATH_END 0xA4E7
"""

from typing import Iterator, Optional, List
from collections import deque
import hashlib
from gt.debug import debug_print_compile
from gt.transport.protocol import (
    WorkerCommand, WorkerBinaryOp, WorkerUnaryOp, WorkerReshapeOp, WorkerSliceOp,
    WorkerGetData, WorkerHotPathStart, WorkerHotPathEnd
)


class InstructionSignature:
    """Lightweight signature for a single instruction."""

    def __init__(self, op_type: str, op_name: str):
        self.op_type = op_type
        self.op_name = op_name
        self._hash = hash((self.op_type, self.op_name))  # Cache hash

    def __eq__(self, other):
        return self.op_type == other.op_type and self.op_name == other.op_name

    def __hash__(self):
        return self._hash

    def __repr__(self):
        return f"{self.op_type}:{self.op_name}"


def compute_sequence_hash(instructions: List[InstructionSignature]) -> int:
    """
    Compute integer hash for a sequence of instructions.

    Uses cached hashes from InstructionSignature objects for O(n) computation.
    Result can be used directly as dict key for O(1) equality checks.
    """
    return hash(tuple(inst._hash for inst in instructions))


class HotPathDetector:
    """
    Stream transformer that detects hot paths and injects markers.

    Usage:
        detector = HotPathDetector(hot_threshold=10)
        for cmd in command_stream:
            for output_cmd in detector.process(cmd):
                yield output_cmd
    """

    def __init__(self, window_size: int = 20, hot_threshold: int = 10, min_sequence_length: int = 3):
        """
        Initialize hot path detector.

        Args:
            window_size: Number of recent instructions to track
            hot_threshold: Number of repetitions before marking as hot
            min_sequence_length: Minimum operations for a sequence to be hot
        """
        self.window_size = window_size
        self.hot_threshold = hot_threshold
        self.min_sequence_length = min_sequence_length

        # Sliding window of recent instructions
        self.window: deque = deque(maxlen=window_size)

        # Track repetition counts for sequences (using integer hashes)
        self.sequence_counts: dict = {}  # int (sequence hash) -> count
        self.sequence_instructions: dict = {}  # int (sequence hash) -> List[InstructionSignature] (for debugging)

        # Active hot sequence tracking
        self.active_hot_sequence_hash: Optional[int] = None
        self.active_hot_sequence_instructions: Optional[List[InstructionSignature]] = None
        self.active_position = 0  # Position within active sequence

        # Stats
        self.total_instructions = 0
        self.hot_instructions = 0

    def _extract_signature(self, cmd: WorkerCommand) -> Optional[InstructionSignature]:
        """Extract operation signature from command (or None for non-compute ops)."""
        if isinstance(cmd, WorkerBinaryOp):
            return InstructionSignature('binary', cmd.op)
        elif isinstance(cmd, WorkerUnaryOp):
            if cmd.input_id is not None:  # Only compute ops
                return InstructionSignature('unary', cmd.op)
        elif isinstance(cmd, WorkerReshapeOp):
            return InstructionSignature('reshape', cmd.op)
        elif isinstance(cmd, WorkerSliceOp):
            return InstructionSignature('slice', 'subscript')
        return None

    def _is_sync_op(self, cmd: WorkerCommand) -> bool:
        """Check if command is a sync point that should reset pattern detection."""
        # GetData is NOT a sync point - it's just reading data and doesn't affect
        # the computation pattern. Treating it as sync prevents hot path detection
        # in typical training loops where you check loss every iteration.
        return False  # No operations currently trigger sync/reset

    def _detect_hot_sequence(self) -> Optional[tuple]:
        """
        Detect if any sequence in the window is hot.

        Returns (hash, instructions) for the longest hot sequence, or None if none found.
        """
        if len(self.window) < self.min_sequence_length:
            return None

        # Try sequences from longest to shortest
        for seq_len in range(len(self.window), self.min_sequence_length - 1, -1):
            # Check if we have enough instructions for this length
            if seq_len > len(self.window):
                continue

            # Get the most recent sequence of this length
            recent_instructions = list(self.window)[-seq_len:]
            seq_hash = compute_sequence_hash(recent_instructions)

            # Check if this sequence is hot
            count = self.sequence_counts.get(seq_hash, 0)
            if count >= self.hot_threshold:
                return (seq_hash, recent_instructions)

        return None

    def _matches_pattern(self, inst: InstructionSignature, pattern: InstructionSignature) -> bool:
        """Check if instruction matches the pattern."""
        return inst == pattern

    def process(self, cmd: WorkerCommand) -> Iterator[WorkerCommand]:
        """
        Process a command and yield output commands (with potential markers).

        Yields:
            Original command plus potential HOTPATH_START/END markers
        """
        # Sync operations reset detection
        if self._is_sync_op(cmd):
            # End any active sequence
            if self.active_hot_sequence_hash is not None:
                # IMPORTANT: Emit END marker to flush the buffer before sync
                seq_id = hex(self.active_hot_sequence_hash)
                yield WorkerHotPathEnd(sequence_id=seq_id)

                self.active_hot_sequence_hash = None
                self.active_hot_sequence_instructions = None
                self.active_position = 0

            # Reset window
            self.window.clear()
            self.sequence_counts.clear()
            self.sequence_instructions.clear()

            # Pass through sync command
            yield cmd
            return

        # Extract signature (skip non-compute ops)
        sig = self._extract_signature(cmd)
        if sig is None:
            yield cmd
            return

        self.total_instructions += 1

        # STATE 1: Are we currently inside a hot sequence?
        if self.active_hot_sequence_hash is not None:
            expected = self.active_hot_sequence_instructions[self.active_position]

            # Check if current instruction matches expected pattern
            if self._matches_pattern(sig, expected):
                # Continue in hot sequence
                self.active_position += 1

                # Yield the command (part of hot sequence)
                yield cmd

                # Check if sequence is complete
                if self.active_position >= len(self.active_hot_sequence_instructions):
                    # Emit END marker
                    seq_id = hex(self.active_hot_sequence_hash)
                    yield WorkerHotPathEnd(sequence_id=seq_id)
                    debug_print_compile(f"Hot sequence {seq_id} complete ({len(self.active_hot_sequence_instructions)} ops)")

                    # Reset for next iteration
                    self.active_hot_sequence_hash = None
                    self.active_hot_sequence_instructions = None
                    self.active_position = 0

                return
            else:
                # Pattern broken - stop tracking this sequence
                debug_print_compile(f"Hot sequence interrupted (expected {expected}, got {sig})")

                # IMPORTANT: Emit END marker to flush the buffer before stopping
                seq_id = hex(self.active_hot_sequence_hash)
                yield WorkerHotPathEnd(sequence_id=seq_id)

                self.active_hot_sequence_hash = None
                self.active_hot_sequence_instructions = None
                self.active_position = 0
                # Fall through to normal processing

        # STATE 2: Not in active sequence - record and detect patterns
        self.window.append(sig)

        # Update sequence counts for all possible sequences ending here
        for seq_len in range(self.min_sequence_length, len(self.window) + 1):
            if seq_len <= len(self.window):
                recent_instructions = list(self.window)[-seq_len:]
                seq_hash = compute_sequence_hash(recent_instructions)
                self.sequence_counts[seq_hash] = self.sequence_counts.get(seq_hash, 0) + 1
                # Store instructions for debugging (only first time)
                if seq_hash not in self.sequence_instructions:
                    self.sequence_instructions[seq_hash] = recent_instructions

        # Check if we should start a hot sequence
        hot_seq = self._detect_hot_sequence()
        if hot_seq is not None:
            seq_hash, seq_instructions = hot_seq
            self.active_hot_sequence_hash = seq_hash
            self.active_hot_sequence_instructions = seq_instructions
            self.active_position = 0

            # Check if current instruction is the START of the hot sequence
            if self._matches_pattern(sig, seq_instructions[0]):
                # Emit START marker
                seq_id = hex(seq_hash)
                yield WorkerHotPathStart(sequence_id=seq_id)
                debug_print_compile(f"Starting hot sequence {seq_id} ({len(seq_instructions)} ops, seen {self.sequence_counts[seq_hash]} times)")

                self.active_position = 1
                self.hot_instructions += 1

        # Always yield the original command
        yield cmd

    def get_stats(self) -> dict:
        """Get detection statistics."""
        # Format top sequences with readable representation
        top_sequences = []
        for seq_hash, count in sorted(self.sequence_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            instructions = self.sequence_instructions.get(seq_hash, [])
            seq_repr = f"Seq({len(instructions)}ops: {' -> '.join(repr(inst) for inst in instructions)})"
            top_sequences.append((seq_repr, count))

        return {
            'total_instructions': self.total_instructions,
            'hot_instructions': self.hot_instructions,
            'unique_sequences': len(self.sequence_counts),
            'hot_sequences': sum(1 for count in self.sequence_counts.values() if count >= self.hot_threshold),
            'hot_threshold': self.hot_threshold,
            'top_sequences': top_sequences
        }
