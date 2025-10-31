"""
Hot path detection for automatic torch.compile() optimization.

Operates on the INSTRUCTION STREAM - individual operations as they arrive.
Detects repeated sequences in the stream and enables compilation.

Key insight: Batching is just transport optimization (message reduction).
Hot path detection should see the raw instruction stream.
"""

from typing import List, Dict, Optional, Tuple
from collections import deque
import hashlib
import time


class InstructionSignature:
    """Signature for a single instruction."""

    def __init__(self, op_type: str, op_name: str, input_count: int):
        """
        Create signature for an instruction.

        Args:
            op_type: 'binary', 'unary', 'reshape', etc.
            op_name: 'matmul', 'relu', 'add', etc.
            input_count: Number of inputs
        """
        self.op_type = op_type
        self.op_name = op_name
        self.input_count = input_count
        self.hash = f"{op_type}:{op_name}:in{input_count}"

    def __eq__(self, other):
        return self.hash == other.hash

    def __hash__(self):
        return hash(self.hash)

    def __repr__(self):
        return self.hash


class StreamSignature:
    """Signature for a sequence of instructions (hot path)."""

    def __init__(self, instructions: List[InstructionSignature]):
        """
        Create signature from instruction sequence.

        Args:
            instructions: List of InstructionSignature objects
        """
        self.instructions = instructions
        self.length = len(instructions)
        self.hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute hash of instruction sequence."""
        sig_str = "|".join(str(inst) for inst in self.instructions)
        return hashlib.md5(sig_str.encode()).hexdigest()[:12]

    def __eq__(self, other):
        return self.hash == other.hash

    def __hash__(self):
        return hash(self.hash)

    def __repr__(self):
        return f"Stream({self.hash}:{self.length}ops)"


class HotPathDetector:
    """
    Detects repeated instruction sequences in the operation stream.

    Tracks individual instructions as they arrive (stream-based).
    Detects when sequences repeat many times.
    """

    def __init__(
        self,
        window_size: int = 20,
        hot_threshold: int = 10,
        min_sequence_length: int = 5,
    ):
        """
        Initialize stream-based hot path detector.

        Args:
            window_size: How many recent instructions to track
            hot_threshold: Number of repetitions before marking as hot
            min_sequence_length: Minimum length of sequence to detect
        """
        self.window_size = window_size
        self.hot_threshold = hot_threshold
        self.min_sequence_length = min_sequence_length

        # Instruction stream tracking
        self.instruction_stream: deque = deque(maxlen=window_size)

        # Pattern detection
        self.sequence_counts: Dict[StreamSignature, int] = {}
        self.hot_sequences: set = set()

        # Current sequence tracking (for matching)
        self.current_sequence: List[InstructionSignature] = []
        self.sequence_start_idx = 0

        # Statistics
        self.total_instructions = 0
        self.hot_instructions_executed = 0

    def record_instruction(
        self,
        op_type: str,
        op_name: str,
        input_count: int
    ) -> Tuple[bool, Optional[int]]:
        """
        Record an instruction in the stream.

        Args:
            op_type: Type of operation ('binary', 'unary', etc.)
            op_name: Name of operation ('matmul', 'relu', etc.)
            input_count: Number of inputs

        Returns:
            (should_compile, sequence_length) - True if we should compile,
            and if so, how many upcoming instructions are part of hot sequence
        """
        self.total_instructions += 1

        # Create instruction signature
        inst = InstructionSignature(op_type, op_name, input_count)

        # Add to stream
        self.instruction_stream.append(inst)
        self.current_sequence.append(inst)

        # Check for repeated sequences
        if len(self.current_sequence) >= self.min_sequence_length:
            # Try to match against known patterns
            seq_sig = StreamSignature(self.current_sequence)

            # Check if this matches a known sequence
            if seq_sig in self.sequence_counts:
                self.sequence_counts[seq_sig] += 1
                count = self.sequence_counts[seq_sig]

                # Mark as hot if threshold reached
                if count >= self.hot_threshold and seq_sig not in self.hot_sequences:
                    self.hot_sequences.add(seq_sig)
                    print(f"[HotPath] Detected hot sequence after {count} reps: {seq_sig}")

                # If this is a hot sequence, signal compilation
                if seq_sig in self.hot_sequences:
                    self.hot_instructions_executed += 1
                    # Return True and remaining length
                    remaining = seq_sig.length - len(self.current_sequence)
                    if remaining <= 0:
                        # Sequence complete, reset
                        self.current_sequence = []
                    return True, seq_sig.length

            else:
                # New sequence, record it
                self.sequence_counts[seq_sig] = 1

            # Sliding window: if sequence gets too long, slide it
            if len(self.current_sequence) > self.window_size // 2:
                # Start new sequence from midpoint
                self.current_sequence = self.current_sequence[self.min_sequence_length:]

        return False, None

    def reset_sequence(self):
        """Reset current sequence tracking (call at sync points)."""
        if self.current_sequence:
            # Record the sequence we just saw
            if len(self.current_sequence) >= self.min_sequence_length:
                seq_sig = StreamSignature(self.current_sequence)
                if seq_sig not in self.sequence_counts:
                    self.sequence_counts[seq_sig] = 0
                self.sequence_counts[seq_sig] += 1

            self.current_sequence = []

    def get_stats(self) -> Dict:
        """Get statistics about detected patterns."""
        return {
            'total_instructions': self.total_instructions,
            'hot_instructions': self.hot_instructions_executed,
            'unique_sequences': len(self.sequence_counts),
            'hot_sequences': len(self.hot_sequences),
            'hot_threshold': self.hot_threshold,
            'window_size': self.window_size,
            'top_sequences': sorted(
                [(seq, count) for seq, count in self.sequence_counts.items()],
                key=lambda x: x[1],
                reverse=True
            )[:5],
        }

    def should_enable_compilation(self) -> bool:
        """Check if any hot sequences have been detected."""
        return len(self.hot_sequences) > 0
