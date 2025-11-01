"""
Hot path detection for automatic torch.compile() optimization.

Operates on the INSTRUCTION STREAM - individual operations as they arrive.
Detects repeated sequences in the stream and enables compilation.

Key insight: Batching is just transport optimization (message reduction).
Hot path detection should see the raw instruction stream.

CRITICAL: Signatures must be based on DEPENDENCY GRAPH structure, not just
operation sequence. The order and dependencies of inputs matter!

Example:
  a = f(b, c)
  e = g(a, d)  # e depends on a
Is NOT the same as:
  a = f(b, c)
  e = g(b, c)  # e doesn't depend on a
"""

from typing import List, Dict, Optional, Tuple, Set
from collections import deque
import hashlib
import time


class InstructionSignature:
    """Signature for a single instruction with dependency tracking."""

    def __init__(self, op_type: str, op_name: str, result_id: str, input_ids: List[str]):
        """
        Create signature for an instruction.

        Args:
            op_type: 'binary', 'unary', 'reshape', etc.
            op_name: 'matmul', 'relu', 'add', etc.
            result_id: ID of the output tensor
            input_ids: List of input tensor IDs (order matters!)
        """
        self.op_type = op_type
        self.op_name = op_name
        self.result_id = result_id
        self.input_ids = input_ids

    def __repr__(self):
        inputs_str = ",".join(self.input_ids)
        return f"{self.result_id}={self.op_name}({inputs_str})"


class StreamSignature:
    """
    Signature for a sequence of instructions based on dependency graph structure.

    Canonicalizes the instruction stream by renormalizing tensor IDs based on
    the dependency graph structure, not the actual ID values.
    """

    def __init__(self, instructions: List[InstructionSignature]):
        """
        Create signature from instruction sequence.

        Args:
            instructions: List of InstructionSignature objects
        """
        self.instructions = instructions
        self.length = len(instructions)
        self.hash = self._compute_graph_hash()

    def _compute_graph_hash(self) -> str:
        """
        Compute hash based on dependency graph structure.

        Renormalizes tensor IDs to create a canonical representation:
        - Inputs are renormalized based on first use
        - Intermediate values are numbered by creation order
        - The graph structure (which op depends on which outputs) is preserved
        """
        if not self.instructions:
            return hashlib.md5(b"").hexdigest()[:12]

        # Build dependency graph and renormalize IDs
        id_mapping = {}  # Old ID -> Canonical ID
        next_input_id = 0
        next_intermediate_id = 0
        canonical_ops = []

        for inst in self.instructions:
            # Renormalize input IDs
            canonical_inputs = []
            for input_id in inst.input_ids:
                if input_id not in id_mapping:
                    # First time seeing this input - it's an external input
                    id_mapping[input_id] = f"in{next_input_id}"
                    next_input_id += 1
                canonical_inputs.append(id_mapping[input_id])

            # Renormalize output ID
            canonical_result = f"t{next_intermediate_id}"
            id_mapping[inst.result_id] = canonical_result
            next_intermediate_id += 1

            # Create canonical operation string
            # Format: "result = op(input1, input2, ...)"
            # This preserves input ORDER and dependencies
            inputs_str = ",".join(canonical_inputs)
            op_str = f"{canonical_result}={inst.op_name}({inputs_str})"
            canonical_ops.append(op_str)

        # Hash the canonical representation
        canonical_str = "|".join(canonical_ops)
        return hashlib.md5(canonical_str.encode()).hexdigest()[:12]

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

    TODO: Advanced graph analysis for independent subgraphs
    --------------------------------------------------------
    Currently we detect hot paths in the instruction stream as-is.
    Future enhancement: Detect when multiple INDEPENDENT dependency graphs
    are interleaved in the stream, and:

    1. Identify independent subgraphs within a sequence
    2. Extract and compile only the hot subgraph(s)
    3. Re-sort the instruction stream to group compiled code together
    4. Execute with partial compilation (compile hot path, stream cold path)

    Example:
        # Interleaved independent graphs
        a = f(b)      # Graph 1
        x = g(y)      # Graph 2 (independent of Graph 1)
        c = h(a)      # Graph 1 continued
        z = k(x)      # Graph 2 continued

    If Graph 1 is hot but Graph 2 is cold, we should:
    - Detect they are independent (no shared tensors)
    - Re-sort to: [Graph 1 ops] [Graph 2 ops]
    - Compile only Graph 1
    - Execute: compiled_fn(b) + stream_execute(Graph 2)

    This maximizes compilation benefits without over-compiling.
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
        result_id: str,
        input_ids: List[str]
    ) -> Tuple[bool, Optional[int]]:
        """
        Record an instruction in the stream.

        Args:
            op_type: Type of operation ('binary', 'unary', etc.)
            op_name: Name of operation ('matmul', 'relu', etc.)
            result_id: ID of the output tensor
            input_ids: List of input tensor IDs (order matters!)

        Returns:
            (should_compile, sequence_length) - True if we should compile,
            and if so, how many upcoming instructions are part of hot sequence
        """
        self.total_instructions += 1

        # Create instruction signature
        inst = InstructionSignature(op_type, op_name, result_id, input_ids)

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
