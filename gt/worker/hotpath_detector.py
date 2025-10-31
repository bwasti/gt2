"""
Hot path detection for automatic torch.compile() optimization.

Operates at the worker level, tracking instruction batches as they execute.
When a batch pattern repeats many times, enables compilation automatically.

Integration: Workers call record_batch() before executing each batch.
"""

from typing import List, Dict, Optional
from collections import defaultdict, deque
import hashlib
import time
import os


class BatchSignature:
    """Signature for an operation batch (for pattern matching)."""

    def __init__(self, operations: List):
        """
        Create signature from batch of operations.

        Args:
            operations: List of Operation objects from gt.worker.engine.base
        """
        self.hash = self._compute_hash(operations)
        self.op_count = len(operations)

    def _compute_hash(self, operations: List) -> str:
        """Compute stable hash of operation batch."""
        # Build signature from operation types and names
        # Similar to _compute_graph_signature in pytorch.py but simpler
        sig_parts = []
        for op in operations:
            sig_parts.append(f"{op.op_type}:{op.op_name}")
            # Include input count for structure
            sig_parts.append(f"in:{len(op.input_ids)}")

        sig_str = "|".join(sig_parts)
        return hashlib.md5(sig_str.encode()).hexdigest()[:12]

    def __eq__(self, other):
        return self.hash == other.hash

    def __hash__(self):
        return hash(self.hash)

    def __repr__(self):
        return f"Batch({self.hash}:{self.op_count}ops)"


class HotPathDetector:
    """
    Detects repeated batch patterns at the worker level.

    Tracks operation batches as they execute and automatically enables
    compilation when patterns repeat frequently.
    """

    def __init__(
        self,
        hot_threshold: int = 10,
        min_ops_per_batch: int = 3,
        enable_auto_compile: bool = True,
    ):
        """
        Initialize hot path detector for a worker.

        Args:
            hot_threshold: Number of repetitions before marking as hot
            min_ops_per_batch: Minimum ops in batch to consider for compilation
            enable_auto_compile: Whether to automatically enable compilation
        """
        # Read from environment if not explicitly set
        if hot_threshold == 10:  # default value
            hot_threshold = int(os.environ.get('GT_HOTPATH_THRESHOLD', '10'))

        self.hot_threshold = hot_threshold
        self.min_ops_per_batch = min_ops_per_batch
        self.enable_auto_compile = enable_auto_compile

        # Pattern tracking
        self.signature_counts: Dict[BatchSignature, int] = defaultdict(int)
        self.hot_signatures: set = set()

        # Performance tracking
        self.signature_times: Dict[BatchSignature, deque] = defaultdict(
            lambda: deque(maxlen=10)
        )
        self.last_batch_time: Optional[float] = None

        # Statistics
        self.total_batches = 0
        self.hot_batches_executed = 0
        self.unique_patterns = 0

    def record_batch(self, operations: List) -> bool:
        """
        Record a batch before execution. Returns whether to compile.

        Args:
            operations: List of Operation objects to execute

        Returns:
            True if this batch should be compiled, False otherwise
        """
        self.total_batches += 1

        # Skip if too few operations
        if len(operations) < self.min_ops_per_batch:
            return False

        # Create signature for this batch
        signature = BatchSignature(operations)

        # Update counts
        self.signature_counts[signature] += 1
        count = self.signature_counts[signature]

        # Update unique pattern count
        self.unique_patterns = len(self.signature_counts)

        # Check if this just became a hot path
        if count == self.hot_threshold and signature not in self.hot_signatures:
            self.hot_signatures.add(signature)
            print(f"[HotPath] Worker detected hot batch after {count} repetitions: {signature}")

        # Track timing
        self.last_batch_time = time.time()

        # Decide whether to compile
        should_compile = (
            self.enable_auto_compile
            and signature in self.hot_signatures
            and count >= self.hot_threshold
        )

        if should_compile:
            self.hot_batches_executed += 1

        return should_compile

    def record_batch_completion(self, operations: List):
        """Record batch completion time for performance tracking."""
        if self.last_batch_time is not None:
            signature = BatchSignature(operations)
            elapsed = time.time() - self.last_batch_time
            self.signature_times[signature].append(elapsed)
            self.last_batch_time = None

    def get_stats(self) -> Dict:
        """Get statistics about detected patterns."""
        return {
            'total_batches': self.total_batches,
            'hot_batches_executed': self.hot_batches_executed,
            'unique_patterns': self.unique_patterns,
            'hot_paths': len(self.hot_signatures),
            'hot_threshold': self.hot_threshold,
            'top_patterns': sorted(
                [(sig, count) for sig, count in self.signature_counts.items()],
                key=lambda x: x[1],
                reverse=True
            )[:5],  # Top 5
        }

    def should_enable_compilation(self) -> bool:
        """Check if any hot paths have been detected."""
        return len(self.hot_signatures) > 0

    def reset(self):
        """Reset all tracking."""
        self.signature_counts.clear()
        self.hot_signatures.clear()
        self.signature_times.clear()
        self.total_batches = 0
        self.hot_batches_executed = 0
        self.unique_patterns = 0
