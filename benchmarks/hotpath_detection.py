"""
Isolated hotpath detection benchmark and correctness test.

Tests the hotpath detector in isolation without full GT system overhead.
Measures:
1. Detection accuracy (% of repeated patterns detected)
2. Performance overhead (ops/sec throughput)
3. Memory usage

Use this for profiling and optimization of hotpath detection logic.
"""

import time
from collections import defaultdict
from gt.worker.hotpath_detector import HotPathDetector, InstructionSignature
from gt.transport.protocol import WorkerBinaryOp, WorkerUnaryOp, WorkerHotPathStart, WorkerHotPathEnd


class SyntheticWorkload:
    """Generate synthetic operation streams with known patterns."""

    def __init__(self):
        self.op_counter = 0

    def make_op(self, op_type, op_name):
        """Create a synthetic operation."""
        self.op_counter += 1
        if op_type == 'binary':
            return WorkerBinaryOp(
                left_id=f"t{self.op_counter}",
                right_id=f"t{self.op_counter+1}",
                op=op_name,
                result_id=f"t{self.op_counter+2}"
            )
        elif op_type == 'unary':
            return WorkerUnaryOp(
                input_id=f"t{self.op_counter}",
                op=op_name,
                result_id=f"t{self.op_counter+1}",
                shape=None,
                dtype='float32',
                axis=None,
                keepdims=False
            )

    def repeated_pattern(self, pattern, repetitions):
        """Generate a repeated pattern.

        Args:
            pattern: List of (op_type, op_name) tuples
            repetitions: Number of times to repeat

        Returns:
            List of operations
        """
        ops = []
        for _ in range(repetitions):
            for op_type, op_name in pattern:
                ops.append(self.make_op(op_type, op_name))
        return ops

    def mixed_workload(self, hot_pattern, hot_reps, cold_ops):
        """Generate workload with hot pattern + cold operations.

        Args:
            hot_pattern: Pattern that should be detected as hot
            hot_reps: Repetitions of hot pattern
            cold_ops: Number of random cold operations between patterns

        Returns:
            List of operations, expected hot operations count
        """
        ops = []
        expected_hot = 0

        # Start with some cold operations
        for _ in range(cold_ops):
            ops.append(self.make_op('binary', 'add'))

        # Repeated hot pattern
        for i in range(hot_reps):
            for op_type, op_name in hot_pattern:
                ops.append(self.make_op(op_type, op_name))
                # After threshold, these should be marked as hot
                if i >= 5:  # Assuming threshold=5
                    expected_hot += 1

            # Some cold ops between pattern repetitions
            if cold_ops > 0 and i < hot_reps - 1:
                for _ in range(cold_ops // 2):
                    ops.append(self.make_op('binary', 'mul'))

        return ops, expected_hot


def test_detection_accuracy():
    """Test that hotpath detector correctly identifies repeated patterns."""
    print("=" * 80)
    print("HOTPATH DETECTION ACCURACY TEST")
    print("=" * 80)
    print()

    workload = SyntheticWorkload()

    # Pattern: matmul -> relu -> matmul (typical MLP layer)
    pattern = [
        ('binary', 'matmul'),
        ('unary', 'relu'),
        ('binary', 'matmul'),
    ]

    test_cases = [
        ("Pure repeated pattern (50 reps)", pattern, 50, 0),
        ("Pattern with noise (20 reps, 2 cold ops)", pattern, 20, 2),
        ("Pattern with more noise (15 reps, 5 cold ops)", pattern, 15, 5),
    ]

    for test_name, test_pattern, reps, cold_ops in test_cases:
        print(f"\nTest: {test_name}")
        print(f"  Pattern: {test_pattern}")
        print(f"  Repetitions: {reps}, Cold ops: {cold_ops}")

        # Generate workload
        ops, expected_hot = workload.mixed_workload(test_pattern, reps, cold_ops)
        print(f"  Total operations: {len(ops)}")
        print(f"  Expected hot operations: {expected_hot}")

        # Run detector
        detector = HotPathDetector(
            window_size=20,
            hot_threshold=5,
            min_sequence_length=3
        )

        # Count hot path markers
        hot_starts = 0
        hot_ends = 0
        ops_in_hot_path = 0
        in_hot_path = False

        for op in ops:
            for output_cmd in detector.process(op):
                if isinstance(output_cmd, WorkerHotPathStart):
                    hot_starts += 1
                    in_hot_path = True
                elif isinstance(output_cmd, WorkerHotPathEnd):
                    hot_ends += 1
                    in_hot_path = False
                elif in_hot_path:
                    ops_in_hot_path += 1

        # Results
        print(f"\n  Results:")
        print(f"    Hot path markers: {hot_starts} START, {hot_ends} END")
        print(f"    Operations in hot paths: {ops_in_hot_path}")

        if expected_hot > 0:
            detection_rate = ops_in_hot_path / expected_hot
            print(f"    Detection rate: {detection_rate:.1%}")

            if detection_rate > 0.8:
                print(f"    ✅ Good detection (>80%)")
            elif detection_rate > 0.5:
                print(f"    ⚠️  Moderate detection (50-80%)")
            else:
                print(f"    ❌ Poor detection (<50%)")

        # Detector stats
        stats = detector.get_stats()
        print(f"\n  Detector stats:")
        print(f"    Total instructions: {stats['total_instructions']}")
        print(f"    Hot instructions: {stats['hot_instructions']}")
        print(f"    Unique sequences tracked: {stats['unique_sequences']}")
        print(f"    Hot sequences (>threshold): {stats['hot_sequences']}")


def benchmark_throughput():
    """Measure hotpath detector throughput (ops/sec)."""
    print("\n" + "=" * 80)
    print("HOTPATH DETECTION THROUGHPUT BENCHMARK")
    print("=" * 80)
    print()

    workload = SyntheticWorkload()

    # Create a workload with repeated pattern
    pattern = [
        ('binary', 'matmul'),
        ('unary', 'relu'),
        ('binary', 'add'),
    ]

    test_configs = [
        ("Small (1K ops)", 1000),
        ("Medium (10K ops)", 10000),
        ("Large (50K ops)", 50000),
    ]

    for test_name, num_ops in test_configs:
        print(f"\n{test_name}:")

        # Generate workload
        reps = num_ops // len(pattern)
        ops = workload.repeated_pattern(pattern, reps)
        print(f"  Operations: {len(ops)}")

        # Benchmark
        detector = HotPathDetector(
            window_size=20,
            hot_threshold=5,
            min_sequence_length=3
        )

        start = time.time()
        for op in ops:
            for _ in detector.process(op):
                pass
        elapsed = time.time() - start

        throughput = len(ops) / elapsed
        latency_us = (elapsed / len(ops)) * 1e6

        print(f"  Time: {elapsed:.3f}s")
        print(f"  Throughput: {throughput:,.0f} ops/sec")
        print(f"  Latency: {latency_us:.2f} µs/op")

        # Stats
        stats = detector.get_stats()
        print(f"  Detection stats:")
        print(f"    Hot instructions: {stats['hot_instructions']}")
        print(f"    Unique sequences: {stats['unique_sequences']}")


def profile_hotspots():
    """Profile the hotpath detector to find bottlenecks."""
    print("\n" + "=" * 80)
    print("HOTPATH DETECTION PROFILING")
    print("=" * 80)
    print()

    import cProfile
    import pstats
    import io

    workload = SyntheticWorkload()

    # Create workload
    pattern = [
        ('binary', 'matmul'),
        ('unary', 'relu'),
        ('binary', 'add'),
        ('unary', 'tanh'),
    ]

    reps = 5000
    ops = workload.repeated_pattern(pattern, reps)
    print(f"Profiling with {len(ops)} operations...")
    print()

    # Profile
    detector = HotPathDetector(
        window_size=20,
        hot_threshold=5,
        min_sequence_length=3
    )

    profiler = cProfile.Profile()
    profiler.enable()

    for op in ops:
        for _ in detector.process(op):
            pass

    profiler.disable()

    # Print stats
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    print(s.getvalue())

    print("\n" + "=" * 80)
    print("TOP 20 BY TOTAL TIME")
    print("=" * 80)
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('tottime')
    ps.print_stats(20)
    print(s.getvalue())


def stress_test():
    """Stress test with very large workload."""
    print("\n" + "=" * 80)
    print("STRESS TEST (100K operations)")
    print("=" * 80)
    print()

    workload = SyntheticWorkload()

    # Create large workload with multiple patterns
    patterns = [
        [('binary', 'matmul'), ('unary', 'relu')],
        [('binary', 'add'), ('unary', 'sigmoid')],
        [('binary', 'matmul'), ('unary', 'tanh'), ('binary', 'mul')],
    ]

    ops = []
    for i in range(100000 // 10):
        pattern = patterns[i % len(patterns)]
        ops.extend([workload.make_op(t, n) for t, n in pattern])

    print(f"Operations: {len(ops)}")
    print("Running detector...")

    detector = HotPathDetector(
        window_size=20,
        hot_threshold=5,
        min_sequence_length=3
    )

    start = time.time()
    hot_markers = 0

    for op in ops:
        for output in detector.process(op):
            if isinstance(output, (WorkerHotPathStart, WorkerHotPathEnd)):
                hot_markers += 1

    elapsed = time.time() - start

    print(f"Time: {elapsed:.3f}s")
    print(f"Throughput: {len(ops)/elapsed:,.0f} ops/sec")
    print(f"Hot path markers: {hot_markers}")

    stats = detector.get_stats()
    print(f"\nStats:")
    print(f"  Total instructions: {stats['total_instructions']}")
    print(f"  Hot instructions: {stats['hot_instructions']}")
    print(f"  Unique sequences: {stats['unique_sequences']}")
    print(f"  Hot sequences: {stats['hot_sequences']}")
    print(f"  Hit rate: {stats['hot_instructions']/stats['total_instructions']:.1%}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == '--accuracy':
            test_detection_accuracy()
        elif mode == '--throughput':
            benchmark_throughput()
        elif mode == '--profile':
            profile_hotspots()
        elif mode == '--stress':
            stress_test()
        else:
            print(f"Unknown mode: {mode}")
            print("Usage: python benchmarks/hotpath_detection.py [--accuracy|--throughput|--profile|--stress]")
    else:
        # Run all tests
        test_detection_accuracy()
        benchmark_throughput()
        stress_test()

        print("\n" + "=" * 80)
        print("COMPLETE")
        print("=" * 80)
        print("\nTo profile: python benchmarks/hotpath_detection.py --profile")
