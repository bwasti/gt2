"""
Profile the matmul operation to find bottlenecks.
"""

import cProfile
import pstats
import io
import gt
import numpy as np


def run_matmul():
    """Run a single matmul to profile."""
    # Single iteration to profile
    a = gt.randn(100, 100)
    b = gt.randn(100, 100)
    c = a @ b
    result = c.data.numpy()
    return result


if __name__ == "__main__":
    print("Profiling matmul operation...")

    # Profile the operation
    profiler = cProfile.Profile()
    profiler.enable()

    result = run_matmul()

    profiler.disable()

    # Print stats
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(30)  # Top 30 functions

    print(s.getvalue())
