"""
Detailed profiling of framework overhead.
"""

import time
import numpy as np
import cProfile
import pstats
import io

def profile_operations():
    """Profile GT operations to find bottlenecks."""
    import gt

    print("Creating tensors...")
    a = gt.randn(1000, 1000)
    b = gt.randn(1000, 1000)

    # Warmup
    for _ in range(10):
        c = a @ b
        total = c.sum()

    print("\nProfiling 100 iterations...")

    pr = cProfile.Profile()
    pr.enable()

    for _ in range(100):
        c = a @ b
        total = c.sum()

    pr.disable()

    # Print stats
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(30)  # Top 30 functions

    print(s.getvalue())

if __name__ == "__main__":
    import gt

    # Auto-start the GT system
    gt.zeros(1, 1)  # Force initialization

    try:
        profile_operations()
    finally:
        # Clean shutdown
        import time
        time.sleep(0.1)  # Let pending operations finish
        print("\nProfile complete!")
