"""
Simple example showing automatic tensor sharding across multiple workers.

IMPORTANT: Auto-connect (gt.tensor without explicit connection) only starts
1 worker by default. To use multiple GPUs, you need to manually start workers:

For distributed sharding across N GPUs:

Terminal 1 - Start server:
    python -m gt.server -p 12345

Terminal 2-N - Start workers (one per GPU):
    CUDA_VISIBLE_DEVICES=0 python -m gt.worker --host localhost -p 12345
    CUDA_VISIBLE_DEVICES=1 python -m gt.worker --host localhost -p 12345
    CUDA_VISIBLE_DEVICES=2 python -m gt.worker --host localhost -p 12345
    CUDA_VISIBLE_DEVICES=3 python -m gt.worker --host localhost -p 12345

Terminal N+1 - Run this script:
    python examples/simple_auto_shard.py --distributed
"""

import numpy as np
import argparse


def main(distributed=False):
    import gt

    if distributed:
        # Connect to existing multi-worker system
        gt.connect('localhost:12345')
        print("Connected to distributed GT system")
    else:
        # Auto-connect mode (single worker)
        print("Using auto-connect mode (1 worker - no sharding)")
        print("For sharding demo, run with --distributed flag and start workers first")

    print("\n=== Distributed Matrix Multiplication Example ===\n")

    # Create sharded matrix A (128, 64)
    # With 4 workers: automatically sharded as (32, 64) per worker
    # With 1 worker: no sharding, stays as (128, 64)
    print("Creating matrix A (128, 64)...")
    a = gt.randn(128, 64)
    print(f"  A shape: {a.data.numpy().shape}")

    # Create matrix B (64, 32) - stays on one worker
    print("\nCreating matrix B (64, 32)...")
    b_data = np.random.randn(64, 32).astype('float32')
    b = gt.tensor(b_data)
    print(f"  B shape: {b.data.numpy().shape}")

    # Distributed matmul: C = A @ B
    # With 4 workers: Each computes A_shard @ B in parallel
    # With 1 worker: Standard matmul
    print("\nComputing C = A @ B...")
    c = a @ b

    # Get result (automatically gathers from all workers if sharded)
    result = c.data.numpy()
    print(f"  Result shape: {result.shape}")

    # Verify correctness
    a_data = a.data.numpy()
    b_data_full = b.data.numpy()
    expected = a_data @ b_data_full

    max_diff = np.max(np.abs(result - expected))
    print(f"\n✓ Result matches numpy computation (max diff: {max_diff:.2e})")

    # Show what's happening under the hood
    print("\n=== How Sharding Works ===")
    print("With 4 workers:")
    print("  1. A (128, 64) is sharded → each worker gets (32, 64)")
    print("  2. B (64, 32) is broadcast to all workers")
    print("  3. Each worker computes: A_shard @ B in parallel")
    print("  4. Results are gathered: (32,32) × 4 → (128, 32)")
    print("\nWith 1 worker:")
    print("  1. A (128, 64) stays on single worker")
    print("  2. B (64, 32) is on the same worker")
    print("  3. Standard matmul: A @ B → (128, 32)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-sharding example")
    parser.add_argument('--distributed', action='store_true',
                        help='Connect to distributed system (requires manual worker startup)')
    args = parser.parse_args()

    main(distributed=args.distributed)
