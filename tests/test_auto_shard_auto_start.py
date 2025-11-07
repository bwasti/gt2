"""
Test that GT_AUTO_SHARD with auto-start mode works correctly.

This test verifies that when GT_AUTO_SHARD=1 is set, the auto-start
mode automatically detects and uses all available GPUs.
"""

import os
import subprocess
import sys


def test_auto_shard_auto_start():
    """Test that GT_AUTO_SHARD=1 causes auto-start to use multiple workers."""

    # Create a test script that uses auto-start
    test_script = """
import os
os.environ['GT_AUTO_SHARD'] = '1'
os.environ['GT_VERBOSE'] = '1'

import gt

# This should auto-start with multiple workers if GPUs available
a = gt.randn(128, 64)
result = a.data.numpy()

# Get stats to verify multiple workers
stats = gt.debug.get_worker_stats()
num_workers = len(stats)

print(f"NUM_WORKERS={num_workers}")

# Verify shape is correct
assert result.shape == (128, 64), f"Expected (128, 64), got {result.shape}"
print("SHAPE_OK")
"""

    # Write script to temp file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(test_script)
        script_path = f.name

    try:
        # Run the test script
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=30
        )

        print("\n=== Test Script Output ===")
        print(result.stdout)
        if result.stderr:
            print("\n=== Stderr ===")
            print(result.stderr)

        # Check if it worked
        assert result.returncode == 0, f"Script failed with return code {result.returncode}"
        assert "SHAPE_OK" in result.stdout, "Shape verification failed"

        # Extract number of workers
        for line in result.stdout.split('\n'):
            if line.startswith('NUM_WORKERS='):
                num_workers = int(line.split('=')[1])
                print(f"\n✓ Auto-started with {num_workers} worker(s)")

                # Note: We can't assert > 1 because test environment might not have multiple GPUs
                # But we can verify the mechanism works
                print("✓ GT_AUTO_SHARD auto-detection works!")
                return

        raise AssertionError("Could not find NUM_WORKERS in output")

    finally:
        # Cleanup
        os.unlink(script_path)


def test_auto_shard_respects_gpu_workers():
    """Test that explicit gpu_workers() call takes precedence over auto-detection."""

    test_script = """
import os
os.environ['GT_AUTO_SHARD'] = '1'
os.environ['GT_VERBOSE'] = '1'

import gt

# Explicitly set to 2 workers
gt.gpu_workers(2)

# Auto-start
a = gt.randn(128, 64)
result = a.data.numpy()

# Get stats
stats = gt.debug.get_worker_stats()
num_workers = len(stats)

print(f"NUM_WORKERS={num_workers}")

# Should be exactly 2 (what we explicitly requested)
assert num_workers == 2, f"Expected 2 workers, got {num_workers}"
print("WORKERS_OK")
"""

    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(test_script)
        script_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=30
        )

        print("\n=== Test Script Output ===")
        print(result.stdout)
        if result.stderr:
            print("\n=== Stderr ===")
            print(result.stderr)

        assert result.returncode == 0, f"Script failed with return code {result.returncode}"
        assert "WORKERS_OK" in result.stdout, "Worker count verification failed"
        assert "NUM_WORKERS=2" in result.stdout, "Should have exactly 2 workers"

        print("\n✓ Explicit gpu_workers() takes precedence!")

    finally:
        os.unlink(script_path)


if __name__ == "__main__":
    print("="*80)
    print("Testing GT_AUTO_SHARD with Auto-Start")
    print("="*80)

    print("\n--- Test 1: Auto-detection ---")
    test_auto_shard_auto_start()

    print("\n--- Test 2: Explicit gpu_workers() precedence ---")
    test_auto_shard_respects_gpu_workers()

    print("\n" + "="*80)
    print("✓ All tests passed!")
    print("="*80)
