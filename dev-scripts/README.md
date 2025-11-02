# Development Scripts

This directory contains one-off scripts used during GT development for debugging, validation, and testing specific issues.

## Contents

- **test_hotpath.py** - Tests hot path detection for torch.compile integration
  - Run with `GT_AUTO_COMPILE=1 python dev-scripts/test_hotpath.py`
  - Validates that repeated operation patterns trigger compilation

- **test_matmul_transpose_batch.py** - Debugging script for batch size bug
  - Used to reproduce and fix matmul + transpose pattern issues
  - Mimics Qwen3 attention mechanism patterns

- **test_subscripting.py** - Validates tensor subscripting semantics
  - Compares GT tensor slicing against PyTorch reference
  - Ensures subscripting behavior matches exactly

## Usage

These scripts are standalone and can be run directly:

```bash
python dev-scripts/test_hotpath.py
python dev-scripts/test_subscripting.py
python dev-scripts/test_matmul_transpose_batch.py
```

## Note

These are development/debugging scripts and may become outdated. For production examples, see `/examples/`.
