# Installation

GT comes batteries-included with all dependencies needed for distributed training, visualization, and monitoring.

## From GitHub (pip)

```bash
# Install GT (batteries included: PyTorch, visualization, monitoring)
pip install git+https://github.com/bwasti/gt.git
```

This is the recommended installation method for most users.

## From Source (editable)

For development or contributions:

```bash
# Clone repository
git clone https://github.com/bwasti/gt.git
cd gt

# Install in editable mode
pip install -e .
```

## Included Dependencies

**Core Dependencies:**
- **PyTorch** - GPU/CPU support, compilation
- **matplotlib** - Timeline visualizations
- **rich + psutil** - Real-time monitoring
- **NumPy** - Numerical operations
- **pytest** - Testing framework
- **pyzmq** - ZeroMQ transport layer
- **pyyaml** - Configuration file parsing

All dependencies are installed automatically - no optional extras needed.

## Verify Installation

Test your installation:

```bash
# Quick test
python -c "import gt; print(gt.randn(2, 2))"

# Should output a 2x2 tensor
```

This auto-starts a local dispatcher and worker, confirming everything is working.

## GPU Setup

GT automatically uses GPUs when available through PyTorch. No additional configuration needed.

To verify GPU access:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Next Steps

- [Usage Guide](usage.md) - Learn how to use GT
- [Quick Start](README.md#quick-start) - Simple examples to get started
- [Contributing](contributing.md) - Set up development environment
