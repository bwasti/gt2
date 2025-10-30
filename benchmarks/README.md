# GT Performance Benchmarks

This directory contains benchmarks to measure the overhead of the GT distributed framework compared to pure PyTorch.

## Files

- `pytorch_baseline.py` - Pure PyTorch benchmarks (baseline performance)
- `gt_benchmark.py` - GT framework benchmarks (with distributed overhead)
- `compare.py` - Side-by-side comparison showing overhead

## Running Benchmarks

### PyTorch Baseline Only
```bash
cd benchmarks
python pytorch_baseline.py
```

### GT Framework Only
First start the GT system, then run:
```bash
cd benchmarks
python gt_benchmark.py
```

### Full Comparison
```bash
cd benchmarks
python compare.py
```

## What's Being Measured

The benchmarks measure the following operations:

1. **Matrix Multiplication** (1000x1000) - Core linear algebra operation
2. **Linear Layer Forward** (128x512->256) - Neural network layer
3. **MSE Loss** (10000 elements) - Loss function computation
4. **Backward Pass** (1000x1000) - Autograd gradient computation
5. **Activation Functions** (10000 elements) - ReLU, Sigmoid, Tanh

## Expected Overhead

The GT framework adds overhead due to:
- Network communication (client <-> dispatcher <-> worker)
- Serialization/deserialization (pickle)
- Protocol overhead (command parsing)
- Round-trip latency

For small operations, this overhead can be significant (100-1000%+).
For larger operations, the overhead becomes negligible as compute dominates.

## Backend

The worker backend can be switched between 'numpy' and 'pytorch':
- **numpy**: Reference implementation, slower compute
- **pytorch**: Production backend, faster compute (CPU/GPU)

To change the backend, edit `simple_launch.py` or `gt/server/__main__.py`.
