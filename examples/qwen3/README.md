# Qwen3 Training Examples

Two training implementations for testing GT performance:

## Quick Start

```bash
# Fast test with nano model (25M params)
python train_gt.py --model-size nano --batch-size 8 --num-samples 100

# Realistic training with small model (135M params)
python train_gt.py --model-size small --batch-size 4 --num-samples 500

# Compare with PyTorch
python train_gt.py --pytorch --model-size nano --batch-size 8 --num-samples 100

# Enable profiling
python train_gt.py --model-size nano --profile
```

## Files

- **train_gt.py**: Optimized training with real TinyStories dataset
- **qwen3_gt.py**: Optimized model (nano/small/medium sizes)
- **train.py**: Original training with synthetic data
- **prepare_data.py**: Dataset preparation utilities

## Model Sizes

| Model  | Params | Hidden | Layers |
|--------|--------|--------|--------|
| nano   | 25M    | 256    | 6      |
| small  | 135M   | 768    | 12     |
| medium | 350M   | 1024   | 24     |

## Dataset: TinyStories

Install optional dependencies:
```bash
pip install datasets transformers
```

Falls back to synthetic data if not installed.
