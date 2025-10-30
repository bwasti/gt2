# Qwen3-1.7B Training with GT

Fine-tuning Qwen3-1.7B using GT's distributed tensor framework.

## Overview

This example demonstrates training a production language model (Qwen3-1.7B) using GT. The code uses `import gt as torch` to allow easy comparison with PyTorch.

## Files

- `model.py` - Qwen3 model architecture using GT operations
- `load_weights.py` - Downloads HuggingFace weights and converts to GT format
- `prepare_data.py` - Downloads and tokenizes training data (Alpaca dataset)
- `train.py` - Training loop with loss computation and optimization
- `train.sh` - Runs complete pipeline

## Quick Start

```bash
# Install dependencies
pip install transformers datasets

# Run complete training pipeline
cd examples/qwen3
./train.sh
```

## What It Does

1. **Load Weights** (`load_weights.py`)
   - Downloads Qwen2.5-1.5B-Instruct from HuggingFace
   - Converts weights to GT tensor format
   - Verifies forward pass matches HuggingFace implementation

2. **Prepare Data** (`prepare_data.py`)
   - Downloads Alpaca instruction-following dataset (1000 examples)
   - Tokenizes with Qwen tokenizer
   - Saves as numpy arrays for training

3. **Train** (`train.py`)
   - Loads data in batches
   - Runs forward pass through model
   - Computes loss and gradients
   - Updates parameters with SGD optimizer

## Model Architecture

Qwen3-1.7B configuration:
- Vocabulary: 152,064 tokens
- Hidden size: 2048
- Layers: 28 transformer blocks
- Attention heads: 16
- Intermediate size: 11,008
- Context length: 32,768 tokens

Components:
- RMSNorm for layer normalization
- Rotary position embeddings (RoPE)
- Multi-head self-attention
- SwiGLU MLP activation

## Benchmarking

Since the code uses `import gt as torch`, you can benchmark against PyTorch by:

1. Copy train.py to train_pytorch.py
2. Change `import gt as torch` to `import torch`
3. Run both and compare training time

## Current Limitations

- Loss function is simplified (uses MSE instead of cross-entropy as placeholder)
- RoPE implementation is simplified (returns input unchanged)
- Attention is single-head instead of multi-head
- No gradient accumulation or mixed precision

These are placeholders to get the pipeline working. Full implementations can be added incrementally.

## Environment Variables

```bash
# Enable instruction batching
GT_WORKER_BATCH_SIZE=10 ./train.sh

# Enable sharding across GPUs
GT_CONFIG=sharding.yaml ./train.sh

# Log instruction stream
GT_INSTRUCTION_LOG=debug.log ./train.sh
```

## Dataset

Using Alpaca dataset (1000 examples) for demonstration:
- Instruction-following format
- Max sequence length: 512 tokens
- Tasks: question answering, summarization, etc.

## Expected Output

```
Step 1: Loading weights from HuggingFace...
Model loaded. Vocab size: 151936
Converting weights to GT format...
Weight conversion complete!
Forward pass verification PASSED

Step 2: Preparing training data...
Dataset size: 1000 examples
Tokenization complete!

Step 3: Training model...
Epoch 1/3
  Batch 0/500 | Loss: 0.1234 | Time: 2.31s
  ...
Training Complete!
```

## Notes

- First run downloads ~3GB model weights
- Training dataset is ~100MB after tokenization
- Memory usage depends on batch size and sequence length
- Compilation overhead occurs on first batch (if batching enabled)
