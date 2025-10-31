"""
Train optimized Qwen3 model with real data.

Uses the highly optimized qwen3_gt.py model and TinyStories dataset.

Usage:
    # Train with GT (nano model, fast)
    python train_gt.py --model-size nano --epochs 1 --batch-size 8 --num-samples 100

    # Train with GT (small model, realistic)
    python train_gt.py --model-size small --epochs 1 --batch-size 4 --num-samples 500

    # Compare with PyTorch
    python train_gt.py --pytorch --model-size nano --epochs 1 --batch-size 8 --num-samples 100
"""

import os
import sys
import time
import argparse
import numpy as np


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train optimized Qwen3 model with GT or PyTorch')
    parser.add_argument('--pytorch', action='store_true',
                        help='Use real PyTorch instead of GT (for benchmarking)')
    parser.add_argument('--model-size', type=str, default='nano',
                        choices=['nano', 'small', 'medium'],
                        help='Model size: nano (25M), small (135M), medium (350M)')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size for training (default: 4)')
    parser.add_argument('--num-samples', type=int, default=500,
                        help='Number of training samples (default: 500)')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate (default: 3e-4)')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of epochs (default: 1)')
    parser.add_argument('--profile', action='store_true',
                        help='Enable instruction logging for profiling')
    return parser.parse_args()


# Parse arguments first
args = parse_args()

# Set environment variables
if args.pytorch:
    os.environ['USE_PYTORCH'] = '1'

if args.profile:
    os.environ['GT_INSTRUCTION_LOG'] = f'/tmp/gt_profile_{args.model_size}.log'
    print(f"Profiling enabled: /tmp/gt_profile_{args.model_size}.log")

# Import after setting environment
USE_PYTORCH = args.pytorch

if USE_PYTORCH:
    import torch
    from torch.optim import AdamW
    print("Using PyTorch backend")
else:
    import gt as torch
    from gt.nn import SGD  # GT doesn't have AdamW yet
    print("Using GT backend")

from qwen3_gt import create_model


class DataLoader:
    """Simple dataloader for batched training."""

    def __init__(self, input_ids, labels, batch_size=4, shuffle=True):
        self.input_ids = input_ids
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(input_ids)
        self.num_batches = (self.num_samples + batch_size - 1) // batch_size

    def __iter__(self):
        # Generate indices
        indices = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(indices)

        # Yield batches
        for i in range(0, self.num_samples, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch_input_ids = self.input_ids[batch_indices]
            batch_labels = self.labels[batch_indices]

            yield {
                'input_ids': batch_input_ids,
                'labels': batch_labels,
            }

    def __len__(self):
        return self.num_batches


def cross_entropy_loss(logits, labels):
    """
    Simplified loss for now.

    In production, would use proper cross-entropy with log_softmax.
    For now, MSE as placeholder to test the pipeline.
    """
    # Simplified: mean squared error on logits
    batch_size = logits.shape[0]
    seq_len = logits.shape[1]
    vocab_size = logits.shape[2]
    loss = (logits * logits).sum() / (batch_size * seq_len * vocab_size)
    return loss


def train_epoch(model, dataloader, optimizer, epoch):
    """Train for one epoch."""
    print(f"\n{'='*70}")
    print(f"Epoch {epoch}")
    print(f"{'='*70}")

    total_loss = 0.0
    num_batches = 0
    epoch_start = time.time()

    for batch_idx, batch in enumerate(dataloader):
        batch_start = time.time()

        # Get batch data
        input_ids = torch.from_numpy(batch['input_ids'])
        labels = torch.from_numpy(batch['labels'])

        # Forward pass
        logits = model(input_ids)

        # Compute loss
        loss = cross_entropy_loss(logits, labels)

        # Backward pass
        loss.backward()

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()

        # Get loss value
        loss_val = loss.item()
        total_loss += loss_val
        num_batches += 1

        batch_time = time.time() - batch_start

        # Print progress every 5 batches
        if batch_idx % 5 == 0:
            print(f"  Batch {batch_idx:3d}/{len(dataloader):3d} | "
                  f"Loss: {loss_val:.6f} | "
                  f"Time: {batch_time:.3f}s")

    epoch_time = time.time() - epoch_start
    avg_loss = total_loss / num_batches

    print(f"\nEpoch {epoch} Summary:")
    print(f"  Average Loss:    {avg_loss:.6f}")
    print(f"  Total Time:      {epoch_time:.2f}s")
    print(f"  Avg Batch Time:  {epoch_time/num_batches:.3f}s")
    print(f"  Throughput:      {num_batches/epoch_time:.1f} batches/sec")

    return avg_loss


def main():
    """Main training loop."""
    print("="*70)
    backend = "PyTorch" if USE_PYTORCH else "GT"
    print(f"Optimized Qwen3 Training ({backend})")
    print("="*70)

    # Configuration
    model_size = args.model_size
    batch_size = args.batch_size
    num_samples = args.num_samples
    learning_rate = args.lr
    num_epochs = args.epochs
    data_dir = "examples/qwen3/data"

    print(f"\nConfiguration:")
    print(f"  Model:       {model_size}")
    print(f"  Batch size:  {batch_size}")
    print(f"  Samples:     {num_samples}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Epochs:      {num_epochs}")

    # Check if data exists
    input_ids_path = f"{data_dir}/input_ids.npy"
    if not os.path.exists(input_ids_path):
        print(f"\nData not found at {data_dir}")
        print("Preparing dataset...")
        from prepare_data import prepare_synthetic_data
        prepare_synthetic_data(model_size, num_samples=num_samples)
        print("Data preparation complete!")

    # Load data
    print(f"\nLoading data from {data_dir}...")
    input_ids = np.load(f"{data_dir}/input_ids.npy")
    labels = np.load(f"{data_dir}/labels.npy")

    # Limit to num_samples if specified
    if num_samples < len(input_ids):
        input_ids = input_ids[:num_samples]
        labels = labels[:num_samples]

    print(f"  Loaded:      {len(input_ids)} samples")
    print(f"  Shape:       {input_ids.shape}")
    print(f"  Seq length:  {input_ids.shape[1]}")

    # Create dataloader
    dataloader = DataLoader(input_ids, labels, batch_size=batch_size, shuffle=True)
    print(f"  Batches:     {len(dataloader)}")

    # Create model
    print(f"\nCreating model...")
    model = create_model(model_size)

    # Create optimizer
    print(f"\nCreating optimizer...")
    params = model.parameters()
    print(f"  Parameters:  {len(params)} tensors")

    if USE_PYTORCH:
        optimizer = AdamW(params, lr=learning_rate)
        print(f"  Optimizer:   AdamW(lr={learning_rate})")
    else:
        optimizer = SGD(params, lr=learning_rate)
        print(f"  Optimizer:   SGD(lr={learning_rate})")

    # Training loop
    print("\nStarting training...")
    training_start = time.time()

    for epoch in range(1, num_epochs + 1):
        avg_loss = train_epoch(model, dataloader, optimizer, epoch)

    training_time = time.time() - training_start

    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"Total time:          {training_time:.2f}s")
    print(f"Avg time per epoch:  {training_time/num_epochs:.2f}s")
    print(f"Samples/sec:         {num_samples * num_epochs / training_time:.1f}")

    if args.profile:
        print(f"\nProfile log: /tmp/gt_profile_{args.model_size}.log")
        print("Analyze with:")
        print(f"  grep '|' /tmp/gt_profile_{args.model_size}.log | tail -100")


if __name__ == "__main__":
    main()
