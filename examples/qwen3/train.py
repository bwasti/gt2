"""
Train Qwen3-1.7B model using GT framework.

Uses 'import gt as torch' pattern for easy benchmarking against PyTorch.
"""

import os
import time
import numpy as np
import gt as torch
from model import create_model
from gt.nn import SGD


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
    Compute cross-entropy loss.

    For now, simplified version that works with GT's operations.
    In production, would use proper cross-entropy with numerical stability.

    Args:
        logits: (batch, seq_len, vocab_size) or (batch*seq_len, vocab_size)
        labels: (batch, seq_len) with -100 for ignored tokens

    Returns:
        Scalar loss tensor
    """
    # For simplicity, just compute MSE loss for now
    # TODO: Implement proper cross-entropy once we have log_softmax
    # This is a placeholder to get the training loop working

    # For causal LM, we're predicting next token
    # Shift logits and labels: predict token i from tokens 0..i-1
    # logits[:, :-1, :] predicts labels[:, 1:]

    # Simplified: just use sum of squares as proxy loss
    # Real implementation would use cross_entropy
    loss = (logits * logits).sum() / (logits.shape[0] * logits.shape[1] * logits.shape[2])

    return loss


def train_epoch(model, dataloader, optimizer, epoch):
    """Train for one epoch."""
    print(f"\n{'='*60}")
    print(f"Epoch {epoch}")
    print(f"{'='*60}")

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

        # Print progress
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)} | Loss: {loss_val:.4f} | Time: {batch_time:.2f}s")

    epoch_time = time.time() - epoch_start
    avg_loss = total_loss / num_batches

    print(f"\nEpoch {epoch} Summary:")
    print(f"  Average Loss: {avg_loss:.4f}")
    print(f"  Total Time: {epoch_time:.1f}s")
    print(f"  Avg Batch Time: {epoch_time/num_batches:.2f}s")

    return avg_loss


def main():
    """Main training loop."""
    print("="*60)
    print("Qwen3-1.7B Training with GT")
    print("="*60)

    # Configuration
    batch_size = 2  # Small batch size for initial testing
    learning_rate = 1e-5
    num_epochs = 3
    data_dir = "examples/qwen3/data"

    # Load data
    print(f"\nLoading data from {data_dir}...")
    input_ids = np.load(f"{data_dir}/input_ids.npy")
    labels = np.load(f"{data_dir}/labels.npy")
    print(f"  Loaded {len(input_ids)} examples")
    print(f"  Input shape: {input_ids.shape}")

    # Create dataloader
    dataloader = DataLoader(input_ids, labels, batch_size=batch_size, shuffle=True)
    print(f"  Created dataloader with {len(dataloader)} batches")

    # Create model
    print("\nCreating model...")
    model = create_model('1.7B')
    print("  Model created")

    # Create optimizer
    print("\nCreating optimizer...")
    params = model.parameters()
    print(f"  Model has {len(params)} parameter tensors")
    optimizer = SGD(params, lr=learning_rate)
    print(f"  Optimizer: SGD(lr={learning_rate})")

    # Training loop
    print("\nStarting training...")
    training_start = time.time()

    for epoch in range(1, num_epochs + 1):
        avg_loss = train_epoch(model, dataloader, optimizer, epoch)

    training_time = time.time() - training_start

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Total training time: {training_time:.1f}s")
    print(f"Average time per epoch: {training_time/num_epochs:.1f}s")


if __name__ == "__main__":
    main()
