"""
Training script with 1F1B (One Forward One Backward) pipeline schedule.

The 1F1B schedule reduces pipeline bubbles by interleaving forward and backward passes:
- Traditional: All forward passes, then all backward passes (big bubble)
- 1F1B: Interleave forward/backward to keep all pipeline stages busy

With 4 pipeline stages and 8 microbatches:
  Stage 0: F0 F1 F2 F3 B0 B1 B2 B3 B4 B5 B6 B7
  Stage 1:    F0 F1 F2 B0 B1 B2 B3 B4 B5 B6 B7
  Stage 2:       F0 F1 B0 B1 B2 B3 B4 B5 B6 B7
  Stage 3:          F0 B0 B1 B2 B3 B4 B5 B6 B7

This keeps GPUs busy and minimizes idle time (bubbles).

Usage:
    python examples/tuning_demo/train.py --steps 5
"""

import os
import time

# Set config path before importing gt
os.environ['GT_CONFIG'] = 'examples/tuning_demo/sharding_config.yaml'

import gt

# Configure for 8 GPUs (auto-start)
gt.gpu_workers(8)
from gt.client import nn
import numpy as np
from model import create_model, count_parameters


class PipelineScheduler:
    """
    1F1B pipeline scheduler with microbatching.

    Key concepts:
    - Microbatch: Small batch processed by one pipeline stage at a time
    - Warmup phase: Fill the pipeline with forward passes
    - Steady state: Interleave 1 forward + 1 backward per microbatch
    - Cooldown phase: Drain the pipeline with remaining backward passes
    """

    def __init__(self, num_stages=4, num_microbatches=8):
        self.num_stages = num_stages
        self.num_microbatches = num_microbatches

        # For proper 1F1B, need at least num_stages microbatches
        # Otherwise, just do all forwards then all backwards
        if num_microbatches < num_stages:
            self.simple_mode = True
            self.warmup_microbatches = num_microbatches
            self.steady_microbatches = 0
            self.cooldown_microbatches = 0
        else:
            self.simple_mode = False
            # Warmup: num_stages-1 forward passes to fill pipeline
            self.warmup_microbatches = num_stages - 1
            # Steady state: alternate forward/backward
            self.steady_microbatches = num_microbatches - self.warmup_microbatches
            # Cooldown: remaining backward passes
            self.cooldown_microbatches = num_stages - 1

    def get_schedule(self):
        """
        Generate 1F1B schedule.

        Returns:
            List of (operation, microbatch_id) tuples
            operation: 'F' (forward) or 'B' (backward)
        """
        schedule = []

        if self.simple_mode:
            # Not enough microbatches for 1F1B - just do all F then all B
            for mb in range(self.num_microbatches):
                schedule.append(('F', mb))
            for mb in range(self.num_microbatches):
                schedule.append(('B', mb))
            return schedule

        # Warmup: Fill pipeline with forwards
        for mb in range(self.warmup_microbatches):
            schedule.append(('F', mb))

        # Steady state: 1 forward, 1 backward
        for mb in range(self.warmup_microbatches, self.num_microbatches):
            schedule.append(('F', mb))
            backward_mb = mb - self.warmup_microbatches
            schedule.append(('B', backward_mb))

        # Cooldown: Drain remaining backwards
        for mb in range(self.steady_microbatches, self.num_microbatches):
            schedule.append(('B', mb))

        return schedule


def create_microbatch(batch_data, microbatch_idx, microbatch_size):
    """Extract microbatch from full batch."""
    start = microbatch_idx * microbatch_size
    end = start + microbatch_size
    return batch_data[start:end]


def train_step_1f1b(model, batch_input, batch_target, learning_rate=0.0001,
                    num_microbatches=8):
    """
    Perform one training step with 1F1B pipeline schedule.

    Args:
        model: MiniGPT model
        batch_input: Input token IDs (batch_size, seq_len)
        batch_target: Target token IDs (batch_size, seq_len)
        learning_rate: Learning rate for optimizer
        num_microbatches: Number of microbatches to split batch into

    Returns:
        Average loss across microbatches
    """
    batch_size = batch_input.shape[0]
    microbatch_size = batch_size // num_microbatches

    scheduler = PipelineScheduler(num_stages=4, num_microbatches=num_microbatches)
    schedule = scheduler.get_schedule()

    # Storage for intermediate activations and losses
    activations = {}
    losses = []

    print(f"\n1F1B Schedule ({num_microbatches} microbatches):")
    print(f"  Warmup: {scheduler.warmup_microbatches} forward passes")
    print(f"  Steady: {scheduler.steady_microbatches} F+B pairs")
    print(f"  Cooldown: {scheduler.cooldown_microbatches} backward passes")
    print(f"  Schedule: {' '.join([f'{op}{mb}' for op, mb in schedule[:10]])}...")

    # Execute schedule
    for step_idx, (operation, mb_idx) in enumerate(schedule):
        if operation == 'F':
            # Forward pass for microbatch
            mb_input = create_microbatch(batch_input, mb_idx, microbatch_size)
            mb_target = create_microbatch(batch_target, mb_idx, microbatch_size)

            # Forward through model
            logits = model(mb_input)

            # Compute loss (simplified - no actual cross-entropy implementation yet)
            # In real version: loss = nn.cross_entropy_loss(logits, mb_target)
            loss = logits.sum() / (batch_size * logits.shape[1])

            # Store for backward pass
            activations[mb_idx] = loss

        elif operation == 'B':
            # Backward pass for microbatch
            loss = activations[mb_idx]
            loss.backward()

            # Accumulate loss for reporting
            losses.append(loss.data.numpy())

            # Clean up activation to save memory
            del activations[mb_idx]

    # Update parameters (simplified SGD)
    for param in model.parameters():
        if param.grad is not None:
            # param = param - lr * param.grad (would need in-place ops)
            # For now, just signal that we would update
            pass

    avg_loss = np.mean(losses)
    return avg_loss


def train(num_steps=10, batch_size=32, seq_len=128, num_microbatches=8):
    """
    Train mini-GPT with 1F1B pipeline schedule.

    Args:
        num_steps: Number of training steps
        batch_size: Global batch size (split into microbatches)
        seq_len: Sequence length
        num_microbatches: Number of microbatches for pipeline parallelism
    """
    print("=" * 80)
    print("8-GPU Distributed Training with 1F1B Pipeline Schedule")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Pipeline stages: 4 (2 layers each)")
    print(f"  Tensor parallel: 2 (within each stage)")
    print(f"  Global batch size: {batch_size}")
    print(f"  Microbatches: {num_microbatches}")
    print(f"  Microbatch size: {batch_size // num_microbatches}")
    print(f"  Sequence length: {seq_len}")

    # GT will auto-start when first tensor is created
    print("\n✓ GT will auto-start with 8 workers...")

    # Create model (triggers auto-start)
    print("\nInitializing model...")
    model = create_model()
    num_params = count_parameters(model)
    print(f"✓ Model created: {num_params:,} parameters (~{num_params/1e6:.1f}M)")

    # Training loop
    print(f"\nTraining for {num_steps} steps...")
    print("-" * 80)

    for step in range(num_steps):
        start_time = time.time()

        # Create dummy batch (in real training, load from dataloader)
        batch_input = np.random.randint(0, 50257, size=(batch_size, seq_len), dtype='int32')
        batch_target = np.random.randint(0, 50257, size=(batch_size, seq_len), dtype='int32')

        # Convert to GT tensors
        batch_input_gt = gt.from_numpy(batch_input.astype('float32'))
        batch_target_gt = gt.from_numpy(batch_target.astype('float32'))

        # Training step with 1F1B schedule
        loss = train_step_1f1b(
            model,
            batch_input_gt,
            batch_target_gt,
            num_microbatches=num_microbatches
        )

        step_time = time.time() - start_time

        print(f"Step {step+1}/{num_steps} | Loss: {loss:.4f} | Time: {step_time:.2f}s")

    print("-" * 80)
    print("\n✓ Training complete!")

    # Print pipeline efficiency stats
    print("\nPipeline Efficiency:")
    scheduler = PipelineScheduler(num_stages=4, num_microbatches=num_microbatches)
    total_ops = len(scheduler.get_schedule())
    ideal_ops = num_microbatches * 2  # F + B for each microbatch
    bubble_ops = total_ops - ideal_ops
    efficiency = (ideal_ops / total_ops) * 100

    print(f"  Total operations: {total_ops}")
    print(f"  Ideal operations: {ideal_ops}")
    print(f"  Bubble overhead: {bubble_ops} operations")
    print(f"  Pipeline efficiency: {efficiency:.1f}%")

    print("\nNote: This is a demo showing the 1F1B schedule structure.")
    print("Full implementation would include:")
    print("  - Actual gradient accumulation across microbatches")
    print("  - Parameter updates with optimizer (Adam/SGD)")
    print("  - Learning rate scheduling")
    print("  - Gradient clipping")
    print("  - Checkpointing")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train with 1F1B pipeline schedule")
    parser.add_argument('--steps', type=int, default=10, help='Number of training steps')
    parser.add_argument('--batch-size', type=int, default=32, help='Global batch size')
    parser.add_argument('--seq-len', type=int, default=128, help='Sequence length')
    parser.add_argument('--microbatches', type=int, default=8, help='Number of microbatches')

    args = parser.parse_args()

    train(
        num_steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_microbatches=args.microbatches
    )
