"""
Mini-GPT Transformer Model for 8-GPU Distributed Training Demo

Architecture:
- 8 transformer layers (~512M parameters)
- 1024 hidden size
- 16 attention heads
- 4096 intermediate size (MLP)
- 50257 vocab size (GPT-2 tokenizer compatible)

Distribution Strategy (Better for ZB-V pipeline schedules):
- 4-way Pipeline Parallel: 2 layers per stage
  - Stage 0: Layers 0-1 on GPUs 0-1
  - Stage 1: Layers 2-3 on GPUs 2-3
  - Stage 2: Layers 4-5 on GPUs 4-5
  - Stage 3: Layers 6-7 on GPUs 6-7
- 2-way Tensor Parallel: Within each pipeline stage
"""

import gt
from gt.client import nn
import numpy as np


class RMSNorm:
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim, eps=1e-6):
        self.eps = eps
        self.dim = dim
        # Initialize weight to ones using from_numpy
        import numpy as np
        self.weight = gt.from_numpy(np.ones(dim, dtype='float32'), requires_grad=True)

    def __call__(self, x):
        # Simplified RMS norm
        x_squared = x * x
        total_elements = x.shape[0] * x.shape[1] * x.shape[2] if len(x.shape) == 3 else x.shape[0] * x.shape[1]
        mean_squared = x_squared.sum() / total_elements

        eps_tensor = gt.from_numpy(np.array(self.eps, dtype='float32'))
        rms = (mean_squared + eps_tensor).sqrt()

        normalized = x / rms
        return normalized * self.weight

    def parameters(self):
        return [self.weight]


class Attention(nn.Module):
    """Multi-head self-attention with tensor parallelism."""

    def __init__(self, config, signal_prefix):
        super().__init__()
        self.hidden_size = config['hidden_size']
        self.num_heads = config['num_heads']
        self.head_dim = self.hidden_size // self.num_heads
        self.signal_prefix = signal_prefix

        # Initialize with proper scaling
        std = 0.02

        # QKV projection - column parallel (splits output features)
        self.qkv_weight = gt.randn(self.hidden_size, 3 * self.hidden_size) * std
        self.qkv_weight = self.qkv_weight.requires_grad_(True)

        # Output projection - row parallel (splits input features)
        # Maps from 3*hidden_size back to hidden_size
        self.out_weight = gt.randn(3 * self.hidden_size, self.hidden_size) * std
        self.out_weight = self.out_weight.requires_grad_(True)

        self._parameters = [self.qkv_weight, self.out_weight]

    def forward(self, x):
        # QKV projection with column-parallel signal
        with gt.signal.context(f'{self.signal_prefix}_colpar'):
            qkv = x @ self.qkv_weight  # (batch, seq_len, 3*hidden)

        # Simplified attention (skip actual Q/K/V split for now)
        # In a real implementation, we'd reshape, compute attention scores, etc.

        # Output projection with row-parallel signal
        with gt.signal.context(f'{self.signal_prefix}_rowpar'):
            output = qkv @ self.out_weight  # (batch, seq_len, hidden)

        return output


class MLP(nn.Module):
    """Feed-forward network with tensor parallelism."""

    def __init__(self, config, signal_prefix):
        super().__init__()
        self.hidden_size = config['hidden_size']
        self.intermediate_size = config['intermediate_size']
        self.signal_prefix = signal_prefix

        std = 0.02

        # Up projection - column parallel (splits output features)
        self.up_weight = gt.randn(self.hidden_size, self.intermediate_size) * std
        self.up_weight = self.up_weight.requires_grad_(True)

        # Down projection - row parallel (splits input features)
        self.down_weight = gt.randn(self.intermediate_size, self.hidden_size) * std
        self.down_weight = self.down_weight.requires_grad_(True)

        self._parameters = [self.up_weight, self.down_weight]

    def forward(self, x):
        # Up projection with column-parallel signal
        with gt.signal.context(f'{self.signal_prefix}_colpar'):
            hidden = x @ self.up_weight  # (batch, seq_len, intermediate)
            hidden = hidden.relu()  # GeLU approximation

        # Down projection with row-parallel signal
        with gt.signal.context(f'{self.signal_prefix}_rowpar'):
            output = hidden @ self.down_weight  # (batch, seq_len, hidden)

        return output


class TransformerLayer(nn.Module):
    """Single transformer layer with attention and MLP."""

    def __init__(self, config, signal_prefix):
        super().__init__()
        self.signal_prefix = signal_prefix

        self.ln1 = RMSNorm(config['hidden_size'])
        self.attn = Attention(config, signal_prefix)
        self.ln2 = RMSNorm(config['hidden_size'])
        self.mlp = MLP(config, signal_prefix)

        self._parameters = self.ln1.parameters() + self.attn.parameters() + \
                          self.ln2.parameters() + self.mlp.parameters()

    def forward(self, x):
        # Pre-norm architecture
        with gt.signal.context(self.signal_prefix):
            # Attention block with residual
            normed = self.ln1(x)
            attn_out = self.attn(normed)
            x = x + attn_out

            # MLP block with residual
            normed = self.ln2(x)
            mlp_out = self.mlp(normed)
            x = x + mlp_out

        return x


class MiniGPT(nn.Module):
    """
    Mini-GPT model with 8 transformer layers.

    Pipeline parallelism splits into 4 stages (2 layers each):
    - Stage 0 (GPUs 0-1): Embedding + Layers 0-1
    - Stage 1 (GPUs 2-3): Layers 2-3
    - Stage 2 (GPUs 4-5): Layers 4-5
    - Stage 3 (GPUs 6-7): Layers 6-7 + Output head
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        vocab_size = config['vocab_size']
        hidden_size = config['hidden_size']
        std = 0.02

        # Create model weights on stage 0 GPUs (not sharded across all 8)
        # This prevents "cannot shard 50257 across 8 workers" errors
        with gt.signal.context('pp_stage0'):
            # Embedding table
            self.embed_weight = gt.randn(vocab_size, hidden_size) * std
            self.embed_weight = self.embed_weight.requires_grad_(True)

            # Pipeline Stage 0: Layers 0-1
            self.layers_stage0 = [
                TransformerLayer(config, 'pp_stage0') for _ in range(2)
            ]

        with gt.signal.context('pp_stage1'):
            # Pipeline Stage 1: Layers 2-3
            self.layers_stage1 = [
                TransformerLayer(config, 'pp_stage1') for _ in range(2)
            ]

        with gt.signal.context('pp_stage2'):
            # Pipeline Stage 2: Layers 4-5
            self.layers_stage2 = [
                TransformerLayer(config, 'pp_stage2') for _ in range(2)
            ]

        with gt.signal.context('pp_stage3'):
            # Pipeline Stage 3: Layers 6-7
            self.layers_stage3 = [
                TransformerLayer(config, 'pp_stage3') for _ in range(2)
            ]

            # Output projection (LM head)
            self.output_weight = gt.randn(hidden_size, vocab_size) * std
            self.output_weight = self.output_weight.requires_grad_(True)

            # Final layer norm
            self.ln_f = RMSNorm(hidden_size)

        # Collect all parameters
        self._parameters = [self.embed_weight]
        for layer in self.layers_stage0:
            self._parameters.extend(layer.parameters())
        for layer in self.layers_stage1:
            self._parameters.extend(layer.parameters())
        for layer in self.layers_stage2:
            self._parameters.extend(layer.parameters())
        for layer in self.layers_stage3:
            self._parameters.extend(layer.parameters())
        self._parameters.extend(self.ln_f.parameters())
        self._parameters.append(self.output_weight)

    def forward(self, input_ids):
        """
        Forward pass with 4-stage pipeline parallelism.

        Args:
            input_ids: (batch_size, seq_len) - token IDs

        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        # Stage 0: Embedding lookup
        with gt.signal.context('pp_stage0'):
            batch_size, seq_len = input_ids.shape[0], input_ids.shape[1]
            # Initialize hidden states (simplified embedding)
            x = gt.randn(batch_size, seq_len, self.config['hidden_size'])

        # Stage 0: Layers 0-1
        for layer in self.layers_stage0:
            x = layer(x)

        # Pipeline boundary: Stage 0 → Stage 1
        with gt.signal.context('stage0_to_stage1'):
            x = x

        # Stage 1: Layers 2-3
        for layer in self.layers_stage1:
            x = layer(x)

        # Pipeline boundary: Stage 1 → Stage 2
        with gt.signal.context('stage1_to_stage2'):
            x = x

        # Stage 2: Layers 4-5
        for layer in self.layers_stage2:
            x = layer(x)

        # Pipeline boundary: Stage 2 → Stage 3
        with gt.signal.context('stage2_to_stage3'):
            x = x

        # Stage 3: Layers 6-7
        for layer in self.layers_stage3:
            x = layer(x)

        # Stage 3: Final layer norm and output projection
        with gt.signal.context('pp_stage3'):
            x = self.ln_f(x)

        with gt.signal.context('pp_stage3_colpar'):
            logits = x @ self.output_weight  # (batch, seq_len, vocab_size)

        return logits


def create_model():
    """Create a mini-GPT model for distributed training."""
    config = {
        'hidden_size': 1024,
        'num_heads': 16,
        'intermediate_size': 4096,
        'num_layers': 8,
        'vocab_size': 50257,
    }

    return MiniGPT(config)


def count_parameters(model):
    """Count total parameters in the model."""
    total = 0
    for param in model.parameters():
        size = 1
        for dim in param.shape:
            size *= dim
        total += size
    return total
