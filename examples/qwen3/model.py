"""
Qwen3 model implementation using GT's PyTorch-like API.

Architecture based on Qwen3-1.7B.
"""

import os
import numpy as np

# Switch between GT and PyTorch for benchmarking
USE_PYTORCH = os.environ.get('USE_PYTORCH', '0') == '1'

if USE_PYTORCH:
    import torch
else:
    import gt as torch


class RMSNorm:
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim, eps=1e-6):
        self.eps = eps
        self.weight = torch.from_numpy(np.ones(dim, dtype='float32')).requires_grad_(True)

    def __call__(self, x):
        # RMS normalization: x / sqrt(mean(x^2) + eps) * weight
        # x shape: (batch, seq_len, dim) or (seq_len, dim)
        # Compute RMS over last dimension

        # Simplified version: compute global RMS
        # TODO: Implement proper per-position RMS norm (requires mean over last dim only)
        x_squared = x * x

        # Count total elements
        if len(x.shape) == 3:
            total_elements = x.shape[0] * x.shape[1] * x.shape[2]
        elif len(x.shape) == 2:
            total_elements = x.shape[0] * x.shape[1]
        else:
            total_elements = x.shape[0]

        mean_x_squared = x_squared.sum() / total_elements
        rms = (mean_x_squared + self.eps).sqrt()
        normed = x / rms
        return normed * self.weight


class RotaryEmbedding:
    """Rotary Position Embeddings (RoPE)."""

    def __init__(self, dim, max_seq_len=2048, base=10000):
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute frequencies
        inv_freq = 1.0 / (base ** (np.arange(0, dim, 2).astype('float32') / dim))
        self.inv_freq = torch.from_numpy(inv_freq)

    def __call__(self, x, seq_len):
        # x shape: (batch, n_heads, seq_len, head_dim)
        # For simplicity, we'll return x as-is for now
        # Full RoPE implementation is complex and requires cos/sin operations
        return x


class Attention:
    """Multi-head attention."""

    def __init__(self, config):
        self.hidden_size = config['hidden_size']
        self.num_heads = config['num_attention_heads']
        self.head_dim = self.hidden_size // self.num_heads

        # Q, K, V projections
        self.q_proj = torch.from_numpy(
            np.random.randn(self.hidden_size, self.hidden_size).astype('float32') * 0.02
        ).requires_grad_(True)
        self.k_proj = torch.from_numpy(
            np.random.randn(self.hidden_size, self.hidden_size).astype('float32') * 0.02
        ).requires_grad_(True)
        self.v_proj = torch.from_numpy(
            np.random.randn(self.hidden_size, self.hidden_size).astype('float32') * 0.02
        ).requires_grad_(True)
        self.o_proj = torch.from_numpy(
            np.random.randn(self.hidden_size, self.hidden_size).astype('float32') * 0.02
        ).requires_grad_(True)

        self.rotary_emb = RotaryEmbedding(self.head_dim)

    def __call__(self, hidden_states):
        # hidden_states shape: (batch, seq_len, hidden_size)

        # Project to Q, K, V
        q = hidden_states @ self.q_proj
        k = hidden_states @ self.k_proj
        v = hidden_states @ self.v_proj

        # For now, simplified attention: just apply projections and skip attention mechanism
        # This preserves shape: (batch, seq_len, hidden_size)
        # TODO: Implement proper multi-head attention with reshape, attention computation, etc.

        # Just pass through V projection (simplified placeholder)
        attn_output = v

        # Output projection
        output = attn_output @ self.o_proj

        return output


class MLP:
    """Feed-forward network."""

    def __init__(self, config):
        self.hidden_size = config['hidden_size']
        self.intermediate_size = config['intermediate_size']

        self.gate_proj = torch.from_numpy(
            np.random.randn(self.hidden_size, self.intermediate_size).astype('float32') * 0.02
        ).requires_grad_(True)
        self.up_proj = torch.from_numpy(
            np.random.randn(self.hidden_size, self.intermediate_size).astype('float32') * 0.02
        ).requires_grad_(True)
        self.down_proj = torch.from_numpy(
            np.random.randn(self.intermediate_size, self.hidden_size).astype('float32') * 0.02
        ).requires_grad_(True)

    def __call__(self, x):
        # SwiGLU activation: gate_proj(x) * swish(up_proj(x))
        gate = x @ self.gate_proj
        up = x @ self.up_proj

        # SwiGLU (simplified - just use gate * up for now)
        hidden = gate * up

        # Down projection
        output = hidden @ self.down_proj

        return output


class TransformerBlock:
    """Single transformer block."""

    def __init__(self, config):
        self.input_layernorm = RMSNorm(config['hidden_size'])
        self.self_attn = Attention(config)
        self.post_attention_layernorm = RMSNorm(config['hidden_size'])
        self.mlp = MLP(config)

    def __call__(self, hidden_states):
        # Self attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states

        # MLP with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Qwen3Model:
    """Qwen3 transformer model."""

    def __init__(self, config):
        self.config = config
        self.vocab_size = config['vocab_size']
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_hidden_layers']

        # Embeddings
        self.embed_tokens = torch.from_numpy(
            np.random.randn(self.vocab_size, self.hidden_size).astype('float32') * 0.02
        ).requires_grad_(True)

        # Transformer layers
        self.layers = [TransformerBlock(config) for _ in range(self.num_layers)]

        # Final layer norm
        self.norm = RMSNorm(self.hidden_size)

    def __call__(self, input_ids):
        # Embed tokens
        # input_ids shape: (batch_size, seq_len)
        # embed_tokens shape: (vocab_size, hidden_size)

        # For now, simplified: just use random slice of embeddings as proxy
        # TODO: Implement proper embedding lookup (requires gather/index operation)
        # This is a placeholder to get the training loop working

        batch_size = input_ids.shape[0] if len(input_ids.shape) > 1 else 1
        seq_len = input_ids.shape[1] if len(input_ids.shape) > 1 else input_ids.shape[0]

        # Take first seq_len rows of embeddings as proxy
        # In production, would use proper gather: embed_tokens[input_ids]
        hidden_states = self.embed_tokens  # (vocab_size, hidden_size)

        # Slice to get (seq_len, hidden_size) - simplified proxy
        # This is not correct but allows us to test the pipeline
        # TODO: Implement gather/index_select operation

        # For now, just use a small slice
        hidden_states = torch.from_numpy(
            np.random.randn(batch_size, seq_len, self.hidden_size).astype('float32') * 0.02
        )

        # Pass through transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states)

        # Final layer norm
        hidden_states = self.norm(hidden_states)

        return hidden_states


class Qwen3ForCausalLM:
    """Qwen3 model with language modeling head."""

    def __init__(self, config):
        self.model = Qwen3Model(config)
        self.vocab_size = config['vocab_size']
        self.hidden_size = config['hidden_size']

        # LM head
        self.lm_head = torch.from_numpy(
            np.random.randn(self.hidden_size, self.vocab_size).astype('float32') * 0.02
        ).requires_grad_(True)

    def __call__(self, input_ids):
        hidden_states = self.model(input_ids)
        logits = hidden_states @ self.lm_head
        return logits

    def parameters(self):
        """Return all trainable parameters."""
        params = []

        # Embeddings
        params.append(self.model.embed_tokens)

        # Layer parameters
        for layer in self.model.layers:
            params.append(layer.self_attn.q_proj)
            params.append(layer.self_attn.k_proj)
            params.append(layer.self_attn.v_proj)
            params.append(layer.self_attn.o_proj)
            params.append(layer.mlp.gate_proj)
            params.append(layer.mlp.up_proj)
            params.append(layer.mlp.down_proj)
            params.append(layer.input_layernorm.weight)
            params.append(layer.post_attention_layernorm.weight)

        # Final norm and LM head
        params.append(self.model.norm.weight)
        params.append(self.lm_head)

        return params


def get_qwen3_config(model_size='1.7B'):
    """Get Qwen3 model configuration."""

    if model_size == 'tiny':
        # Tiny config for testing
        return {
            'vocab_size': 1000,
            'hidden_size': 64,
            'intermediate_size': 128,
            'num_hidden_layers': 2,
            'num_attention_heads': 4,
            'max_position_embeddings': 512,
            'rms_norm_eps': 1e-6,
        }
    elif model_size == '1.7B':
        return {
            'vocab_size': 152064,
            'hidden_size': 2048,
            'intermediate_size': 11008,
            'num_hidden_layers': 28,
            'num_attention_heads': 16,
            'max_position_embeddings': 32768,
            'rms_norm_eps': 1e-6,
        }
    else:
        raise ValueError(f"Unknown model size: {model_size}")


def create_model(model_size='1.7B'):
    """Create a Qwen3 model."""
    config = get_qwen3_config(model_size)
    return Qwen3ForCausalLM(config)
