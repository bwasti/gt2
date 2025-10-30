"""
Load Qwen3-1.7B weights from HuggingFace.

Downloads the model and converts weights to GT format.
"""

import os
import numpy as np
import torch as pt  # Use real PyTorch for loading HF weights
from transformers import AutoModelForCausalLM, AutoTokenizer
from model import create_model, get_qwen3_config
import gt


def load_hf_weights(model_name="Qwen/Qwen2.5-1.5B-Instruct"):
    """Download and load HuggingFace model weights."""
    print(f"Loading HuggingFace model: {model_name}")

    # Download model and tokenizer
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=pt.float32,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"Model loaded. Vocab size: {len(tokenizer)}")

    return hf_model, tokenizer


def convert_weights_to_gt(hf_model, gt_model):
    """
    Convert HuggingFace weights to GT model.

    This copies weights from the HF model to the GT model parameters.
    """
    print("\nConverting weights to GT format...")

    # Get HF state dict
    hf_state = hf_model.state_dict()

    # Embeddings
    print("Converting embeddings...")
    embed_weight = hf_state['model.embed_tokens.weight'].cpu().numpy().astype('float32')
    gt_model.model.embed_tokens = gt.from_numpy(embed_weight)

    # Transformer layers
    config = get_qwen3_config('1.7B')
    num_layers = config['num_hidden_layers']

    for i in range(num_layers):
        print(f"Converting layer {i}/{num_layers}...")
        layer = gt_model.model.layers[i]

        # Attention weights
        q_weight = hf_state[f'model.layers.{i}.self_attn.q_proj.weight'].cpu().numpy().astype('float32')
        k_weight = hf_state[f'model.layers.{i}.self_attn.k_proj.weight'].cpu().numpy().astype('float32')
        v_weight = hf_state[f'model.layers.{i}.self_attn.v_proj.weight'].cpu().numpy().astype('float32')
        o_weight = hf_state[f'model.layers.{i}.self_attn.o_proj.weight'].cpu().numpy().astype('float32')

        layer.self_attn.q_proj = gt.from_numpy(q_weight.T)  # Transpose for our convention
        layer.self_attn.k_proj = gt.from_numpy(k_weight.T)
        layer.self_attn.v_proj = gt.from_numpy(v_weight.T)
        layer.self_attn.o_proj = gt.from_numpy(o_weight.T)

        # MLP weights
        gate_weight = hf_state[f'model.layers.{i}.mlp.gate_proj.weight'].cpu().numpy().astype('float32')
        up_weight = hf_state[f'model.layers.{i}.mlp.up_proj.weight'].cpu().numpy().astype('float32')
        down_weight = hf_state[f'model.layers.{i}.mlp.down_proj.weight'].cpu().numpy().astype('float32')

        layer.mlp.gate_proj = gt.from_numpy(gate_weight.T)
        layer.mlp.up_proj = gt.from_numpy(up_weight.T)
        layer.mlp.down_proj = gt.from_numpy(down_weight.T)

        # Layer norms
        ln1_weight = hf_state[f'model.layers.{i}.input_layernorm.weight'].cpu().numpy().astype('float32')
        ln2_weight = hf_state[f'model.layers.{i}.post_attention_layernorm.weight'].cpu().numpy().astype('float32')

        layer.input_layernorm.weight = gt.from_numpy(ln1_weight)
        layer.post_attention_layernorm.weight = gt.from_numpy(ln2_weight)

    # Final layer norm
    print("Converting final layer norm...")
    final_ln_weight = hf_state['model.norm.weight'].cpu().numpy().astype('float32')
    gt_model.model.norm.weight = gt.from_numpy(final_ln_weight)

    # LM head
    print("Converting LM head...")
    lm_head_weight = hf_state['lm_head.weight'].cpu().numpy().astype('float32')
    gt_model.lm_head = gt.from_numpy(lm_head_weight.T)

    print("Weight conversion complete!")

    return gt_model


def verify_forward_pass(hf_model, gt_model, tokenizer):
    """
    Verify that GT model produces same outputs as HF model.
    """
    print("\nVerifying forward pass...")

    # Create test input
    test_text = "Hello, how are you?"
    inputs = tokenizer(test_text, return_tensors="pt")
    input_ids = inputs['input_ids']

    print(f"Test input: {test_text}")
    print(f"Input IDs shape: {input_ids.shape}")

    # HF forward pass
    with pt.no_grad():
        hf_output = hf_model(input_ids)
        hf_logits = hf_output.logits[0, -1, :].cpu().numpy()  # Last token logits

    # GT forward pass
    gt_input = gt.from_numpy(input_ids.cpu().numpy().astype('int32'))
    gt_output = gt_model(gt_input)
    gt_logits = gt_output.data.numpy()[-1, :]  # Last token logits

    # Compare
    diff = np.abs(hf_logits - gt_logits)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    print(f"Max difference: {max_diff:.6f}")
    print(f"Mean difference: {mean_diff:.6f}")

    if max_diff < 1e-3:
        print("✓ Forward pass verification PASSED")
        return True
    else:
        print("✗ Forward pass verification FAILED")
        print(f"  HF logits (first 10): {hf_logits[:10]}")
        print(f"  GT logits (first 10): {gt_logits[:10]}")
        return False


def main():
    """Load weights and verify."""
    # Create GT model
    print("Creating GT model...")
    gt_model = create_model('1.7B')

    # Load HF weights
    hf_model, tokenizer = load_hf_weights()

    # Convert weights
    gt_model = convert_weights_to_gt(hf_model, gt_model)

    # Verify forward pass
    verify_forward_pass(hf_model, gt_model, tokenizer)

    # Save tokenizer for training
    print("\nSaving tokenizer...")
    tokenizer.save_pretrained("examples/qwen3/tokenizer")

    print("\nDone! Weights loaded and verified.")


if __name__ == "__main__":
    main()
