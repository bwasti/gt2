"""
Prepare training dataset for Qwen3 fine-tuning.

Downloads and prepares a small instruction-following dataset.
"""

import json
import os
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np


def download_dataset(dataset_name="tatsu-lab/alpaca", split="train", max_samples=1000):
    """
    Download instruction-following dataset.

    Using Alpaca dataset as it's small and good for fine-tuning demos.
    """
    print(f"Downloading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split=split)

    # Take subset for faster training
    if len(dataset) > max_samples:
        dataset = dataset.select(range(max_samples))

    print(f"Dataset size: {len(dataset)} examples")

    return dataset


def format_prompt(example):
    """
    Format example as instruction-following prompt.

    Alpaca format:
    - instruction: The task description
    - input: Optional context
    - output: Expected response
    """
    instruction = example['instruction']
    input_text = example.get('input', '')
    output = example['output']

    if input_text:
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"

    return prompt


def tokenize_dataset(dataset, tokenizer, max_length=512):
    """
    Tokenize dataset for training.

    Returns:
        input_ids: List of tokenized sequences
        labels: Same as input_ids (for causal LM)
    """
    print(f"Tokenizing dataset (max_length={max_length})...")

    tokenized_examples = []

    for i, example in enumerate(dataset):
        if i % 100 == 0:
            print(f"  Tokenized {i}/{len(dataset)} examples")

        # Format prompt
        text = format_prompt(example)

        # Tokenize
        tokens = tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_tensors='np'
        )

        input_ids = tokens['input_ids'][0]
        attention_mask = tokens['attention_mask'][0]

        # For causal LM, labels are same as input_ids
        # We'll mask padding tokens with -100 (ignored by loss)
        labels = input_ids.copy()
        labels[attention_mask == 0] = -100

        tokenized_examples.append({
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
        })

    print(f"Tokenization complete!")

    return tokenized_examples


def save_dataset(tokenized_examples, output_dir="examples/qwen3/data"):
    """Save tokenized dataset to disk."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"Saving dataset to {output_dir}...")

    # Save as numpy arrays
    input_ids = np.array([ex['input_ids'] for ex in tokenized_examples], dtype='int32')
    labels = np.array([ex['labels'] for ex in tokenized_examples], dtype='int32')
    attention_mask = np.array([ex['attention_mask'] for ex in tokenized_examples], dtype='int32')

    np.save(f"{output_dir}/input_ids.npy", input_ids)
    np.save(f"{output_dir}/labels.npy", labels)
    np.save(f"{output_dir}/attention_mask.npy", attention_mask)

    print(f"Saved {len(tokenized_examples)} examples")
    print(f"  input_ids shape: {input_ids.shape}")
    print(f"  labels shape: {labels.shape}")


def create_synthetic_data(num_samples=100, seq_length=128, vocab_size=1000):
    """
    Create synthetic data for testing (used with tiny model).

    Returns random token IDs - just for testing the training pipeline.
    """
    print(f"Creating synthetic data for testing...")
    print(f"  {num_samples} samples, seq_length={seq_length}, vocab_size={vocab_size}")

    # Random token IDs
    input_ids = np.random.randint(0, vocab_size, size=(num_samples, seq_length), dtype='int32')
    labels = np.random.randint(0, vocab_size, size=(num_samples, seq_length), dtype='int32')

    # Create tokenized examples in same format
    tokenized_examples = []
    for i in range(num_samples):
        tokenized_examples.append({
            'input_ids': input_ids[i],
            'labels': labels[i],
            'attention_mask': np.ones(seq_length, dtype='int32'),
        })

    return tokenized_examples


def prepare_tinystories_dataset(num_samples=1000, seq_length=256):
    """
    Download and prepare TinyStories dataset.

    TinyStories is a dataset of simple stories written in a simplified vocabulary,
    designed for training small language models.

    Args:
        num_samples: Number of samples to use
        seq_length: Maximum sequence length
    """
    print(f"Downloading TinyStories dataset...")
    print(f"  num_samples: {num_samples}")
    print(f"  seq_length: {seq_length}")

    try:
        from datasets import load_dataset
        from transformers import GPT2Tokenizer

        # Download TinyStories dataset
        dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

        # Load GPT-2 tokenizer (50257 vocab size)
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        # Process samples
        input_ids_list = []
        labels_list = []

        print("Tokenizing stories...")
        for i, example in enumerate(dataset):
            if i >= num_samples:
                break

            text = example['text']

            # Tokenize
            encoded = tokenizer(
                text,
                max_length=seq_length,
                truncation=True,
                padding='max_length',
                return_tensors='np'
            )

            input_ids = encoded['input_ids'][0]
            # For causal LM, labels are the same as input_ids (shifted during loss computation)
            labels = input_ids.copy()

            input_ids_list.append(input_ids)
            labels_list.append(labels)

            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{num_samples} stories")

        # Convert to numpy arrays
        input_ids_array = np.stack(input_ids_list)
        labels_array = np.stack(labels_list)

        print(f"Dataset prepared: {input_ids_array.shape}")

        # Save
        tokenized = {
            'input_ids': input_ids_array,
            'labels': labels_array
        }
        save_dataset(tokenized)

        return tokenized

    except ImportError as e:
        print(f"Error: Missing dependencies for real dataset: {e}")
        print("Install with: pip install datasets transformers")
        print("Falling back to synthetic data...")
        return create_synthetic_data(num_samples=num_samples, seq_length=seq_length, vocab_size=50257)


def prepare_synthetic_data(model_size='tiny', num_samples=100):
    """
    Prepare synthetic data for training (reusable function).

    Args:
        model_size: Model size ('tiny', 'nano', 'small', etc.)
        num_samples: Number of training samples to generate
    """
    if model_size == 'tiny':
        # Create synthetic data for testing
        tokenized = create_synthetic_data(num_samples=num_samples, seq_length=128, vocab_size=1000)
        save_dataset(tokenized)
    elif model_size in ('nano', 'small', 'medium'):
        # Use TinyStories for realistic models
        tokenized = prepare_tinystories_dataset(num_samples=num_samples, seq_length=256)
    else:
        raise ValueError(f"prepare_synthetic_data doesn't support model_size='{model_size}'")


def main():
    """Download and prepare dataset."""
    # Check if using tiny model (no tokenizer available)
    tokenizer_path = "examples/qwen3/tokenizer"
    model_size = os.environ.get('MODEL_SIZE', 'tiny')

    if model_size == 'tiny':
        # Create synthetic data for testing
        print("Using TINY model - creating synthetic data for testing")
        prepare_synthetic_data(model_size)
        print("\nSynthetic dataset ready!")
        print("Training data ready at: examples/qwen3/data/")
        return

    # Full model path - need real tokenizer
    if not os.path.exists(tokenizer_path):
        print(f"Tokenizer not found at {tokenizer_path}")
        print("Please run load_weights.py first to download the tokenizer")
        return

    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # Download dataset
    dataset = download_dataset(max_samples=1000)

    # Tokenize
    tokenized = tokenize_dataset(dataset, tokenizer, max_length=512)

    # Save
    save_dataset(tokenized)

    print("\nDataset preparation complete!")
    print("Training data ready at: examples/qwen3/data/")


if __name__ == "__main__":
    main()
