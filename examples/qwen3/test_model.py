"""
Simple test to verify Qwen3 model works with GT.
"""

import numpy as np
import gt as torch
from model import create_model

print("Creating Qwen3 tiny model (for testing)...")
model = create_model('tiny')
print("Model created successfully!")

print("\nCreating test input...")
batch_size = 2
seq_len = 10
input_ids = torch.from_numpy(np.random.randint(0, 1000, size=(batch_size, seq_len), dtype='int32'))
print(f"Input shape: {input_ids.shape}")

print("\nRunning forward pass...")
output = model(input_ids)
print(f"Output shape: {output.shape}")

print("\nGetting output data...")
output_data = output.data.numpy()
print(f"Output data shape: {output_data.shape}")
print(f"Output data sample (first 5 values): {output_data.flatten()[:5]}")

print("\n✓ Model test passed!")
