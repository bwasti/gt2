#!/bin/bash

# Train Qwen3-1.7B model using GT framework
# This script runs all steps: weight loading, data prep, and training

set -e  # Exit on error

echo "========================================"
echo "Qwen3-1.7B Training Pipeline"
echo "========================================"

# Check dependencies
echo ""
echo "Checking dependencies..."
python -c "import transformers" 2>/dev/null || {
    echo "ERROR: transformers not installed"
    echo "Install with: pip install transformers"
    exit 1
}
python -c "import datasets" 2>/dev/null || {
    echo "ERROR: datasets not installed"
    echo "Install with: pip install datasets"
    exit 1
}

# Step 1: Load weights from HuggingFace
echo ""
echo "Step 1: Loading weights from HuggingFace..."
echo "========================================"
python examples/qwen3/load_weights.py

# Step 2: Prepare training data
echo ""
echo "Step 2: Preparing training data..."
echo "========================================"
python examples/qwen3/prepare_data.py

# Step 3: Train the model
echo ""
echo "Step 3: Training model..."
echo "========================================"
python examples/qwen3/train.py

echo ""
echo "========================================"
echo "Training pipeline complete!"
echo "========================================"
