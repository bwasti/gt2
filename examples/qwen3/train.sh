#!/bin/bash

# Train Qwen3-1.7B model using GT framework
# This script runs all steps: weight loading, data prep, and training

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "========================================"
echo "Qwen3-1.7B Training Pipeline"
echo "========================================"
echo "Running from: $SCRIPT_DIR"

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
python load_weights.py

# Step 2: Prepare training data
echo ""
echo "Step 2: Preparing training data..."
echo "========================================"
python prepare_data.py

# Step 3: Train the model
echo ""
echo "Step 3: Training model..."
echo "========================================"
python train.py

echo ""
echo "========================================"
echo "Training pipeline complete!"
echo "========================================"
