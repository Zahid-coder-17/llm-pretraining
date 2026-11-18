#!/bin/bash
set -e

echo "=========================================================="
echo "  STARTING MARIA CONVERSATIONAL ALIGNMENT PIPELINE"
echo "=========================================================="

# Step 1: Build 10,000 short-turn conversational dataset
echo "[Step 1/3] Building clean 10k conversational dataset..."
python3 build_conversational_patch_dataset.py

# Step 2: Run EOS Up-Weighted Fine-Tuning
echo "[Step 2/3] Running EOS up-weighted SFT training..."
python3 train_conversational_patch.py

# Step 3: Verify the 4 Advisor Prompts
echo "[Step 3/3] Running verification test suite..."
python3 verify_patch.py

echo "=========================================================="
echo "  ALIGNMENT COMPLETE! Maria Instruct V2 is ready."
echo "=========================================================="
