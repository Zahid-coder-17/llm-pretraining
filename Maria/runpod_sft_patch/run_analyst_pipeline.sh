#!/bin/bash
set -e

echo "=========================================================="
echo "  STARTING MARIA ANALYST V3 PIPELINE"
echo "=========================================================="

echo "[1/3] Building Multi-Task Dataset..."
python3 build_maria_analyst_dataset.py

echo "[2/3] Training Maria Analyst V3 on NVIDIA H100..."
python3 train_maria_analyst_v3.py

echo "[3/3] Running Verification Suite across all 4 pillars..."
python3 verify_maria_analyst.py

echo "=========================================================="
echo "  ALL PILLARS ALIGNED! Maria Analyst V3 is ready."
echo "=========================================================="
