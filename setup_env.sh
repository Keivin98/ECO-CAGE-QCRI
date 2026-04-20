#!/bin/bash
set -e

echo "=========================================="
echo "Setting up ECO-CAGE-QCRI environment"
echo "=========================================="

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cage

echo ""
echo "[1/2] Installing fast-hadamard-transform..."
cd /export/home/keisufaj/optimization/ECO-CAGE-QCRI/fast-hadamard-transform
pip install -e .
cd /export/home/keisufaj/optimization/ECO-CAGE-QCRI

echo ""
echo "[2/2] Setting up datasets directory..."

# Create local datasets directory
mkdir -p ./datasets

# Update the test script to use local datasets
export DATASETS_DIR="/export/home/keisufaj/optimization/ECO-CAGE-QCRI/datasets"

echo ""
echo "=========================================="
echo "Setup complete!"
echo ""
echo "Datasets will be downloaded to: ${DATASETS_DIR}"
echo "The first run will download C4 dataset automatically (may take a few minutes)"
echo ""
echo "To use in scripts, set:"
echo "  export DATASETS_DIR=\"${DATASETS_DIR}\""
echo "=========================================="
