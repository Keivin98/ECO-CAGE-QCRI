#!/bin/bash
# ==============================
# TEMPORARY DIRECTORY ISOLATION & CLEANUP
# ==============================
# Prevents /tmp pollution on SLURM nodes by redirecting all temp/cache dirs
# to job-specific local storage with automatic cleanup.
#
# Usage: source run_scripts/utils/setup_tmp_cleanup.sh
#
# Why: Torch compile, wandb, and triton create thousands of temp files that
# can fill /tmp and put nodes into drain mode. This isolates and cleans them.
# ==============================

echo "========================================"
echo "Setting up temporary directory isolation"
echo "========================================"

# ==============================
# DETERMINE TMP LOCATION
# ==============================

# Priority: /scratch (Panther) > /mnt/localssd (GCP) > SLURM_TMPDIR > /tmp
if [ -d "/scratch" ]; then
    export JOB_TMP=/scratch/${USER}/tmp_${SLURM_JOB_ID}
    TMP_LOCATION="/scratch (node-local)"
elif [ -d "/mnt/localssd" ]; then
    export JOB_TMP=/mnt/localssd/${USER}/tmp_${SLURM_JOB_ID}
    TMP_LOCATION="/mnt/localssd (GCP SSD)"
elif [ -n "$SLURM_TMPDIR" ]; then
    export JOB_TMP=$SLURM_TMPDIR/${USER}_${SLURM_JOB_ID}
    TMP_LOCATION="SLURM_TMPDIR"
else
    export JOB_TMP=/tmp/${USER}_${SLURM_JOB_ID}
    TMP_LOCATION="/tmp (fallback)"
fi

mkdir -p "$JOB_TMP" || {
    echo "ERROR: Failed to create JOB_TMP at $JOB_TMP"
    exit 1
}

echo "Temp location: $JOB_TMP ($TMP_LOCATION)"

# ==============================
# REDIRECT ALL TEMP DIRECTORIES
# ==============================

# System temp dirs
export TMPDIR="$JOB_TMP"
export TMP="$JOB_TMP"
export TEMP="$JOB_TMP"

# PyTorch / Triton / TorchInductor (main culprits!)
export TORCHINDUCTOR_CACHE_DIR="$JOB_TMP/torchinductor"
export TRITON_CACHE_DIR="$JOB_TMP/triton"
export TORCH_HOME="$JOB_TMP/torch"

# WandB - isolate cache, keep data persistent
export WANDB_DIR="$JOB_TMP/wandb"
export WANDB_CACHE_DIR="$JOB_TMP/wandb_cache"
# Note: WANDB_DATA_DIR not set - uses default ~/wandb for persistence

# HuggingFace - redirect cache but preserve token location
# CRITICAL: HF_HOME must stay in home directory for authentication tokens!
# Only redirect the heavy cache files (models/datasets)
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"  # Keep tokens accessible
export TRANSFORMERS_CACHE="$JOB_TMP/hf/transformers"    # Redirect models to temp
export HF_DATASETS_CACHE="$JOB_TMP/hf/datasets"         # Redirect datasets to temp

# General XDG cache - but HF token lookup uses HF_HOME, not XDG
export XDG_CACHE_HOME="$JOB_TMP/.cache"

# Python bytecode cache
export PYTHONPYCACHEPREFIX="$JOB_TMP/pycache"

# ==============================
# CREATE DIRECTORY STRUCTURE
# ==============================

mkdir -p \
  "$TORCHINDUCTOR_CACHE_DIR" \
  "$TRITON_CACHE_DIR" \
  "$TORCH_HOME" \
  "$WANDB_DIR" \
  "$WANDB_CACHE_DIR" \
  "$XDG_CACHE_HOME" \
  "$PYTHONPYCACHEPREFIX" \
  "$TRANSFORMERS_CACHE" \
  "$HF_DATASETS_CACHE"

echo "Created temp directories:"
echo "  - Torch: $TORCHINDUCTOR_CACHE_DIR"
echo "  - Triton: $TRITON_CACHE_DIR"
echo "  - WandB: $WANDB_DIR"
echo "  - HF models: $TRANSFORMERS_CACHE (tokens: $HF_HOME)"
echo "  - Cache: $XDG_CACHE_HOME"

# ==============================
# INITIAL DISK USAGE
# ==============================

echo ""
echo "Initial disk usage:"
df -h "$JOB_TMP" 2>/dev/null | tail -1 || echo "  (disk info unavailable)"

# ==============================
# CLEANUP FUNCTION
# ==============================

cleanup_tmp() {
  echo ""
  echo "========================================"
  echo "Cleaning up temporary files..."
  echo "========================================"

  if [ -d "$JOB_TMP" ]; then
    # Show final size before cleanup
    FINAL_SIZE=$(du -sh "$JOB_TMP" 2>/dev/null | cut -f1)
    echo "Total temp files created: $FINAL_SIZE"

    # Show breakdown of largest dirs (helpful for debugging)
    echo "Largest temp directories:"
    du -sh "$JOB_TMP"/*/ 2>/dev/null | sort -rh | head -5 || echo "  (none)"

    # Remove everything
    rm -rf "$JOB_TMP"

    if [ $? -eq 0 ]; then
      echo "✓ Successfully cleaned: $JOB_TMP"
    else
      echo "✗ Warning: Cleanup may have failed for $JOB_TMP"
    fi
  else
    echo "Temp directory already removed or doesn't exist"
  fi

  echo "Cleanup complete!"
  echo "========================================"
}

# ==============================
# REGISTER CLEANUP TRAP
# ==============================

# Trap multiple signals for robust cleanup:
# - EXIT: normal termination
# - SIGTERM: SLURM cancellation
# - SIGINT: Ctrl+C
trap cleanup_tmp EXIT SIGTERM SIGINT

echo "✓ Cleanup trap registered"
echo "  Will clean on: EXIT, SIGTERM, SIGINT"
echo "========================================"
echo ""
