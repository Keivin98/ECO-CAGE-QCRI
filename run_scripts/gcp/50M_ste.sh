#!/bin/bash
# 50M STE Baseline (GCP H100)
# GPU Requirements: 2× H100 (40GB or 80GB)
# Runtime: ~5-6 hours
# Usage: CUDA_VISIBLE_DEVICES=0,1 bash run_scripts/gcp/50M_ste.sh > logs/50M_ste.out 2>&1 &

set -e  # Exit on error

echo "=========================================="
echo "50M STE Baseline Experiment"
echo "=========================================="

# Activate conda environment
eval "$('/home/local/QCRI/kisufaj/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
conda activate /home/local/QCRI/kisufaj/miniconda3/envs/qwen/

# ========================================
# TEMP DIRECTORY ISOLATION & CLEANUP
# ========================================
# Prevents /tmp pollution from torchinductor, wandb, triton

echo "Setting up temporary directory isolation..."

# Use PID for unique job ID on GCP
JOB_ID=$$

# Priority: /mnt/localssd (GCP SSD) > /scratch > /tmp
if [ -d "/mnt/localssd" ]; then
    export JOB_TMP=/mnt/localssd/${USER}/tmp_${JOB_ID}
elif [ -d "/scratch" ]; then
    export JOB_TMP=/scratch/${USER}/tmp_${JOB_ID}
else
    export JOB_TMP=/tmp/${USER}_tmp_${JOB_ID}
fi

mkdir -p "$JOB_TMP"

# Redirect all temp directories
export TMPDIR="$JOB_TMP"
export TMP="$JOB_TMP"
export TEMP="$JOB_TMP"
export TORCHINDUCTOR_CACHE_DIR="$JOB_TMP/torchinductor"
export TRITON_CACHE_DIR="$JOB_TMP/triton"
export TORCH_HOME="$JOB_TMP/torch"
export WANDB_DIR="$JOB_TMP/wandb"
export WANDB_CACHE_DIR="$JOB_TMP/wandb_cache"
export XDG_CACHE_HOME="$JOB_TMP/.cache"
export PYTHONPYCACHEPREFIX="$JOB_TMP/pycache"

# HuggingFace - CRITICAL: Keep HF_HOME in home directory for authentication tokens!
# Only redirect the heavy cache files (models/datasets) to temp
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"  # Tokens stay accessible
export TRANSFORMERS_CACHE="$JOB_TMP/hf/transformers"
export HF_DATASETS_CACHE="$JOB_TMP/hf/datasets"

mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$TORCH_HOME" \
         "$WANDB_DIR" "$WANDB_CACHE_DIR" "$XDG_CACHE_HOME" "$PYTHONPYCACHEPREFIX" \
         "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

echo "Temp directory: $JOB_TMP"

# Cleanup function
cleanup_tmp() {
  echo "Cleaning up temporary files..."
  if [ -d "$JOB_TMP" ]; then
    FINAL_SIZE=$(du -sh "$JOB_TMP" 2>/dev/null | cut -f1)
    echo "Temp files created: $FINAL_SIZE"
    rm -rf "$JOB_TMP"
    echo "✓ Cleaned: $JOB_TMP"
  fi
}

# Register cleanup on exit
trap cleanup_tmp EXIT SIGTERM SIGINT

# ========================================
# ENVIRONMENT SETUP
# ========================================

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export TORCH_COMPILE=0
export TORCH_COMPILE_DISABLE=1
export TORCHINDUCTOR_DISABLE_TRITON=1

export WANDB_ENTITY="keisufaj-hamad-bin-khalifa-university"
export WANDB_PROJECT="ECO0-SCALING"

# Dataset location (adjust if needed for GCP)
export DATASETS_DIR="/image-generation/kisufaj/optimization/cage/CAGE/datasets"

# ========================================
# 50M MODEL CONFIGURATION
# ========================================

MODEL_SIZE="50M"
export N_LAYER=7
export N_EMBD=768
export N_HEAD=6

# Batch config (same as other 50M experiments)
export BATCH_SIZE=64
export ACC_STEPS=8
export SEQUENCE_LENGTH=512

# Training config - 5B tokens (matches 50M baseline)
TOKENS=5000000000
TOKENS_PER_ITER=$((BATCH_SIZE * ACC_STEPS * SEQUENCE_LENGTH))
ITERATIONS=$((TOKENS / TOKENS_PER_ITER))
WARMUP_STEPS=$((ITERATIONS / 10))

echo "Training for ${ITERATIONS} iterations (${TOKENS} tokens = 5B)"
echo "Warmup: ${WARMUP_STEPS} steps"

# Fixed hyperparams
BETA1=0.9
BETA2=0.95

# STE Configuration (standard QAT without CAGE curvature)
METHOD="STE-4bit"
OPT="adamw"
W_QUANT="Q99FP4Quantizer"
A_QUANT="NoQuantizer"
W_QUANT_KWARGS='{"bits":4,"percentile":90.0}'  # P90 (validated optimal)
USE_CAGE="False"  # Critical: STE does NOT use CAGE correction
LR=0.00093  # Same as CAGE/FP16 at 50M

WANDB_PREFIX="${MODEL_SIZE}-${METHOD}-LR=${LR}-P90.0-BS=${BATCH_SIZE}x${ACC_STEPS}-ITER=${ITERATIONS}"

# ========================================
# RUN TRAINING
# ========================================

echo "=========================================="
echo "Configuration:"
echo "  Model: ${MODEL_SIZE} (7 layers, 768 embd, 6 heads)"
echo "  Method: ${METHOD}"
echo "  Optimizer: ${OPT}"
echo "  Learning Rate: ${LR}"
echo "  Percentile: 90.0"
echo "  Use CAGE: ${USE_CAGE}"
echo "  Batch: ${BATCH_SIZE} × ${ACC_STEPS} = $((BATCH_SIZE * ACC_STEPS))"
echo "  Sequence: ${SEQUENCE_LENGTH}"
echo "  Iterations: ${ITERATIONS} (${TOKENS} tokens)"
echo ""
echo "Expected Results:"
echo "  - STE should match CAGE: ~23.1-23.2 PPL"
echo "  - At 30M: STE (29.07) ≈ CAGE (29.08) ✓"
echo "  - Validates curvature correction minimal benefit"
echo "=========================================="

cd /image-generation/kisufaj/optimization/cage/CAGE

torchrun --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    --nproc_per_node=2 src/main.py \
    --distributed-backend nccl \
    --dataset c4 \
    --datasets-dir ${DATASETS_DIR} \
    --model llama \
    --opt ${OPT} \
    --use-cage ${USE_CAGE} \
    --compile \
    --acc-steps ${ACC_STEPS} \
    --batch-size ${BATCH_SIZE} \
    --sequence-length ${SEQUENCE_LENGTH} \
    --wandb \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-run-prefix "${WANDB_PREFIX}" \
    --seed 0 \
    --n-layer ${N_LAYER} \
    --n-embd ${N_EMBD} \
    --n-head ${N_HEAD} \
    --warmup-steps ${WARMUP_STEPS} \
    --iterations ${ITERATIONS} \
    --lr ${LR} \
    --w-quant ${W_QUANT} \
    --w-quant-kwargs "${W_QUANT_KWARGS}" \
    --a-quant ${A_QUANT} \
    --beta1 ${BETA1} \
    --beta2 ${BETA2}

echo "=========================================="
echo "50M STE Experiment Complete!"
echo "=========================================="
echo "Check WandB: ${WANDB_PROJECT} / ${WANDB_PREFIX}"
echo ""
echo "Compare with 50M CAGE results:"
echo "  CAGE 4-bit: 23.14 PPL (LR=0.00093)"
echo "  Expected STE: ~23.1-23.2 PPL"
echo ""
echo "If STE ≈ CAGE, confirms curvature correction minimal!"
echo "=========================================="
