#!/bin/bash
# 300M Scaling: FP16 + ECO-0 (GCP H100)
# GPU Requirements: 4× H100 (2 GPUs per method)
# Runtime: ~24-30 hours per method (can run in parallel)
# Usage: bash run_scripts/gcp/300M_fp16_eco0.sh [1|2] > logs/300M_method_X.out 2>&1 &
#   Method 1: FP16 Adam (baseline)
#   Method 2: ECO-0 4-bit

set -e  # Exit on error

# Check for method argument
if [ $# -eq 0 ]; then
    echo "Usage: $0 [1|2]"
    echo "  1: FP16 Adam baseline"
    echo "  2: ECO-0 4-bit"
    echo ""
    echo "Example (run both in parallel):"
    echo "  CUDA_VISIBLE_DEVICES=4,5 bash $0 1 > logs/300M_fp16.out 2>&1 &"
    echo "  CUDA_VISIBLE_DEVICES=6,7 bash $0 2 > logs/300M_eco0.out 2>&1 &"
    exit 1
fi

METHOD_ID=$1

echo "=========================================="
echo "300M Scaling Experiment"
echo "=========================================="

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cage

# Environment setup
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((29502 + METHOD_ID))  # Unique port per method
export TORCH_COMPILE=0
export TORCH_COMPILE_DISABLE=1
export TORCHINDUCTOR_DISABLE_TRITON=1

export WANDB_ENTITY="keisufaj-hamad-bin-khalifa-university"
export WANDB_PROJECT="ECO0-SCALING"

# Dataset location (adjust if needed for GCP)
export DATASETS_DIR="/export/home/keisufaj/optimization/ECO-CAGE-QCRI/datasets"

# ========================================
# 300M MODEL CONFIGURATION
# ========================================

MODEL_SIZE="300M"
export N_LAYER=12
export N_EMBD=1536
export N_HEAD=12

# Batch config (adjusted for 300M - smaller batch to fit in memory)
export BATCH_SIZE=24
export ACC_STEPS=16
export SEQUENCE_LENGTH=512

# Training config - 15B tokens (1.5× from 100M for better convergence)
TOKENS=15000000000
TOKENS_PER_ITER=$((BATCH_SIZE * ACC_STEPS * SEQUENCE_LENGTH))
ITERATIONS=$((TOKENS / TOKENS_PER_ITER))
WARMUP_STEPS=$((ITERATIONS / 10))

echo "Training for ${ITERATIONS} iterations (${TOKENS} tokens = 15B)"
echo "Warmup: ${WARMUP_STEPS} steps"
echo "Effective batch size: $((BATCH_SIZE * ACC_STEPS)) tokens"

# Fixed hyperparams
BETA1=0.9
BETA2=0.95

# ========================================
# METHOD SELECTION
# ========================================

case ${METHOD_ID} in
    1)
        METHOD="FP16-Adam"
        OPT="adamw"
        W_QUANT="NoQuantizer"
        A_QUANT="NoQuantizer"
        W_QUANT_KWARGS='{"bits":16}'
        USE_CAGE="False"
        # LR prediction: 100M used 0.0006, 300M is 3× → scale by ~0.65
        LR=0.0004
        echo "Testing FP16 Adam (baseline)"
        echo "LR prediction: 0.0006 × 0.65 ≈ 0.0004"
        ;;
    2)
        METHOD="ECO0-4bit"
        OPT="eco0m-rooh"
        W_QUANT="Q99FP4Quantizer"
        A_QUANT="NoQuantizer"
        W_QUANT_KWARGS='{"bits":4,"percentile":90.0}'
        USE_CAGE="False"
        # LR prediction: 100M best was 0.007 (pending refinement), scale by ~0.65-0.7
        LR=0.0045
        echo "Testing ECO-0 4-bit (ours)"
        echo "LR prediction: 0.007 × 0.65 ≈ 0.0045"
        echo "Note: May need tuning based on 100M refinement results"
        ;;
    *)
        echo "Invalid method ID: ${METHOD_ID}"
        echo "Valid options: 1 (FP16) or 2 (ECO-0)"
        exit 1
        ;;
esac

WANDB_PREFIX="${MODEL_SIZE}-${METHOD}-LR=${LR}-BS=${BATCH_SIZE}x${ACC_STEPS}-ITER=${ITERATIONS}"

# ========================================
# RUN TRAINING
# ========================================

echo "=========================================="
echo "Configuration:"
echo "  Model: ${MODEL_SIZE} (12 layers, 1536 embd, 12 heads)"
echo "  Method: ${METHOD}"
echo "  Optimizer: ${OPT}"
echo "  Learning Rate: ${LR}"
echo "  Batch: ${BATCH_SIZE} × ${ACC_STEPS} = $((BATCH_SIZE * ACC_STEPS))"
echo "  Sequence: ${SEQUENCE_LENGTH}"
echo "  Iterations: ${ITERATIONS} (${TOKENS} tokens)"
echo ""
echo "Scaling progression:"
echo "  30M  → 50M  (1.67×)"
echo "  50M  → 100M (2×)"
echo "  100M → 300M (3×) ← CURRENT"
echo ""
echo "Expected memory savings (ECO-0 vs FP16):"
echo "  ~3-4 GB (linear scaling from 100M's 1.64 GB)"
echo "=========================================="

cd /export/home/keisufaj/optimization/ECO-CAGE-QCRI

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
echo "300M ${METHOD} Experiment Complete!"
echo "=========================================="
echo "Check WandB: ${WANDB_PROJECT} / ${WANDB_PREFIX}"
echo ""
echo "Next steps:"
echo "  1. Compare ECO-0 vs FP16 at 300M"
echo "  2. Validate memory scaling (expect ~3-4 GB savings)"
echo "  3. Check if performance gap to baseline continues narrowing"
echo "  4. Update paper with 300M results"
echo "=========================================="
