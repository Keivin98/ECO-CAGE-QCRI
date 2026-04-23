#!/bin/bash -l
#SBATCH -J 100M_eco0_lr_refine
#SBATCH -o outs/100M_eco0_lr_refine_%A_%a.out
#SBATCH -e outs/100M_eco0_lr_refine_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --gres gpu:H200_141GB:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=32000
#SBATCH -A H200
#SBATCH -q h200_qos
#SBATCH --array=1-2

# 100M ECO-0 LR Refinement
# Current results:
#   ECO    LR=0.005  → 20.11 PPL (best overall)
#   ECO-0  LR=0.007  → 21.35 PPL (current ECO-0 best)
#   ECO-0  LR=0.0085 → 21.41 PPL (worse than 0.007)
#
# Testing lower LRs to close gap with ECO:
#   Task 1: LR=0.006  (midpoint between ECO's 0.005 and our 0.007)
#   Task 2: LR=0.0065 (split difference 0.006-0.007)

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cage

# Setup isolated temp directories with automatic cleanup
source /export/home/keisufaj/optimization/ECO-CAGE-QCRI/run_scripts/utils/setup_tmp_cleanup.sh

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((29500 + SLURM_ARRAY_TASK_ID))
export TORCH_COMPILE=0
export TORCH_COMPILE_DISABLE=1
export TORCHINDUCTOR_DISABLE_TRITON=1

export WANDB_ENTITY="keisufaj-hamad-bin-khalifa-university"
export WANDB_PROJECT="ECO0-SCALING"

# Copy datasets to fast scratch space (job-specific to avoid collisions)
export DATASETS_DIR="/scratch/keisufaj_datasets_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Copying datasets to ${DATASETS_DIR}..."

# Check if already exists (in case of restart)
if [ -d "${DATASETS_DIR}/c4" ]; then
    echo "Dataset already exists, skipping copy"
else
    mkdir -p ${DATASETS_DIR}
    rsync -a --info=progress2 /export/home/keisufaj/optimization/ECO-CAGE-QCRI/datasets/ ${DATASETS_DIR}/
    echo "Dataset copy complete!"
fi

# ========================================
# 100M MODEL CONFIGURATION
# ========================================

MODEL_SIZE="100M"
export N_LAYER=8
export N_EMBD=1024
export N_HEAD=8

# Batch config (same as other 100M experiments)
export BATCH_SIZE=32
export ACC_STEPS=16
export SEQUENCE_LENGTH=512

# Training config - 10B tokens (CAGE's proven config)
TOKENS=10000000000
TOKENS_PER_ITER=$((BATCH_SIZE * ACC_STEPS * SEQUENCE_LENGTH))
ITERATIONS=$((TOKENS / TOKENS_PER_ITER))
WARMUP_STEPS=$((ITERATIONS / 10))

echo "Training for ${ITERATIONS} iterations (${TOKENS} tokens = 10B)"
echo "Warmup: ${WARMUP_STEPS} steps"

# Fixed hyperparams
BETA1=0.9
BETA2=0.95

# Fixed quantization config
W_QUANT="Q99FP4Quantizer"
A_QUANT="NoQuantizer"
W_QUANT_KWARGS='{"bits":4,"percentile":90.0}'  # P90 (optimal)
USE_CAGE="False"

# ========================================
# METHOD SELECTION - ECO-0 LR REFINEMENT
# ========================================

METHOD="ECO0-4bit"
OPT="eco0m-rooh"

case ${SLURM_ARRAY_TASK_ID} in
    1)
        LR=0.006
        echo "Testing ECO-0 with LR=0.006 (lower than current best 0.007)"
        ;;
    2)
        LR=0.008
        echo "Testing ECO-0 with LR=0.008 (midpoint 0.006-0.007)"
        ;;
    *)
        echo "Invalid task ID: ${SLURM_ARRAY_TASK_ID}"
        echo "Valid range: 1-2"
        exit 1
        ;;
esac

WANDB_PREFIX="${MODEL_SIZE}-${METHOD}-LR=${LR}-P90.0-BS=${BATCH_SIZE}x${ACC_STEPS}-ITER=${ITERATIONS}"

# ========================================
# RUN TRAINING
# ========================================

echo "=========================================="
echo "100M ECO-0 LR Refinement"
echo "=========================================="
echo "Model: ${MODEL_SIZE} (8 layers, 1024 embd, 8 heads)"
echo "Method: ${METHOD}"
echo "Optimizer: ${OPT}"
echo "Learning Rate: ${LR}"
echo "Percentile: 90.0 (validated optimal)"
echo "Batch: ${BATCH_SIZE} × ${ACC_STEPS} = $((BATCH_SIZE * ACC_STEPS))"
echo "Sequence: ${SEQUENCE_LENGTH}"
echo "Iterations: ${ITERATIONS} (${TOKENS} tokens = 10B)"
echo ""
echo "Current results to beat:"
echo "  ECO    LR=0.005  → 20.11 PPL ⭐ (best overall)"
echo "  ECO-0  LR=0.007  → 21.35 PPL (current ECO-0 best)"
echo "  ECO-0  LR=0.0085 → 21.41 PPL (worse)"
echo ""
echo "Goal: Close the 1.24 PPL gap to ECO"
echo "Hypothesis: Optimal LR is between 0.005-0.007"
echo "=========================================="

torchrun --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    --nproc_per_node=2 /export/home/keisufaj/optimization/ECO-CAGE-QCRI/src/main.py \
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
echo "Job ${SLURM_ARRAY_TASK_ID} (${METHOD}, LR=${LR}) complete!"
echo "Check WandB project: ECO0-SCALING"
echo "Run name: ${WANDB_PREFIX}"
echo ""
echo "Next steps based on results:"
echo "  If LR=0.006 or 0.0065 substantially improves:"
echo "    → ECO-0 can match ECO at 100M scale ✅"
echo "  If both fail to improve over 21.35:"
echo "    → Test LR=0.0055 (closer to ECO's 0.005)"
echo "    OR accept 21.35 as near-optimal and proceed to 500M"
echo ""
echo "Expected memory: ~18.8 GB (vs ECO's 20.5 GB = 8% savings)"
echo "=========================================="

# Clean up scratch space (job-specific directory)
echo "Cleaning up ${DATASETS_DIR}..."
rm -rf ${DATASETS_DIR}
echo "Cleanup complete!"
