#!/bin/bash -l
#SBATCH -J eco_lr_ablation_50M
#SBATCH -o outs/eco_lr_ablation_50M_%A_%a.out
#SBATCH -e outs/eco_lr_ablation_50M_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --gres gpu:H200_141GB:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=32000
#SBATCH -A H200
#SBATCH -q h200_qos
#SBATCH --array=1-4

# ECO LR Ablation at 50M
# Goal: Find optimal LR for ECO to ensure fair comparison with ECO0
# Testing: LR ∈ {0.006, 0.00625, 0.008, 0.01} at P90 percentile
#
# Context:
# - ECO0 best: LR=0.01, P90 → Loss 3.221
# - ECO baseline: LR=0.00484, P90 → Loss 3.238
# - Need to verify if ECO can improve with higher LR

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

# Copy datasets to fast scratch space
echo "Copying datasets to /scratch..."
rm -rf /scratch/keisufaj_datasets
mkdir -p /scratch/keisufaj_datasets
rsync -a --info=progress2 /export/home/keisufaj/optimization/ECO-CAGE-QCRI/datasets/ /scratch/keisufaj_datasets/
export DATASETS_DIR="/scratch/keisufaj_datasets"
echo "Dataset copy complete!"

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

# Training config - 5B tokens
TOKENS=5000000000
TOKENS_PER_ITER=$((BATCH_SIZE * ACC_STEPS * SEQUENCE_LENGTH))
ITERATIONS=$((TOKENS / TOKENS_PER_ITER))
WARMUP_STEPS=$((ITERATIONS / 10))

echo "Training for ${ITERATIONS} iterations (${TOKENS} tokens)"
echo "Warmup: ${WARMUP_STEPS} steps"

# Fixed hyperparams
BETA1=0.9
BETA2=0.95

# ========================================
# LR SELECTION
# ========================================

case ${SLURM_ARRAY_TASK_ID} in
    1)
        LR=0.006
        echo "Testing ECO with LR=0.006 (moderate increase from 0.00484)"
        ;;
    2)
        LR=0.00625
        echo "Testing ECO with LR=0.00625 (30M best, unscaled)"
        ;;
    3)
        LR=0.008
        echo "Testing ECO with LR=0.008 (significant increase)"
        ;;
    4)
        LR=0.01
        echo "Testing ECO with LR=0.01 (match ECO0's best LR)"
        ;;
    *)
        echo "Invalid task ID: ${SLURM_ARRAY_TASK_ID}"
        echo "Valid range: 1-4"
        exit 1
        ;;
esac

# ECO configuration (fixed for all runs)
METHOD="ECO-4bit"
OPT="eco"
W_QUANT="Q99FP4Quantizer"
A_QUANT="NoQuantizer"
W_QUANT_KWARGS='{"bits":4,"percentile":90.0}'  # P90 (optimal)
USE_CAGE="False"

WANDB_PREFIX="${MODEL_SIZE}-${METHOD}-LR=${LR}-P90.0-BS=${BATCH_SIZE}x${ACC_STEPS}-ITER=${ITERATIONS}"

# ========================================
# RUN TRAINING
# ========================================

echo "=========================================="
echo "ECO LR Ablation - 50M Model"
echo "=========================================="
echo "Model: ${MODEL_SIZE} (7 layers, 768 embd, 6 heads)"
echo "Method: ${METHOD}"
echo "Optimizer: ${OPT}"
echo "Learning Rate: ${LR}"
echo "Percentile: 90.0 (optimal)"
echo "Batch: ${BATCH_SIZE} × ${ACC_STEPS} = $((BATCH_SIZE * ACC_STEPS))"
echo "Sequence: ${SEQUENCE_LENGTH}"
echo "Iterations: ${ITERATIONS} (${TOKENS} tokens = 5B)"
echo ""
echo "Comparison targets:"
echo "  - ECO baseline: LR=0.00484, Loss=3.238"
echo "  - ECO0 best: LR=0.01, Loss=3.221"
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
echo "Job ${SLURM_ARRAY_TASK_ID} (LR=${LR}) complete!"
echo "Check WandB project: ECO0-SCALING"
echo "Run name: ${WANDB_PREFIX}"
echo "=========================================="

# Clean up scratch space
echo "Cleaning up /scratch..."
rm -rf /scratch/keisufaj_datasets
echo "Cleanup complete!"
