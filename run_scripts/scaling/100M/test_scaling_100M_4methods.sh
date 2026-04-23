#!/bin/bash -l
#SBATCH -J scaling_100M
#SBATCH -o outs/scaling_100M_%A_%a.out
#SBATCH -e outs/scaling_100M_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --gres gpu:H200_141GB:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=32000
#SBATCH -A H200
#SBATCH -q h200_qos
#SBATCH --array=1-4

# Scaling experiment: 100M model (2× from 50M)
# Testing: FP16, CAGE, ECO, ECO0 (complete comparison)
# LR scaling: ×0.71 (1/sqrt(2)) from 50M values

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
# 100M MODEL CONFIGURATION
# ========================================

MODEL_SIZE="100M"
export N_LAYER=8
export N_EMBD=1024
export N_HEAD=8

# Batch config (same effective batch as 50M)
export BATCH_SIZE=32
export ACC_STEPS=16
export SEQUENCE_LENGTH=512

# Training config - 10B tokens (CAGE's proven config)
TOKENS=10000000000
TOKENS_PER_ITER=$((BATCH_SIZE * ACC_STEPS * SEQUENCE_LENGTH))
ITERATIONS=$((TOKENS / TOKENS_PER_ITER))
WARMUP_STEPS=$((ITERATIONS / 10))

echo "Training for ${ITERATIONS} iterations (${TOKENS} tokens)"
echo "Warmup: ${WARMUP_STEPS} steps"

# Fixed hyperparams
BETA1=0.9
BETA2=0.95

# ========================================
# METHOD SELECTION
# ========================================
# Using CAGE's proven LRs for 100M
# FP16/CAGE: 0.0006 (CAGE's tested baseline)
# ECO/ECO0: Scaled maintaining ratio from 50M experiments

case ${SLURM_ARRAY_TASK_ID} in
    1)
        METHOD="FP16-Adam"
        OPT="adamw"
        W_QUANT="NoQuantizer"
        A_QUANT="NoQuantizer"
        W_QUANT_KWARGS='{"bits":16}'
        USE_CAGE="False"
        LR=0.0006
        echo "Testing FP16 Adam (baseline, CAGE's proven LR)"
        ;;
    2)
        METHOD="CAGE-4bit"
        OPT="adamw"
        W_QUANT="Q99FP4Quantizer"
        A_QUANT="NoQuantizer"
        W_QUANT_KWARGS='{"bits":4,"percentile":90.0}'
        USE_CAGE="True"
        LR=0.0006
        echo "Testing CAGE (QAT baseline, LR=0.0006)"
        ;;
    3)
        METHOD="ECO-4bit"
        OPT="eco"
        W_QUANT="Q99FP4Quantizer"
        A_QUANT="NoQuantizer"
        W_QUANT_KWARGS='{"bits":4,"percentile":90.0}'
        USE_CAGE="False"
        LR=0.0031
        echo "Testing ECO (LR=0.0031, scaled from 50M ratio)"
        ;;
    4)
        METHOD="ECO0-4bit"
        OPT="eco0m-rooh"
        W_QUANT="Q99FP4Quantizer"
        A_QUANT="NoQuantizer"
        W_QUANT_KWARGS='{"bits":4,"percentile":90.0}'
        USE_CAGE="False"
        # Conservative: scaled from 0.006 at 50M
        LR=0.0042
        echo "Testing ECO0 (ours, LR=0.0042, conservative)"
        echo "NOTE: Using 0.006×0.7 from 50M ablation (conservative)"
        ;;
    *)
        echo "Invalid task ID: ${SLURM_ARRAY_TASK_ID}"
        echo "Valid range: 1-4"
        exit 1
        ;;
esac

WANDB_PREFIX="${MODEL_SIZE}-${METHOD}-LR=${LR}-BS=${BATCH_SIZE}x${ACC_STEPS}-ITER=${ITERATIONS}"

# ========================================
# RUN TRAINING
# ========================================

echo "=========================================="
echo "Model: ${MODEL_SIZE} (8 layers, 1024 embd, 8 heads)"
echo "Method: ${METHOD}"
echo "Learning Rate: ${LR}"
echo "Batch: ${BATCH_SIZE} × ${ACC_STEPS} = $((BATCH_SIZE * ACC_STEPS))"
echo "Sequence: ${SEQUENCE_LENGTH}"
echo "Iterations: ${ITERATIONS} (${TOKENS} tokens = 10B)"
echo ""
echo "Comparison to 50M:"
echo "  - Params: 50M → 100M (2× scale)"
echo "  - Tokens: 5B → 10B (2× more training)"
echo "  - LR baseline: 0.0006 (CAGE's proven config)"
echo "  - Expected memory: ~30-32 GB per GPU"
echo "  - Expected ECO0 savings: ~2 GB vs FP16/CAGE"
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
echo "Job ${SLURM_ARRAY_TASK_ID} (${METHOD}) complete!"
echo "Check WandB project: ECO0-SCALING"
echo "Expected results:"
echo "  - FP16 baseline: ~XX perplexity"
echo "  - CAGE: ~+0.5-1.0 perplexity vs FP16"
echo "  - ECO0 savings: ~2 GB memory vs FP16/CAGE"
echo "=========================================="

# Clean up scratch space
echo "Cleaning up /scratch..."
rm -rf /scratch/keisufaj_datasets
echo "Cleanup complete!"
