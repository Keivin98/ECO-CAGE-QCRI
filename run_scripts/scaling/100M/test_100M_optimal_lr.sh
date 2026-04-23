#!/bin/bash -l
#SBATCH -J 100M_optimal_lr
#SBATCH -o outs/100M_optimal_lr_%A_%a.out
#SBATCH -e outs/100M_optimal_lr_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --gres gpu:H200_141GB:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=32000
#SBATCH -A H200
#SBATCH -q h200_qos
#SBATCH --array=1-2

# 100M Optimal LR Test
# Testing predicted optimal LRs based on 50M results
# ECO0: 0.007 (scaled from 50M's 0.01)
# ECO:  0.005 (predicted from ECO0/ECO ratio at 50M)

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cage

# Setup isolated temp directories with automatic cleanup (HF auth fixed!)
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
# METHOD SELECTION
# ========================================

case ${SLURM_ARRAY_TASK_ID} in
    1)
        METHOD="ECO0-4bit"
        OPT="eco0m-rooh"
        LR=0.007
        echo "Testing ECO0 with LR=0.007 (predicted optimal from 50M)"
        ;;
    2)
        METHOD="ECO0-4bit"
        OPT="eco0m-rooh"
        LR=0.0085
        echo "Testing ECO0 with LR=0.008 (predicted optimal from 50M)"
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
echo "100M Optimal LR Test"
echo "=========================================="
echo "Model: ${MODEL_SIZE} (8 layers, 1024 embd, 8 heads)"
echo "Method: ${METHOD}"
echo "Optimizer: ${OPT}"
echo "Learning Rate: ${LR} (predicted optimal)"
echo "Percentile: 90.0 (validated optimal)"
echo "Batch: ${BATCH_SIZE} × ${ACC_STEPS} = $((BATCH_SIZE * ACC_STEPS))"
echo "Sequence: ${SEQUENCE_LENGTH}"
echo "Iterations: ${ITERATIONS} (${TOKENS} tokens = 10B)"
echo ""
echo "Baselines to compare:"
echo "  - Current ECO0: LR=0.0042 → 22.39 PPL (under-tuned)"
echo "  - Current ECO:  LR=0.0031 → 21.26 PPL (under-tuned)"
echo "  - CAGE:         LR=0.0006 → 19.18 PPL"
echo "  - FP16:         LR=0.0006 → 17.93 PPL"
echo ""
echo "Expected: ECO0 (0.007) should beat ECO (0.005) by ~0.4 PPL"
echo "          and narrow gap to CAGE significantly"
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
echo "Next: Compare with baseline results to validate:"
echo "  1. ECO0 (0.007) > ECO0 (0.0042) baseline"
echo "  2. ECO0 (0.007) > ECO (0.005)"
echo "  3. Gap to CAGE narrows from 3.21 PPL baseline"
echo "=========================================="

# Clean up scratch space (job-specific directory)
echo "Cleaning up ${DATASETS_DIR}..."
rm -rf ${DATASETS_DIR}
echo "Cleanup complete!"
