#!/bin/bash -l
#SBATCH -J 500M_optimal_lr
#SBATCH -o outs/500M_optimal_lr_%A_%a.out
#SBATCH -e outs/500M_optimal_lr_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --gres gpu:H200_141GB:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000
#SBATCH -A H200
#SBATCH -q h200_qos
#SBATCH --array=1-4

# 500M Optimal LR Test
# Testing: FP16, CAGE, ECO0 (2 LRs)
# Scaling: 5× from 100M → LR scale by 1/√5 = 0.447

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
# 500M MODEL CONFIGURATION
# ========================================
# Architecture: ~500M params (5× from 100M)
# Scaling from 100M: layers 8→16, embd 1024→1280, heads 8→10

MODEL_SIZE="500M"
export N_LAYER=16
export N_EMBD=1280
export N_HEAD=10

# Batch config - reduce batch size for memory
export BATCH_SIZE=16
export ACC_STEPS=32
export SEQUENCE_LENGTH=512

# Training config - 20B tokens (2× from 100M for better convergence)
TOKENS=20000000000
TOKENS_PER_ITER=$((BATCH_SIZE * ACC_STEPS * SEQUENCE_LENGTH))
ITERATIONS=$((TOKENS / TOKENS_PER_ITER))
WARMUP_STEPS=$((ITERATIONS / 10))

echo "Training for ${ITERATIONS} iterations (${TOKENS} tokens = 20B)"
echo "Warmup: ${WARMUP_STEPS} steps"

# Fixed hyperparams
BETA1=0.9
BETA2=0.95

# Fixed quantization config
W_QUANT="Q99FP4Quantizer"
A_QUANT="NoQuantizer"
W_QUANT_KWARGS='{"bits":4,"percentile":90.0}'  # P90 (optimal)

# ========================================
# METHOD SELECTION
# ========================================
# LR scaling from 100M (assuming 0.007 optimal):
# 100M→500M: 5× scale → 1/√5 = 0.447 → 0.007 × 0.447 = 0.00313

case ${SLURM_ARRAY_TASK_ID} in
    1)
        METHOD="FP16-Adam"
        OPT="adamw"
        W_QUANT="NoQuantizer"
        W_QUANT_KWARGS='{"bits":16}'
        USE_CAGE="False"
        LR=0.00027  # 0.0006 × 0.447 from 100M
        echo "Testing FP16 Adam (baseline, scaled from 100M)"
        ;;
    2)
        METHOD="CAGE-4bit"
        OPT="adamw"
        USE_CAGE="True"
        LR=0.00027  # Same as FP16
        echo "Testing CAGE (QAT baseline, scaled from 100M)"
        ;;
    3)
        METHOD="ECO0-4bit"
        OPT="eco0m-rooh"
        USE_CAGE="False"
        LR=0.003  # Conservative: 0.007 × 0.447 ≈ 0.003
        echo "Testing ECO0 with LR=0.003 (conservative, scaled from 100M)"
        ;;
    4)
        METHOD="ECO0-4bit"
        OPT="eco0m-rooh"
        USE_CAGE="False"
        LR=0.004  # Aggressive: testing higher
        echo "Testing ECO0 with LR=0.004 (aggressive)"
        ;;
    *)
        echo "Invalid task ID: ${SLURM_ARRAY_TASK_ID}"
        echo "Valid range: 1-4"
        exit 1
        ;;
esac

WANDB_PREFIX="${MODEL_SIZE}-${METHOD}-LR=${LR}-P90.0-BS=${BATCH_SIZE}x${ACC_STEPS}-ITER=${ITERATIONS}"

# ========================================
# RUN TRAINING
# ========================================

echo "=========================================="
echo "500M Scaling Test"
echo "=========================================="
echo "Model: ${MODEL_SIZE} (16 layers, 1280 embd, 10 heads)"
echo "Method: ${METHOD}"
echo "Optimizer: ${OPT}"
echo "Learning Rate: ${LR}"
echo "Percentile: 90.0 (validated optimal)"
echo "Batch: ${BATCH_SIZE} × ${ACC_STEPS} = $((BATCH_SIZE * ACC_STEPS))"
echo "Sequence: ${SEQUENCE_LENGTH}"
echo "Iterations: ${ITERATIONS} (${TOKENS} tokens = 20B)"
echo ""
echo "Scaling from 100M (5× parameters):"
echo "  - Architecture: 8L/1024E/8H → 16L/1280E/10H"
echo "  - LR scaling: ×0.447 (1/√5)"
echo "  - Tokens: 10B → 20B (2× for better convergence)"
echo "  - Expected memory: ~40-50 GB per GPU"
echo "  - Expected ECO0 savings: ~5-8 GB vs FP16/CAGE"
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
echo "Expected scaling behavior:"
echo "  - Gap to CAGE should continue narrowing"
echo "  - Memory savings scale to ~5-8 GB"
echo "  - ECO0 should remain competitive at 500M scale"
echo "=========================================="

# Clean up scratch space (job-specific directory)
echo "Cleaning up ${DATASETS_DIR}..."
rm -rf ${DATASETS_DIR}
echo "Cleanup complete!"
