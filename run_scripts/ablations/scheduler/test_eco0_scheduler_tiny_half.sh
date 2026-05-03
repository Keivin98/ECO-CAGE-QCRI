#!/bin/bash -l
#SBATCH -J eco0_sched_tiny
#SBATCH -o outs/eco0_sched_tiny_%A_%a.out
#SBATCH -e outs/eco0_sched_tiny_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --exclude=crirdchpxd006   
#SBATCH --gres=gpu:H200_141GB:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16000
#SBATCH -A H200_16GPUs
#SBATCH -q h200_qos_16_gpus
#SBATCH --array=2,4,6,8,10,12,13-16 

# ECO-0 Scheduler Ablation: Cosine Decay vs Constant LR after Warmup
# Tests both FP4 (Q99FP4Quantizer) and INT4 (Q99IntQuantizer) variants
# Model: tiny (3 layers, 128 embd, 2 heads)
# Runtime: ~3 min per job, all 12 can run in parallel on H200s (16 GPU limit)
# NOTE: Using preemptable queue (H200_16GPUs) - normal H200 jobs can preempt these

# ========================================
# EXPERIMENT MATRIX
# ========================================
# Task 1: FP4 + cos + LR=0.01
# Task 2: FP4 + static-eco + LR=0.01
# Task 3: FP4 + cos + LR=0.012
# Task 4: FP4 + static-eco + LR=0.012
# Task 5: FP4 + cos + LR=0.008
# Task 6: FP4 + static-eco + LR=0.008
# Task 7: INT4 + cos + LR=0.01
# Task 8: INT4 + static-eco + LR=0.01
# Task 9: INT4 + cos + LR=0.012
# Task 10: INT4 + static-eco + LR=0.012
# Task 11: INT4 + cos + LR=0.008
# Task 12: INT4 + static-eco + LR=0.008

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cage

# Setup isolated temp directories with automatic cleanup
source /export/home/keisufaj/optimization/ECO-CAGE-QCRI/run_scripts/utils/setup_tmp_cleanup.sh

# ========================================
# ENVIRONMENT SETUP
# ========================================

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((29600 + SLURM_ARRAY_TASK_ID))
export TORCH_COMPILE=0
export TORCH_COMPILE_DISABLE=1
export TORCHINDUCTOR_DISABLE_TRITON=1

export WANDB_ENTITY="keisufaj-hamad-bin-khalifa-university"
export WANDB_PROJECT="ECO0-SCHEDULER-ABLATION"

# Use datasets directly from source (no scratch copy)
export DATASETS_DIR="/export/home/keisufaj/optimization/ECO-CAGE-QCRI/datasets"

# ========================================
# TINY MODEL CONFIGURATION
# ========================================

MODEL_SIZE="tiny"
export N_LAYER=3
export N_EMBD=128
export N_HEAD=2

export BATCH_SIZE=128
export ACC_STEPS=4
export SEQUENCE_LENGTH=512

# Training config - 5000 iterations
ITERATIONS=5000
WARMUP_STEPS=$((ITERATIONS / 10))  # 500 steps

# Fixed hyperparams
BETA1=0.9
BETA2=0.95
OPT="eco0m-rooh"

# ========================================
# TASK SELECTION
# ========================================

case ${SLURM_ARRAY_TASK_ID} in
    1)
        QUANT_MODE="FP4"
        SCHEDULER="cos"
        LR=0.01
        ;;
    2)
        QUANT_MODE="FP4"
        SCHEDULER="half-eco0"
        LR=0.01
        ;;
    3)
        QUANT_MODE="FP4"
        SCHEDULER="cos"
        LR=0.012
        ;;
    4)
        QUANT_MODE="FP4"
        SCHEDULER="half-eco0"
        LR=0.012
        ;;
    5)
        QUANT_MODE="FP4"
        SCHEDULER="cos"
        LR=0.008
        ;;
    6)
        QUANT_MODE="FP4"
        SCHEDULER="half-eco0"
        LR=0.008
        ;;
    7)
        QUANT_MODE="INT4"
        SCHEDULER="cos"
        LR=0.01
        ;;
    8)
        QUANT_MODE="INT4"
        SCHEDULER="half-eco0"
        LR=0.01
        ;;
    9)
        QUANT_MODE="INT4"
        SCHEDULER="cos"
        LR=0.012
        ;;
    10)
        QUANT_MODE="INT4"
        SCHEDULER="half-eco0"
        LR=0.012
        ;;
    11)
        QUANT_MODE="INT4"
        SCHEDULER="cos"
        LR=0.008
        ;;
    12)
        QUANT_MODE="INT4"
        SCHEDULER="half-eco0"
        LR=0.008
        ;;
    13)
        QUANT_MODE="FP4"
        SCHEDULER="half-eco0"
        LR=0.005
        ;;
    14)
        QUANT_MODE="INT4"
        SCHEDULER="half-eco0"
        LR=0.005
        ;;
    15)
        QUANT_MODE="FP4"
        SCHEDULER="half-eco0"
        LR=0.0025
        ;;
    16)
        QUANT_MODE="INT4"
        SCHEDULER="half-eco0"
        LR=0.0025
        ;;
esac

if [ "$QUANT_MODE" = "FP4" ]; then
    W_QUANT="Q99FP4Quantizer"
    W_QUANT_KWARGS='{"bits":4,"percentile":90.0}'
else
    W_QUANT="Q99IntQuantizer"
    W_QUANT_KWARGS='{"bits":4,"percentile":90.0}'
fi

WANDB_PREFIX="${MODEL_SIZE}-ECO0-${QUANT_MODE}-sched=${SCHEDULER}-LR=${LR}-ITER=${ITERATIONS}"

# ========================================
# RUN TRAINING
# ========================================

echo "=========================================="
echo "ECO-0 Scheduler Ablation - Task ${SLURM_ARRAY_TASK_ID}/12"
echo "=========================================="
echo "Configuration:"
echo "  Model: ${MODEL_SIZE} (${N_LAYER} layers, ${N_EMBD} embd, ${N_HEAD} heads)"
echo "  Quantization: ${QUANT_MODE}"
echo "  Scheduler: ${SCHEDULER}"
echo "  Learning Rate: ${LR}"
echo "  Iterations: ${ITERATIONS} (warmup: ${WARMUP_STEPS})"
echo "  Batch: ${BATCH_SIZE} × ${ACC_STEPS} = $((BATCH_SIZE * ACC_STEPS))"
echo ""
echo "Testing hypothesis:"
echo "  half-eco0 (constant LR after warmup) should match/beat cos (decay to LR/10)"
echo "  Reason: ECO-0's variance from current grads doesn't accumulate like Adam"
echo "=========================================="

torchrun --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    --nproc_per_node=1 /export/home/keisufaj/optimization/ECO-CAGE-QCRI/src/main.py \
    --distributed-backend nccl \
    --dataset c4 \
    --datasets-dir ${DATASETS_DIR} \
    --model llama \
    --opt ${OPT} \
    --use-cage False \
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
    --scheduler ${SCHEDULER} \
    --w-quant ${W_QUANT} \
    --w-quant-kwargs "${W_QUANT_KWARGS}" \
    --a-quant NoQuantizer \
    --beta1 ${BETA1} \
    --beta2 ${BETA2}

echo "=========================================="
echo "Task ${SLURM_ARRAY_TASK_ID}/12 Complete!"
echo "=========================================="
echo "${QUANT_MODE} | ${SCHEDULER} | LR=${LR}"
echo ""
echo "Check WandB: ${WANDB_PROJECT} / ${WANDB_PREFIX}"
echo ""
echo "After all 12 jobs finish:"
echo "  1. Compare validation loss: cos vs half-eco0 at each (LR, QUANT_MODE)"
echo "  2. If half-eco0 wins/ties → validates constant LR hypothesis"
echo "  3. Test winner on 30M model"
echo "=========================================="
