#!/bin/bash -l
#SBATCH -J 300M_int4
#SBATCH -o outs/300M_int4_%A_%a.out
#SBATCH -e outs/300M_int4_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --exclude=crirdchpxd001
#SBATCH --gres=gpu:H200_141GB:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=64000
#SBATCH -A H200_16GPUs
#SBATCH -q h200_qos_16_gpus
#SBATCH --array=7-7

# ==========================================================================
# 300M INT4 rerun (Table tab_300m: NEW)
# ==========================================================================
# Tracking ID: EXP-300M-INT4-RERUN
#
# Config: 12 layers, 1536 dim, 12 heads (~300M params)
# Batch: 64 x 8 = 512 effective batch (same as all prior scales).
# With nproc=4: backend redistributes to per-GPU batch=64, acc=2.
# Each GPU sees 64 samples per forward pass.
#
# Memory expectation (per H200, 141 GB):
#   - 100M with per-GPU batch=32: ~20 GB used
#   - 300M with per-GPU batch=64: estimated ~100-120 GB. Tight but should fit.
#   - Fallback if OOM: change to batch=32, acc=16 (per-GPU batch=32).
#
# Token budget: 15B (matches prior 300M FP4 attempt; allows comparison).
#   - tokens_per_iter = 64 * 8 * 512 = 262,144
#   - iter = 15e9 / 262,144 = 57,220
#   - Runtime estimate: ~11-14h per task on 4x H200.
#
# Matrix:
#   1: FP16 Adam   cos       LR=0.0003     (sqrt-scale from 100M's 0.0006)
#   2: CAGE INT4   cos       LR=0.0003
#   3: ECO   INT4  half-eco0 LR=0.004      (50M=0.004, 100M sweep ~0.005)
#   4: ECO   INT4  half-eco0 LR=0.005
#   5: ECO-0 INT4  half-eco0 LR=0.003      (50M=0.005, 100M=0.005)
#   6: ECO-0 INT4  half-eco0 LR=0.004
#   7: STE  INT4   cos       LR=0.0003     (added after main sweep; matches CAGE config)
#
# Tasks 1-6 already complete; current sbatch run only dispatches task 7
# (--array=7-7). To rerun the full sweep change to --array=1-7.
#
# Skipping ECO-0 LR=0.005+ in the main sweep: stability ceiling drops with
# scale (LR=0.006/0.007 unstable at 100M). 0.003 and 0.004 are safer at 300M.
# A separate follow-up script (test_300M_eco0_lr_followup.sh) tests LR=0.0045
# and LR=0.005 to bracket the ECO-0 optimum above 0.004.
#
# Total wall-clock per task: ~12h on 4x H200.
# ==========================================================================

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cage

source /export/home/keisufaj/optimization/ECO-CAGE-QCRI/run_scripts/utils/setup_tmp_cleanup.sh

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((30700 + SLURM_ARRAY_TASK_ID))
export TORCH_COMPILE=0
export TORCH_COMPILE_DISABLE=1
export TORCHINDUCTOR_DISABLE_TRITON=1

export WANDB_ENTITY="keisufaj-hamad-bin-khalifa-university"
export WANDB_PROJECT="ECO0-300M-INT4-RERUN"

# Copy datasets to fast scratch space (job-specific to avoid collisions)
export DATASETS_DIR="/scratch/keisufaj_datasets_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
if [ -d "${DATASETS_DIR}/c4" ]; then
    echo "Dataset already exists, skipping copy"
else
    mkdir -p ${DATASETS_DIR}
    rsync -a --info=progress2 /export/home/keisufaj/optimization/ECO-CAGE-QCRI/datasets/ ${DATASETS_DIR}/
fi

# 300M model config
MODEL_SIZE="300M"
export N_LAYER=12
export N_EMBD=1536
export N_HEAD=12
export BATCH_SIZE=64
export ACC_STEPS=8
export SEQUENCE_LENGTH=512

# 15B tokens (matches prior 300M FP4 attempt for direct comparison)
TOKENS=15000000000
TOKENS_PER_ITER=$((BATCH_SIZE * ACC_STEPS * SEQUENCE_LENGTH))
ITERATIONS=$((TOKENS / TOKENS_PER_ITER))
WARMUP_STEPS=$((ITERATIONS / 10))

echo "Training for ${ITERATIONS} iterations (${TOKENS} tokens = 15B)"

BETA1=0.9
BETA2=0.95

# Quantizer default for INT4 tasks (overridden for task 1)
W_QUANT="Q99IntQuantizer"
W_QUANT_KWARGS='{"bits":4,"percentile":99.5}'

# ========================================
# TASK SELECTION
# ========================================

case ${SLURM_ARRAY_TASK_ID} in
    1)  # FP16 Adam (no quantization)
        METHOD="FP16-Adam"
        OPT="adamw"
        USE_CAGE="False"
        SCHEDULER="cos"
        LR=0.0003
        W_QUANT="NoQuantizer"
        W_QUANT_KWARGS='{"bits":16}'
        ;;
    2)  # CAGE INT4
        METHOD="CAGE-4bit-INT4"
        OPT="adamw"
        USE_CAGE="True"
        SCHEDULER="cos"
        LR=0.0003
        ;;
    3)  # ECO LR sweep: 0.004
        METHOD="ECO-4bit-INT4"
        OPT="eco"
        USE_CAGE="False"
        SCHEDULER="half-eco0"
        LR=0.004
        ;;
    4)  # ECO LR sweep: 0.005
        METHOD="ECO-4bit-INT4"
        OPT="eco"
        USE_CAGE="False"
        SCHEDULER="half-eco0"
        LR=0.005
        ;;
    5)  # ECO-0 LR sweep: 0.003
        METHOD="ECO0-4bit-INT4"
        OPT="eco0m-rooh"
        USE_CAGE="False"
        SCHEDULER="half-eco0"
        LR=0.003
        ;;
    6)  # ECO-0 LR sweep: 0.004
        METHOD="ECO0-4bit-INT4"
        OPT="eco0m-rooh"
        USE_CAGE="False"
        SCHEDULER="half-eco0"
        LR=0.004
        ;;
    7)  # STE INT4 baseline (added after main sweep; fills the only missing 300M cell)
        METHOD="STE-4bit-INT4"
        OPT="adamw"
        USE_CAGE="False"
        SCHEDULER="cos"
        LR=0.0003
        ;;
esac

WANDB_PREFIX="${MODEL_SIZE}-${METHOD}-sched=${SCHEDULER}-LR=${LR}-ITER=${ITERATIONS}"

echo "=========================================="
echo "300M INT4 Rerun — Task ${SLURM_ARRAY_TASK_ID}/7"
echo "  Method:    ${METHOD}"
echo "  Optimizer: ${OPT}"
echo "  Scheduler: ${SCHEDULER}"
echo "  LR:        ${LR}"
echo "  Quantizer: ${W_QUANT}"
echo "  Iter:      ${ITERATIONS} (${TOKENS} tokens)"
echo "  Batch:     ${BATCH_SIZE} x ${ACC_STEPS} = $((BATCH_SIZE * ACC_STEPS)) (effective)"
echo "  GPUs:      4"
echo "=========================================="

torchrun --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    --nproc_per_node=4 /export/home/keisufaj/optimization/ECO-CAGE-QCRI/src/main.py \
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
    --scheduler ${SCHEDULER} \
    --w-quant ${W_QUANT} \
    --w-quant-kwargs "${W_QUANT_KWARGS}" \
    --a-quant NoQuantizer \
    --beta1 ${BETA1} \
    --beta2 ${BETA2}

echo "Task ${SLURM_ARRAY_TASK_ID}/6 complete: ${METHOD} + ${SCHEDULER} + LR=${LR}"

# Cleanup scratch
rm -rf ${DATASETS_DIR}
