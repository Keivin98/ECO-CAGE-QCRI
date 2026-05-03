#!/bin/bash -l
#SBATCH -J 300M_eco0_lr
#SBATCH -o outs/300M_eco0_lr_%A_%a.out
#SBATCH -e outs/300M_eco0_lr_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --exclude=crirdchpxd001
#SBATCH --gres=gpu:H200_141GB:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=64000
#SBATCH -A H200_16GPUs
#SBATCH -q h200_qos_16_gpus
#SBATCH --array=1-2

# ==========================================================================
# 300M ECO-0 LR follow-up — bracket the optimum above 0.004
# ==========================================================================
# Tracking ID: EXP-300M-ECO0-LR-FOLLOWUP
#
# Current 300M ECO-0 sweep (from EXP-300M-INT4-RERUN):
#   LR=0.003 → 17.54 PPL
#   LR=0.004 → 17.01 PPL  ← best, NOT bracketed above
#
# The 0.003→0.004 improvement (-0.53 PPL) suggests the curve is still
# descending past 0.004. We test LR=0.0045 and LR=0.005 to either:
#   (a) find a better number than 17.01, or
#   (b) cleanly bracket 0.004 as the optimum (both worse), or
#   (c) hit the 300M stability ceiling (LR=0.005 spikes/diverges, useful
#       Limitations data point).
#
# Same config as EXP-300M-INT4-RERUN: 12L 1536D 12H, 15B tokens, INT4 P99.5,
# half-eco0 scheduler, batch 64x8 effective 512, 4 GPUs/task.
# ==========================================================================

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cage

source /export/home/keisufaj/optimization/ECO-CAGE-QCRI/run_scripts/utils/setup_tmp_cleanup.sh

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((30900 + SLURM_ARRAY_TASK_ID))
export TORCH_COMPILE=0
export TORCH_COMPILE_DISABLE=1
export TORCHINDUCTOR_DISABLE_TRITON=1

export WANDB_ENTITY="keisufaj-hamad-bin-khalifa-university"
export WANDB_PROJECT="ECO0-300M-INT4-RERUN"

# Copy datasets to fast scratch space (job-specific)
export DATASETS_DIR="/scratch/keisufaj_datasets_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
if [ -d "${DATASETS_DIR}/c4" ]; then
    echo "Dataset already exists, skipping copy"
else
    mkdir -p ${DATASETS_DIR}
    rsync -a --info=progress2 /export/home/keisufaj/optimization/ECO-CAGE-QCRI/datasets/ ${DATASETS_DIR}/
fi

# 300M model config (matches EXP-300M-INT4-RERUN exactly)
MODEL_SIZE="300M"
export N_LAYER=12
export N_EMBD=1536
export N_HEAD=12
export BATCH_SIZE=64
export ACC_STEPS=8
export SEQUENCE_LENGTH=512

TOKENS=15000000000
TOKENS_PER_ITER=$((BATCH_SIZE * ACC_STEPS * SEQUENCE_LENGTH))
ITERATIONS=$((TOKENS / TOKENS_PER_ITER))
WARMUP_STEPS=$((ITERATIONS / 10))

BETA1=0.9
BETA2=0.95

# Common config (all tasks)
W_QUANT="Q99IntQuantizer"
W_QUANT_KWARGS='{"bits":4,"percentile":99.5}'
SCHEDULER="half-eco0"
USE_CAGE="False"
OPT="eco0m-rooh"
METHOD="ECO0-4bit-INT4"

case ${SLURM_ARRAY_TASK_ID} in
    1)  # LR=0.0045 — between known good (0.004) and risky (0.005)
        LR=0.0045
        ;;
    2)  # LR=0.005 — tests if optimum extends here AND stability ceiling
        LR=0.005
        ;;
esac

WANDB_PREFIX="${MODEL_SIZE}-${METHOD}-sched=${SCHEDULER}-LR=${LR}-ITER=${ITERATIONS}"

echo "=========================================="
echo "300M ECO-0 LR Follow-up — Task ${SLURM_ARRAY_TASK_ID}/2"
echo "  Method:    ${METHOD}"
echo "  Scheduler: ${SCHEDULER}"
echo "  LR:        ${LR}"
echo "  Goal:      bracket above current 300M best (LR=0.004 → 17.01 PPL)"
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

echo "Task ${SLURM_ARRAY_TASK_ID}/2 complete: ${METHOD} half-eco0 LR=${LR}"

rm -rf ${DATASETS_DIR}
