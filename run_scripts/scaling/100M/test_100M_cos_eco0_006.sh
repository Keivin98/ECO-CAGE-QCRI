#!/bin/bash -l
#SBATCH -J 100M_cos006
#SBATCH -o outs/100M_cos006_%A_%a.out
#SBATCH -e outs/100M_cos006_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --exclude=crirdchpxd001
#SBATCH --gres=gpu:H200_141GB:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=32000
#SBATCH -A H200_16GPUs
#SBATCH -q h200_qos_16_gpus
#SBATCH --array=1-1

# ==========================================================================
# 100M ECO-0 cos LR=0.006 — close the upper bracket.
# ==========================================================================
# Tracking ID: EXP-100M-COS-ECO0-006
#
# Existing 100M cos ECO-0 sweep (bracketed below):
#   LR=0.003 -> 21.88 PPL
#   LR=0.004 -> 20.03 PPL
#   LR=0.005 -> 19.37 PPL  (current best, bracketed below only)
#
# This adds the upper bracket point to confirm 0.005 is the optimum.
#   Task 1: ECO-0 INT4 cos LR=0.006
#
# Same config as EXP-100M-INT4-RERUN: 8 layers, 1024 dim, 8 heads,
# 10B tokens (38,146 iter), batch 32x16 effective 512, seed 0.
# 2 GPUs, ~7-8h.
# ==========================================================================

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cage

source /export/home/keisufaj/optimization/ECO-CAGE-QCRI/run_scripts/utils/setup_tmp_cleanup.sh

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=31750
export TORCH_COMPILE=0
export TORCH_COMPILE_DISABLE=1
export TORCHINDUCTOR_DISABLE_TRITON=1

export WANDB_ENTITY="keisufaj-hamad-bin-khalifa-university"
export WANDB_PROJECT="ECO0-100M-INT4-RERUN"

export DATASETS_DIR="/scratch/keisufaj_datasets_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
if [ -d "${DATASETS_DIR}/c4" ]; then
    echo "Dataset already exists, skipping copy"
else
    mkdir -p ${DATASETS_DIR}
    rsync -a --info=progress2 /export/home/keisufaj/optimization/ECO-CAGE-QCRI/datasets/ ${DATASETS_DIR}/
fi

MODEL_SIZE="100M"
export N_LAYER=8
export N_EMBD=1024
export N_HEAD=8
export BATCH_SIZE=32
export ACC_STEPS=16
export SEQUENCE_LENGTH=512

TOKENS=10000000000
TOKENS_PER_ITER=$((BATCH_SIZE * ACC_STEPS * SEQUENCE_LENGTH))
ITERATIONS=$((TOKENS / TOKENS_PER_ITER))
WARMUP_STEPS=$((ITERATIONS / 10))

BETA1=0.9
BETA2=0.95

W_QUANT="Q99IntQuantizer"
W_QUANT_KWARGS='{"bits":4,"percentile":99.5}'
SCHEDULER="cos"
USE_CAGE="False"

METHOD="ECO0-4bit-INT4"
OPT="eco0m-rooh"
LR=0.006

WANDB_PREFIX="${MODEL_SIZE}-${METHOD}-sched=${SCHEDULER}-LR=${LR}-ITER=${ITERATIONS}"

echo "=========================================="
echo "100M ECO-0 cos LR=0.006 — upper bracket closer"
echo "  Method: ${METHOD}  LR: ${LR}  Scheduler: ${SCHEDULER}"
echo "  Existing bracket: 0.003 -> 21.88, 0.004 -> 20.03, 0.005 -> 19.37 (current best)"
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
    --scheduler ${SCHEDULER} \
    --w-quant ${W_QUANT} \
    --w-quant-kwargs "${W_QUANT_KWARGS}" \
    --a-quant NoQuantizer \
    --beta1 ${BETA1} \
    --beta2 ${BETA2}

echo "100M ECO-0 cos LR=0.006 complete"

rm -rf ${DATASETS_DIR}
