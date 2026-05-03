#!/bin/bash -l
#SBATCH -J 300M_coslr
#SBATCH -o outs/300M_coslr_%A_%a.out
#SBATCH -e outs/300M_coslr_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --exclude=crirdchpxd001
#SBATCH --gres=gpu:H200_141GB:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=64000
#SBATCH -A H200_16GPUs
#SBATCH -q h200_qos_16_gpus
#SBATCH --array=1-4

# ==========================================================================
# 300M cos LR sweep — bracket the optimum for ECO-0 and ECO under cos at
# the headline scale. Most paper-defining cell.
# ==========================================================================
# Tracking ID: EXP-300M-COS-LR-SWEEP
#
# Existing 300M cos data (single point per method):
#   ECO-0 cos LR=0.004 -> 15.96 PPL  (beats half-eco0 best 17.01 by 1.05)
#   ECO   cos LR=0.005 -> 16.69 PPL  (beats half-eco0 best 18.25 by 1.56)
#
# This script adds 2 new LRs per method to form a 3-point figure cell.
#
# Bracketing strategy:
#   - ECO-0: symmetric around existing 0.004 -> add 0.003 and 0.005.
#     Mirrors half-eco0 bracket at 300M ({0.003, 0.004}).
#   - ECO: symmetric around existing 0.005 -> add 0.004 and 0.006.
#     ECO has m+v buffers and cos amplifies EF late in training, so
#     leaning low/symmetric is safer than pushing max_lr up.
#
# Tasks (4):
#   1: ECO-0 INT4 cos LR=0.003
#   2: ECO-0 INT4 cos LR=0.005
#   3: ECO   INT4 cos LR=0.004
#   4: ECO   INT4 cos LR=0.006
#
# Same config as EXP-300M-INT4-RERUN: 12 layers, 1536 dim, 12 heads,
# 15B tokens (57,220 iter), batch 64x8 effective 512, seed 0.
# 4 GPUs each, ~12h per task. Total footprint: 16 GPUs (fills the QoS).
# ==========================================================================

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cage

source /export/home/keisufaj/optimization/ECO-CAGE-QCRI/run_scripts/utils/setup_tmp_cleanup.sh

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((31800 + SLURM_ARRAY_TASK_ID))
export TORCH_COMPILE=0
export TORCH_COMPILE_DISABLE=1
export TORCHINDUCTOR_DISABLE_TRITON=1

export WANDB_ENTITY="keisufaj-hamad-bin-khalifa-university"
export WANDB_PROJECT="ECO0-300M-INT4-RERUN"

export DATASETS_DIR="/scratch/keisufaj_datasets_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
if [ -d "${DATASETS_DIR}/c4" ]; then
    echo "Dataset already exists, skipping copy"
else
    mkdir -p ${DATASETS_DIR}
    rsync -a --info=progress2 /export/home/keisufaj/optimization/ECO-CAGE-QCRI/datasets/ ${DATASETS_DIR}/
fi

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

W_QUANT="Q99IntQuantizer"
W_QUANT_KWARGS='{"bits":4,"percentile":99.5}'
SCHEDULER="cos"
USE_CAGE="False"

case ${SLURM_ARRAY_TASK_ID} in
    1)  # ECO-0 cos: bracket below existing
        METHOD="ECO0-4bit-INT4"
        OPT="eco0m-rooh"
        LR=0.003
        ;;
    2)  # ECO-0 cos: bracket above existing
        METHOD="ECO0-4bit-INT4"
        OPT="eco0m-rooh"
        LR=0.005
        ;;
    3)  # ECO cos: bracket below existing
        METHOD="ECO-4bit-INT4"
        OPT="eco"
        LR=0.004
        ;;
    4)  # ECO cos: bracket above existing
        METHOD="ECO-4bit-INT4"
        OPT="eco"
        LR=0.006
        ;;
esac

WANDB_PREFIX="${MODEL_SIZE}-${METHOD}-sched=${SCHEDULER}-LR=${LR}-ITER=${ITERATIONS}"

echo "=========================================="
echo "300M cos LR sweep — Task ${SLURM_ARRAY_TASK_ID}/4"
echo "  Method:    ${METHOD}"
echo "  Optimizer: ${OPT}"
echo "  Scheduler: ${SCHEDULER}"
echo "  LR:        ${LR}"
echo "  Iter:      ${ITERATIONS} (${TOKENS} tokens)"
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

echo "Task ${SLURM_ARRAY_TASK_ID}/4 complete: ${METHOD} cos LR=${LR}"

rm -rf ${DATASETS_DIR}
