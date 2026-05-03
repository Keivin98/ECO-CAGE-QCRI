#!/bin/bash -l
#SBATCH -J 100M_coslr
#SBATCH -o outs/100M_coslr_%A_%a.out
#SBATCH -e outs/100M_coslr_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --exclude=crirdchpxd001
#SBATCH --gres=gpu:H200_141GB:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=32000
#SBATCH -A H200_16GPUs
#SBATCH -q h200_qos_16_gpus
#SBATCH --array=1-4

# ==========================================================================
# 100M cos LR sweep — bracket the optimum for ECO-0 and ECO under cos so
# the paper figure has 3 LR points per scale.
# ==========================================================================
# Tracking ID: EXP-100M-COS-LR-SWEEP
#
# Existing 100M cos data (single point per method):
#   ECO-0 cos LR=0.004 -> 20.03 PPL  (beats half-eco0 best 20.65 by 0.62)
#   ECO   cos LR=0.004 -> 20.42 PPL  (beats half-eco0 best 22.24 by 1.82)
#
# This script adds 2 new LRs per method to form a 3-point figure cell.
#
# Bracketing strategy:
#   - ECO-0: symmetric around existing 0.004 -> add 0.003 and 0.005.
#     Mirrors the half-eco0 bracket at 100M ({0.003, 0.004, 0.005}).
#   - ECO: symmetric around existing 0.004 -> add 0.003 and 0.005.
#     Mirrors the half-eco0 100M ECO best LR; cos's late-training EF
#     amplification means lean-low is safer than pushing max_lr up.
#
# Tasks (4):
#   1: ECO-0 INT4 cos LR=0.003
#   2: ECO-0 INT4 cos LR=0.005
#   3: ECO   INT4 cos LR=0.003
#   4: ECO   INT4 cos LR=0.005
#
# Same config as EXP-100M-INT4-RERUN: 8 layers, 1024 dim, 8 heads,
# 10B tokens (38,146 iter), batch 32x16 effective 512, seed 0.
# 2 GPUs each, ~12-15h per task.
# ==========================================================================

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cage

source /export/home/keisufaj/optimization/ECO-CAGE-QCRI/run_scripts/utils/setup_tmp_cleanup.sh

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((31700 + SLURM_ARRAY_TASK_ID))
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
        LR=0.003
        ;;
    4)  # ECO cos: bracket above existing
        METHOD="ECO-4bit-INT4"
        OPT="eco"
        LR=0.005
        ;;
esac

WANDB_PREFIX="${MODEL_SIZE}-${METHOD}-sched=${SCHEDULER}-LR=${LR}-ITER=${ITERATIONS}"

echo "=========================================="
echo "100M cos LR sweep — Task ${SLURM_ARRAY_TASK_ID}/4"
echo "  Method: ${METHOD}  LR: ${LR}  Scheduler: ${SCHEDULER}"
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

echo "Task ${SLURM_ARRAY_TASK_ID}/4 complete: ${METHOD} cos LR=${LR}"

rm -rf ${DATASETS_DIR}
