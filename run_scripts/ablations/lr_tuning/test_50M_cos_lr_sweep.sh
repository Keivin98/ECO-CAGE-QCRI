#!/bin/bash -l
#SBATCH -J 50M_coslr
#SBATCH -o outs/50M_coslr_%A_%a.out
#SBATCH -e outs/50M_coslr_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --exclude=crirdchpxd001
#SBATCH --gres=gpu:H200_141GB:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=32000
#SBATCH -A H200_16GPUs
#SBATCH -q h200_qos_16_gpus
#SBATCH --array=1-4

# ==========================================================================
# 50M cos LR sweep — bracket the optimum for ECO-0 and ECO under cos so
# the paper figure has 3 LR points per scale (analogous to the half-eco0
# ECO-0 LR figure but for cos).
# ==========================================================================
# Tracking ID: EXP-50M-COS-LR-SWEEP
#
# Existing 50M cos data (single point per method):
#   ECO-0 cos LR=0.005 -> 23.73 PPL  (beats half-eco0 best 24.23 at same LR)
#   ECO   cos LR=0.004 -> 26.63 PPL  (loses to half-eco0 25.50 at same LR)
#
# This script adds 2 new LRs per method to form a 3-point figure cell.
#
# Bracketing strategy:
#   - ECO-0: symmetric around existing 0.005 -> add 0.004 and 0.006.
#     Cross-scale ECO-0 cos opt trend is monotonic down with scale
#     (30M=0.008, 100M=0.004, 300M=0.004), so 50M should land between
#     0.004 and 0.008. Symmetric ±25% bracket around 0.005 is right;
#     no signal that would justify reaching 0.007.
#   - ECO: symmetric around existing 0.004 -> add 0.003 and 0.005.
#     The 30M cos-at-top data point isn't strong enough to justify
#     pushing upward at 50M+; ECO has m+v buffers and cos already
#     amplifies EF late in training, so leaning low is safer.
#
# Tasks (4):
#   1: ECO-0 INT4 cos LR=0.004
#   2: ECO-0 INT4 cos LR=0.006
#   3: ECO   INT4 cos LR=0.003
#   4: ECO   INT4 cos LR=0.005
#
# Same config as EXP-50M-INT4-RERUN: 7 layers, 768 dim, 6 heads,
# 5B tokens (19,073 iter), batch 64x8 effective 512, seed 0.
# 2 GPUs each, ~3-4h per task.
# ==========================================================================

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cage

source /export/home/keisufaj/optimization/ECO-CAGE-QCRI/run_scripts/utils/setup_tmp_cleanup.sh

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((31600 + SLURM_ARRAY_TASK_ID))
export TORCH_COMPILE=0
export TORCH_COMPILE_DISABLE=1
export TORCHINDUCTOR_DISABLE_TRITON=1

export WANDB_ENTITY="keisufaj-hamad-bin-khalifa-university"
export WANDB_PROJECT="ECO0-50M-INT4-RERUN"

export DATASETS_DIR="/scratch/keisufaj_datasets_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
if [ -d "${DATASETS_DIR}/c4" ]; then
    echo "Dataset already exists, skipping copy"
else
    mkdir -p ${DATASETS_DIR}
    rsync -a --info=progress2 /export/home/keisufaj/optimization/ECO-CAGE-QCRI/datasets/ ${DATASETS_DIR}/
fi

MODEL_SIZE="50M"
export N_LAYER=7
export N_EMBD=768
export N_HEAD=6
export BATCH_SIZE=64
export ACC_STEPS=8
export SEQUENCE_LENGTH=512

TOKENS=5000000000
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
        LR=0.004
        ;;
    2)  # ECO-0 cos: bracket above existing
        METHOD="ECO0-4bit-INT4"
        OPT="eco0m-rooh"
        LR=0.006
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
echo "50M cos LR sweep — Task ${SLURM_ARRAY_TASK_ID}/4"
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
