#!/bin/bash -l
#SBATCH -J 300M_cos
#SBATCH -o outs/300M_cos_%A_%a.out
#SBATCH -e outs/300M_cos_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --exclude=crirdchpxd001
#SBATCH --gres=gpu:H200_141GB:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=64000
#SBATCH -A H200_16GPUs
#SBATCH -q h200_qos_16_gpus
#SBATCH --array=1-2

# ==========================================================================
# 300M cos follow-up — re-run ECO-0 and ECO at the half-eco0 best LR
# but with the cos scheduler that beat half-eco0 at 30M.
# ==========================================================================
# Tracking ID: EXP-300M-COS-FOLLOWUP
#
# Why: at 30M the cos scheduler beat half-eco0 for ECO-0 by ~1 PPL, and
# we need to know whether that crossover holds at the 300M headline scale
# (paper Table 3). 300M is the most paper-defining number — running cos
# here first dominates the priority.
#
# 300M half-eco0 reference (from EXP-300M-INT4-RERUN + LR follow-up):
#   ECO-0 LR=0.003 -> ...  LR=0.004 -> ...  (best so far at 0.004)
#   ECO   LR=0.004 -> ...  LR=0.005 -> ...  (best so far at 0.005)
#
# This script re-runs each best LR under cos:
#   1: ECO-0 INT4 cos LR=0.004    ← priority 1 (single most paper-defining cell)
#   2: ECO   INT4 cos LR=0.005    ← priority 4 in the cross-scale priority list
#
# Same config as EXP-300M-INT4-RERUN: 12 layers, 1536 dim, 12 heads,
# 15B tokens (57,220 iter), batch 64x8 effective 512 (per-GPU 64, acc=2),
# 4 GPUs each, ~12h per task.
#
# Total GPU footprint: 2 tasks x 4 GPUs = 8 GPUs (fits 16-GPU QoS alongside
# the 50M+100M cos follow-ups, 4x2=8, total 16).
# ==========================================================================

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cage

source /export/home/keisufaj/optimization/ECO-CAGE-QCRI/run_scripts/utils/setup_tmp_cleanup.sh

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((31300 + SLURM_ARRAY_TASK_ID))
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

# 300M model config (matches EXP-300M-INT4-RERUN)
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

echo "Training for ${ITERATIONS} iterations (${TOKENS} tokens = 15B)"

BETA1=0.9
BETA2=0.95

W_QUANT="Q99IntQuantizer"
W_QUANT_KWARGS='{"bits":4,"percentile":99.5}'
SCHEDULER="cos"
USE_CAGE="False"

case ${SLURM_ARRAY_TASK_ID} in
    1)  # ECO-0 cos at half-eco0 best LR
        METHOD="ECO0-4bit-INT4"
        OPT="eco0m-rooh"
        LR=0.004
        ;;
    2)  # ECO cos at half-eco0 best LR
        METHOD="ECO-4bit-INT4"
        OPT="eco"
        LR=0.005
        ;;
esac

WANDB_PREFIX="${MODEL_SIZE}-${METHOD}-sched=${SCHEDULER}-LR=${LR}-ITER=${ITERATIONS}"

echo "=========================================="
echo "300M cos follow-up — Task ${SLURM_ARRAY_TASK_ID}/2"
echo "  Method:    ${METHOD}"
echo "  Optimizer: ${OPT}"
echo "  Scheduler: ${SCHEDULER} (vs half-eco0 reference)"
echo "  LR:        ${LR} (matches half-eco0 best LR for this method/scale)"
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

echo "Task ${SLURM_ARRAY_TASK_ID}/2 complete: ${METHOD} cos LR=${LR}"

rm -rf ${DATASETS_DIR}
