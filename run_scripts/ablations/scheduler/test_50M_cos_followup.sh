#!/bin/bash -l
#SBATCH -J 50M_cos
#SBATCH -o outs/50M_cos_%A_%a.out
#SBATCH -e outs/50M_cos_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --exclude=crirdchpxd001
#SBATCH --gres=gpu:H200_141GB:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=32000
#SBATCH -A H200_16GPUs
#SBATCH -q h200_qos_16_gpus
#SBATCH --array=1-2

# ==========================================================================
# 50M cos follow-up — re-run ECO-0 and ECO at the half-eco0 best LR
# but with the cos scheduler that beat half-eco0 at 30M.
# ==========================================================================
# Tracking ID: EXP-50M-COS-FOLLOWUP
#
# Why: completes the cross-scale cos vs half-eco0 comparison alongside
# the 100M and 300M follow-ups. 50M is the bottom of the scaling table.
#
# 50M half-eco0 reference (from EXP-50M-INT4-RERUN + LR follow-up):
#   ECO-0 best (half-eco0):  LR=0.005  (24.62 PPL provisional)
#   ECO   best (half-eco0):  LR=0.004  (LR follow-up confirmed below 0.005)
#
# This script re-runs each best LR under cos:
#   1: ECO-0 INT4 cos LR=0.005   ← priority 3 in cross-scale list
#   2: ECO   INT4 cos LR=0.004   ← priority 6 in cross-scale list
#
# Same config as EXP-50M-INT4-RERUN: 7 layers, 768 dim, 6 heads,
# 5B tokens (19,073 iter), batch 64x8 effective 512, seed 0.
# 2 GPUs each, ~20-24h per task.
# ==========================================================================

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cage

source /export/home/keisufaj/optimization/ECO-CAGE-QCRI/run_scripts/utils/setup_tmp_cleanup.sh

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((31500 + SLURM_ARRAY_TASK_ID))
export TORCH_COMPILE=0
export TORCH_COMPILE_DISABLE=1
export TORCHINDUCTOR_DISABLE_TRITON=1

export WANDB_ENTITY="keisufaj-hamad-bin-khalifa-university"
export WANDB_PROJECT="ECO0-50M-INT4-RERUN"

# Copy datasets to fast scratch space (job-specific)
export DATASETS_DIR="/scratch/keisufaj_datasets_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
if [ -d "${DATASETS_DIR}/c4" ]; then
    echo "Dataset already exists, skipping copy"
else
    mkdir -p ${DATASETS_DIR}
    rsync -a --info=progress2 /export/home/keisufaj/optimization/ECO-CAGE-QCRI/datasets/ ${DATASETS_DIR}/
fi

# 50M model config (matches EXP-50M-INT4-RERUN)
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

echo "Training for ${ITERATIONS} iterations (${TOKENS} tokens = 5B)"

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
        LR=0.005
        ;;
    2)  # ECO cos at half-eco0 best LR
        METHOD="ECO-4bit-INT4"
        OPT="eco"
        LR=0.004
        ;;
esac

WANDB_PREFIX="${MODEL_SIZE}-${METHOD}-sched=${SCHEDULER}-LR=${LR}-ITER=${ITERATIONS}"

echo "=========================================="
echo "50M cos follow-up — Task ${SLURM_ARRAY_TASK_ID}/2"
echo "  Method:    ${METHOD}"
echo "  Optimizer: ${OPT}"
echo "  Scheduler: ${SCHEDULER} (vs half-eco0 reference)"
echo "  LR:        ${LR} (matches half-eco0 best LR for this method/scale)"
echo "  Iter:      ${ITERATIONS} (${TOKENS} tokens)"
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

echo "Task ${SLURM_ARRAY_TASK_ID}/2 complete: ${METHOD} cos LR=${LR}"

rm -rf ${DATASETS_DIR}
