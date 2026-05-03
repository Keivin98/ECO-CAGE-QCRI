#!/bin/bash -l
#SBATCH -J 100M_cos
#SBATCH -o outs/100M_cos_%A_%a.out
#SBATCH -e outs/100M_cos_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --exclude=crirdchpxd001
#SBATCH --gres=gpu:H200_141GB:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=32000
#SBATCH -A H200_16GPUs
#SBATCH -q h200_qos_16_gpus
#SBATCH --array=1-2

# ==========================================================================
# 100M cos follow-up — re-run ECO-0 and ECO at the half-eco0 best LR
# but with the cos scheduler that beat half-eco0 at 30M.
# ==========================================================================
# Tracking ID: EXP-100M-COS-FOLLOWUP
#
# Why: confirms whether the 30M cos > half-eco0 crossover holds across
# scales. 100M is the mid-scale data point in the cross-scale story.
#
# 100M half-eco0 reference (from EXP-100M-INT4-RERUN):
#   ECO-0 best (half-eco0):  LR=0.004
#   ECO   best (half-eco0):  LR=0.004
#
# This script re-runs each best LR under cos:
#   1: ECO-0 INT4 cos LR=0.004   ← priority 2 in cross-scale list
#   2: ECO   INT4 cos LR=0.004   ← priority 5 in cross-scale list
#
# Same config as EXP-100M-INT4-RERUN: 8 layers, 1024 dim, 8 heads,
# 10B tokens (38,146 iter), batch 32x16 effective 512, seed 0.
# 2 GPUs each, ~12-15h per task.
# ==========================================================================

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cage

source /export/home/keisufaj/optimization/ECO-CAGE-QCRI/run_scripts/utils/setup_tmp_cleanup.sh

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((31400 + SLURM_ARRAY_TASK_ID))
export TORCH_COMPILE=0
export TORCH_COMPILE_DISABLE=1
export TORCHINDUCTOR_DISABLE_TRITON=1

export WANDB_ENTITY="keisufaj-hamad-bin-khalifa-university"
export WANDB_PROJECT="ECO0-100M-INT4-RERUN"

# Copy datasets to fast scratch space (job-specific)
export DATASETS_DIR="/scratch/keisufaj_datasets_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
if [ -d "${DATASETS_DIR}/c4" ]; then
    echo "Dataset already exists, skipping copy"
else
    mkdir -p ${DATASETS_DIR}
    rsync -a --info=progress2 /export/home/keisufaj/optimization/ECO-CAGE-QCRI/datasets/ ${DATASETS_DIR}/
fi

# 100M model config (matches EXP-100M-INT4-RERUN)
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

echo "Training for ${ITERATIONS} iterations (${TOKENS} tokens = 10B)"

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
        LR=0.004
        ;;
esac

WANDB_PREFIX="${MODEL_SIZE}-${METHOD}-sched=${SCHEDULER}-LR=${LR}-ITER=${ITERATIONS}"

echo "=========================================="
echo "100M cos follow-up — Task ${SLURM_ARRAY_TASK_ID}/2"
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
