#!/bin/bash -l
#SBATCH -J 50M_int4_lr
#SBATCH -o outs/50M_int4_lr_%A_%a.out
#SBATCH -e outs/50M_int4_lr_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --exclude=crirdchpxd001
#SBATCH --gres=gpu:H200_141GB:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=32000
#SBATCH -A H200_16GPUs
#SBATCH -q h200_qos_16_gpus
#SBATCH --array=1-4

# ==========================================================================
# 50M INT4 LR follow-up — bracket ECO / ECO-0 optima from below at 50M
# ==========================================================================
# Paper ref: Table 2 (tab_50m_scaling.tex) — pins int-perp headlines
# Tracking ID: EXP-50M-INT4-LR-FOLLOWUP
#
# 30M showed ECO/ECO-0 optima at the LOW end of their tested LR grid. 50M
# scaling usually requires LR slightly lower than 30M (by sqrt(scale) or so),
# so we bracket below the 50M main sweep.
#
# Existing 50M main sweep LRs (already running):
#   - ECO   : {0.005, 0.00625, 0.008}
#   - ECO-0 : {0.006, 0.008, 0.010}
#
# Follow-up (this script):
#   1: ECO   INT4 + half-eco0 + LR=0.003
#   2: ECO   INT4 + half-eco0 + LR=0.004
#   3: ECO-0 INT4 + half-eco0 + LR=0.004
#   4: ECO-0 INT4 + half-eco0 + LR=0.005
#
# Same config as EXP-50M-INT4-RERUN: 19,073 iter (5B tokens), INT4 P99.5, seed 0.
# ==========================================================================

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cage

source /export/home/keisufaj/optimization/ECO-CAGE-QCRI/run_scripts/utils/setup_tmp_cleanup.sh

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((30400 + SLURM_ARRAY_TASK_ID))
export TORCH_COMPILE=0
export TORCH_COMPILE_DISABLE=1
export TORCHINDUCTOR_DISABLE_TRITON=1

export WANDB_ENTITY="keisufaj-hamad-bin-khalifa-university"
export WANDB_PROJECT="ECO0-50M-INT4-RERUN"

# Copy datasets to fast scratch space (job-specific to avoid collisions)
export DATASETS_DIR="/scratch/keisufaj_datasets_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
if [ -d "${DATASETS_DIR}/c4" ]; then
    echo "Dataset already exists, skipping copy"
else
    mkdir -p ${DATASETS_DIR}
    rsync -a --info=progress2 /export/home/keisufaj/optimization/ECO-CAGE-QCRI/datasets/ ${DATASETS_DIR}/
fi

# 50M model config (matches EXP-50M-INT4-RERUN exactly)
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
SCHEDULER="half-eco0"
USE_CAGE="False"

case ${SLURM_ARRAY_TASK_ID} in
    1)
        METHOD="ECO-4bit-INT4"
        OPT="eco"
        LR=0.003
        ;;
    2)
        METHOD="ECO-4bit-INT4"
        OPT="eco"
        LR=0.004
        ;;
    3)
        METHOD="ECO0-4bit-INT4"
        OPT="eco0m-rooh"
        LR=0.004
        ;;
    4)
        METHOD="ECO0-4bit-INT4"
        OPT="eco0m-rooh"
        LR=0.005
        ;;
esac

WANDB_PREFIX="${MODEL_SIZE}-${METHOD}-sched=${SCHEDULER}-LR=${LR}-ITER=${ITERATIONS}"

echo "=========================================="
echo "50M INT4 LR follow-up — Task ${SLURM_ARRAY_TASK_ID}/4"
echo "  Method: ${METHOD} | LR: ${LR}"
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

echo "Task ${SLURM_ARRAY_TASK_ID}/4 complete: ${METHOD} + LR=${LR}"

rm -rf ${DATASETS_DIR}
