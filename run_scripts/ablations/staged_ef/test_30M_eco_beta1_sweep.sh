#!/bin/bash -l
#SBATCH -J eco_beta1
#SBATCH -o outs/eco_beta1_%A_%a.out
#SBATCH -e outs/eco_beta1_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --exclude=crirdchpxd001
#SBATCH --gres=gpu:H200_141GB:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=32000
#SBATCH -A H200_16GPUs
#SBATCH -q h200_qos_16_gpus
#SBATCH --array=1-4

# ==========================================================================
# 30M ECO β1 sweep at the validated best LR.
# ==========================================================================
# Tracking ID: EXP-30M-ECO-BETA1
#
# Why: ECO's error-feedback formula has a `(1 - 1/β1)` factor, which means
# β1 directly controls how aggressively the EF correction enters m. We want
# to see how performance / stability depends on β1, particularly toward the
# low-momentum end where the formula's behavior changes:
#
#   β1 → 1   : correction → 0 (no EF, just standard Adam)
#   β1 = 0.9 : (1 - 1/β1) = -0.111  ← production setting
#   β1 = 0.8 : (1 - 1/β1) = -0.250
#   β1 = 0.5 : (1 - 1/β1) = -1.000
#   β1 → 0   : correction → -∞·e   (formula diverges; ECO validates β1>0)
#
# β1=0 is REJECTED by ECO's __init__ (`raise ValueError` if β1 ≤ 0). The
# closest stable "near-zero" we can pass is β1=0.05, which gives
# (1 - 1/0.05) = -19. Already aggressive but finite.
#
# Tasks (all cos, INT4 P99.5, 30M, seed 0, 3B tokens, LR=0.003):
#   1: β1 = 0.05  (near-zero substitute for 0; expect possibly unstable)
#   2: β1 = 0.5
#   3: β1 = 0.8
#   4: β1 = 0.9   (production reference: should reproduce the 28.75 PPL)
#
# LR=0.003 is the validated 30M ECO best from prior cos sweeps (matches the
# half-eco0 best LR; cos at 0.003 is a close match). Change LR if you want
# to test β1 sensitivity at a different operating point.
#
# 4 tasks × 2 GPUs = 8 GPUs. ~1h per task. Total ~1h wall clock (parallel).
# ==========================================================================

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cage

source /export/home/keisufaj/optimization/ECO-CAGE-QCRI/run_scripts/utils/setup_tmp_cleanup.sh

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((33500 + SLURM_ARRAY_TASK_ID))
export TORCH_COMPILE=0
export TORCH_COMPILE_DISABLE=1
export TORCHINDUCTOR_DISABLE_TRITON=1

export WANDB_ENTITY="keisufaj-hamad-bin-khalifa-university"
export WANDB_PROJECT="ECO0-30M-ECO-BETA1"

export DATASETS_DIR="/scratch/keisufaj_datasets_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
if [ -d "${DATASETS_DIR}/c4" ]; then
    echo "Dataset already exists, skipping copy"
else
    mkdir -p ${DATASETS_DIR}
    rsync -a --info=progress2 /export/home/keisufaj/optimization/ECO-CAGE-QCRI/datasets/ ${DATASETS_DIR}/
fi

MODEL_SIZE="30M"
export N_LAYER=6
export N_EMBD=640
export N_HEAD=5
export BATCH_SIZE=64
export ACC_STEPS=8
export SEQUENCE_LENGTH=512

TOKENS=3000000000
TOKENS_PER_ITER=$((BATCH_SIZE * ACC_STEPS * SEQUENCE_LENGTH))
ITERATIONS=$((TOKENS / TOKENS_PER_ITER))
WARMUP_STEPS=$((ITERATIONS / 10))

LR=0.003
BETA2=0.95
SCHEDULER="cos"
W_QUANT="Q99IntQuantizer"
W_QUANT_KWARGS='{"bits":4,"percentile":99.5}'
USE_CAGE="False"
OPT="eco"
METHOD="ECO-4bit-INT4"

# ECO requires β1 > 0; β1=0.05 stands in for "essentially no momentum".
case ${SLURM_ARRAY_TASK_ID} in
    1)  BETA1=0.05 ; LABEL="beta1=0.05_near_zero" ;;
    2)  BETA1=0.5  ; LABEL="beta1=0.5" ;;
    3)  BETA1=0.8  ; LABEL="beta1=0.8" ;;
    4)  BETA1=0.9  ; LABEL="beta1=0.9_production" ;;
esac

WANDB_PREFIX="${MODEL_SIZE}-${METHOD}-${LABEL}-LR=${LR}-ITER=${ITERATIONS}"

echo "=========================================="
echo "30M ECO β1 sweep — Task ${SLURM_ARRAY_TASK_ID}/4"
echo "  Method:    ${METHOD}"
echo "  β1:        ${BETA1}  (${LABEL})"
echo "  LR:        ${LR}  (validated 30M ECO best)"
echo "  Scheduler: ${SCHEDULER}"
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

echo "Task ${SLURM_ARRAY_TASK_ID}/4 complete: ${METHOD} β1=${BETA1} LR=${LR}"

rm -rf ${DATASETS_DIR}
