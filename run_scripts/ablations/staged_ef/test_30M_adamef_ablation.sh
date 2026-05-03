#!/bin/bash -l
#SBATCH -J adamef
#SBATCH -o outs/adamef_%A_%a.out
#SBATCH -e outs/adamef_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --exclude=crirdchpxd001
#SBATCH --gres=gpu:H200_141GB:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=32000
#SBATCH -A H200_16GPUs
#SBATCH -q h200_qos_16_gpus
#SBATCH --array=1-12

# ==========================================================================
# AdamEF staged ablation at 30M — faithful test of Nikdan's exact-EF theorem.
# ==========================================================================
# Tracking ID: EXP-30M-ADAMEF
#
# This is the same 4-stage progression as the Adam0 staged ablation, but
# using STANDARD Adam (separate v EMA from g², not the Adam0 factorized
# variance). v is independent of the EF correction, so Nikdan eq 30 applies
# cleanly.
#
# Stage 1: master FP, no EF                      (storage: m + v + θ_FP)
# Stage 2: quantized + exact EF    (storage: m + v + θ̂ + e_prev)
# Stage 3: quantized + approx EF                 (storage: m + v + θ̂)
# Stage 4: quantized + no EF                     (storage: m + v + θ̂)
#
# Hypothesis (with proper init from Nikdan eq 31): Stage 1 ≡ Stage 2
# trajectory-identically. The Adam0 version of this ablation showed
# stage 2 NEVER matched stage 1 because Adam0's denom is contaminated by
# the EF (factorized v from m). With real Adam (v from g² EMA), denom is
# EF-free and the equivalence should hold.
#
# Three small learning rates around the standard Adam optimum at 30M
# (validated optimum LR=0.0012 for FP16 Adam): {0.0008, 0.0012, 0.0020}.
#
# Tasks GROUPED BY LR so the first 4 give a complete 4-stage comparison at
# LR=0.0008 even if later tasks are still queued:
#   1: stage 1 LR=0.0008    5: stage 1 LR=0.0012    9:  stage 1 LR=0.0020
#   2: stage 2 LR=0.0008    6: stage 2 LR=0.0012    10: stage 2 LR=0.0020
#   3: stage 3 LR=0.0008    7: stage 3 LR=0.0012    11: stage 3 LR=0.0020
#   4: stage 4 LR=0.0008    8: stage 4 LR=0.0012    12: stage 4 LR=0.0020
#
# 12 tasks × 2 GPUs = 24 GPU-demand vs 16-GPU QoS cap. First ~8 dispatch
# immediately (covers LR=0.0008 fully + half of LR=0.0012). Each task ~1h.
# Full sweep done in ~2-3h.
# ==========================================================================

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cage

source /export/home/keisufaj/optimization/ECO-CAGE-QCRI/run_scripts/utils/setup_tmp_cleanup.sh

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((33000 + SLURM_ARRAY_TASK_ID))
export TORCH_COMPILE=0
export TORCH_COMPILE_DISABLE=1
export TORCHINDUCTOR_DISABLE_TRITON=1

export WANDB_ENTITY="keisufaj-hamad-bin-khalifa-university"
export WANDB_PROJECT="ECO0-30M-ADAMEF"

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

BETA1=0.9
BETA2=0.95
SCHEDULER="cos"
W_QUANT="Q99IntQuantizer"
W_QUANT_KWARGS='{"bits":4,"percentile":99.5}'
USE_CAGE="False"
OPT="adam-ef"

# Map task -> (stage, LR), grouped by LR (ascending) so the first 4 tasks
# give a complete 4-stage comparison at LR=0.0008.
case ${SLURM_ARRAY_TASK_ID} in
    # ---- LR = 0.0008 (small LR, the regime where Adam0 stage 2 died) ----
    1)  STAGE=1; STAGE_NAME="masterFP_noEF"  ; LR=0.0008 ;;
    2)  STAGE=2; STAGE_NAME="quant_exactEF"  ; LR=0.0008 ;;
    3)  STAGE=3; STAGE_NAME="quant_approxEF" ; LR=0.0008 ;;
    4)  STAGE=4; STAGE_NAME="quant_noEF"     ; LR=0.0008 ;;
    # ---- LR = 0.0012 (validated standard-Adam optimum at 30M) ----
    5)  STAGE=1; STAGE_NAME="masterFP_noEF"  ; LR=0.0012 ;;
    6)  STAGE=2; STAGE_NAME="quant_exactEF"  ; LR=0.0012 ;;
    7)  STAGE=3; STAGE_NAME="quant_approxEF" ; LR=0.0012 ;;
    8)  STAGE=4; STAGE_NAME="quant_noEF"     ; LR=0.0012 ;;
    # ---- LR = 0.0020 (above optimum, regularization regime) ----
    9)  STAGE=1; STAGE_NAME="masterFP_noEF"  ; LR=0.0020 ;;
    10) STAGE=2; STAGE_NAME="quant_exactEF"  ; LR=0.0020 ;;
    11) STAGE=3; STAGE_NAME="quant_approxEF" ; LR=0.0020 ;;
    12) STAGE=4; STAGE_NAME="quant_noEF"     ; LR=0.0020 ;;
esac

EXTRA_FLAGS=""

WANDB_PREFIX="${MODEL_SIZE}-adamef-stage=${STAGE}_${STAGE_NAME}-LR=${LR}-ITER=${ITERATIONS}"

echo "=========================================="
echo "AdamEF ablation — Task ${SLURM_ARRAY_TASK_ID}/12"
echo "  Stage:    ${STAGE} (${STAGE_NAME})"
echo "  LR:       ${LR}"
echo "  Extra:    ${EXTRA_FLAGS:-none}"
echo "=========================================="

torchrun --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    --nproc_per_node=2 /export/home/keisufaj/optimization/ECO-CAGE-QCRI/src/main.py \
    --distributed-backend nccl \
    --dataset c4 \
    --datasets-dir ${DATASETS_DIR} \
    --model llama \
    --opt ${OPT} \
    --ablation-stage ${STAGE} \
    ${EXTRA_FLAGS} \
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

echo "Task ${SLURM_ARRAY_TASK_ID}/12 complete: stage=${STAGE} LR=${LR}"

rm -rf ${DATASETS_DIR}
