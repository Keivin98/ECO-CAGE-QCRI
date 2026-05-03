#!/bin/bash -l
#SBATCH -J staged_ef
#SBATCH -o outs/staged_ef_%A_%a.out
#SBATCH -e outs/staged_ef_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --exclude=crirdchpxd001
#SBATCH --gres=gpu:H200_141GB:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=32000
#SBATCH -A H200_16GPUs
#SBATCH -q h200_qos_16_gpus
#SBATCH --array=1-4

# ==========================================================================
# Staged error-feedback ablation at 30M.
# Single optimizer class (`Adam0Staged`); stage selects post-update behavior.
# ==========================================================================
# Tracking ID: EXP-30M-STAGED-EF
#
# All four stages share the same Adam0 mechanism (no v buffer; factorized
# variance from the current first moment) and STE forward. They differ only
# in storage and EF policy:
#
#   stage 1: Master FP θ̃, no EF              -> no quantization residual exists
#   stage 2: Quantized θ̂ + exact two-term EF -> "ECO-exact" (e_prev buffer)
#   stage 3: Quantized θ̂ + approximate EF    -> production ECO-0
#   stage 4: Quantized θ̂ + no EF             -> naive (residual lost each step)
#
# Hypothesis (PPL, lower=better): 1 ≈ 2 < 3 < 4.
#   1 vs 4: how much EF actually buys
#   2 vs 3: cost of the e_t ≈ e_{t+1} approximation
#   1 vs 2: theoretical claim that exact EF matches master-weight trajectory
#
# Same config as EXP-30M-INT4-RERUN: 6 layers / 640 dim / 5 heads,
# 3B tokens (11,444 iter), batch 64x8 effective 512, INT4 P99.5, seed 0.
# LR=0.008 (the validated ECO-0 cos optimum at 30M; common across stages so
# the comparison is apples-to-apples on storage/EF, not LR tuning).
# 4 tasks × 2 GPUs = 8 GPUs. ~1h per task.
# ==========================================================================

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cage

source /export/home/keisufaj/optimization/ECO-CAGE-QCRI/run_scripts/utils/setup_tmp_cleanup.sh

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((32500 + SLURM_ARRAY_TASK_ID))
export TORCH_COMPILE=0
export TORCH_COMPILE_DISABLE=1
export TORCHINDUCTOR_DISABLE_TRITON=1

export WANDB_ENTITY="keisufaj-hamad-bin-khalifa-university"
export WANDB_PROJECT="ECO0-30M-STAGED-EF"

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
LR=0.008
SCHEDULER="cos"
W_QUANT="Q99IntQuantizer"
W_QUANT_KWARGS='{"bits":4,"percentile":99.5}'
USE_CAGE="False"
OPT="eco0m-staged"

case ${SLURM_ARRAY_TASK_ID} in
    1)  STAGE=1; STAGE_NAME="masterFP_noEF" ;;
    2)  STAGE=2; STAGE_NAME="quant_exactEF" ;;
    3)  STAGE=3; STAGE_NAME="quant_approxEF" ;;
    4)  STAGE=4; STAGE_NAME="quant_noEF" ;;
esac

WANDB_PREFIX="${MODEL_SIZE}-staged-stage=${STAGE}_${STAGE_NAME}-LR=${LR}-ITER=${ITERATIONS}"

echo "=========================================="
echo "Staged EF ablation — Task ${SLURM_ARRAY_TASK_ID}/4"
echo "  Stage:     ${STAGE} (${STAGE_NAME})"
echo "  Optimizer: ${OPT}"
echo "  LR:        ${LR}"
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
    --ablation-stage ${STAGE} \
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

echo "Task ${SLURM_ARRAY_TASK_ID}/4 complete: stage=${STAGE} (${STAGE_NAME})"

rm -rf ${DATASETS_DIR}
