#!/bin/bash -l
#SBATCH -J staged_lr
#SBATCH -o outs/staged_lr_%A_%a.out
#SBATCH -e outs/staged_lr_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --exclude=crirdchpxd001
#SBATCH --gres=gpu:H200_141GB:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=32000
#SBATCH -A H200_16GPUs
#SBATCH -q h200_qos_16_gpus
#SBATCH --array=1-6

# ==========================================================================
# Staged-EF ablation, LR follow-up for stages 1 & 2.
# ==========================================================================
# Tracking ID: EXP-30M-STAGED-EF-LR
#
# At fixed LR=0.008 we observed: stage 3 < stage 2 < stage 1 << stage 4
# (lower=better). Two surprises to disentangle from LR mismatch:
#   (a) stage 1 (master FP, no EF) is worse than stage 3 (production ECO-0).
#       LR=0.008 was tuned for stage 3's regime (quantized storage + EF).
#       Stage 1 likely wants a lower LR more typical of master-weight Adam.
#   (b) stage 2 (exact EF) is slightly worse than stage 3 (approx EF).
#       Hypothesis: exact form has higher per-step variance under stochastic
#       quantization noise. Lower LR may help by shrinking the step-noise.
#
# This script reruns ONLY stages 1 and 2 at three lower LRs derived from
# the original 0.008 by /1.5, /2, /4. Stage 3 already has 25.80 at 0.008
# (skip); stage 4 is catastrophic regardless (skip).
#
# Tasks (all cos, INT4 P99.5, 30M, seed 0, 3B tokens):
#   1: stage 1 (masterFP)   LR = 0.008/1.5 ≈ 0.006
#   2: stage 1 (masterFP)   LR = 0.008/2   = 0.004
#   3: stage 1 (masterFP)   LR = 0.008/4   = 0.002
#   4: stage 2 (exactEF)    LR = 0.008/1.5 ≈ 0.006
#   5: stage 2 (exactEF)    LR = 0.008/2   = 0.004
#   6: stage 2 (exactEF)    LR = 0.008/4   = 0.002
#
# 6 tasks × 2 GPUs = 12 GPUs (fits 16-GPU QoS easily). ~1h per task.
# ==========================================================================

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cage

source /export/home/keisufaj/optimization/ECO-CAGE-QCRI/run_scripts/utils/setup_tmp_cleanup.sh

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((32600 + SLURM_ARRAY_TASK_ID))
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
SCHEDULER="cos"
W_QUANT="Q99IntQuantizer"
W_QUANT_KWARGS='{"bits":4,"percentile":99.5}'
USE_CAGE="False"
OPT="eco0m-staged"

case ${SLURM_ARRAY_TASK_ID} in
    1)  STAGE=1; STAGE_NAME="masterFP_noEF"  ; LR=0.006 ; DIV=1.5 ;;
    2)  STAGE=1; STAGE_NAME="masterFP_noEF"  ; LR=0.004   ; DIV=2   ;;
    3)  STAGE=1; STAGE_NAME="masterFP_noEF"  ; LR=0.002   ; DIV=4   ;;
    4)  STAGE=2; STAGE_NAME="quant_exactEF"  ; LR=0.006 ; DIV=1.5 ;;
    5)  STAGE=2; STAGE_NAME="quant_exactEF"  ; LR=0.004   ; DIV=2   ;;
    6)  STAGE=2; STAGE_NAME="quant_exactEF"  ; LR=0.002   ; DIV=4   ;;
esac

WANDB_PREFIX="${MODEL_SIZE}-staged-stage=${STAGE}_${STAGE_NAME}-LR=${LR}_div${DIV}-ITER=${ITERATIONS}"

echo "=========================================="
echo "Staged EF LR follow-up — Task ${SLURM_ARRAY_TASK_ID}/6"
echo "  Stage:     ${STAGE} (${STAGE_NAME})"
echo "  LR:        ${LR}  (= 0.008 / ${DIV})"
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

echo "Task ${SLURM_ARRAY_TASK_ID}/6 complete: stage=${STAGE} LR=${LR}"

rm -rf ${DATASETS_DIR}
