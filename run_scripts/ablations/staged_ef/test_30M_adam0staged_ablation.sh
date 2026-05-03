#!/bin/bash -l
#SBATCH -J adam0_staged
#SBATCH -o outs/adam0_staged_%A_%a.out
#SBATCH -e outs/adam0_staged_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --exclude=crirdchpxd001
#SBATCH --gres=gpu:H200_141GB:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=32000
#SBATCH -A H200_16GPUs
#SBATCH -q h200_qos_16_gpus
#SBATCH --array=1-12

# ==========================================================================
# Adam0Staged 4-stage ablation at 30M (clean residual-buffer version).
# ==========================================================================
# Tracking ID: EXP-30M-ADAM0-STAGED
#
# Same 4-stage progression as the AdamEF ablation, but using the Adam0
# mechanism (factorized v from current m, no separate v EMA buffer) instead
# of standard Adam. This is the "Adam0 + staged" version:
#
#   stage 1: master FP, no EF                      (storage: m + θ_FP)
#   stage 2: quantized + residual buffer           (storage: m + θ̂ + residual)
#   stage 3: quantized + ECO approx EF             (= production ECO-0)
#   stage 4: quantized + no EF                     (storage: m + θ̂)
#
# Differences vs AdamEF ablation at 30M:
#   - Adam0 mechanism (denom from rank-1 factorized m², no v EMA)
#   - LRs scaled to Adam0's regime: ~0.008 instead of ~0.0012
#     (Adam0 needs higher LR because m has no (1-β1) factor)
#
# LRs grouped — first 4 tasks give a complete 4-stage comparison at the
# validated production LR=0.008:
#
#   1: stage 1 LR=0.006     5: stage 1 LR=0.008     9:  stage 1 LR=0.010
#   2: stage 2 LR=0.006     6: stage 2 LR=0.008     10: stage 2 LR=0.010
#   3: stage 3 LR=0.006     7: stage 3 LR=0.008     11: stage 3 LR=0.010
#   4: stage 4 LR=0.006     8: stage 4 LR=0.008     12: stage 4 LR=0.010
#
# 12 tasks × 2 GPUs = 24 GPU-demand vs 16-GPU QoS cap. First ~8 dispatch
# immediately. Each task ~1h.
# ==========================================================================

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cage

source /export/home/keisufaj/optimization/ECO-CAGE-QCRI/run_scripts/utils/setup_tmp_cleanup.sh

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((34000 + SLURM_ARRAY_TASK_ID))
export TORCH_COMPILE=0
export TORCH_COMPILE_DISABLE=1
export TORCHINDUCTOR_DISABLE_TRITON=1

export WANDB_ENTITY="keisufaj-hamad-bin-khalifa-university"
export WANDB_PROJECT="ECO0-30M-ADAM0-STAGED"

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
    # ---- LR = 0.006 (below ECO-0 optimum) ----
    1)  STAGE=1; STAGE_NAME="masterFP_noEF"  ; LR=0.006 ;;
    2)  STAGE=2; STAGE_NAME="quant_residual" ; LR=0.006 ;;
    3)  STAGE=3; STAGE_NAME="quant_approxEF" ; LR=0.006 ;;
    4)  STAGE=4; STAGE_NAME="quant_noEF"     ; LR=0.006 ;;
    # ---- LR = 0.008 (validated production ECO-0 optimum at 30M) ----
    5)  STAGE=1; STAGE_NAME="masterFP_noEF"  ; LR=0.008 ;;
    6)  STAGE=2; STAGE_NAME="quant_residual" ; LR=0.008 ;;
    7)  STAGE=3; STAGE_NAME="quant_approxEF" ; LR=0.008 ;;
    8)  STAGE=4; STAGE_NAME="quant_noEF"     ; LR=0.008 ;;
    # ---- LR = 0.010 (above optimum) ----
    9)  STAGE=1; STAGE_NAME="masterFP_noEF"  ; LR=0.010 ;;
    10) STAGE=2; STAGE_NAME="quant_residual" ; LR=0.010 ;;
    11) STAGE=3; STAGE_NAME="quant_approxEF" ; LR=0.010 ;;
    12) STAGE=4; STAGE_NAME="quant_noEF"     ; LR=0.010 ;;
esac

WANDB_PREFIX="${MODEL_SIZE}-adam0staged-stage=${STAGE}_${STAGE_NAME}-LR=${LR}-ITER=${ITERATIONS}"

echo "=========================================="
echo "Adam0Staged 30M ablation — Task ${SLURM_ARRAY_TASK_ID}/12"
echo "  Stage:    ${STAGE} (${STAGE_NAME})"
echo "  LR:       ${LR}"
echo "  Optimizer: ${OPT}  (Adam0 mechanism + staged behavior)"
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

echo "Task ${SLURM_ARRAY_TASK_ID}/12 complete: stage=${STAGE} LR=${LR}"

rm -rf ${DATASETS_DIR}
