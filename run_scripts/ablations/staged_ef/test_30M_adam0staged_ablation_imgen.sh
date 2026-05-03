#!/bin/bash -l
#SBATCH -J adam0_staged_imgen
#SBATCH -o outs/adam0_staged_imgen_%A_%a.out
#SBATCH -e outs/adam0_staged_imgen_%A_%a.err
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128000
#SBATCH --array=8-12

# ==========================================================================
# Adam0Staged ablation, tasks 8-12, on the image-generation cluster.
# ==========================================================================
# Tracking ID: EXP-30M-ADAM0-STAGED-IMGEN
#
# Tasks (matches the indices in test_30M_adam0staged_ablation.sh exactly,
# so the wandb run names line up with the rest of the sweep):
#
#   8:  stage 4 LR=0.008  (no EF, validated production LR cell)
#   9:  stage 1 LR=0.010  (master FP, above optimum)
#   10: stage 2 LR=0.010  (residual buffer, above optimum)
#   11: stage 3 LR=0.010  (approx EF, above optimum)
#   12: stage 4 LR=0.010  (no EF, above optimum)
#
# Differences from the H200 / Panther version of this script:
#   - No /scratch local copy (cluster has no local scratch); reads datasets
#     directly from /image-generation NFS.
#   - All compile/triton caches go to /image-generation/.tmp/ (NFS) so they
#     don't pollute /tmp on the shared node.
#   - Uses the qwen conda env at /home/local/QCRI/kisufaj/miniconda3/envs/qwen/.
#   - No nodelist / exclude clauses (let SLURM place wherever it can).
#   - No partition / QoS pinned; if your cluster needs one, add it via
#     `sbatch -p <partition>` at submit time.
#
# 5 tasks × 2 GPUs = 10 GPUs of demand.
# ==========================================================================

# ---------- conda env (qwen) ----------
eval "$('/home/local/QCRI/kisufaj/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
conda activate /home/local/QCRI/kisufaj/miniconda3/envs/qwen/

# ---------- distributed + compile/triton caches on /image-generation NFS ----------
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((34500 + SLURM_ARRAY_TASK_ID))
export TORCH_COMPILE=0
export TORCH_COMPILE_DISABLE=1
export TORCHINDUCTOR_DISABLE_TRITON=1
# Per-job cache dirs on NFS (avoid clobbering between concurrent array tasks).
export TORCHINDUCTOR_CACHE_DIR=/image-generation/.tmp/torchinductor_${USER}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
export TRITON_CACHE_DIR=/image-generation/.tmp/triton_${USER}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"

# ---------- wandb ----------
export WANDB_ENTITY="keisufaj-hamad-bin-khalifa-university"
export WANDB_PROJECT="ECO0-30M-ADAM0-STAGED"

# ---------- code + dataset paths (NFS, no local copy) ----------
PROJECT_ROOT="/image-generation/kisufaj/optimization/cage/CAGE"
export DATASETS_DIR="${PROJECT_ROOT}/datasets"

# ---------- model config (matches Panther test_30M_adam0staged_ablation.sh) ----------
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

# ---------- task -> (stage, LR) mapping (indices 8..12 only) ----------
case ${SLURM_ARRAY_TASK_ID} in
    8)  STAGE=4; STAGE_NAME="quant_noEF"     ; LR=0.008 ;;
    9)  STAGE=1; STAGE_NAME="masterFP_noEF"  ; LR=0.010 ;;
    10) STAGE=2; STAGE_NAME="quant_residual" ; LR=0.010 ;;
    11) STAGE=3; STAGE_NAME="quant_approxEF" ; LR=0.010 ;;
    12) STAGE=4; STAGE_NAME="quant_noEF"     ; LR=0.010 ;;
    *)
        echo "ERROR: This script only handles tasks 8-12, got ${SLURM_ARRAY_TASK_ID}"
        exit 1
        ;;
esac

WANDB_PREFIX="${MODEL_SIZE}-adam0staged-stage=${STAGE}_${STAGE_NAME}-LR=${LR}-ITER=${ITERATIONS}"

echo "=========================================="
echo "Adam0Staged 30M (image-generation) — Task ${SLURM_ARRAY_TASK_ID}"
echo "  Stage:     ${STAGE} (${STAGE_NAME})"
echo "  LR:        ${LR}"
echo "  Project:   ${PROJECT_ROOT}"
echo "  Datasets:  ${DATASETS_DIR}  (read directly from NFS, no local copy)"
echo "  Caches:    ${TORCHINDUCTOR_CACHE_DIR}"
echo "=========================================="

cd "${PROJECT_ROOT}"

torchrun --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    --nproc_per_node=2 "${PROJECT_ROOT}/src/main.py" \
    --distributed-backend nccl \
    --dataset c4 \
    --datasets-dir "${DATASETS_DIR}" \
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

echo "Task ${SLURM_ARRAY_TASK_ID} complete: stage=${STAGE} LR=${LR}"

# Clean up this task's compile/triton caches on NFS (don't accumulate).
rm -rf "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}" 2>/dev/null || true
