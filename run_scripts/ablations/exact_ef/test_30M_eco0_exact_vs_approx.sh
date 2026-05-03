#!/bin/bash -l
#SBATCH -J 30M_exact_ef
#SBATCH -o outs/30M_exact_ef_%A_%a.out
#SBATCH -e outs/30M_exact_ef_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --exclude=crirdchpxd001
#SBATCH --gres=gpu:H200_141GB:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=32000
#SBATCH -A H200_16GPUs
#SBATCH -q h200_qos_16_gpus
#SBATCH --array=1-3

# ==========================================================================
# 30M ECO-0 (exact ECO error feedback) vs current approximate ECO-0
# ==========================================================================
# Tracking ID: EXP-30M-ECO0-EXACT-EF
#
# Hypothesis: the e_t ≈ e_{t+1} approximation in the standard ECO update
# is responsible for a large fraction of the gap to CAGE/master-weight
# training. Storing the previous error e_t (one extra buffer, 4 B/param)
# and using the exact two-term update from Nikdan et al. 2026 Appendix A
# should close most of that gap, while keeping ECO-0 well below CAGE's
# memory footprint.
#
# Existing 30M ECO-0 (approximate, --opt eco0m-rooh) sweep at INT4+P99.5+half-eco0:
#   LR=0.005: 27.40   LR=0.006: 27.37   LR=0.008: 26.76 (best)   LR=0.010: 27.92
#
# This script runs the EXACT variant (--opt eco0m-rooh-exact) at a tight
# bracket around the approximate optimum:
#   1: LR=0.006
#   2: LR=0.008  ← matches approx best LR
#   3: LR=0.010
#
# Each task: 11,444 iter (3B tokens), 2 GPUs, ~1h.
# All three dispatch in parallel; full result in ~1-1.5h.
#
# Memory check: e_prev adds 4 B/param → 30M × 4 B = 120 MB extra. Negligible.
# ==========================================================================

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cage

source /export/home/keisufaj/optimization/ECO-CAGE-QCRI/run_scripts/utils/setup_tmp_cleanup.sh

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((31100 + SLURM_ARRAY_TASK_ID))
export TORCH_COMPILE=0
export TORCH_COMPILE_DISABLE=1
export TORCHINDUCTOR_DISABLE_TRITON=1

export WANDB_ENTITY="keisufaj-hamad-bin-khalifa-university"
export WANDB_PROJECT="ECO0-30M-INT4-RERUN"

# Copy datasets to fast scratch space (job-specific)
export DATASETS_DIR="/scratch/keisufaj_datasets_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
if [ -d "${DATASETS_DIR}/c4" ]; then
    echo "Dataset already exists, skipping copy"
else
    mkdir -p ${DATASETS_DIR}
    rsync -a --info=progress2 /export/home/keisufaj/optimization/ECO-CAGE-QCRI/datasets/ ${DATASETS_DIR}/
fi

# 30M model config (matches EXP-30M-INT4-RERUN exactly)
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

# Common config (same recipe as the approximate ECO-0 best-bracket runs)
W_QUANT="Q99IntQuantizer"
W_QUANT_KWARGS='{"bits":4,"percentile":99.5}'
SCHEDULER="half-eco0"
USE_CAGE="False"
OPT="eco0m-rooh-exact"   # ← the new exact-EF variant
METHOD="ECO0-EXACT-4bit-INT4"

case ${SLURM_ARRAY_TASK_ID} in
    1)  LR=0.006 ;;   # below approx optimum
    2)  LR=0.008 ;;   # at approx optimum (head-to-head)
    3)  LR=0.010 ;;   # above approx optimum
esac

WANDB_PREFIX="${MODEL_SIZE}-${METHOD}-sched=${SCHEDULER}-LR=${LR}-ITER=${ITERATIONS}"

echo "=========================================="
echo "30M ECO-0 EXACT-EF — Task ${SLURM_ARRAY_TASK_ID}/3"
echo "  Method:    ${METHOD}"
echo "  Optimizer: ${OPT}  (exact two-term ECO error feedback)"
echo "  Scheduler: ${SCHEDULER}"
echo "  LR:        ${LR}"
echo "  Approx ECO-0 reference at this LR:"
echo "    LR=0.006 → 27.37 PPL,  LR=0.008 → 26.76 PPL,  LR=0.010 → 27.92 PPL"
echo "  Memory cost: +4 B/param vs approximate ECO-0 (e_prev buffer)"
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

echo "Task ${SLURM_ARRAY_TASK_ID}/3 complete: ${METHOD} half-eco0 LR=${LR}"

rm -rf ${DATASETS_DIR}
