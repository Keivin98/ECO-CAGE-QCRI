#!/bin/bash -l
#SBATCH -J eco_matched_lr
#SBATCH -o outs/eco_matched_lr_%A_%a.out
#SBATCH -e outs/eco_matched_lr_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --exclude=crirdchpxd001
#SBATCH --gres=gpu:H200_141GB:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=64000
#SBATCH -A H200_16GPUs
#SBATCH -q h200_qos_16_gpus
#SBATCH --array=1-8

# ==========================================================================
# ECO and ECO-0 with CAGE-matched LR (per ECO paper convention)
# ==========================================================================
# Tracking ID: EXP-ECO-MATCHED-CAGE-LR
#
# Why this experiment exists:
#   The ECO paper (Nikdan et al., 2026) runs ECO with the SAME LR as the
#   Adam-master-weights baseline, NOT a scaled-up LR. Our previous
#   experiments tuned ECO/ECO-0 LR independently and found:
#     30M:  CAGE LR=0.0012 vs ECO LR=0.003 (2.5x)
#     50M:  CAGE LR=0.00093 vs ECO LR=0.004 (4.3x)
#     100M: CAGE LR=0.0006 vs ECO LR=0.004 (6.7x)
#     300M: CAGE LR=0.0003 vs ECO LR=0.005 (16x)
#   The ratio grows with scale, suggesting our independent tuning may have
#   drifted from the regime where ECO's theoretical guarantees hold.
#
#   This experiment runs ECO and ECO-0 at the EXACT SAME LR as CAGE at every
#   scale, with the same scheduler (cos), to provide a paper data point of
#   "what happens if you follow ECO paper conventions verbatim?"
#
# Expected outcome: under-tuned (much worse than our half-eco0 + tuned LR
# results), but provides honest comparison for paper.
#
# Matrix (8 tasks x 4 GPUs = 32 GPUs total; QoS=16, so 2 waves):
#   1: 30M  ECO    cos LR=0.0012      (matches 30M CAGE/Adam LR)
#   2: 30M  ECO-0  cos LR=0.0012
#   3: 50M  ECO    cos LR=0.00093     (matches 50M CAGE/Adam LR)
#   4: 50M  ECO-0  cos LR=0.00093
#   5: 100M ECO    cos LR=0.0006      (matches 100M CAGE/Adam LR)
#   6: 100M ECO-0  cos LR=0.0006
#   7: 300M ECO    cos LR=0.0003      (matches 300M CAGE/Adam LR)
#   8: 300M ECO-0  cos LR=0.0003
#
# All tasks: INT4 + P99.5 (paper default), cos scheduler, batch=64x8=512.
# Token budgets match prior runs at each scale (3B/5B/10B/15B).
# ==========================================================================

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cage

source /export/home/keisufaj/optimization/ECO-CAGE-QCRI/run_scripts/utils/setup_tmp_cleanup.sh

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((30800 + SLURM_ARRAY_TASK_ID))
export TORCH_COMPILE=0
export TORCH_COMPILE_DISABLE=1
export TORCHINDUCTOR_DISABLE_TRITON=1

export WANDB_ENTITY="keisufaj-hamad-bin-khalifa-university"
export WANDB_PROJECT="ECO0-MATCHED-CAGE-LR"

# Copy datasets to fast scratch space (job-specific to avoid collisions)
export DATASETS_DIR="/scratch/keisufaj_datasets_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
if [ -d "${DATASETS_DIR}/c4" ]; then
    echo "Dataset already exists, skipping copy"
else
    mkdir -p ${DATASETS_DIR}
    rsync -a --info=progress2 /export/home/keisufaj/optimization/ECO-CAGE-QCRI/datasets/ ${DATASETS_DIR}/
fi

# Common batch config (effective batch 512)
export BATCH_SIZE=64
export ACC_STEPS=8
export SEQUENCE_LENGTH=512

# Common hyperparams
BETA1=0.9
BETA2=0.95
SCHEDULER="cos"     # matches CAGE / Adam / ECO paper convention
USE_CAGE="False"
W_QUANT="Q99IntQuantizer"
W_QUANT_KWARGS='{"bits":4,"percentile":99.5}'

# ========================================
# TASK SELECTION (scale + method)
# ========================================

case ${SLURM_ARRAY_TASK_ID} in
    1)  # 30M ECO
        MODEL_SIZE="30M"
        export N_LAYER=6 N_EMBD=640 N_HEAD=5
        TOKENS=3000000000           # 3B tokens
        OPT="eco"
        LR=0.0012
        ;;
    2)  # 30M ECO-0
        MODEL_SIZE="30M"
        export N_LAYER=6 N_EMBD=640 N_HEAD=5
        TOKENS=3000000000
        OPT="eco0m-rooh"
        LR=0.0024
        ;;
    3)  # 50M ECO
        MODEL_SIZE="50M"
        export N_LAYER=7 N_EMBD=768 N_HEAD=6
        TOKENS=5000000000           # 5B tokens
        OPT="eco"
        LR=0.00093
        ;;
    4)  # 50M ECO-0
        MODEL_SIZE="50M"
        export N_LAYER=7 N_EMBD=768 N_HEAD=6
        TOKENS=5000000000
        OPT="eco0m-rooh"
        LR=0.00186
        ;;
    5)  # 100M ECO
        MODEL_SIZE="100M"
        export N_LAYER=8 N_EMBD=1024 N_HEAD=8
        TOKENS=10000000000          # 10B tokens
        OPT="eco"
        LR=0.0006
        ;;
    6)  # 100M ECO-0
        MODEL_SIZE="100M"
        export N_LAYER=8 N_EMBD=1024 N_HEAD=8
        TOKENS=10000000000
        OPT="eco0m-rooh"
        LR=0.0012
        ;;
    7)  # 300M ECO
        MODEL_SIZE="300M"
        export N_LAYER=12 N_EMBD=1536 N_HEAD=12
        TOKENS=15000000000          # 15B tokens
        OPT="eco"
        LR=0.0003
        ;;
    8)  # 300M ECO-0
        MODEL_SIZE="300M"
        export N_LAYER=12 N_EMBD=1536 N_HEAD=12
        TOKENS=15000000000
        OPT="eco0m-rooh"
        LR=0.0006
        ;;
esac

# Compute iterations from tokens (effective batch 512)
TOKENS_PER_ITER=$((BATCH_SIZE * ACC_STEPS * SEQUENCE_LENGTH))
ITERATIONS=$((TOKENS / TOKENS_PER_ITER))
WARMUP_STEPS=$((ITERATIONS / 10))

# Resolve method tag
if [ "${OPT}" = "eco" ]; then
    METHOD="ECO-4bit-INT4"
else
    METHOD="ECO0-4bit-INT4"
fi

WANDB_PREFIX="${MODEL_SIZE}-${METHOD}-MATCHED-sched=${SCHEDULER}-LR=${LR}-ITER=${ITERATIONS}"

echo "=========================================="
echo "ECO Matched-LR — Task ${SLURM_ARRAY_TASK_ID}/8"
echo "  Scale:     ${MODEL_SIZE} (${N_LAYER}L, ${N_EMBD}D, ${N_HEAD}H)"
echo "  Method:    ${METHOD}"
echo "  Optimizer: ${OPT}"
echo "  Scheduler: ${SCHEDULER}"
echo "  LR:        ${LR}  (matches CAGE/Adam at this scale)"
echo "  Tokens:    ${TOKENS}"
echo "  Iter:      ${ITERATIONS}  (warmup ${WARMUP_STEPS})"
echo "=========================================="

torchrun --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    --nproc_per_node=4 /export/home/keisufaj/optimization/ECO-CAGE-QCRI/src/main.py \
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

echo "Task ${SLURM_ARRAY_TASK_ID}/8 complete: ${MODEL_SIZE} ${METHOD} cos LR=${LR}"

rm -rf ${DATASETS_DIR}
