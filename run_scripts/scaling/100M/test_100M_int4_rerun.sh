#!/bin/bash -l
#SBATCH -J 100M_int4
#SBATCH -o outs/100M_int4_%A_%a.out
#SBATCH -e outs/100M_int4_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --exclude=crirdchpxd001
#SBATCH --gres=gpu:H200_141GB:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=32000
#SBATCH -A H200_16GPUs
#SBATCH -q h200_qos_16_gpus
#SBATCH --array=1-9

# ==========================================================================
# 100M INT4 rerun (Table 3: tab_100m_scaling) + ECO/ECO-0 LR sweeps
# ==========================================================================
# Paper table targeted: tab_100m_scaling.tex (int-val/int-perp columns)
# Tracking ID: EXP-100M-INT4-RERUN
#
# LR sweep design informed by 30M results (bracketed) and 50M partial:
#   30M ECO-0 best: LR=0.008 (bracketed); 50M ECO-0 best so far: 0.008
#   30M ECO best:   LR=0.003 (flat in 0.003-0.005); 50M ECO best so far: 0.005
#
# Standard LR scaling for Adam-based methods at 100M is 0.0006 (sqrt-scale
# from 30M's 0.0012). For ECO/ECO-0, the scale-down between 50M and 100M
# was previously 0.7-0.8x, so we predict:
#   - ECO-0 100M: ~0.005-0.006 (slightly below 30M optimum of 0.008)
#   - ECO   100M: ~0.0025-0.004
#
# Matrix (8 tasks x 2 GPUs = 16 GPUs, fits 16-GPU QoS):
#   1: FP16 Adam       cos       LR=0.0006     (no quant baseline, 10B tokens)
#   2: STE  INT4       cos       LR=0.0006     (Adam-based)
#   3: CAGE INT4       cos       LR=0.0006     (Adam-based)
#   4: ECO  INT4       half-eco0 LR=0.003      (LR sweep low)
#   5: ECO  INT4       half-eco0 LR=0.004      (LR sweep mid)
#   6: ECO  INT4       half-eco0 LR=0.005      (LR sweep high)
#   7: ECO-0 INT4      half-eco0 LR=0.005      (LR sweep low)
#   8: ECO-0 INT4      half-eco0 LR=0.006      (LR sweep mid; predicted optimum)
#
# Skipping ECO-0 LR=0.008: at 50M LR=0.010 already diverged; LR=0.008 may
# also be unstable at 100M (stability ceiling drops with scale).
# Each task: ~38k iter (10B tokens), runtime ~12-15h on 2x H200.
# ==========================================================================

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cage

source /export/home/keisufaj/optimization/ECO-CAGE-QCRI/run_scripts/utils/setup_tmp_cleanup.sh

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((30500 + SLURM_ARRAY_TASK_ID))
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

# 100M model config
MODEL_SIZE="100M"
export N_LAYER=8
export N_EMBD=1024
export N_HEAD=8
export BATCH_SIZE=32
export ACC_STEPS=16
export SEQUENCE_LENGTH=512

# 10B tokens (matches prior 100M experiments)
TOKENS=10000000000
TOKENS_PER_ITER=$((BATCH_SIZE * ACC_STEPS * SEQUENCE_LENGTH))
ITERATIONS=$((TOKENS / TOKENS_PER_ITER))
WARMUP_STEPS=$((ITERATIONS / 10))

echo "Training for ${ITERATIONS} iterations (${TOKENS} tokens = 10B)"

BETA1=0.9
BETA2=0.95

# Quantizer default for INT4 tasks (overridden for task 1)
W_QUANT="Q99IntQuantizer"
W_QUANT_KWARGS='{"bits":4,"percentile":99.5}'

# ========================================
# TASK SELECTION
# ========================================

case ${SLURM_ARRAY_TASK_ID} in
    1)  # FP16 Adam (no quantization) — refresh at 10B tokens
        METHOD="FP16-Adam"
        OPT="adamw"
        USE_CAGE="False"
        SCHEDULER="cos"
        LR=0.0006
        W_QUANT="NoQuantizer"
        W_QUANT_KWARGS='{"bits":16}'
        ;;
    2)  # STE INT4
        METHOD="STE-4bit-INT4"
        OPT="adamw"
        USE_CAGE="False"
        SCHEDULER="cos"
        LR=0.0006
        ;;
    3)  # CAGE INT4
        METHOD="CAGE-4bit-INT4"
        OPT="adamw"
        USE_CAGE="True"
        SCHEDULER="cos"
        LR=0.0006
        ;;
    4)  # ECO LR sweep: 0.003
        METHOD="ECO-4bit-INT4"
        OPT="eco"
        USE_CAGE="False"
        SCHEDULER="half-eco0"
        LR=0.001
        ;;
    5)  # ECO LR sweep: 0.004
        METHOD="ECO-4bit-INT4"
        OPT="eco"
        USE_CAGE="False"
        SCHEDULER="half-eco0"
        LR=0.002
        ;;
    6)  # ECO LR sweep: 0.005
        METHOD="ECO-4bit-INT4"
        OPT="eco"
        USE_CAGE="False"
        SCHEDULER="half-eco0"
        LR=0.004
        ;;
    7)  # ECO-0 LR sweep: 0.005
        METHOD="ECO0-4bit-INT4"
        OPT="eco0m-rooh"
        USE_CAGE="False"
        SCHEDULER="half-eco0"
        LR=0.005
        ;;
    8)  # ECO-0 LR sweep: 0.006 (predicted optimum)
        METHOD="ECO0-4bit-INT4"
        OPT="eco0m-rooh"
        USE_CAGE="False"
        SCHEDULER="half-eco0"
        LR=0.006
        ;;
    9)  # ECO-0 LR sweep: 0.006 (predicted optimum)
        METHOD="ECO0-4bit-INT4"
        OPT="eco0m-rooh"
        USE_CAGE="False"
        SCHEDULER="half-eco0"
        LR=0.007
        ;;
esac

WANDB_PREFIX="${MODEL_SIZE}-${METHOD}-sched=${SCHEDULER}-LR=${LR}-ITER=${ITERATIONS}"

echo "=========================================="
echo "100M INT4 Rerun — Task ${SLURM_ARRAY_TASK_ID}/8"
echo "  Method:    ${METHOD}"
echo "  Optimizer: ${OPT}"
echo "  Scheduler: ${SCHEDULER}"
echo "  LR:        ${LR}"
echo "  Quantizer: ${W_QUANT}"
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

echo "Task ${SLURM_ARRAY_TASK_ID}/8 complete: ${METHOD} + ${SCHEDULER} + LR=${LR}"

rm -rf ${DATASETS_DIR}
