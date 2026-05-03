#!/bin/bash -l
#SBATCH -J 50M_int4
#SBATCH -o outs/50M_int4_%A_%a.out
#SBATCH -e outs/50M_int4_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --exclude=crirdchpxd006
#SBATCH --gres=gpu:H200_141GB:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=32000
#SBATCH -A H200_16GPUs
#SBATCH -q h200_qos_16_gpus
#SBATCH --array=1-8

# ==========================================================================
# 50M INT4 rerun (Table 2 refresh) + ECO/ECO-0 LR ablations under half-eco0
# ==========================================================================
# Paper table targeted: tab_50m_scaling.tex (int-val / int-perp columns)
# Tracking ID: EXP-50M-INT4-RERUN
#
# Matrix:
#   1: STE   + INT4 + cos      + LR=0.00093     (baseline rerun)
#   2: CAGE  + INT4 + cos      + LR=0.00093     (baseline rerun)
#   3: ECO   + INT4 + half-eco0 + LR=0.005      (LR ablation)
#   4: ECO   + INT4 + half-eco0 + LR=0.00625    (LR ablation, prior optimum)
#   5: ECO   + INT4 + half-eco0 + LR=0.008      (LR ablation)
#   6: ECO-0 + INT4 + half-eco0 + LR=0.006      (LR ablation)
#   7: ECO-0 + INT4 + half-eco0 + LR=0.008      (LR ablation)
#   8: ECO-0 + INT4 + half-eco0 + LR=0.010      (LR ablation, prior optimum)
#
# Skipping LR=0.012 for ECO-0 because 30M showed it to be unstable under
# half-eco0 (see Limitations in paper). 50M is larger → likely worse.
#
# Each task: 19,073 iter ≈ 5B tokens, ~20-24h on 2× H200.
# 8 tasks × 2 GPUs = 16 GPUs concurrent, exactly fits 16-GPU QoS.
# ==========================================================================

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cage

source /export/home/keisufaj/optimization/ECO-CAGE-QCRI/run_scripts/utils/setup_tmp_cleanup.sh

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((30000 + SLURM_ARRAY_TASK_ID))
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

# 50M model config (matches prior 50M experiments)
MODEL_SIZE="50M"
export N_LAYER=7
export N_EMBD=768
export N_HEAD=6
export BATCH_SIZE=64
export ACC_STEPS=8
export SEQUENCE_LENGTH=512

# 5B tokens (matches prior 50M runs)
TOKENS=5000000000
TOKENS_PER_ITER=$((BATCH_SIZE * ACC_STEPS * SEQUENCE_LENGTH))
ITERATIONS=$((TOKENS / TOKENS_PER_ITER))
WARMUP_STEPS=$((ITERATIONS / 10))

echo "Training for ${ITERATIONS} iterations (${TOKENS} tokens)"

BETA1=0.9
BETA2=0.95

# All quantized methods use INT4 + P90
W_QUANT="Q99IntQuantizer"
W_QUANT_KWARGS='{"bits":4,"percentile":99.5}'

# ========================================
# TASK SELECTION
# ========================================

case ${SLURM_ARRAY_TASK_ID} in
    1)  # STE baseline
        METHOD="STE-4bit-INT4"
        OPT="adamw"
        USE_CAGE="False"
        SCHEDULER="cos"
        LR=0.00093
        ;;
    2)  # CAGE baseline
        METHOD="CAGE-4bit-INT4"
        OPT="adamw"
        USE_CAGE="True"
        SCHEDULER="cos"
        LR=0.00093
        ;;
    3)  # ECO LR sweep: 0.005
        METHOD="ECO-4bit-INT4"
        OPT="eco"
        USE_CAGE="False"
        SCHEDULER="half-eco0"
        LR=0.005
        ;;
    4)  # ECO LR sweep: 0.00625 (prior FP4 optimum)
        METHOD="ECO-4bit-INT4"
        OPT="eco"
        USE_CAGE="False"
        SCHEDULER="half-eco0"
        LR=0.00625
        ;;
    5)  # ECO LR sweep: 0.008
        METHOD="ECO-4bit-INT4"
        OPT="eco"
        USE_CAGE="False"
        SCHEDULER="half-eco0"
        LR=0.008
        ;;
    6)  # ECO-0 LR sweep: 0.006
        METHOD="ECO0-4bit-INT4"
        OPT="eco0m-rooh"
        USE_CAGE="False"
        SCHEDULER="half-eco0"
        LR=0.006
        ;;
    7)  # ECO-0 LR sweep: 0.008
        METHOD="ECO0-4bit-INT4"
        OPT="eco0m-rooh"
        USE_CAGE="False"
        SCHEDULER="half-eco0"
        LR=0.008
        ;;
    8)  # ECO-0 LR sweep: 0.010 (prior FP4 optimum)
        METHOD="ECO0-4bit-INT4"
        OPT="eco0m-rooh"
        USE_CAGE="False"
        SCHEDULER="half-eco0"
        LR=0.010
        ;;
esac

WANDB_PREFIX="${MODEL_SIZE}-${METHOD}-sched=${SCHEDULER}-LR=${LR}-ITER=${ITERATIONS}"

echo "=========================================="
echo "50M INT4 Rerun — Task ${SLURM_ARRAY_TASK_ID}/8"
echo "  Method:    ${METHOD}"
echo "  Optimizer: ${OPT}"
echo "  Scheduler: ${SCHEDULER}"
echo "  LR:        ${LR}"
echo "  Quantizer: INT4 (Q99IntQuantizer, P90)"
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

# Cleanup scratch
rm -rf ${DATASETS_DIR}
