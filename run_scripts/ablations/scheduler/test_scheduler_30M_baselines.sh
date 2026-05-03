#!/bin/bash -l
#SBATCH -J sched_30M_bl
#SBATCH -o outs/sched_30M_bl_%A_%a.out
#SBATCH -e outs/sched_30M_bl_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --exclude=crirdchpxd006
#SBATCH --gres=gpu:H200_141GB:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=32000
#SBATCH -A H200_16GPUs
#SBATCH -q h200_qos_16_gpus
#SBATCH --array=1-8

# ==========================================================================
# EXPERIMENT (C): Validate cos > half-eco0 for non-ECO-0 methods at 30M
# ==========================================================================
# Hypothesis: cos (decay to LR/10) beats half-eco0 (decay to LR/2) for
# Adam/STE/CAGE/ECO because their v-buffers accumulate (beta2=0.95), so
# LR decay is needed late in training to compensate.
#
# Methods tested:
#   FP16 Adam (no quantization)
#   STE 4-bit (Q99FP4, no CAGE correction)
#   CAGE 4-bit (Q99FP4, with CAGE)
#   ECO 4-bit (Q99FP4, error feedback in momentum)
#
# Each at its 30M-validated optimal LR, compared under cos vs half-eco0.
#
# Paper reference: experiments_log.md → EXP-C-SCHEDULER-BASELINES-30M
# Expected outcome: If hypothesis holds, cos wins for all 4 methods.
# ==========================================================================

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cage

source /export/home/keisufaj/optimization/ECO-CAGE-QCRI/run_scripts/utils/setup_tmp_cleanup.sh

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((29800 + SLURM_ARRAY_TASK_ID))
export TORCH_COMPILE=0
export TORCH_COMPILE_DISABLE=1
export TORCHINDUCTOR_DISABLE_TRITON=1

export WANDB_ENTITY="keisufaj-hamad-bin-khalifa-university"
export WANDB_PROJECT="ECO0-SCHEDULER-ABLATION"

# Copy datasets to fast scratch space
export DATASETS_DIR="/scratch/keisufaj_datasets_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
if [ -d "${DATASETS_DIR}/c4" ]; then
    echo "Dataset already exists, skipping copy"
else
    mkdir -p ${DATASETS_DIR}
    rsync -a --info=progress2 /export/home/keisufaj/optimization/ECO-CAGE-QCRI/datasets/ ${DATASETS_DIR}/
fi

# 30M model config
MODEL_SIZE="30M"
export N_LAYER=6
export N_EMBD=640
export N_HEAD=5
export BATCH_SIZE=64
export ACC_STEPS=8
export SEQUENCE_LENGTH=512

ITERATIONS=5000
WARMUP_STEPS=500

BETA1=0.9
BETA2=0.95

# Task mapping: (method × scheduler)
# Adam-based methods use LR=0.0012 (established 30M default)
# ECO uses LR=0.00625 (established 30M default)
case ${SLURM_ARRAY_TASK_ID} in
    1)  # FP16 Adam + cos
        METHOD="FP16-Adam"; OPT="adamw"; LR=0.0012
        W_QUANT="NoQuantizer"; W_QUANT_KWARGS='{"bits":16}'
        USE_CAGE="False"; SCHEDULER="cos"
        ;;
    2)  # FP16 Adam + half-eco0
        METHOD="FP16-Adam"; OPT="adamw"; LR=0.0012
        W_QUANT="NoQuantizer"; W_QUANT_KWARGS='{"bits":16}'
        USE_CAGE="False"; SCHEDULER="half-eco0"
        ;;
    3)  # STE 4-bit + cos
        METHOD="STE-4bit"; OPT="adamw"; LR=0.0012
        W_QUANT="Q99FP4Quantizer"; W_QUANT_KWARGS='{"bits":4,"percentile":90.0}'
        USE_CAGE="False"; SCHEDULER="cos"
        ;;
    4)  # STE 4-bit + half-eco0
        METHOD="STE-4bit"; OPT="adamw"; LR=0.0012
        W_QUANT="Q99FP4Quantizer"; W_QUANT_KWARGS='{"bits":4,"percentile":90.0}'
        USE_CAGE="False"; SCHEDULER="half-eco0"
        ;;
    5)  # CAGE 4-bit + cos
        METHOD="CAGE-4bit"; OPT="adamw"; LR=0.0012
        W_QUANT="Q99FP4Quantizer"; W_QUANT_KWARGS='{"bits":4,"percentile":90.0}'
        USE_CAGE="True"; SCHEDULER="cos"
        ;;
    6)  # CAGE 4-bit + half-eco0
        METHOD="CAGE-4bit"; OPT="adamw"; LR=0.0012
        W_QUANT="Q99FP4Quantizer"; W_QUANT_KWARGS='{"bits":4,"percentile":90.0}'
        USE_CAGE="True"; SCHEDULER="half-eco0"
        ;;
    7)  # ECO 4-bit + cos
        METHOD="ECO-4bit"; OPT="eco"; LR=0.00625
        W_QUANT="Q99FP4Quantizer"; W_QUANT_KWARGS='{"bits":4,"percentile":90.0}'
        USE_CAGE="False"; SCHEDULER="cos"
        ;;
    8)  # ECO 4-bit + half-eco0
        METHOD="ECO-4bit"; OPT="eco"; LR=0.00625
        W_QUANT="Q99FP4Quantizer"; W_QUANT_KWARGS='{"bits":4,"percentile":90.0}'
        USE_CAGE="False"; SCHEDULER="half-eco0"
        ;;
esac

WANDB_PREFIX="${MODEL_SIZE}-${METHOD}-sched=${SCHEDULER}-LR=${LR}"

echo "=========================================="
echo "Scheduler Validation ${SLURM_ARRAY_TASK_ID}/8"
echo "  Method: ${METHOD} | Scheduler: ${SCHEDULER} | LR: ${LR}"
echo "  Hypothesis: cos should beat half-eco0 for this method"
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

echo "Task ${SLURM_ARRAY_TASK_ID}/8 Complete: ${METHOD} + ${SCHEDULER} + LR=${LR}"

# Cleanup scratch space
rm -rf ${DATASETS_DIR}
