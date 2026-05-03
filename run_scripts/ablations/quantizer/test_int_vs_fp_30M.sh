#!/bin/bash -l
#SBATCH -J int_vs_fp_30M
#SBATCH -o outs/int_vs_fp_30M_%A_%a.out
#SBATCH -e outs/int_vs_fp_30M_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --exclude=crirdchpxd001
#SBATCH --gres=gpu:H200_141GB:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=32000
#SBATCH -A H200_16GPUs
#SBATCH -q h200_qos_16_gpus
#SBATCH --array=3-3

# ==========================================================================
# EXPERIMENT (D): INT4 vs FP4 rematch at 30M for ECO-0
# ==========================================================================
# Tiny-scale shock: INT4 beat FP4 by 6-9 PPL at every matched (sched, LR).
# This contradicts the paper's premise that FP4's non-uniform codebook
# matches weight distributions better.
#
# If INT4 wins at 30M too → paper narrative needs revision.
# If FP4 wins at 30M → tiny-scale quirk (weight distribution differs).
#
# Config: ECO-0 × {INT4, FP4} × {LR=0.01, LR=0.012} × half-eco0 scheduler
#
# Paper reference: experiments_log.md → EXP-D-INT-VS-FP-30M
# ==========================================================================

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cage

source /export/home/keisufaj/optimization/ECO-CAGE-QCRI/run_scripts/utils/setup_tmp_cleanup.sh

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((29900 + SLURM_ARRAY_TASK_ID))
export TORCH_COMPILE=0
export TORCH_COMPILE_DISABLE=1
export TORCHINDUCTOR_DISABLE_TRITON=1

export WANDB_ENTITY="keisufaj-hamad-bin-khalifa-university"
export WANDB_PROJECT="ECO0-QUANTIZER-ABLATION"

# Copy datasets to fast scratch space
export DATASETS_DIR="/scratch/keisufaj_datasets_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
if [ -d "${DATASETS_DIR}/c4" ]; then
    echo "Dataset already exists, skipping copy"
else
    mkdir -p ${DATASETS_DIR}
    rsync -a --info=progress2 /export/home/keisufaj/optimization/ECO-CAGE-QCRI/datasets/ ${DATASETS_DIR}/
fi

# 30M model config (matches baseline comparison)
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
OPT="eco0m-rooh"
SCHEDULER="half-eco0"  # Validated best scheduler for ECO-0 at tiny

case ${SLURM_ARRAY_TASK_ID} in
    1)  # FP4 at LR=0.01 (baseline — 30M ECO-0 winner was LR=0.01)
        QUANT_MODE="FP4"
        W_QUANT="Q99FP4Quantizer"
        W_QUANT_KWARGS='{"bits":4,"percentile":90.0}'
        LR=0.01
        ;;
    2)  # INT4 at LR=0.01
        QUANT_MODE="INT4"
        W_QUANT="Q99IntQuantizer"
        W_QUANT_KWARGS='{"bits":4,"percentile":90.0}'
        LR=0.01
        ;;
    3)  # FP4 at LR=0.012 (tiny winner LR)
        QUANT_MODE="FP4"
        W_QUANT="Q99FP4Quantizer"
        W_QUANT_KWARGS='{"bits":4,"percentile":90.0}'
        LR=0.012
        ;;
    4)  # INT4 at LR=0.012 (tiny winner: 58.10 PPL)
        QUANT_MODE="INT4"
        W_QUANT="Q99IntQuantizer"
        W_QUANT_KWARGS='{"bits":4,"percentile":90.0}'
        LR=0.012
        ;;
esac

WANDB_PREFIX="${MODEL_SIZE}-ECO0-${QUANT_MODE}-sched=${SCHEDULER}-LR=${LR}"

echo "=========================================="
echo "INT vs FP Rematch ${SLURM_ARRAY_TASK_ID}/4"
echo "  Quantizer: ${QUANT_MODE} | Scheduler: ${SCHEDULER} | LR: ${LR}"
echo "  Tiny result to compare: INT4+half+0.012=58.10, FP4+half+0.012=66.11"
echo "=========================================="

torchrun --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    --nproc_per_node=2 /export/home/keisufaj/optimization/ECO-CAGE-QCRI/src/main.py \
    --distributed-backend nccl \
    --dataset c4 \
    --datasets-dir ${DATASETS_DIR} \
    --model llama \
    --opt ${OPT} \
    --use-cage False \
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

echo "Task ${SLURM_ARRAY_TASK_ID}/4 Complete: ${QUANT_MODE} + LR=${LR}"

# Cleanup
rm -rf ${DATASETS_DIR}
