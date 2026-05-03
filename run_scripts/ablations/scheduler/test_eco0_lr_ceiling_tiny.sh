#!/bin/bash -l
#SBATCH -J eco0_lr_ceiling
#SBATCH -o outs/eco0_lr_ceiling_%A_%a.out
#SBATCH -e outs/eco0_lr_ceiling_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --exclude=crirdchpxd006
#SBATCH --gres=gpu:H200_141GB:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16000
#SBATCH -A H200_16GPUs
#SBATCH -q h200_qos_16_gpus
#SBATCH --array=1-4

# ==========================================================================
# EXPERIMENT (A): ECO-0 LR ceiling with half-eco0 on tiny model
# ==========================================================================
# Goal: Find where ECO-0 breaks as LR increases past 0.012.
# Previous winner: INT4 + half-eco0 + LR=0.012 → 58.10 PPL
# The monotonic 0.008→0.010→0.012 trend suggests ceiling is higher.
#
# Paper reference: experiments_log.md → EXP-A-LR-CEILING
# ==========================================================================

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cage

source /export/home/keisufaj/optimization/ECO-CAGE-QCRI/run_scripts/utils/setup_tmp_cleanup.sh

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((29700 + SLURM_ARRAY_TASK_ID))
export TORCH_COMPILE=0
export TORCH_COMPILE_DISABLE=1
export TORCHINDUCTOR_DISABLE_TRITON=1

export WANDB_ENTITY="keisufaj-hamad-bin-khalifa-university"
export WANDB_PROJECT="ECO0-SCHEDULER-ABLATION"

export DATASETS_DIR="/export/home/keisufaj/optimization/ECO-CAGE-QCRI/datasets"

# Tiny model config
MODEL_SIZE="tiny"
export N_LAYER=3
export N_EMBD=128
export N_HEAD=2
export BATCH_SIZE=128
export ACC_STEPS=4
export SEQUENCE_LENGTH=512

ITERATIONS=5000
WARMUP_STEPS=$((ITERATIONS / 10))

BETA1=0.9
BETA2=0.95
OPT="eco0m-rooh"

# Fixed config: INT4 + half-eco0 (current winner)
QUANT_MODE="INT4"
SCHEDULER="half-eco0"
W_QUANT="Q99IntQuantizer"
W_QUANT_KWARGS='{"bits":4,"percentile":90.0}'

case ${SLURM_ARRAY_TASK_ID} in
    1) LR=0.014 ;;
    2) LR=0.016 ;;
    3) LR=0.020 ;;
    4) LR=0.025 ;;
esac

WANDB_PREFIX="${MODEL_SIZE}-ECO0-${QUANT_MODE}-sched=${SCHEDULER}-LR=${LR}-ITER=${ITERATIONS}"

echo "=========================================="
echo "LR Ceiling Test ${SLURM_ARRAY_TASK_ID}/4"
echo "  Config: INT4 + half-eco0 + LR=${LR}"
echo "  Previous best: INT4+half-eco0+LR=0.012 → 58.10 PPL"
echo "=========================================="

torchrun --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    --nproc_per_node=1 /export/home/keisufaj/optimization/ECO-CAGE-QCRI/src/main.py \
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

echo "Task ${SLURM_ARRAY_TASK_ID}/4 Complete: LR=${LR}"
