#!/bin/bash -l
#SBATCH -J 30M_eco0_cos
#SBATCH -o outs/30M_eco0_cos_%A_%a.out
#SBATCH -e outs/30M_eco0_cos_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --exclude=crirdchpxd001
#SBATCH --gres=gpu:H200_141GB:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=32000
#SBATCH -A H200_16GPUs
#SBATCH -q h200_qos_16_gpus
#SBATCH --array=1-3

# ==========================================================================
# 30M ECO-0 cos sweep — paired with existing half-eco0 sweep for a 30M
# scheduler-comparison figure (analogous to tiny figure but with 30M PPLs)
# ==========================================================================
# Tracking ID: EXP-30M-ECO0-COS-FIGURE
#
# Existing 30M ECO-0 + half-eco0 + INT4 + P99.5 data (from prior runs):
#   LR=0.005: 27.40 PPL
#   LR=0.006: 27.37 PPL
#   LR=0.008: 26.76 PPL  ← bracketed optimum
#   LR=0.010: 27.92 PPL
#
# This run produces matched cos data at LR ∈ {0.006, 0.008, 0.010} —
# same 3 points as the half-eco0 U-shape brackets — for the figure.
#
# Same config as EXP-30M-INT4-RERUN: 11,444 iter (3B tokens), INT4 P99.5,
# batch 64x8 effective 512, seed 0.
# ==========================================================================

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cage

source /export/home/keisufaj/optimization/ECO-CAGE-QCRI/run_scripts/utils/setup_tmp_cleanup.sh

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((31000 + SLURM_ARRAY_TASK_ID))
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

# 30M model config (matches EXP-30M-INT4-RERUN)
MODEL_SIZE="30M"
export N_LAYER=6
export N_EMBD=640
export N_HEAD=5
export BATCH_SIZE=64
export ACC_STEPS=8
export SEQUENCE_LENGTH=512

# 3B tokens (matches EXP-30M-INT4-RERUN)
TOKENS=3000000000
TOKENS_PER_ITER=$((BATCH_SIZE * ACC_STEPS * SEQUENCE_LENGTH))
ITERATIONS=$((TOKENS / TOKENS_PER_ITER))
WARMUP_STEPS=$((ITERATIONS / 10))

BETA1=0.9
BETA2=0.95

# Common config (all tasks)
W_QUANT="Q99IntQuantizer"
W_QUANT_KWARGS='{"bits":4,"percentile":99.5}'
SCHEDULER="cos"           # the variable we're testing
USE_CAGE="False"
OPT="eco0m-rooh"
METHOD="ECO0-4bit-INT4"

case ${SLURM_ARRAY_TASK_ID} in
    1)  LR=0.006 ;;
    2)  LR=0.008 ;;
    3)  LR=0.010 ;;
esac

WANDB_PREFIX="${MODEL_SIZE}-${METHOD}-sched=${SCHEDULER}-LR=${LR}-ITER=${ITERATIONS}"

echo "=========================================="
echo "30M ECO-0 cos figure pair — Task ${SLURM_ARRAY_TASK_ID}/3"
echo "  Method:    ${METHOD}"
echo "  Scheduler: ${SCHEDULER} (paired against existing half-eco0 sweep)"
echo "  LR:        ${LR}"
echo "  Goal:      ECO-0 cos vs half-eco0 figure at 30M (matched LRs 0.006/0.008/0.010)"
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

echo "Task ${SLURM_ARRAY_TASK_ID}/3 complete: ${METHOD} cos LR=${LR}"

rm -rf ${DATASETS_DIR}
