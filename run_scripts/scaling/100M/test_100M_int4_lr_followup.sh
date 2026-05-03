#!/bin/bash -l
#SBATCH -J 100M_int4_lr
#SBATCH -o outs/100M_int4_lr_%A_%a.out
#SBATCH -e outs/100M_int4_lr_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --exclude=crirdchpxd001
#SBATCH --gres=gpu:H200_141GB:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=32000
#SBATCH -A H200_16GPUs
#SBATCH -q h200_qos_16_gpus
#SBATCH --array=1-4

# ==========================================================================
# 100M INT4 LR follow-up — bracket ECO-0 below and ECO above
# ==========================================================================
# Paper ref: Table 3 (tab_100m_scaling.tex) — pins ECO/ECO-0 100M headlines
# Tracking ID: EXP-100M-INT4-LR-FOLLOWUP
#
# State after EXP-100M-INT4-RERUN:
#
# ECO-0 INT4 100M (LR=0.005 → 21.21 stable; 0.006, 0.007 → spikes/unstable):
#   The stability ceiling is even lower at 100M than at 50M (which was 0.010).
#   Need to test BELOW 0.005 to bracket optimum from below.
#
# ECO INT4 100M (curve monotonically improving with LR up to 0.004 → 22.24):
#   The user's LR grid {0.001, 0.002, 0.004} did not bracket the optimum.
#   Need to test ABOVE 0.004 to find the peak.
#
# Matrix (4 tasks x 2 GPUs = 8 GPUs):
#   1: ECO-0 INT4 + half-eco0 + LR=0.003   (below current best 0.005)
#   2: ECO-0 INT4 + half-eco0 + LR=0.004   (below current best 0.005)
#   3: ECO   INT4 + half-eco0 + LR=0.005   (above current best 0.004)
#   4: ECO   INT4 + half-eco0 + LR=0.006   (above current best 0.004)
#
# Same config as EXP-100M-INT4-RERUN: 38,146 iter (10B tokens), INT4 P99.5.
# ==========================================================================

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cage

source /export/home/keisufaj/optimization/ECO-CAGE-QCRI/run_scripts/utils/setup_tmp_cleanup.sh

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((30600 + SLURM_ARRAY_TASK_ID))
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

TOKENS=10000000000
TOKENS_PER_ITER=$((BATCH_SIZE * ACC_STEPS * SEQUENCE_LENGTH))
ITERATIONS=$((TOKENS / TOKENS_PER_ITER))
WARMUP_STEPS=$((ITERATIONS / 10))

BETA1=0.9
BETA2=0.95

W_QUANT="Q99IntQuantizer"
W_QUANT_KWARGS='{"bits":4,"percentile":99.5}'
SCHEDULER="half-eco0"
USE_CAGE="False"

case ${SLURM_ARRAY_TASK_ID} in
    1)  # ECO-0 LR=0.003 (below current best 0.005)
        METHOD="ECO0-4bit-INT4"
        OPT="eco0m-rooh"
        LR=0.003
        ;;
    2)  # ECO-0 LR=0.004 (below current best 0.005)
        METHOD="ECO0-4bit-INT4"
        OPT="eco0m-rooh"
        LR=0.004
        ;;
    3)  # ECO LR=0.005 (above current best 0.004)
        METHOD="ECO-4bit-INT4"
        OPT="eco"
        LR=0.005
        ;;
    4)  # ECO LR=0.006 (above current best 0.004)
        METHOD="ECO-4bit-INT4"
        OPT="eco"
        LR=0.006
        ;;
esac

WANDB_PREFIX="${MODEL_SIZE}-${METHOD}-sched=${SCHEDULER}-LR=${LR}-ITER=${ITERATIONS}"

echo "=========================================="
echo "100M INT4 LR follow-up — Task ${SLURM_ARRAY_TASK_ID}/4"
echo "  Method: ${METHOD} | LR: ${LR}"
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

echo "Task ${SLURM_ARRAY_TASK_ID}/4 complete: ${METHOD} + LR=${LR}"

rm -rf ${DATASETS_DIR}
