#!/bin/bash -l
#SBATCH -J 30M_seeds
#SBATCH -o outs/30M_seeds_%A_%a.out
#SBATCH -e outs/30M_seeds_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --exclude=crirdchpxd001
#SBATCH --gres=gpu:H200_141GB:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=32000
#SBATCH -A H200_16GPUs
#SBATCH -q h200_qos_16_gpus
#SBATCH --array=1-4

# ==========================================================================
# 30M ECO-0 vs CAGE — multi-seed validation of the headline result.
# ==========================================================================
# Tracking ID: EXP-30M-ECO0-CAGE-MULTISEED
#
# Why: the central paper claim "ECO-0 beats CAGE at 30M" rests on a
# 0.23 PPL gap (ECO-0 cos LR=0.008 = 25.80 vs CAGE INT4 cos LR=0.0012
# = 26.03), both single-seed. If inter-seed variance is comparable to
# the gap, the claim is noise. Two extra seeds per method (4 runs total)
# is ~4 GPU-hours and either confirms or kills the headline.
#
# Existing single-seed numbers (seed=0):
#   ECO-0 INT4 cos LR=0.008 -> 25.80 PPL
#   CAGE  INT4 cos LR=0.0012 -> 26.03 PPL
#   gap = -0.23 (ECO-0 wins)
#
# This script adds seeds 1 and 2 for each method:
#   1: ECO-0 INT4 cos LR=0.008 seed=1
#   2: ECO-0 INT4 cos LR=0.008 seed=2
#   3: CAGE  INT4 cos LR=0.0012 seed=1
#   4: CAGE  INT4 cos LR=0.0012 seed=2
#
# Decision rule after results:
#   - mean(ECO-0) < mean(CAGE) by > 1·max(stddev): claim holds, ship it
#   - within 1·stddev: claim is noise, soften to "competitive with CAGE"
#   - mean(ECO-0) > mean(CAGE): claim flips, rewrite around it
#
# Same config as EXP-30M-INT4-RERUN: 6 layers, 640 dim, 5 heads,
# 3B tokens (11,444 iter), batch 64x8 effective 512.
# 2 GPUs each, ~1h per task. 4 tasks × 2 GPUs = 8 GPUs concurrent.
# ==========================================================================

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cage

source /export/home/keisufaj/optimization/ECO-CAGE-QCRI/run_scripts/utils/setup_tmp_cleanup.sh

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((31900 + SLURM_ARRAY_TASK_ID))
export TORCH_COMPILE=0
export TORCH_COMPILE_DISABLE=1
export TORCHINDUCTOR_DISABLE_TRITON=1

export WANDB_ENTITY="keisufaj-hamad-bin-khalifa-university"
export WANDB_PROJECT="ECO0-30M-INT4-RERUN"

export DATASETS_DIR="/scratch/keisufaj_datasets_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
if [ -d "${DATASETS_DIR}/c4" ]; then
    echo "Dataset already exists, skipping copy"
else
    mkdir -p ${DATASETS_DIR}
    rsync -a --info=progress2 /export/home/keisufaj/optimization/ECO-CAGE-QCRI/datasets/ ${DATASETS_DIR}/
fi

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

W_QUANT="Q99IntQuantizer"
W_QUANT_KWARGS='{"bits":4,"percentile":99.5}'
SCHEDULER="cos"

case ${SLURM_ARRAY_TASK_ID} in
    1)  # ECO-0 cos LR=0.008 seed=1
        METHOD="ECO0-4bit-INT4"
        OPT="eco0m-rooh"
        USE_CAGE="False"
        LR=0.008
        SEED=1
        ;;
    2)  # ECO-0 cos LR=0.008 seed=2
        METHOD="ECO0-4bit-INT4"
        OPT="eco0m-rooh"
        USE_CAGE="False"
        LR=0.008
        SEED=2
        ;;
    3)  # CAGE INT4 cos LR=0.0012 seed=1
        METHOD="CAGE-4bit-INT4"
        OPT="adamw"
        USE_CAGE="True"
        LR=0.0012
        SEED=1
        ;;
    4)  # CAGE INT4 cos LR=0.0012 seed=2
        METHOD="CAGE-4bit-INT4"
        OPT="adamw"
        USE_CAGE="True"
        LR=0.0012
        SEED=2
        ;;
esac

WANDB_PREFIX="${MODEL_SIZE}-${METHOD}-sched=${SCHEDULER}-LR=${LR}-seed=${SEED}-ITER=${ITERATIONS}"

echo "=========================================="
echo "30M multi-seed validation — Task ${SLURM_ARRAY_TASK_ID}/4"
echo "  Method: ${METHOD}  LR: ${LR}  Seed: ${SEED}"
echo "  Reference (seed=0):"
echo "    ECO-0 cos LR=0.008 -> 25.80"
echo "    CAGE  cos LR=0.0012 -> 26.03"
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
    --seed ${SEED} \
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

echo "Task ${SLURM_ARRAY_TASK_ID}/4 complete: ${METHOD} cos LR=${LR} seed=${SEED}"

rm -rf ${DATASETS_DIR}
