#!/bin/bash -l
#SBATCH -J pct_int4
#SBATCH -o outs/pct_int4_%A_%a.out
#SBATCH -e outs/pct_int4_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --exclude=crirdchpxd001
#SBATCH --gres=gpu:H200_141GB:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16000
#SBATCH -A H200_16GPUs
#SBATCH -q h200_qos_16_gpus
#SBATCH --array=2-3

# ==========================================================================
# Percentile ablation under INT4 quantizer (tiny model)
# ==========================================================================
# Paper ref: Section 4.11 percentile ablation / fig_percentile_ablation.tex
# Tracking ID: EXP-PERCENTILE-INT4-TINY
#
# Re-does the 13-point percentile sweep under INT4 to verify P90 is still
# optimal after the quantizer switch. If INT4's optimal percentile differs,
# we need to update the paper's Method section (P90 was motivated by FP4's
# non-uniform codebook).
#
# Uses our validated tiny ECO-0 best recipe instead of the original FP4
# sweep's arbitrary LR:
#   - Quantizer:  Q99IntQuantizer (was Q99FP4Quantizer in the original)
#   - Scheduler:  half-eco0       (was cos in the original)
#   - LR:         0.012           (was 0.005 in the original — suboptimal)
#   - Iterations: 5000            (was 1000 — matches EXP-SCHEDULER-TINY)
#   - Batch/GPU config tuned for 16-GPU QoS (1 GPU/task, batch=128, acc=4)
# Effective batch 512 preserved throughout.
# ==========================================================================

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cage

source /export/home/keisufaj/optimization/ECO-CAGE-QCRI/run_scripts/utils/setup_tmp_cleanup.sh

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((30200 + SLURM_ARRAY_TASK_ID))
export TORCH_COMPILE=0
export TORCH_COMPILE_DISABLE=1
export TORCHINDUCTOR_DISABLE_TRITON=1

export WANDB_ENTITY="keisufaj-hamad-bin-khalifa-university"
export WANDB_PROJECT="INT4-PERCENTILE-SWEEP"
export DATASETS_DIR="/export/home/keisufaj/optimization/ECO-CAGE-QCRI/datasets"

# Tiny model (matches original percentile ablation exactly)
export N_LAYER=3
export N_EMBD=128
export N_HEAD=2

# Effective batch 512 on 1 GPU (original sweep used 64 x 4 x 2 GPUs)
export BATCH_SIZE=128
export ACC_STEPS=4
export SEQUENCE_LENGTH=512

# Validated tiny ECO-0 best recipe (from EXP-SCHEDULER-TINY-HALF)
LR=0.012
BETA1=0.9
BETA2=0.95
ITERATIONS=5000
WARMUP_STEPS=$((ITERATIONS / 10))
OPT="eco0m-rooh"
SCHEDULER="half-eco0"

# 13 percentile points matching the existing figure
case ${SLURM_ARRAY_TASK_ID} in
    1)  PERCENTILE=85.0   ;;
    2)  PERCENTILE=90.0   ;;
    3)  PERCENTILE=92.0   ;;
    4)  PERCENTILE=93.0   ;;
    5)  PERCENTILE=95.0   ;;
    6)  PERCENTILE=96.0   ;;
    7)  PERCENTILE=97.0   ;;
    8)  PERCENTILE=98.0   ;;
    9)  PERCENTILE=99.0   ;;
    10) PERCENTILE=99.5   ;;
    11) PERCENTILE=99.9   ;;
    12) PERCENTILE=99.99  ;;
    13) PERCENTILE=100.0  ;;
esac

WANDB_PREFIX="tiny-ECO0-INT4-sched=${SCHEDULER}-P${PERCENTILE}-LR=${LR}-ITER=${ITERATIONS}"

echo "=========================================="
echo "INT4 Percentile Sweep — Task ${SLURM_ARRAY_TASK_ID}/13"
echo "  Percentile: P${PERCENTILE}"
echo "  Quantizer:  Q99IntQuantizer"
echo "  Scheduler:  ${SCHEDULER}"
echo "  LR:         ${LR}"
echo "  Iterations: ${ITERATIONS}"
echo "  Goal: find optimal INT4 percentile under our validated tiny recipe"
echo "=========================================="

torchrun --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    --nproc_per_node=1 /export/home/keisufaj/optimization/ECO-CAGE-QCRI/src/main.py \
    --distributed-backend nccl \
    --dataset c4 \
    --datasets-dir ${DATASETS_DIR} \
    --model llama \
    --opt ${OPT} \
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
    --w-quant Q99IntQuantizer \
    --w-quant-kwargs "{\"bits\":4,\"percentile\":${PERCENTILE}}" \
    --a-quant NoQuantizer \
    --beta1 ${BETA1} \
    --beta2 ${BETA2}

echo "Task ${SLURM_ARRAY_TASK_ID}/13 complete: P${PERCENTILE}"
