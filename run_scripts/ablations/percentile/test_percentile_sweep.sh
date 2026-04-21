#!/bin/bash -l
#SBATCH -J percentile_sweep
#SBATCH -o outs/percentile_sweep_%A_%a.out
#SBATCH -e outs/percentile_sweep_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --gres gpu:H200_141GB:2
#SBATCH --mem=10000
#SBATCH -A H200
#SBATCH -q h200_qos
#SBATCH --array=1-6

# Test different percentile values for FP4 scale calibration
# Hypothesis: 99th percentile might clip too much or too little

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cage

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((29500 + SLURM_ARRAY_TASK_ID))
export TORCH_COMPILE=0
export TORCH_COMPILE_DISABLE=1
export TORCHINDUCTOR_DISABLE_TRITON=1

export WANDB_ENTITY="keisufaj-hamad-bin-khalifa-university"
export WANDB_PROJECT="FP4-PERCENTILE-SWEEP"
export DATASETS_DIR="/export/home/keisufaj/optimization/ECO-CAGE-QCRI/datasets"

# Model config - tiny for fast testing
export N_LAYER=3
export N_EMBD=128
export N_HEAD=2
export BATCH_SIZE=64
export ACC_STEPS=4
export SEQUENCE_LENGTH=512

# Fixed hyperparams
LR=0.005
BETA1=0.9
BETA2=0.95
ITERATIONS=1000
WARMUP_STEPS=100
OPT="eco0m-rooh"  # Use ECO-0 since it performed better

# Select percentile based on array task ID
case ${SLURM_ARRAY_TASK_ID} in
    1)
        PERCENTILE=95.0
        echo "Testing 95th percentile (more aggressive clipping)"
        ;;
    2)
        PERCENTILE=99.0
        echo "Testing 99th percentile (current baseline)"
        ;;
    3)
        PERCENTILE=99.5
        echo "Testing 99.5th percentile"
        ;;
    4)
        PERCENTILE=99.9
        echo "Testing 99.9th percentile"
        ;;
    5)
        PERCENTILE=99.99
        echo "Testing 99.99th percentile (nearly min-max)"
        ;;
    6)
        PERCENTILE=100.0
        echo "Testing 100th percentile (true min-max, no clipping)"
        ;;
esac

WANDB_PREFIX="ECO0-P${PERCENTILE}-LR=${LR}-BETA1=${BETA1}"

torchrun --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    --nproc_per_node=2 ./src/main.py \
    --distributed-backend nccl \
    --dataset c4 \
    --datasets-dir ${DATASETS_DIR} \
    --model llama \
    --opt ${OPT} \
    --compile \
    --acc-steps ${ACC_STEPS} \
    --batch-size ${BATCH_SIZE} \
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
    --w-quant Q99FP4Quantizer \
    --w-quant-kwargs "{\"bits\":4,\"percentile\":${PERCENTILE},\"recalibrate_interval\":0}" \
    --a-quant NoQuantizer \
    --beta1 ${BETA1} \
    --beta2 ${BETA2}

echo "Job ${SLURM_ARRAY_TASK_ID} (percentile=${PERCENTILE}) complete!"
