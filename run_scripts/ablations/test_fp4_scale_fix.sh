#!/bin/bash -l
#SBATCH -J fp4_scale_test
#SBATCH -o outs/fp4_scale_test_%A_%a.out
#SBATCH -e outs/fp4_scale_test_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --gres gpu:H200_141GB:2
#SBATCH --mem=10000
#SBATCH -A H200
#SBATCH -q h200_qos
#SBATCH --array=1-4

# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh

conda activate cage

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((29500 + SLURM_ARRAY_TASK_ID))
export TORCH_COMPILE=0
export TORCH_COMPILE_DISABLE=1
export TORCHINDUCTOR_DISABLE_TRITON=1

export WANDB_ENTITY="keisufaj-hamad-bin-khalifa-university"
export WANDB_PROJECT="FP4-SCALE-DEBUG"
export DATASETS_DIR="/export/home/keisufaj/optimization/ECO-CAGE-QCRI/datasets"

# Test parameters
MODEL_SIZE="tiny"
LR=0.005
BETA1=0.9
BETA2=0.95
ITERATIONS=1000  # Short run for quick test
WARMUP_STEPS=100

# Model config for tiny
export N_LAYER=3
export N_EMBD=128
export N_HEAD=2
export BATCH_SIZE=64
export ACC_STEPS=4
export SEQUENCE_LENGTH=512

# Select configuration based on array task ID
case ${SLURM_ARRAY_TASK_ID} in
    1)
        echo "Running Test 1/4: ECO + Frozen Scale (baseline)"
        OPT="eco"
        RECALIBRATE=0
        WANDB_PREFIX="ECO-FROZEN-SCALE-LR=${LR}-BETA1=${BETA1}"
        ;;
    2)
        echo "Running Test 2/4: ECO + Dynamic Scale (recalibrate every 100 steps)"
        OPT="eco"
        RECALIBRATE=100
        WANDB_PREFIX="ECO-DYNAMIC-SCALE-LR=${LR}-BETA1=${BETA1}"
        ;;
    3)
        echo "Running Test 3/4: ECO-0 + Frozen Scale (baseline)"
        OPT="eco0-rooh"
        RECALIBRATE=0
        WANDB_PREFIX="ECO0-FROZEN-SCALE-LR=${LR}-BETA1=${BETA1}"
        ;;
    4)
        echo "Running Test 4/4: ECO-0 + Dynamic Scale (recalibrate every 100 steps)"
        OPT="eco0-rooh"
        RECALIBRATE=100
        WANDB_PREFIX="ECO0-DYNAMIC-SCALE-LR=${LR}-BETA1=${BETA1}"
        ;;
esac

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
    --w-quant-kwargs "{\"bits\":4,\"recalibrate_interval\":${RECALIBRATE}}" \
    --a-quant NoQuantizer \
    --beta1 ${BETA1} \
    --beta2 ${BETA2}

echo "Job ${SLURM_ARRAY_TASK_ID} complete! Check WandB: ${WANDB_PROJECT}"
