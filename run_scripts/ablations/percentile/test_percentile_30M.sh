#!/bin/bash -l
#SBATCH -J percentile_30M
#SBATCH -o outs/percentile_30M_%A_%a.out
#SBATCH -e outs/percentile_30M_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --gres gpu:H200_141GB:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=32000
#SBATCH -A H200
#SBATCH -q h200_qos
#SBATCH --array=5-6

# Validate P90 percentile finding on 30M model
# Testing both ECO and ECO-0 optimizers with P90, P95, P99

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cage

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((29500 + SLURM_ARRAY_TASK_ID))
export TORCH_COMPILE=0
export TORCH_COMPILE_DISABLE=1
export TORCHINDUCTOR_DISABLE_TRITON=1

export WANDB_ENTITY="keisufaj-hamad-bin-khalifa-university"
export WANDB_PROJECT="FP4-PERCENTILE-30M"

# Copy datasets to fast scratch space (H200 NVMe)
echo "Copying datasets to /scratch..."
mkdir -p /scratch/keisufaj_datasets
rsync -a --info=progress2 /export/home/keisufaj/optimization/ECO-CAGE-QCRI/datasets/ /scratch/keisufaj_datasets/
export DATASETS_DIR="/scratch/keisufaj_datasets"
echo "Dataset copy complete!"

# 30M model config (from train.sh)
export N_LAYER=6
export N_EMBD=640
export N_HEAD=5
export BATCH_SIZE=64
export ACC_STEPS=8
export SEQUENCE_LENGTH=512

# Fixed hyperparams
LR=0.005
BETA1=0.9
BETA2=0.95
ITERATIONS=5000  # More iterations for 30M model
WARMUP_STEPS=500

# Select optimizer and percentile based on array task ID
case ${SLURM_ARRAY_TASK_ID} in
    1)
        OPT="eco"
        PERCENTILE=90.0
        echo "Testing ECO with P90 (optimal from tiny model)"
        ;;
    2)
        OPT="eco"
        PERCENTILE=95.0
        echo "Testing ECO with P95"
        ;;
    3)
        OPT="eco"
        PERCENTILE=99.0
        echo "Testing ECO with P99 (baseline)"
        ;;
    4)
        OPT="eco0m-rooh"
        PERCENTILE=90.0
        echo "Testing ECO-0 with P90 (optimal from tiny model)"
        ;;
    5)
        OPT="eco0m-rooh"
        PERCENTILE=95.0
        echo "Testing ECO-0 with P95"
        ;;
    6)
        OPT="eco0m-rooh"
        PERCENTILE=99.0
        echo "Testing ECO-0 with P99 (baseline)"
        ;;
esac

WANDB_PREFIX="30M-${OPT}-P${PERCENTILE}-LR=${LR}-BETA1=${BETA1}"

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

echo "Job ${SLURM_ARRAY_TASK_ID} (${OPT} with P${PERCENTILE}) complete!"

# Cleanup scratch space
echo "Cleaning up /scratch..."
rm -rf /scratch/keisufaj_datasets
echo "Cleanup complete!"
