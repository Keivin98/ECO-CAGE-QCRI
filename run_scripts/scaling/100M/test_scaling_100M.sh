#!/bin/bash -l
#SBATCH -J scaling_100M
#SBATCH -o outs/scaling_100M_%A_%a.out
#SBATCH -e outs/scaling_100M_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --gres gpu:H200_141GB:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=32000
#SBATCH -A H200
#SBATCH -q h200_qos
#SBATCH --array=1-3

# Scaling experiment: 100M model
# Testing: FP32, ECO, ECO0 (core trio for scaling analysis)

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cage

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((29500 + SLURM_ARRAY_TASK_ID))
export TORCH_COMPILE=0
export TORCH_COMPILE_DISABLE=1
export TORCHINDUCTOR_DISABLE_TRITON=1

export WANDB_ENTITY="keisufaj-hamad-bin-khalifa-university"
export WANDB_PROJECT="ECO0-SCALING"

# Copy datasets to fast scratch space
echo "Copying datasets to /scratch..."
mkdir -p /scratch/keisufaj_datasets
rsync -a --info=progress2 /export/home/keisufaj/optimization/ECO-CAGE-QCRI/datasets/ /scratch/keisufaj_datasets/
export DATASETS_DIR="/scratch/keisufaj_datasets"
echo "Dataset copy complete!"

# ========================================
# MODEL SIZE CONFIGURATION
# ========================================

MODEL_SIZE="100M"
export N_LAYER=8
export N_EMBD=1024
export N_HEAD=8

# Batch config (adjusted for 100M size)
export BATCH_SIZE=32
export ACC_STEPS=16
export SEQUENCE_LENGTH=512

# Training config
ITERATIONS=5000
WARMUP_STEPS=500
BETA1=0.9
BETA2=0.95

# ========================================
# METHOD SELECTION
# ========================================

case ${SLURM_ARRAY_TASK_ID} in
    1)
        METHOD="FP32-Adam"
        OPT="adamw"
        W_QUANT="NoQuantizer"
        A_QUANT="NoQuantizer"
        W_QUANT_KWARGS='{"bits":32}'
        USE_CAGE="False"
        LR=0.0012
        echo "Testing FP32 Adam (baseline)"
        ;;
    2)
        METHOD="ECO-4bit"
        OPT="eco"
        W_QUANT="Q99FP4Quantizer"
        A_QUANT="NoQuantizer"
        W_QUANT_KWARGS='{"bits":4,"percentile":90.0}'
        USE_CAGE="False"
        LR=0.005
        echo "Testing ECO (comparison)"
        ;;
    3)
        METHOD="ECO0-4bit"
        OPT="eco0m-rooh"
        W_QUANT="Q99FP4Quantizer"
        A_QUANT="NoQuantizer"
        W_QUANT_KWARGS='{"bits":4,"percentile":90.0}'
        USE_CAGE="False"
        LR=0.005
        echo "Testing ECO0 (ours)"
        ;;
esac

WANDB_PREFIX="${MODEL_SIZE}-${METHOD}-LR=${LR}-BS=${BATCH_SIZE}x${ACC_STEPS}"

# ========================================
# RUN TRAINING
# ========================================

echo "=========================================="
echo "Model: ${MODEL_SIZE}"
echo "Method: ${METHOD}"
echo "Batch: ${BATCH_SIZE} × ${ACC_STEPS} = $((BATCH_SIZE * ACC_STEPS))"
echo "Sequence: ${SEQUENCE_LENGTH}"
echo "=========================================="

torchrun --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    --nproc_per_node=2 ./src/main.py \
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
    --w-quant ${W_QUANT} \
    --w-quant-kwargs "${W_QUANT_KWARGS}" \
    --a-quant ${A_QUANT} \
    --beta1 ${BETA1} \
    --beta2 ${BETA2}

echo "=========================================="
echo "Job complete: ${MODEL_SIZE} ${METHOD}"
echo "=========================================="

# Cleanup scratch space
echo "Cleaning up /scratch..."
rm -rf /scratch/keisufaj_datasets
echo "Cleanup complete!"
