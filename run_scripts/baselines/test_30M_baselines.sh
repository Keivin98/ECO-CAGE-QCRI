#!/bin/bash -l
#SBATCH -J 30M_baselines
#SBATCH -o outs/30M_baselines_%A_%a.out
#SBATCH -e outs/30M_baselines_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --gres gpu:H200_141GB:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=32000
#SBATCH -A H200
#SBATCH -q h200_qos
#SBATCH --array=1-2

# Comprehensive baseline comparison on 30M model
# Testing: FP32, FP16, STE, CAGE, ECO, ECO0
# All quantized methods use P90 percentile

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cage

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((29500 + SLURM_ARRAY_TASK_ID))
export TORCH_COMPILE=0
export TORCH_COMPILE_DISABLE=1
export TORCHINDUCTOR_DISABLE_TRITON=1

export WANDB_ENTITY="keisufaj-hamad-bin-khalifa-university"
export WANDB_PROJECT="ECO0-30M-BASELINES"

# Copy datasets to fast scratch space
echo "Copying datasets to /scratch..."
mkdir -p /scratch/keisufaj_datasets
rsync -a --info=progress2 /export/home/keisufaj/optimization/ECO-CAGE-QCRI/datasets/ /scratch/keisufaj_datasets/
export DATASETS_DIR="/scratch/keisufaj_datasets"
echo "Dataset copy complete!"

# 30M model config
export N_LAYER=6
export N_EMBD=640
export N_HEAD=5
export BATCH_SIZE=64
export ACC_STEPS=8
export SEQUENCE_LENGTH=512

# Training hyperparams (common)
BETA1=0.9
BETA2=0.95
ITERATIONS=5000
WARMUP_STEPS=500

# Select method based on array task ID
case ${SLURM_ARRAY_TASK_ID} in
    1)
        METHOD="FP32-Adam"
        OPT="adamw"
        W_QUANT="NoQuantizer"
        A_QUANT="NoQuantizer"
        W_QUANT_KWARGS='{"bits":32}'
        USE_CAGE="False"
        LR=0.0012  # Standard LR for 30M Adam
        echo "Testing FP32 Adam (upper bound, no quantization)"
        ;;
    2)
        METHOD="FP16-Adam"
        OPT="adamw"
        W_QUANT="NoQuantizer"  # Will use FP16 via mixed precision
        A_QUANT="NoQuantizer"
        W_QUANT_KWARGS='{"bits":16}'
        USE_CAGE="False"
        LR=0.0012  # Same as FP32
        echo "Testing FP16 Adam (mixed precision baseline)"
        ;;
    3)
        METHOD="STE-4bit"
        OPT="adamw"
        W_QUANT="Q99FP4Quantizer"
        A_QUANT="NoQuantizer"
        W_QUANT_KWARGS='{"bits":4,"percentile":90.0}'
        USE_CAGE="False"
        LR=0.0012  # Same as Adam (no error feedback)
        echo "Testing Simple STE (4-bit, no error feedback)"
        ;;
    4)
        METHOD="CAGE-4bit"
        OPT="adamw"
        W_QUANT="Q99FP4Quantizer"
        A_QUANT="NoQuantizer"
        W_QUANT_KWARGS='{"bits":4,"percentile":90.0}'
        USE_CAGE="True"
        LR=0.0012  # CAGE paper uses similar LRs
        echo "Testing CAGE (4-bit + curvature correction)"
        ;;
    5)
        METHOD="ECO-4bit"
        OPT="eco"
        W_QUANT="Q99FP4Quantizer"
        A_QUANT="NoQuantizer"
        W_QUANT_KWARGS='{"bits":4,"percentile":90.0}'
        USE_CAGE="False"
        LR=0.005  # ECO uses higher LR (from validation experiments)
        echo "Testing ECO (4-bit, error feedback in momentum)"
        ;;
    6)
        METHOD="ECO0-4bit"
        OPT="eco0m-rooh"
        W_QUANT="Q99FP4Quantizer"
        A_QUANT="NoQuantizer"
        W_QUANT_KWARGS='{"bits":4,"percentile":90.0}'
        USE_CAGE="False"
        LR=0.005  # ECO0 needs higher LR (gradient decay scaling)
        echo "Testing ECO0 (4-bit, error feedback in gradients)"
        ;;
esac

WANDB_PREFIX="30M-${METHOD}-LR=${LR}-BETA1=${BETA1}"

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

echo "Job ${SLURM_ARRAY_TASK_ID} (${METHOD}) complete!"

# Cleanup scratch space
echo "Cleaning up /scratch..."
rm -rf /scratch/keisufaj_datasets
echo "Cleanup complete!"
