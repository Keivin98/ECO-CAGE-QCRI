#!/bin/bash

# Local runner for 50M scaling experiments (without SLURM)
# Usage: ./run_scaling_50M_local.sh 1  (for job 1)
#        ./run_scaling_50M_local.sh 2  (for job 2)
#        etc.

if [ -z "$1" ]; then
    echo "Usage: $0 <task_id>"
    echo "Example: $0 1"
    echo ""
    echo "Task IDs:"
    echo "  1 - FP16 Adam"
    echo "  2 - CAGE 4-bit"
    echo "  3 - ECO 4-bit"
    echo "  4 - ECO0 4-bit"
    exit 1
fi

export SLURM_ARRAY_TASK_ID=$1

# Scaling experiment: 50M model
# Testing: FP16, CAGE, ECO, ECO0
# LR scaled by 1/sqrt(50M/30M) = 0.775

eval "$('/home/local/QCRI/kisufaj/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
conda activate /home/local/QCRI/kisufaj/miniconda3/envs/qwen/

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((29500 + SLURM_ARRAY_TASK_ID))
export TORCH_COMPILE=0
export TORCH_COMPILE_DISABLE=1
export TORCHINDUCTOR_DISABLE_TRITON=1

export WANDB_ENTITY="keisufaj-hamad-bin-khalifa-university"
export WANDB_PROJECT="ECO0-SCALING"

# Skip scratch copy if not available, use direct path
export DATASETS_DIR="/image-generation/kisufaj/optimization/cage/CAGE/datasets"

# ========================================
# 50M MODEL CONFIGURATION
# ========================================

MODEL_SIZE="50M"
export N_LAYER=7
export N_EMBD=768
export N_HEAD=6

# Batch config
export BATCH_SIZE=64
export ACC_STEPS=8
export SEQUENCE_LENGTH=512

# Training config - calculate iterations from tokens
TOKENS=5000000000  # 5B tokens
TOKENS_PER_ITER=$((BATCH_SIZE * ACC_STEPS * SEQUENCE_LENGTH))
ITERATIONS=$((TOKENS / TOKENS_PER_ITER))
WARMUP_STEPS=$((ITERATIONS / 10))

echo "Training for ${ITERATIONS} iterations (${TOKENS} tokens)"
echo "Warmup: ${WARMUP_STEPS} steps"

# Fixed hyperparams
BETA1=0.9
BETA2=0.95

# ========================================
# METHOD SELECTION
# ========================================

case ${SLURM_ARRAY_TASK_ID} in
    1)
        METHOD="FP16-Adam"
        OPT="adamw"
        W_QUANT="NoQuantizer"
        A_QUANT="NoQuantizer"
        W_QUANT_KWARGS='{"bits":16}'
        USE_CAGE="False"
        LR=0.00093
        echo "Testing FP16 Adam (baseline, LR scaled by 0.775)"
        ;;
    2)
        METHOD="CAGE-4bit"
        OPT="adamw"
        W_QUANT="Q99FP4Quantizer"
        A_QUANT="NoQuantizer"
        W_QUANT_KWARGS='{"bits":4,"percentile":90.0}'
        USE_CAGE="True"
        LR=0.00093
        echo "Testing CAGE (QAT baseline, LR=0.00093)"
        ;;
    3)
        METHOD="ECO-4bit"
        OPT="eco"
        W_QUANT="Q99FP4Quantizer"
        A_QUANT="NoQuantizer"
        W_QUANT_KWARGS='{"bits":4,"percentile":90.0}'
        USE_CAGE="False"
        LR=0.00484
        echo "Testing ECO (LR=0.00484, scaled from 0.00625)"
        ;;
    4)
        METHOD="ECO0-4bit"
        OPT="eco0m-rooh"
        W_QUANT="Q99FP4Quantizer"
        A_QUANT="NoQuantizer"
        W_QUANT_KWARGS='{"bits":4,"percentile":90.0}'
        USE_CAGE="False"
        LR=0.00775
        echo "Testing ECO0 (ours, LR=0.00775, scaled from 0.01)"
        ;;
    *)
        echo "Invalid task ID: ${SLURM_ARRAY_TASK_ID}"
        echo "Valid range: 1-4"
        exit 1
        ;;
esac

WANDB_PREFIX="${MODEL_SIZE}-${METHOD}-LR=${LR}-BS=${BATCH_SIZE}x${ACC_STEPS}-ITER=${ITERATIONS}"

# ========================================
# RUN TRAINING
# ========================================

echo "=========================================="
echo "Model: ${MODEL_SIZE}"
echo "Config: ${N_LAYER} layers, ${N_EMBD} embd, ${N_HEAD} heads"
echo "Method: ${METHOD}"
echo "Batch: ${BATCH_SIZE} × ${ACC_STEPS} = $((BATCH_SIZE * ACC_STEPS))"
echo "Sequence: ${SEQUENCE_LENGTH}"
echo "Iterations: ${ITERATIONS} (${TOKENS} tokens)"
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
echo "Job ${SLURM_ARRAY_TASK_ID} (${METHOD}) complete!"
echo "Peak memory logged to WandB: memory/peak_gb"
echo "=========================================="

