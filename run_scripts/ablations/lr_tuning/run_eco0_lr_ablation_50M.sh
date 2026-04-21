#!/bin/bash

# ECO0 Learning Rate + Percentile Ablation at 50M
# Testing LR={0.006, 0.0065} × Percentile={90, 95}
# Usage: ./run_eco0_lr_ablation_50M.sh 1  (LR=0.006, P90)
#        ./run_eco0_lr_ablation_50M.sh 2  (LR=0.0065, P90)
#        ./run_eco0_lr_ablation_50M.sh 3  (LR=0.006, P95)
#        ./run_eco0_lr_ablation_50M.sh 4  (LR=0.0065, P95)

if [ -z "$1" ]; then
    echo "Usage: $0 <task_id>"
    echo "Example: $0 1"
    echo ""
    echo "Task IDs:"
    echo "  1 - ECO0 LR=0.006, P90"
    echo "  2 - ECO0 LR=0.0065, P90"
    echo "  3 - ECO0 LR=0.006, P95"
    echo "  4 - ECO0 LR=0.0065, P95"
    echo ""
    echo "Context: Original ECO0 LR=0.00775 fell behind ECO (0.00484) in final steps"
    echo "         Testing lower LRs + P95 (ECO0's preferred percentile at 30M)"
    exit 1
fi

export SLURM_ARRAY_TASK_ID=$1

eval "$('/home/local/QCRI/kisufaj/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
conda activate /home/local/QCRI/kisufaj/miniconda3/envs/qwen/

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((29600 + SLURM_ARRAY_TASK_ID))
export TORCH_COMPILE=0
export TORCH_COMPILE_DISABLE=1
export TORCHINDUCTOR_DISABLE_TRITON=1

export WANDB_ENTITY="keisufaj-hamad-bin-khalifa-university"
export WANDB_PROJECT="ECO0-SCALING"

export DATASETS_DIR="/image-generation/kisufaj/optimization/cage/CAGE/datasets"

# ========================================
# 50M MODEL CONFIGURATION (same as original)
# ========================================

MODEL_SIZE="50M"
export N_LAYER=7
export N_EMBD=768
export N_HEAD=6

# Batch config
export BATCH_SIZE=64
export ACC_STEPS=8
export SEQUENCE_LENGTH=512

# Training config - 5B tokens
TOKENS=5000000000
TOKENS_PER_ITER=$((BATCH_SIZE * ACC_STEPS * SEQUENCE_LENGTH))
ITERATIONS=$((TOKENS / TOKENS_PER_ITER))
WARMUP_STEPS=$((ITERATIONS / 10))

echo "Training for ${ITERATIONS} iterations (${TOKENS} tokens)"
echo "Warmup: ${WARMUP_STEPS} steps"

# Fixed hyperparams
BETA1=0.9
BETA2=0.95

# ECO0 config (constant across all runs)
METHOD_BASE="ECO0-4bit"
OPT="eco0m-rooh"
W_QUANT="Q99FP4Quantizer"
A_QUANT="NoQuantizer"
USE_CAGE="False"

# ========================================
# LEARNING RATE + PERCENTILE SELECTION
# ========================================

case ${SLURM_ARRAY_TASK_ID} in
    1)
        LR=0.006
        PERCENTILE=90.0
        W_QUANT_KWARGS='{"bits":4,"percentile":90.0}'
        echo "Testing ECO0 with LR=0.006, P90"
        echo "Context: Most conservative LR + standard percentile"
        ;;
    2)
        LR=0.0065
        PERCENTILE=90.0
        W_QUANT_KWARGS='{"bits":4,"percentile":90.0}'
        echo "Testing ECO0 with LR=0.0065, P90"
        echo "Context: Middle ground LR + standard percentile"
        ;;
    3)
        LR=0.006
        PERCENTILE=95.0
        W_QUANT_KWARGS='{"bits":4,"percentile":95.0}'
        echo "Testing ECO0 with LR=0.006, P95"
        echo "Context: Conservative LR + ECO0's preferred percentile from 30M"
        ;;
    4)
        LR=0.0065
        PERCENTILE=95.0
        W_QUANT_KWARGS='{"bits":4,"percentile":95.0}'
        echo "Testing ECO0 with LR=0.0065, P95"
        echo "Context: Middle LR + ECO0's preferred percentile from 30M"
        ;;
    *)
        echo "Invalid task ID: ${SLURM_ARRAY_TASK_ID}"
        echo "Valid range: 1-4"
        exit 1
        ;;
esac

METHOD="${METHOD_BASE}-LR=${LR}-P${PERCENTILE}"
WANDB_PREFIX="${MODEL_SIZE}-${METHOD}-BS=${BATCH_SIZE}x${ACC_STEPS}-ITER=${ITERATIONS}"

# ========================================
# RUN TRAINING
# ========================================

echo "=========================================="
echo "ABLATION: ECO0 LR + Percentile Sweep"
echo "=========================================="
echo "Model: ${MODEL_SIZE}"
echo "Config: ${N_LAYER} layers, ${N_EMBD} embd, ${N_HEAD} heads"
echo "Method: ${METHOD}"
echo "Learning Rate: ${LR}"
echo "Percentile: P${PERCENTILE}"
echo "Batch: ${BATCH_SIZE} × ${ACC_STEPS} = $((BATCH_SIZE * ACC_STEPS))"
echo "Sequence: ${SEQUENCE_LENGTH}"
echo "Iterations: ${ITERATIONS} (${TOKENS} tokens)"
echo ""
echo "Comparison context:"
echo "  - ECO (LR=0.00484, P90): val_loss=3.189"
echo "  - ECO0 (LR=0.00775, P90): val_loss=3.196 [FAILED]"
echo "  - 30M ECO0: P95 slightly better than P90 (3.538 vs 3.541)"
echo "  - This run (LR=${LR}, P${PERCENTILE}): Testing..."
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
echo "ECO0 Ablation (LR=${LR}, P${PERCENTILE}) complete!"
echo "Check WandB project: ECO0-SCALING"
echo "Compare with:"
echo "  - 50M-ECO-4bit-LR=0.00484 (val: 3.189)"
echo "  - 50M-ECO0-4bit-LR=0.00775 (val: 3.196)"
echo "=========================================="
