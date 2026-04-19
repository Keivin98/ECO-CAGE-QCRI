#!/bin/bash

set -e

export VOCAB_SIZE=32000
export BATCH_SIZE=64
export ACC_STEPS=8
export SEQUENCE_LENGTH=512
export DATASET="c4"


set_model_config() {
    local prefix="$1"
    case "$prefix" in
        "tiny")
            export N_LAYER=3
            export N_EMBD=128
            export N_HEAD=2
            # export LR=0.005
            export TOKENS=${TOKENS:-1000000000} # 1B
            ;;
        "30M")
            export N_LAYER=6
            export N_EMBD=640
            export N_HEAD=5
            export LR=${LR:-0.0012}
            export TOKENS=${TOKENS:-3000000000} # 3B
            ;;
        "50M")
            export N_LAYER=7
            export N_EMBD=768
            export N_HEAD=6
            export LR=${LR:-0.0012}
            export TOKENS=${TOKENS:-5000000000} # 5B
            ;;
        "100M")
            export N_LAYER=8
            export N_EMBD=1024
            export N_HEAD=8
            export LR=${LR:-0.0006}
            export TOKENS=${TOKENS:-10000000000} # 10B
            ;;
        "200M")
            export N_LAYER=10
            export N_EMBD=1280
            export N_HEAD=10
            export LR=${LR:-0.0003}
            export TOKENS=${TOKENS:-20000000000} # 20B
            ;;
        "430M")
            export N_LAYER=13
            export N_EMBD=1664
            export N_HEAD=13
            export LR=${LR:-0.00015}
            export TOKENS=${TOKENS:-43000000000} # 43B
            ;;
        "800M")
            export N_LAYER=16
            export N_EMBD=2048
            export N_HEAD=16
            export LR=${LR:-0.000075}
            export TOKENS=${TOKENS:-80000000000} # 80B
            ;;
        "1700M")
            export N_LAYER=20
            export N_EMBD=2688
            export N_HEAD=21
            export LR=${LR:-0.0000375}
            export TOKENS=${TOKENS:-10750000000} # 10.75B
            ;;
        "3200M")
            export N_LAYER=28
            export N_EMBD=3072
            export N_HEAD=24
            export LR=${LR:-0.000075}
            export TOKENS=${TOKENS:-20000000000} # 20B
            ;;
        *)
            echo "Unknown model prefix: $prefix"
            return 1
            ;;
    esac
}

export W_QUANT="HadamardMSEQuantizer"
export A_QUANT="HadamardMSEQuantizer"
export W_BITS=4
export A_BITS=4


export WANDB_ENTITY="keisufaj-hamad-bin-khalifa-university"
# export WANDB_PROJECT="RUSH-GraDe-pretraining"
# export WANDB_PROJECT="CAGE-QCRI-Momentum"
export WANDB_PROJECT="ECO0"

if [ -z "${NUM_GPUS}" ]; then
    if [ -n "${CUDA_VISIBLE_DEVICES}" ]; then
        export NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | grep -cv '^$')
        if [ "${NUM_GPUS}" -eq 0 ]; then
            export NUM_GPUS=1
        fi
    elif command -v nvidia-smi &> /dev/null; then
        export NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
        if [ "${NUM_GPUS}" -eq 0 ]; then
            export NUM_GPUS=1
        fi
    else
        export NUM_GPUS=1
    fi
else
    export NUM_GPUS=1
fi

export OPT="adamw"
# export OPT="qcriadamw"
# export OPT="qcriadamwgradaccumulator"
# export OPT="naive"

export USE_CAGE=False
export CAGE_LAMBDA=10
export CAGE_SILENCE_RATIO=0.8
export CAGE_SCHEDULE="linear_ramp"
export CAGE_TRACK_STATS=True
export CARRY_DECAY=1

export SEED=0
export RUN_ID=$RANDOM

# Check for dataset directories in order of preference
export DATASETS_DIR="/image-generation/kisufaj/optimization/cage/CAGE/datasets"

echo "Using datasets from $DATASETS_DIR"


for arg in "$@"; do
    if [[ $arg == --*=* ]]; then
        key="${arg#--}"
        key="${key%%=*}"
        value="${arg#*=}"
        key=$(echo "$key" | tr '[:lower:]' '[:upper:]' | tr '-' '_')
        export "$key"="$value"
    fi
done

set_model_config ${MODEL_SIZE_PREFIX}

if [[ -n "${TOKENS}" && "${TOKENS}" =~ ^([0-9]+([.][0-9]+)?)B$ ]]; then
    TOKENS_NUM="${BASH_REMATCH[1]}"
    TOKENS=$(awk "BEGIN {printf \"%.0f\", ${TOKENS_NUM} * 1000000000}")
    export TOKENS
elif [[ -n "${TPP}" ]]; then
    MODEL_SIZE_NUM="${MODEL_SIZE_PREFIX//[!0-9]/}"
    export TOKENS=$(awk "BEGIN {printf \"%.0f\", ${MODEL_SIZE_NUM} * ${TPP} * 1000000}")
fi

ITERATIONS=$((TOKENS / (BATCH_SIZE * ACC_STEPS * SEQUENCE_LENGTH)))

# ITERATIONS=5000
WARMUP_STEPS=$((ITERATIONS / 10))
# WARMUP_STEPS=$((ITERATIONS / 2))

# WANDB_PREFIX="NAIVE-QUANTIZER-LLAMA-${MODEL_SIZE_PREFIX}-CAGE=${USE_CAGE}-CAGE_L=${CAGE_LAMBDA}-CAGE_S_R=${CAGE_SILENCE_RATIO}-CAGE_SCH=${CAGE_SCHEDULE}-${W_QUANT}@${W_BITS}:${A_QUANT}@${A_BITS}-${DATASET}-${RUN_ID}"
# WANDB_PREFIX="NAIVE QUANTIZATION (8 bits): WEIGHTS + ACTIVATIONS"
# WANDB_PREFIX="QCRI ADAMW GRAD ACCUMULATOR: WEIGHTS + ACTIVATIONS-CARRY_DECAY=${CARRY_DECAY}-TAU=${W_QUANT_KWARGS}-${RUN_ID}"
WANDB_PREFIX="CAGE-LR=${LR}-BETA1=${BETA1}-BETA2=${BETA2}-${RUN_ID}"
# Quantizer kwargs - just bits for dynamic Q99IntQuantizer
# export W_QUANT_KWARGS="{\"bits\": ${W_BITS}}"
# export A_QUANT_KWARGS="{\"bits\": ${A_BITS}}"

# torchrun --master_addr="${MASTER_ADDR:-127.0.0.1}" \
#     --master_port="${MASTER_PORT:-29501}" \
#     --nproc_per_node=${NUM_GPUS} ./src/main.py \
#     --distributed-backend nccl \
#     --dataset ${DATASET} \
#     --datasets-dir ${DATASETS_DIR} \
#     --model llama \
#     --opt ${OPT} \
#     --use-cage ${USE_CAGE} \
#     --cage-lambda ${CAGE_LAMBDA} \
#     --cage-silence-ratio ${CAGE_SILENCE_RATIO} \
#     --cage-schedule ${CAGE_SCHEDULE} \
#     --cage-track-stats ${CAGE_TRACK_STATS} \
#     --compile \
#     --latest-ckpt-interval 1000 \
#     --acc-steps ${ACC_STEPS} \
#     --batch-size ${BATCH_SIZE} \
#     --seed ${SEED} \
#     --wandb \
#     --wandb-project "${WANDB_PROJECT}" \
#     --wandb-run-prefix "${WANDB_PREFIX}" \
#     --n-layer ${N_LAYER} \
#     --n-embd ${N_EMBD} \
#     --n-head ${N_HEAD} \
#     --warmup-steps ${WARMUP_STEPS} \
#     --iterations ${ITERATIONS} \
#     --lr ${LR} \
#     --w-quant ${W_QUANT} \
#     --w-quant-kwargs "${W_QUANT_KWARGS}" \
#     --a-quant ${A_QUANT} \
#     --a-quant-kwargs "${A_QUANT_KWARGS}" \
#     --carry-decay ${CARRY_DECAY}


torchrun --master_addr="${MASTER_ADDR:-127.0.0.1}" \
    --master_port="${MASTER_PORT:-29501}" \
    --nproc_per_node=${NUM_GPUS} ./src/main.py \
    --distributed-backend nccl \
    --dataset ${DATASET} \
    --datasets-dir ${DATASETS_DIR} \
    --model llama \
    --opt ${OPT} \
    --use-cage ${USE_CAGE} \
    --cage-lambda ${CAGE_LAMBDA} \
    --cage-silence-ratio ${CAGE_SILENCE_RATIO} \
    --cage-schedule ${CAGE_SCHEDULE} \
    --cage-track-stats ${CAGE_TRACK_STATS} \
    --compile \
    --latest-ckpt-interval 1000 \
    --acc-steps ${ACC_STEPS} \
    --batch-size ${BATCH_SIZE} \
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
    --w-quant ${W_QUANT} \
    --a-quant ${A_QUANT} \
    --beta1 ${BETA1} \
    --beta2 ${BETA2}
