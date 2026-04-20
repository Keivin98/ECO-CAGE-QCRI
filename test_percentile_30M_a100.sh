#!/bin/bash -l
#SBATCH -J percentile_30M_a100
#SBATCH -o outs/percentile_30M_a100_%A_%a.out
#SBATCH -e outs/percentile_30M_a100_%A_%a.err
#SBATCH -p gpu-A100
#SBATCH --gres=gpu:2
#SBATCH --mem=10000
#SBATCH -A A100
#SBATCH -q a100_qos
#SBATCH --array=1-2

# Run remaining ECO0 experiments (P95, P99) on A100
# Jobs 5-6 from original H200 script

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cage_cu128

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((29500 + SLURM_ARRAY_TASK_ID))
export TORCH_COMPILE=0
export TORCH_COMPILE_DISABLE=1
export TORCHINDUCTOR_DISABLE_TRITON=1

export WANDB_ENTITY="keisufaj-hamad-bin-khalifa-university"
export WANDB_PROJECT="FP4-PERCENTILE-30M"
export DATASETS_DIR="/export/home/keisufaj/optimization/ECO-CAGE-QCRI/datasets"

# 30M model config
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
ITERATIONS=5000
WARMUP_STEPS=500

# Map array 1-2 to original jobs 5-6
case ${SLURM_ARRAY_TASK_ID} in
    1)
        OPT="eco0m-rooh"
        PERCENTILE=95.0
        echo "Testing ECO-0 with P95 (on A100)"
        ;;
    2)
        OPT="eco0m-rooh"
        PERCENTILE=99.0
        echo "Testing ECO-0 with P99 (on A100)"
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
    --w-quant-kwargs "{\"bits\":4,\"percentile\":${PERCENTILE}}" \
    --a-quant NoQuantizer \
    --beta1 ${BETA1} \
    --beta2 ${BETA2}

echo "Job ${SLURM_ARRAY_TASK_ID} (ECO0 P${PERCENTILE} on A100) complete!"
