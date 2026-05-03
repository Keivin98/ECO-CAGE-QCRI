#!/bin/bash -l
#SBATCH -J upper_close
#SBATCH -o outs/upper_close_%A_%a.out
#SBATCH -e outs/upper_close_%A_%a.err
#SBATCH -p gpu-H200
#SBATCH --exclude=crirdchpxd001
#SBATCH --gres=gpu:H200_141GB:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=32000
#SBATCH -A H200_16GPUs
#SBATCH -q h200_qos_16_gpus
#SBATCH --array=1-12

# ==========================================================================
# Upper-bracket closer + 30M baseline LR sweeps (Adam / CAGE / STE)
# ==========================================================================
# Tracking ID: EXP-COS-UPPER-BRACKET
#
# Two purposes in one array:
#
# (a) Close upper LR brackets where existing 3-pt cos sweeps put the
#     best LR at the top of the tested range (i.e., we don't yet know if
#     a higher LR would do better):
#       30M ECO   : 0.003=28.75, 0.004=28.91, 0.005=28.81  -> opt ≥ 0.005
#       50M ECO-0 : 0.004=24.731, 0.005=23.73, 0.006=23.314 -> opt ≥ 0.006
#       50M ECO   : 0.004=26.63, 0.005=24.277               -> opt ≥ 0.005
#       100M ECO  : 0.004=20.42, 0.005=20.312               -> opt ≥ 0.005
#       100M ECO-0: already running 0.006 elsewhere (job 283435_1, skip)
#       300M      : both bracketed (skip)
#
# (b) 30M LR sweep for the three Adam-based baselines (FP16 Adam, CAGE INT4,
#     STE INT4). Currently we only have a single-LR point per method at
#     LR=0.0012 — adding LR=0.0008 and LR=0.0016 gives a clean 3-pt bracket
#     at ±33% around 0.0012.
#
# Tasks (all cos, P99.5 for INT4, seed 0):
#   1: 30M ECO   INT4 LR=0.006
#   2: 30M ECO   INT4 LR=0.007
#   3: 30M Adam  FP16 LR=0.0008
#   4: 30M Adam  FP16 LR=0.0016
#   5: 30M CAGE  INT4 LR=0.0008
#   6: 30M CAGE  INT4 LR=0.0016
#   7: 30M STE   INT4 LR=0.0008
#   8: 30M STE   INT4 LR=0.0016
#   9: 50M ECO   INT4 LR=0.006
#  10: 50M ECO   INT4 LR=0.007
#  11: 50M ECO-0 INT4 LR=0.007
#  12: 100M ECO  INT4 LR=0.006
#
# 12 tasks × 2 GPUs = 24 GPUs of demand. The 16-GPU QoS will start ~8 tasks
# immediately; the rest queue. 30M tasks finish in ~1h freeing 8 GPUs, so
# everything dispatches within 1-2h of submission.
# ==========================================================================

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cage

source /export/home/keisufaj/optimization/ECO-CAGE-QCRI/run_scripts/utils/setup_tmp_cleanup.sh

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((32000 + SLURM_ARRAY_TASK_ID))
export TORCH_COMPILE=0
export TORCH_COMPILE_DISABLE=1
export TORCHINDUCTOR_DISABLE_TRITON=1

export WANDB_ENTITY="keisufaj-hamad-bin-khalifa-university"

# Copy datasets to fast scratch space (job-specific)
export DATASETS_DIR="/scratch/keisufaj_datasets_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
if [ -d "${DATASETS_DIR}/c4" ]; then
    echo "Dataset already exists, skipping copy"
else
    mkdir -p ${DATASETS_DIR}
    rsync -a --info=progress2 /export/home/keisufaj/optimization/ECO-CAGE-QCRI/datasets/ ${DATASETS_DIR}/
fi

export SEQUENCE_LENGTH=512
BETA1=0.9
BETA2=0.95
SCHEDULER="cos"

# Default INT4 P99.5 quantizer for INT4 tasks; overridden per-task for FP16.
W_QUANT="Q99IntQuantizer"
W_QUANT_KWARGS='{"bits":4,"percentile":99.5}'

case ${SLURM_ARRAY_TASK_ID} in
    1)  # 30M ECO INT4 cos LR=0.006
        MODEL_SIZE="30M"; N_LAYER=6; N_EMBD=640; N_HEAD=5
        BATCH_SIZE=64; ACC_STEPS=8
        TOKENS=3000000000
        METHOD="ECO-4bit-INT4"; OPT="eco"; USE_CAGE="False"; LR=0.006
        WANDB_PROJECT="ECO0-30M-INT4-RERUN"
        ;;
    2)  # 30M ECO INT4 cos LR=0.007
        MODEL_SIZE="30M"; N_LAYER=6; N_EMBD=640; N_HEAD=5
        BATCH_SIZE=64; ACC_STEPS=8
        TOKENS=3000000000
        METHOD="ECO-4bit-INT4"; OPT="eco"; USE_CAGE="False"; LR=0.007
        WANDB_PROJECT="ECO0-30M-INT4-RERUN"
        ;;
    3)  # 30M FP16 Adam cos LR=0.0008
        MODEL_SIZE="30M"; N_LAYER=6; N_EMBD=640; N_HEAD=5
        BATCH_SIZE=64; ACC_STEPS=8
        TOKENS=3000000000
        METHOD="FP16-Adam"; OPT="adamw"; USE_CAGE="False"; LR=0.0008
        W_QUANT="NoQuantizer"; W_QUANT_KWARGS='{"bits":16}'
        WANDB_PROJECT="ECO0-30M-INT4-RERUN"
        ;;
    4)  # 30M FP16 Adam cos LR=0.0016
        MODEL_SIZE="30M"; N_LAYER=6; N_EMBD=640; N_HEAD=5
        BATCH_SIZE=64; ACC_STEPS=8
        TOKENS=3000000000
        METHOD="FP16-Adam"; OPT="adamw"; USE_CAGE="False"; LR=0.0016
        W_QUANT="NoQuantizer"; W_QUANT_KWARGS='{"bits":16}'
        WANDB_PROJECT="ECO0-30M-INT4-RERUN"
        ;;
    5)  # 30M CAGE INT4 cos LR=0.0008
        MODEL_SIZE="30M"; N_LAYER=6; N_EMBD=640; N_HEAD=5
        BATCH_SIZE=64; ACC_STEPS=8
        TOKENS=3000000000
        METHOD="CAGE-4bit-INT4"; OPT="adamw"; USE_CAGE="True"; LR=0.0008
        WANDB_PROJECT="ECO0-30M-INT4-RERUN"
        ;;
    6)  # 30M CAGE INT4 cos LR=0.0016
        MODEL_SIZE="30M"; N_LAYER=6; N_EMBD=640; N_HEAD=5
        BATCH_SIZE=64; ACC_STEPS=8
        TOKENS=3000000000
        METHOD="CAGE-4bit-INT4"; OPT="adamw"; USE_CAGE="True"; LR=0.0016
        WANDB_PROJECT="ECO0-30M-INT4-RERUN"
        ;;
    7)  # 30M STE INT4 cos LR=0.0008
        MODEL_SIZE="30M"; N_LAYER=6; N_EMBD=640; N_HEAD=5
        BATCH_SIZE=64; ACC_STEPS=8
        TOKENS=3000000000
        METHOD="STE-4bit-INT4"; OPT="adamw"; USE_CAGE="False"; LR=0.0008
        WANDB_PROJECT="ECO0-30M-INT4-RERUN"
        ;;
    8)  # 30M STE INT4 cos LR=0.0016
        MODEL_SIZE="30M"; N_LAYER=6; N_EMBD=640; N_HEAD=5
        BATCH_SIZE=64; ACC_STEPS=8
        TOKENS=3000000000
        METHOD="STE-4bit-INT4"; OPT="adamw"; USE_CAGE="False"; LR=0.0016
        WANDB_PROJECT="ECO0-30M-INT4-RERUN"
        ;;
    9)  # 50M ECO INT4 cos LR=0.006
        MODEL_SIZE="50M"; N_LAYER=7; N_EMBD=768; N_HEAD=6
        BATCH_SIZE=64; ACC_STEPS=8
        TOKENS=5000000000
        METHOD="ECO-4bit-INT4"; OPT="eco"; USE_CAGE="False"; LR=0.006
        WANDB_PROJECT="ECO0-50M-INT4-RERUN"
        ;;
    10) # 50M ECO INT4 cos LR=0.007
        MODEL_SIZE="50M"; N_LAYER=7; N_EMBD=768; N_HEAD=6
        BATCH_SIZE=64; ACC_STEPS=8
        TOKENS=5000000000
        METHOD="ECO-4bit-INT4"; OPT="eco"; USE_CAGE="False"; LR=0.007
        WANDB_PROJECT="ECO0-50M-INT4-RERUN"
        ;;
    11) # 50M ECO-0 INT4 cos LR=0.007
        MODEL_SIZE="50M"; N_LAYER=7; N_EMBD=768; N_HEAD=6
        BATCH_SIZE=64; ACC_STEPS=8
        TOKENS=5000000000
        METHOD="ECO0-4bit-INT4"; OPT="eco0m-rooh"; USE_CAGE="False"; LR=0.007
        WANDB_PROJECT="ECO0-50M-INT4-RERUN"
        ;;
    12) # 100M ECO INT4 cos LR=0.006
        MODEL_SIZE="100M"; N_LAYER=8; N_EMBD=1024; N_HEAD=8
        BATCH_SIZE=32; ACC_STEPS=16
        TOKENS=10000000000
        METHOD="ECO-4bit-INT4"; OPT="eco"; USE_CAGE="False"; LR=0.006
        WANDB_PROJECT="ECO0-100M-INT4-RERUN"
        ;;
esac

export N_LAYER N_EMBD N_HEAD BATCH_SIZE ACC_STEPS
export WANDB_PROJECT

TOKENS_PER_ITER=$((BATCH_SIZE * ACC_STEPS * SEQUENCE_LENGTH))
ITERATIONS=$((TOKENS / TOKENS_PER_ITER))
WARMUP_STEPS=$((ITERATIONS / 10))

WANDB_PREFIX="${MODEL_SIZE}-${METHOD}-sched=${SCHEDULER}-LR=${LR}-ITER=${ITERATIONS}"

echo "=========================================="
echo "Upper-bracket closer — Task ${SLURM_ARRAY_TASK_ID}/12"
echo "  Scale:     ${MODEL_SIZE}"
echo "  Method:    ${METHOD}"
echo "  Optimizer: ${OPT}  (use_cage=${USE_CAGE})"
echo "  Quantizer: ${W_QUANT}"
echo "  Scheduler: ${SCHEDULER}"
echo "  LR:        ${LR}"
echo "  Iter:      ${ITERATIONS} (${TOKENS} tokens)"
echo "  Project:   ${WANDB_PROJECT}"
echo "=========================================="

torchrun --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    --nproc_per_node=2 /export/home/keisufaj/optimization/ECO-CAGE-QCRI/src/main.py \
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
    --scheduler ${SCHEDULER} \
    --w-quant ${W_QUANT} \
    --w-quant-kwargs "${W_QUANT_KWARGS}" \
    --a-quant NoQuantizer \
    --beta1 ${BETA1} \
    --beta2 ${BETA2}

echo "Task ${SLURM_ARRAY_TASK_ID}/12 complete: ${MODEL_SIZE} ${METHOD} cos LR=${LR}"

rm -rf ${DATASETS_DIR}
