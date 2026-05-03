#!/bin/bash -l
#SBATCH -J dyn_smoke
#SBATCH -o outs/dyn_smoke_%j.out
#SBATCH -e outs/dyn_smoke_%j.err
#SBATCH -p gpu-H200
#SBATCH --exclude=crirdchpxd001
#SBATCH --gres=gpu:H200_141GB:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16000
#SBATCH -A H200_16GPUs
#SBATCH -q h200_qos_16_gpus
#SBATCH --time=00:15:00

# ==========================================================================
# Smoke test for the per-bucket dynamics logging added to base.py + adam0.py.
#
# - Tiny model (3 layers, 128 dim, 2 heads, ~0.5M params)
# - 30 iterations with log_interval=2 -> multiple log iters exercised
# - ECO-0 + INT4 (exercises v_rel_err / e_norm / cos_g_e / EF magnitude paths)
# - Single GPU, runs in ~1-2 minutes
# - Reads dataset in-place (no rsync to /scratch -- throwaway run)
# - Logs to wandb project ECO0-SMOKE-TEST (separate from the real runs)
#
# Submit:
#   sbatch run_scripts/utils/smoke_test_dynamics_logging.sh
# ==========================================================================

set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cage

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29900
export TORCH_COMPILE=0
export TORCH_COMPILE_DISABLE=1
export TORCHINDUCTOR_DISABLE_TRITON=1

export WANDB_ENTITY="keisufaj-hamad-bin-khalifa-university"
export WANDB_PROJECT="ECO0-SMOKE-TEST"

# Read dataset in place; no scratch copy for this throwaway run.
DATASETS_DIR="/export/home/keisufaj/optimization/ECO-CAGE-QCRI/datasets"

# Tiny model
N_LAYER=3
N_EMBD=128
N_HEAD=2
BATCH_SIZE=8
ACC_STEPS=2
SEQUENCE_LENGTH=256

ITERATIONS=30
WARMUP_STEPS=5

LR=0.008
BETA1=0.9
BETA2=0.95

WANDB_PREFIX="smoke-dyn-logging-eco0-int4-iter=${ITERATIONS}"

echo "=========================================="
echo "Dynamics-logging smoke test"
echo "  Model:     tiny (${N_LAYER}L / ${N_EMBD}d / ${N_HEAD}h)"
echo "  Iter:      ${ITERATIONS} (log_interval=2, eval_interval=10)"
echo "  Optimizer: eco0m-rooh, scheduler cos, LR=${LR}"
echo "  Datasets:  ${DATASETS_DIR}  (no scratch copy)"
echo "  WandB:     ${WANDB_PROJECT}/${WANDB_PREFIX}"
echo "=========================================="

torchrun --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    --nproc_per_node=1 /export/home/keisufaj/optimization/ECO-CAGE-QCRI/src/main.py \
    --distributed-backend nccl \
    --dataset c4 \
    --datasets-dir "${DATASETS_DIR}" \
    --model llama \
    --opt eco0m-rooh \
    --use-cage False \
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
    --eval-interval 10 \
    --log-interval 2 \
    --lr ${LR} \
    --scheduler cos \
    --w-quant Q99IntQuantizer \
    --w-quant-kwargs '{"bits":4,"percentile":99.5}' \
    --a-quant NoQuantizer \
    --beta1 ${BETA1} \
    --beta2 ${BETA2}

echo ""
echo "Smoke test done. In wandb (${WANDB_PROJECT}/${WANDB_PREFIX}), verify:"
echo "  - dyn/{bucket}/theta_norm, grad_norm, delta_norm, eff_step_size"
echo "  - dyn/{bucket}/v_rel_err/mean, e_norm/mean, cos_g_e/mean, etc."
echo "  - hist/{bucket}/weights_iter_0  AND  hist/{bucket}/weights_iter_final"
echo "  - setup/grad_noise_scale"
echo "  - no Python errors / no crashes / loss decreases over 30 iters"
