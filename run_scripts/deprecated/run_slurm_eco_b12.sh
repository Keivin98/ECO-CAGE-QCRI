#!/bin/bash
set -euo pipefail

slot="${1:-}"

case "$slot" in
  "1")
    export CUDA_VISIBLE_DEVICES=0,1
    export MASTER_PORT=29600
    script="train_eco0m-rooh.sh"
    model="50M"
    lr="0.00484"
    beta1="0.9"
    ;;
  "2")
    export CUDA_VISIBLE_DEVICES=2,3
    export MASTER_PORT=29602
    script="train_eco0m-rooh.sh"
    model="100M"
    lr="0.00342"
    beta1="0.9"
    ;;
  "3")
    export CUDA_VISIBLE_DEVICES=4,5
    export MASTER_PORT=29604
    script="train_eco.sh"
    model="50M"
    lr="0.00387"
    beta1="0.8"
    ;;
  "4")
    export CUDA_VISIBLE_DEVICES=6,7
    export MASTER_PORT=29606
    script="train_eco.sh"
    model="100M"
    lr="0.00274"
    beta1="0.8"
    ;;
  *)
    echo "Usage: bash run_scaled_eco_split.sh {1|2|3|4}"
    exit 1
    ;;
esac

eval "$('/home/local/QCRI/kisufaj/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
conda activate /home/local/QCRI/kisufaj/miniconda3/envs/qwen/

export MASTER_ADDR=127.0.0.1
export TORCH_COMPILE=0
export TORCH_COMPILE_DISABLE=1
export TORCHINDUCTOR_DISABLE_TRITON=1
export TORCHINDUCTOR_CACHE_DIR=/image-generation/.tmp/torchinductor_$USER
export TRITON_CACHE_DIR=/image-generation/.tmp/triton_$USER

echo "=================================================="
echo "slot=$slot"
echo "script=$script"
echo "model=$model"
echo "lr=$lr"
echo "beta1=$beta1"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "MASTER_PORT=$MASTER_PORT"
echo "=================================================="

bash "$script" \
  --model-size-prefix="$model" \
  --lr="$lr" \
  --w-bits=4 \
  --a-bits=4 \
  --cage-lambda=2.5 \
  --cage-silence-ratio=0.95 \
  --batch-size=64 \
  --acc-steps=8 \
  --w-quant=Q99FP4Quantizer \
  --a-quant=NoQuantizer \
  --w-quant-kwargs="{\"bits\":4}" \
  --a-quant-kwargs="{\"bits\":4,\"calibrate_once\":true}" \
  --beta1="$beta1" \
  --beta2="0.95"