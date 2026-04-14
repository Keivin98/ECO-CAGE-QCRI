#!/bin/bash

eval "$('/home/local/QCRI/kisufaj/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"

conda activate /home/local/QCRI/kisufaj/miniconda3/envs/qwen/

export CUDA_VISIBLE_DEVICES=4,5,6,7
export MASTER_PORT=29500
export MASTER_ADDR=127.0.0.1 
export TORCH_COMPILE=0
export TORCH_COMPILE_DISABLE=1
export TORCHINDUCTOR_DISABLE_TRITON=1
export TORCHINDUCTOR_CACHE_DIR=/image-generation/.tmp/torchinductor_$USER
export TRITON_CACHE_DIR=/image-generation/.tmp/triton_$USER
export HF_HOME=/home/local/QCRI/kisufaj/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/hub
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
# bash train.sh --model-size-prefix=50M --w-bits=4 --a-bits=4 --cage-lambda=2.5 --cage-silence-ratio=0.95 --batch-size=64 --acc-steps=8 --w-quant=NoQuantizer --a-quant=NoQuantizer

mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

TAUS=(0.50 0.60 0.70 0.80)

# 3 LR candidates per TAU (same index as TAUS)
LRS_0=(0.00135 0.00180 0.00220 0.00280)
LRS_1=(0.00160 0.00220 0.00280 0.00350)
LRS_2=(0.00200 0.00260 0.00340 0.00420)

for i in "${!TAUS[@]}"; do
  TAU="${TAUS[$i]}"

  for LR in "${LRS_0[$i]}" "${LRS_1[$i]}" "${LRS_2[$i]}"; do
    echo "===================================================="
    echo "Running TAU=$TAU | LR=$LR"
    echo "===================================================="

    bash train_no_activations.sh \
      --model-size-prefix=50M \
      --w-bits=4 \
      --a-bits=4 \
      --cage-lambda=2.5 \
      --cage-silence-ratio=0.95 \
      --batch-size=64 \
      --acc-steps=8 \
      --w-quant=Q99IntQuantizer \
      --a-quant=NoQuantizer \
      --carry-decay=1.0 \
      --lr="$LR" \
      --w-quant-kwargs="{\"bits\":4,\"tau\":$TAU}" \
      --a-quant-kwargs="{\"bits\":4,\"calibrate_once\":true,\"tau\":$TAU}"
  done
done
