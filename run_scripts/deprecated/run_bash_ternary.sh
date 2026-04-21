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

TAUS=(0.9)
# TAUS=(0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95 0.99)

LRS=(0.00090 0.00105 0.00120)
# LRS=(0.00008 0.00012 0.00018 0.00026 0.00038 0.00055 0.00075 0.00090 0.00105 0.00120 0.00160 0.00210 0.00250)


nT=${#TAUS[@]}
nL=${#LRS[@]}

echo "TAUS: ${TAUS[*]}"
echo "LRS : ${LRS[*]}"
echo "Total runs = $((nT * nL))"

run_one () {
  local TAU="$1"
  local LR="$2"

  echo "===================================================="
  echo "Running TAU=$TAU | LR=$LR"
  echo "===================================================="

  # Optional: nice run naming (if your train script passes through to wandb)
  export WANDB_NAME="50M_w4_a4_tau${TAU}_lr${LR}"
  export WANDB_RUN_NAME="$WANDB_NAME"

  # bash train_no_activations.sh \
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
}

# Diagonal traversal of the full grid:
# iterate sums s = i + j, so you hit (tau0,lr0), (tau1,lr1), ... early.
total=0
for s in $(seq 0 $((nT + nL - 2))); do
  for i in $(seq 0 $((nT - 1))); do
    j=$((s - i))
    if [[ $j -ge 0 && $j -lt $nL ]]; then
      TAU="${TAUS[$i]}"
      LR="${LRS[$j]}"
      total=$((total + 1))
      echo "[${total}/$((nT*nL))] diagonal s=$s (i=$i,j=$j)"
      run_one "$TAU" "$LR"
    fi
  done
done

echo "Done. Ran $total experiments."