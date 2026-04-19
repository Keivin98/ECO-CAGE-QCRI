#!/bin/bash


eval "$('/home/local/QCRI/kisufaj/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
conda activate /home/local/QCRI/kisufaj/miniconda3/envs/qwen/

export CUDA_VISIBLE_DEVICES=6,7
export MASTER_PORT=29503
export MASTER_ADDR=127.0.0.1
export TORCH_COMPILE=0
export TORCH_COMPILE_DISABLE=1
export TORCHINDUCTOR_DISABLE_TRITON=1
export TORCHINDUCTOR_CACHE_DIR=/image-generation/.tmp/torchinductor_$USER
export TRITON_CACHE_DIR=/image-generation/.tmp/triton_$USER

lrs=(0.0012 0.0025 0.005 0.00625 0.01 0.02 0.03 0.05)
taus=(0.5)
beta1s=(0.8 0.9)
beta2s=(0.95)

for lr in "${lrs[@]}"; do
  for tau in "${taus[@]}"; do
    for beta1 in "${beta1s[@]}"; do
      for beta2 in "${beta2s[@]}"; do
      echo "Running: lr=$lr tau=$tau beta1=$beta1 beta2=$beta2"

      bash train.sh \
        --model-size-prefix="tiny" \
        --lr="$lr" \
        --w-bits=4 \
        --a-bits=4 \
        --use-cage=True \
        --cage-lambda=2.5 \
        --cage-silence-ratio=0.9 \
        --batch-size=64 \
        --acc-steps=8 \
        --w-quant=HadamardMSEQuantizer \
        --a-quant=NoQuantizer \
        --beta1="$beta1" \
        --beta2="$beta2"
      done
    done
  done
done