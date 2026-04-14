#!/bin/bash
#SBATCH -w a3mega2new-a3meganodeset-0
#SBATCH --gres gpu:2
#SBATCH --output=outs/%j.out
#SBATCH -c 16 #number of CPUs needed, default is 1
#SBATCH --mem 300000MB #amount of memory needed, default is 4096 MB per core
eval "$('/home/local/QCRI/kisufaj/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
conda activate /home/local/QCRI/kisufaj/miniconda3/envs/qwen/

export CUDA_VISIBLE_DEVICES=0,1
export MASTER_PORT=29501
export MASTER_ADDR=127.0.0.1
export TORCH_COMPILE=0
export TORCH_COMPILE_DISABLE=1
export TORCHINDUCTOR_DISABLE_TRITON=1
export TORCHINDUCTOR_CACHE_DIR=/image-generation/.tmp/torchinductor_$USER
export TRITON_CACHE_DIR=/image-generation/.tmp/triton_$USER

lrs=(0.0012 0.0025 0.005 0.01 0.02)
taus=(0.5)
beta1s=(0.9)
beta2s=(0.98)

for lr in "${lrs[@]}"; do
  for tau in "${taus[@]}"; do
    for beta1 in "${beta1s[@]}"; do
      for beta2 in "${beta2s[@]}"; do
      echo "Running: lr=$lr tau=$tau beta1=$beta1 beta2=$beta2"

      bash train_eco.sh \
        --model-size-prefix="tiny" \
        --lr="$lr" \
        --w-bits=4 \
        --a-bits=4 \
        --cage-lambda=2.5 \
        --cage-silence-ratio=0.95 \
        --batch-size=64 \
        --acc-steps=8 \
        --w-quant=Q99IntQuantizer \
        --a-quant=NoQuantizer \
        --w-quant-kwargs="{\"bits\":4,\"tau\":$tau}" \
        --a-quant-kwargs="{\"bits\":4,\"calibrate_once\":true,\"tau\":$tau}" \
        --beta1="$beta1" \
        --beta2="$beta2"
      done
    done
  done
done

for lr in "${lrs[@]}"; do
  for tau in "${taus[@]}"; do
    for beta1 in "${beta1s[@]}"; do
      for beta2 in "${beta2s[@]}"; do
      echo "Running: lr=$lr tau=$tau beta1=$beta1 beta2=$beta2"

      bash train_eco0m.sh \
        --model-size-prefix=tiny \
        --lr="$lr" \
        --w-bits=4 \
        --a-bits=4 \
        --cage-lambda=2.5 \
        --cage-silence-ratio=0.95 \
        --batch-size=64 \
        --acc-steps=8 \
        --w-quant=Q99IntQuantizer \
        --a-quant=NoQuantizer \
        --w-quant-kwargs="{\"bits\":4,\"tau\":$tau}" \
        --a-quant-kwargs="{\"bits\":4,\"calibrate_once\":true,\"tau\":$tau}" \
        --beta1="$beta1" \
        --beta2="$beta2"
      done
    done
  done
done

