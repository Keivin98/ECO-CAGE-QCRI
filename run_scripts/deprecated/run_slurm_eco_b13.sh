#!/bin/bash
set -euo pipefail

slot="${1:-}"

if [[ "$slot" == "1" ]]; then
  export CUDA_VISIBLE_DEVICES=0,1
  lrs=(0.0006 0.0012)
  beta1s=(0.8 0.9)
  export MASTER_PORT=29504
elif [[ "$slot" == "2" ]]; then
  export CUDA_VISIBLE_DEVICES=2,3
  lrs=(0.0025 0.005)
  beta1s=(0.8 0.9)
  export MASTER_PORT=29506
elif [[ "$slot" == "3" ]]; then
  export CUDA_VISIBLE_DEVICES=4,5
  lrs=(0.00625 0.01)
  beta1s=(0.8 0.9)
  export MASTER_PORT=29507
elif [[ "$slot" == "4" ]]; then
  export CUDA_VISIBLE_DEVICES=6,7
  lrs=(0.02 0.03)
  beta1s=(0.8 0.9)
  export MASTER_PORT=29508
else
  echo "Usage: bash run.sh {1|2|3|4}"
  exit 1
fi

eval "$('/home/local/QCRI/kisufaj/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
conda activate /home/local/QCRI/kisufaj/miniconda3/envs/qwen/

export MASTER_ADDR=127.0.0.1
export TORCH_COMPILE=0
export TORCH_COMPILE_DISABLE=1
export TORCHINDUCTOR_DISABLE_TRITON=1
export TORCHINDUCTOR_CACHE_DIR=/image-generation/.tmp/torchinductor_$USER
export TRITON_CACHE_DIR=/image-generation/.tmp/triton_$USER

taus=(0.5)
beta2s=(0.95)

echo "Running slot $slot"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "MASTER_PORT=$MASTER_PORT"
echo "beta1s=${beta1s[*]}"

for lr in "${lrs[@]}"; do
  for tau in "${taus[@]}"; do
    for beta1 in "${beta1s[@]}"; do
      for beta2 in "${beta2s[@]}"; do
      echo "Running: lr=$lr tau=$tau beta1=$beta1 beta2=$beta2"

      bash train_eco0m-rooh.sh \
        --model-size-prefix="30M" \
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

      bash train_eco.sh \
        --model-size-prefix="30M" \
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
        --beta2="$beta2"
      done
    done
  done
done

# for lr in "${lrs[@]}"; do
#   for tau in "${taus[@]}"; do
#     for beta1 in "${beta1s[@]}"; do
#       for beta2 in "${beta2s[@]}"; do
#       echo "Running: lr=$lr tau=$tau beta1=$beta1 beta2=$beta2"

#       bash train.sh \
#         --model-size-prefix="30M" \
#         --lr="$lr" \
#         --w-bits=4 \
#         --a-bits=4 \
#         --use-cage=True \
#         --cage-lambda=2.5 \
#         --cage-silence-ratio=0.9 \
#         --batch-size=64 \
#         --acc-steps=8 \
#         --w-quant=HadamardMSEQuantizer \
#         --a-quant=NoQuantizer \
#         --beta1="$beta1" \
#         --beta2="$beta2"
#       done
#     done
#   done
# done