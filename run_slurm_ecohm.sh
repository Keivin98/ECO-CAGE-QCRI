#!/bin/bash -l
#SBATCH --gres gpu:4
#SBATCH --output=outs/%j.out
#SBATCH -c 16 #number of CPUs needed, default is 1
#SBATCH --mem 200000MB #amount of memory needed, default is 4096 MB per core

eval "$('/home/local/QCRI/kisufaj/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"

conda activate /home/local/QCRI/kisufaj/miniconda3/envs/qwen/

# export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_PORT=29501
export MASTER_ADDR=127.0.0.1 
export TORCH_COMPILE=0
export TORCH_COMPILE_DISABLE=1
export TORCHINDUCTOR_DISABLE_TRITON=1
export TORCHINDUCTOR_CACHE_DIR=/image-generation/.tmp/torchinductor_$USER
export TRITON_CACHE_DIR=/image-generation/.tmp/triton_$USER
# bash train.sh --model-size-prefix=50M --w-bits=4 --a-bits=4 --cage-lambda=2.5 --cage-silence-ratio=0.95 --batch-size=64 --acc-steps=8 --w-quant=NoQuantizer --a-quant=NoQuantizer


bash train_ecohm.sh \
    --model-size-prefix=50M \
    --w-bits=4 \
    --a-bits=4 \
    --cage-lambda=2.5 \
    --cage-silence-ratio=0.95 \
    --batch-size=64 \
    --acc-steps=8 \
    --w-quant=Q99IntQuantizer \
    --a-quant=NoQuantizer \
    --w-quant-kwargs='{"bits":4,"tau":0.5}' \
    --a-quant-kwargs='{"bits":4,"calibrate_once":true,"tau":0.5}'