#!/bin/bash
set -euo pipefail

# ----------------------------
# Choose which chunk to run:
#   PART=1  -> first 14
#   PART=2  -> middle 14
#   PART=3  -> last 14
# If PART is not set, script prints all 42 pairs and exits.
# ----------------------------
PART="${PART:-}"

# --------- your missing pairs (42 total) ----------
# Format: "tau lr"
MISSING=(
  # tau=0.99
  "0.99 0.00026"
  "0.99 0.00038"

  # tau=0.95
  "0.95 0.00038"
  "0.95 0.00055"

  # tau=0.90
  "0.9 0.00055"
  "0.9 0.00075"
  "0.9 0.00090"
  "0.9 0.00105"
  "0.9 0.00250"

  # tau=0.80
  "0.8 0.00090"
  "0.8 0.00105"
  "0.8 0.00120"
  "0.8 0.00250"

  # tau=0.70
  "0.7 0.00090"
  "0.7 0.00105"
  "0.7 0.00120"
  "0.7 0.00160"
  "0.7 0.00250"

  # tau=0.60
  "0.6 0.00105"
  "0.6 0.00160"
  "0.6 0.00210"
  "0.6 0.00250"

  # tau=0.50
  "0.5 0.00075"
  "0.5 0.00090"
  "0.5 0.00105"
  "0.5 0.00120"
  "0.5 0.00160"
  "0.5 0.00210"
  "0.5 0.00250"

  # tau=0.40
  "0.4 0.00090"
  "0.4 0.00105"
  "0.4 0.00120"
  "0.4 0.00160"
  "0.4 0.00210"
  "0.4 0.00250"

  # tau=0.30
  "0.3 0.00105"
  "0.3 0.00120"
  "0.3 0.00160"
  "0.3 0.00210"
  "0.3 0.00250"

  # tau=0.20
  "0.2 0.00210"
  "0.2 0.00250"
)

TOTAL="${#MISSING[@]}"
if [[ "$TOTAL" -ne 42 ]]; then
  echo "ERROR: expected 42 missing pairs, got $TOTAL" >&2
  exit 1
fi

# Print-only mode
if [[ -z "${PART}" ]]; then
  echo "Missing pairs ($TOTAL):"
  for p in "${MISSING[@]}"; do echo "$p"; done
  echo
  echo "Run a chunk with: PART=1 $0   (or PART=2 / PART=3)"
  exit 0
fi

if [[ "$PART" != "1" && "$PART" != "2" && "$PART" != "3" ]]; then
  echo "ERROR: PART must be 1, 2, or 3" >&2
  exit 1
fi

# Compute slice [start, end)
CHUNK=14
START=$(( (PART - 1) * CHUNK ))
END=$(( START + CHUNK ))

echo "Running chunk PART=$PART: indices [$START, $END) out of $TOTAL"

# ----------------------------
# Your environment setup (copying your existing sweep setup)
# ----------------------------
eval "$('/home/local/QCRI/kisufaj/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
conda activate /home/local/QCRI/kisufaj/miniconda3/envs/qwen/

export CUDA_VISIBLE_DEVICES=4,5,6,7
export MASTER_PORT=29501
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

mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

run_one () {
  local TAU="$1"
  local LR="$2"

  echo "===================================================="
  echo "Running TAU=$TAU | LR=$LR"
  echo "===================================================="

  export WANDB_NAME="50M_w4_a4_tau${TAU}_lr${LR}"
  export WANDB_RUN_NAME="$WANDB_NAME"

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

# ----------------------------
# Execute selected slice
# ----------------------------
idx=0
for ((k=START; k<END; k++)); do
  pair="${MISSING[$k]}"
  TAU="$(awk '{print $1}' <<< "$pair")"
  LR="$(awk '{print $2}' <<< "$pair")"
  idx=$((idx+1))
  echo "[$idx/$CHUNK] $TAU $LR"
  run_one "$TAU" "$LR"
done

echo "Done PART=$PART."