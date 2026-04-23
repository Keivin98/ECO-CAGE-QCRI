# GCP H100 Experiments - Quick Start Guide

Scripts for running experiments on GCP H100s without SLURM.

## Available Scripts

### 1. 50M STE Baseline
**File:** `50M_ste.sh`  
**GPUs:** 2× H100  
**Runtime:** ~5-6 hours  
**Purpose:** Fill missing STE baseline at 50M scale

```bash
# Create logs directory
mkdir -p logs

# Run on GPUs 0,1
CUDA_VISIBLE_DEVICES=0,1 bash run_scripts/gcp/50M_ste.sh > logs/50M_ste.out 2>&1 &

# Monitor progress
tail -f logs/50M_ste.out
```

**Expected result:** STE ~23.1-23.2 PPL (should match CAGE's 23.14)

---

### 2. 100M STE Baseline
**File:** `100M_ste.sh`  
**GPUs:** 2× H100  
**Runtime:** ~10-12 hours  
**Purpose:** Fill missing STE baseline at 100M scale

```bash
# Run on GPUs 2,3
CUDA_VISIBLE_DEVICES=2,3 bash run_scripts/gcp/100M_ste.sh > logs/100M_ste.out 2>&1 &

# Monitor progress
tail -f logs/100M_ste.out
```

**Expected result:** STE ~19.2-19.3 PPL (should match CAGE's 19.18)

---

### 3. 300M Scaling (FP16 + ECO-0)
**File:** `300M_fp16_eco0.sh`  
**GPUs:** 4× H100 total (2 per method)  
**Runtime:** ~24-30 hours per method  
**Purpose:** Next scaling point, validates linear memory scaling

```bash
# Method 1: FP16 Adam baseline (GPUs 4,5)
CUDA_VISIBLE_DEVICES=4,5 bash run_scripts/gcp/300M_fp16_eco0.sh 1 > logs/300M_fp16.out 2>&1 &

# Method 2: ECO-0 4-bit (GPUs 6,7)
CUDA_VISIBLE_DEVICES=6,7 bash run_scripts/gcp/300M_fp16_eco0.sh 2 > logs/300M_eco0.out 2>&1 &

# Monitor both
tail -f logs/300M_fp16.out
tail -f logs/300M_eco0.out
```

**Expected results:**
- FP16: ~15-16 PPL (baseline)
- ECO-0: ~16-17 PPL (competitive with ~3-4 GB memory savings)

---

## Running All Experiments in Parallel (8 H100s)

**Optimal GPU allocation:**

```bash
# Create logs directory
mkdir -p logs

# Launch all 4 experiments in parallel
CUDA_VISIBLE_DEVICES=0,1 bash run_scripts/gcp/50M_ste.sh > logs/50M_ste.out 2>&1 &
CUDA_VISIBLE_DEVICES=2,3 bash run_scripts/gcp/100M_ste.sh > logs/100M_ste.out 2>&1 &
CUDA_VISIBLE_DEVICES=4,5 bash run_scripts/gcp/300M_fp16_eco0.sh 1 > logs/300M_fp16.out 2>&1 &
CUDA_VISIBLE_DEVICES=6,7 bash run_scripts/gcp/300M_fp16_eco0.sh 2 > logs/300M_eco0.out 2>&1 &

# Check all jobs are running
jobs

# Monitor all logs in separate terminals/tmux panes
watch -n 60 tail -n 20 logs/50M_ste.out
watch -n 60 tail -n 20 logs/100M_ste.out
watch -n 60 tail -n 20 logs/300M_fp16.out
watch -n 60 tail -n 20 logs/300M_eco0.out
```

**Timeline:**
- **~6 hours:** 50M STE completes
- **~12 hours:** 100M STE completes
- **~30 hours:** 300M experiments complete

---

## Monitoring

### Check GPU usage
```bash
watch -n 1 nvidia-smi
```

### Check WandB
All experiments log to: **ECO0-SCALING** project

**Run name format:**
- `50M-STE-4bit-LR=0.00093-P90.0-BS=64x8-ITER=19073`
- `100M-STE-4bit-LR=0.0006-P90.0-BS=32x16-ITER=38146`
- `300M-FP16-Adam-LR=0.0004-BS=24x16-ITER=39062`
- `300M-ECO0-4bit-LR=0.0045-BS=24x16-ITER=39062`

### Check experiment progress
```bash
# Count iterations completed (search for "iter" in logs)
grep -o "iter [0-9]*" logs/50M_ste.out | tail -1

# Check final validation perplexity
grep "final-val/perplexity" logs/50M_ste.out
```

---

## Configuration Summary

| Experiment | Model | Tokens | Iterations | Batch | LR | Expected PPL |
|------------|-------|--------|------------|-------|-----|--------------|
| 50M STE | 7L/768D/6H | 5B | 19,073 | 64×8 | 0.00093 | ~23.15 |
| 100M STE | 8L/1024D/8H | 10B | 38,146 | 32×16 | 0.0006 | ~19.20 |
| 300M FP16 | 12L/1536D/12H | 15B | 39,062 | 24×16 | 0.0004 | ~15-16 |
| 300M ECO-0 | 12L/1536D/12H | 15B | 39,062 | 24×16 | 0.0045 | ~16-17 |

---

## Troubleshooting

### Out of memory
If you see CUDA OOM errors:
1. Reduce batch size in the script
2. Increase gradient accumulation steps to maintain effective batch
3. Check other processes using GPUs: `nvidia-smi`

### Dataset not found
Adjust `DATASETS_DIR` in each script to point to your C4 dataset location.

### Port already in use
Each script uses a unique port (29500, 29501, 29502+method_id). If you see port conflicts:
1. Check existing processes: `lsof -i :29500`
2. Modify `MASTER_PORT` in the script

### WandB authentication
If WandB fails to authenticate:
```bash
wandb login
# Or set WANDB_API_KEY in your environment
export WANDB_API_KEY=your_key_here
```

---

## Expected Impact on Paper

After these experiments complete, you'll have:

**Complete baseline table:**
| Scale | FP16 | STE | CAGE | ECO | ECO-0 |
|-------|------|-----|------|-----|-------|
| 30M | ✅ | ✅ | ✅ | ✅ | ✅ |
| 50M | ✅ | **NEW** ✅ | ✅ | ✅ | ✅ |
| 100M | ✅ | **NEW** ✅ | ✅ | ✅ | 🔄 |
| 300M | **NEW** ✅ | - | - | - | **NEW** ✅ |

**Key findings to document:**
1. **STE ≈ CAGE across scales** → validates curvature correction minimal
2. **300M scaling point** → shows memory advantage grows linearly
3. **Memory savings:** 30M (1GB) → 50M (1GB) → 100M (1.6GB) → 300M (~3-4GB)
4. **Performance trend:** Gap to CAGE narrows or stable?

**Estimated paper completion:**
- After 100M LR refinement + these experiments: ~80% complete
- Missing: Conclusion, figures, maybe 500M/1B
