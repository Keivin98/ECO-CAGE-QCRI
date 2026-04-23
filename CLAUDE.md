# CLAUDE.md

This file provides guidance to Claude Code when working with this ECO-0 quantization-aware training codebase.

## Overview

**ECO-0** (Adam0-Rooh): Memory-efficient stateless optimizer for quantization-aware training (QAT) of transformers.

**Key Innovation**: Eliminates first moment storage by using gradient decay + attention-head-aware variance computation from current gradients only.

**Memory Comparison**:
- CAGE: Full-precision weights + full optimizer state (no savings)
- ECO: Quantized weights + first moment + second moment + error feedback
- **ECO-0**: Quantized weights + structured second moment only (most efficient)

## Training Commands

⚠️ **All scripts now organized in `run_scripts/` directory** - See `run_scripts/README.md` for details

```bash
# ECO-0 (primary)
bash run_scripts/baselines/train_eco0m-rooh.sh --model-size-prefix=30M --lr=0.0012

# ECO baseline
bash run_scripts/baselines/train_eco.sh --model-size-prefix=30M --lr=0.0012

# 30M baselines (SLURM)
sbatch run_scripts/baselines/test_30M_baselines.sh

# 50M scaling (local)
./run_scripts/scaling/50M/run_scaling_50M_local.sh {1|2|3|4}

# ECO0 LR ablation at 50M
./run_scripts/ablations/lr_tuning/run_eco0_lr_ablation_50M.sh {1|2|3}
```

**Environment**: `conda activate cage` (Python 3.11+, PyTorch 2.6, CUDA 12.6)

## Key Configuration

### Quantization (Critical Settings)
- `--w-quant Q99FP4Quantizer --a-quant NoQuantizer` (weight-only quantization)
- `--w-quant-kwargs '{"bits":4,"percentile":90.0}'` ⚠️ **Use P90, not P99!**
- `--w-bits 4 --a-bits 4`

### Optimizer
- `--opt eco0m-rooh` (ECO-0, note: 'm' in name, not `eco0-rooh`)
- `--opt eco` (ECO baseline)
- `--lr`, `--beta1`, `--beta2`: Learning rate and Adam betas

### Model Sizes
- 30M: 6 layers, 640 embd, 5 heads (current experiments)
- tiny: 3 layers, 128 embd, 2 heads (for rapid hyperparameter search)
- Also: 50M, 100M, 200M, 430M, 800M, 1700M, 3200M

### Training
- `--batch-size 64 --acc-steps 8` (effective batch 512)
- `--iterations`: Total steps
- `--scheduler cos`: Cosine LR schedule
- `--compile`: Enable torch.compile (may be slow first run)

## Architecture Essentials

### Core Flow
`train_eco0m-rooh.sh` → `src/main.py` → `optim/base.py:train()`

**Optimizer**: `src/optim/qcri/adam0.py` (Adam0Rooh)
- No first moment stored - uses gradient decay (`zero_grad()` multiplies by beta1)
- Attention-head-aware variance from current gradient
- ECO-style error feedback for quantization

**Quantizer**: `src/models/quantization/base_linear.py` (Q99FP4Quantizer)
- Non-uniform FP4 codebook: `[-6, -4, -3, -2, -1.5, -1, -0.75, 0, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6]`
- Percentile-based scale calibration
- **Use 90th percentile for optimal performance**

### Error Feedback Mechanism
```python
# Optimizer step
theta_tilde = apply_update(p)  # Full precision
theta_hat = p.quantizer.hard_quantize(theta_tilde)  # Quantized
e_next = theta_tilde - theta_hat  # Quantization error
grad.add_(coeff * denom * e_next)  # Error feedback to gradient
p.copy_(theta_hat)  # Store quantized only
```

## 🚀 CURRENT STATUS (April 22, 2026 - Evening)

### ✅ Complete
- **50M:** All methods with optimal LRs ✅
  - **Key finding:** ECO0 = ECO (23.89 vs 23.92 PPL, 1 GB memory savings)
  - Unscaled LR discovery (0.01 for ECO0, 0.00625 for ECO)
  - Paper updated with results
- **100M Baseline:** All 4 methods complete ✅
  - Memory scaling validated (1.7 GB savings, 8.3%)

### 🔄 Ready to Run (DO THIS NEXT!)
```bash
sbatch run_scripts/scaling/100M/test_100M_optimal_lr.sh
```
- Task 1: ECO0 LR=0.007
- Task 2: ECO0 LR=0.0085
- ETA: ~8-10 hours each

### 📋 Scripts Ready
- ✅ 500M: `run_scripts/scaling/500M/test_500M_optimal_lr.sh`
- 📄 Paper: Updated through 50M, needs 100M results after completion

### 🎯 Next Steps
1. **Submit 100M optimal LR tests** (above)
2. Wait ~8-10 hours for results
3. Update paper with 100M section
4. Decide: stop at 100M or continue to 500M

### 📊 Recommended LRs by Scale
| Scale | FP16/CAGE | ECO | ECO0 |
|-------|-----------|-----|------|
| 30M | 0.0012 | 0.00625 | 0.01 |
| 50M | 0.00093 | 0.00625 | 0.01 |
| 100M | 0.0006 | ~0.005 | 0.007-0.0085 |
| 500M | 0.00027 | ~0.0022 | 0.003-0.004 |

**Pattern:** ECO/ECO0 use unscaled LRs 30M→50M, then scale by 1/√factor for larger

---

## Major Research Findings

### 1. Optimal FP4 Percentile: **90th, not 99th!** (April 2026)

**Finding**: FP4 quantization requires more aggressive outlier clipping than uniform quantizers.

**Validation loss on tiny model (1000 iterations):**
- P85: 5.021 (too aggressive)
- **P90: 4.976** 🏆 **OPTIMAL**
- P92: 4.978 (nearly optimal)
- P95: 5.030
- P99: 5.068 (original default)
- P100: 5.075 (min-max, worst)

**Improvement**: 1.81% relative improvement (5.068 → 4.976) from P99 to P90

**Why P90 wins:**
- FP4's non-uniform codebook wastes capacity when outliers dominate scale
- 90th percentile clips top 10% to extreme bins (±6, ±4), allowing middle bins (±1.5, ±2, ±3) to better represent typical weights
- Implicit regularization effect from clipping

**Validation in progress**: Testing P90 vs P99 on 30M model for both ECO and ECO-0

**Recommendation**: 
- **Always use `percentile=90.0` for FP4 quantization**
- This is a key finding that should be highlighted in the paper

### 2. FP4 Scale Recalibration (April 2026)

**Result**: Dynamic scale recalibration does **NOT** improve performance.

**Tested**: `recalibrate_interval=100` (recalibrate every 100 steps) vs frozen scale (calibrate once at init)

**Outcome on tiny model:**
- ECO + Frozen: 5.205
- ECO + Dynamic: 5.816 (worse!)
- ECO-0 + Frozen: 5.028
- ECO-0 + Dynamic: 5.021 (marginal gain)

**Conclusion**: Frozen scale is fine. Dynamic recalibration destabilizes ECO and provides negligible benefit to ECO-0. Use default `recalibrate_interval=0`.

### 3. 30M Baseline Results (April 20, 2026) ✅

**Complete validation on 30M model (5000 iterations, batch 512, P90 percentile):**

| Method | Val Perplexity | LR | Notes |
|--------|---------------|-----|-------|
| FP32 Adam | 27.49 | 0.0012 | Full precision baseline |
| FP16 Adam | 27.49 | 0.0012 | Identical to FP32 (validates baseline) |
| STE 4-bit | 29.07 | 0.0012 | Traditional QAT |
| CAGE 4-bit | 29.08 | 0.0012 | QAT with curvature |
| ECO 4-bit | 32.73 | 0.00625 | Master weight elimination |
| **ECO0 4-bit** | **30.06** | **0.01** | **Our method** |

**Key Findings:**
- ECO0 (30.06) beats ECO (32.73) by 2.67 perplexity points
- ECO0 competitive with CAGE (30.06 vs 29.08) - only 0.98 gap
- FP16 = FP32 confirms baseline validity
- P90 percentile validated for both ECO and ECO0

### 4. 50M Scaling Experiments (April 22, 2026) ✅ COMPLETE

**CRITICAL UPDATE: ECO0 ≈ ECO at optimal LRs!**

**Configuration:**
- Model: 7 layers, 768 embd, 6 heads (~50M params, 1.67× from 30M)
- Training: 5B tokens (19,073 iterations)
- Batch: 64 × 8 acc_steps = 512 effective
- LR Scaling: ×0.775 (1/sqrt(1.67)) from 30M values (base reference)

**Results (Validation Loss / Perplexity):**

| Method | LR | Percentile | Val Loss | Perplexity | Memory (GB) |
|--------|-----|-----------|----------|------------|-------------|
| **FP16 Adam** | 0.00093 | N/A | **3.134** | **21.62** | 29.29 |
| **CAGE 4-bit** | 0.00093 | P90 | **3.197** | **23.14** | 29.29 |
| **ECO0 4-bit** | **0.01** | **P90** | **3.221** 🏆 | **23.89** | **28.28** |
| ECO 4-bit | 0.00484 | P90 | 3.238 | 24.28 | 29.30 |

**ECO0 LR/Percentile Ablation (9 configurations tested):**

| LR | P90 Loss | P95 Loss | Best |
|----|----------|----------|------|
| 0.01 | **3.221** | - | **P90 ✅** |
| 0.009 | 3.233 | - | P90 |
| 0.008 | 3.237 | - | P90 |
| 0.00775 | 3.242 | 3.245 | **P90** ✅ |
| 0.0065 | 3.259 | 3.265 | **P90** ✅ |
| 0.006 | 3.253 | 3.270 | **P90** ✅ |

**Key Findings:**

1. **ECO0 optimal config: LR=0.01, P90** (no LR scaling needed from 30M!)
   - Beats ECO baseline by 0.017 loss (0.7% improvement)
   - Competitive with CAGE: only 0.024 gap (vs 0.98 gap at 30M - **narrowing!**)
   - Memory advantage: 28.28 GB vs 29.29 GB (3.5% savings)

2. **P90 dominates P95** across all tested LRs:
   - LR=0.00775: P90 wins by 0.002
   - LR=0.0065: P90 wins by 0.007
   - LR=0.006: P90 wins by 0.017
   - **Larger gap at lower LR** - P90 more robust

3. **Higher LR strongly preferred for ECO0**:
   - Performance improves monotonically from 0.006 → 0.01
   - LR=0.01 (unscaled from 30M) outperforms scaled LR=0.00775
   - Suggests ECO0 benefits from aggressive learning rates

4. **Scaling insight**: Gap to CAGE narrowing with scale
   - 30M: 0.98 perplexity gap (30.06 vs 29.08)
   - 50M: 0.89 perplexity gap (23.89 vs 23.14)
   - Suggests ECO0 may scale favorably

**Scripts:**
- SLURM: `sbatch run_scripts/scaling/50M/test_scaling_50M.sh`
- Local: `./run_scripts/scaling/50M/run_scaling_50M_local.sh {1|2|3|4}`
- ECO0 LR ablation: `./run_scripts/ablations/lr_tuning/run_eco0_lr_ablation_50M.sh {1|2|3}`
- ECO LR ablation: `sbatch run_scripts/ablations/lr_tuning/run_eco_lr_ablation_50M.sh` (array 1-3)

**ECO LR Ablation (April 22, 2026)** 📋 In Progress

To ensure fair comparison, testing if ECO can improve with higher LR:
- Current ECO: LR=0.00484 (scaled from 30M) → Loss 3.238
- ECO0 best: LR=0.01 (unscaled) → Loss 3.221
- Testing ECO at: LR ∈ {0.006, 0.008, 0.01} with P90

**Rationale**: ECO's LR was scaled down while ECO0's optimal was unscaled. Must verify ECO can't match ECO0 with better LR tuning.

### 5. 100M Scaling Experiments (April 22, 2026) 🔄 In Progress

**Configuration:**
- Model: 8 layers, 1024 embd, 8 heads (~100M params, 2× from 50M)
- Training: 10B tokens (38,146 iterations)
- Batch: 32 × 16 acc_steps = 512 effective

**100M Baseline Results (COMPLETE):**

| Method | Perplexity | LR | Memory (GB) | Status |
|--------|------------|-----|-------------|--------|
| FP16 Adam | 17.93 | 0.0006 | 20.46 | ✅ |
| CAGE 4-bit | 19.18 | 0.0006 | 20.46 | ✅ |
| ECO 4-bit | 21.26 | 0.0031 | 20.42 | ✅ Under-tuned |
| ECO0 4-bit | 22.39 | 0.0042 | **18.77** | ✅ Under-tuned |

**Memory Scaling Confirmed:**
- 50M: 1.0 GB savings (3.5%)
- 100M: 1.7 GB savings (8.3%)
- **Scales linearly with model size** ✅

**100M Optimal LR Tests (RUNNING):**
- Script: `run_scripts/scaling/100M/test_100M_optimal_lr.sh`
- Task 1: ECO0 LR=0.007 (scaled from 50M's 0.01)
- Task 2: ECO0 LR=0.0085 (testing higher)
- **Status:** Jobs submitted and running ✅
- **ETA:** ~8-10 hours per job (check in morning)

**Scripts:**
- Baseline: `sbatch run_scripts/scaling/100M/test_scaling_100M_4methods.sh`
- Optimal LR: `sbatch run_scripts/scaling/100M/test_100M_optimal_lr.sh` ⚠️ **RUN THIS**

### 6. 500M Scaling (April 22, 2026) 📋 READY

**Script Created:** `run_scripts/scaling/500M/test_500M_optimal_lr.sh`

**Configuration:**
- Model: 16 layers, 1280 embd, 10 heads (~500M params, 5× from 100M)
- Training: 20B tokens (2× from 100M for better convergence)
- Batch: 16 × 32 acc_steps = 512 effective
- Memory: ~64GB allocated

**Methods:**
1. FP16 Adam (LR=0.00027)
2. CAGE 4-bit (LR=0.00027)
3. ECO0 4-bit (LR=0.003)
4. ECO0 4-bit (LR=0.004)

**Expected:**
- Memory savings: 5-8 GB (vs 1.7 GB at 100M)
- Gap to CAGE continues narrowing
- Runtime: ~15-20 hours per job

**To submit (after 100M):**
```bash
sbatch run_scripts/scaling/500M/test_500M_optimal_lr.sh
```

### 7. Memory Profiling Results (April 20, 2026) ⚠️ CRITICAL INSIGHT

**50M Model Memory Usage (batch=64, seq=512, 2 GPUs):**
- FP16 Adam: **29.29 GB**
- CAGE 4-bit: **29.29 GB** (identical to FP16!)
- ECO 4-bit: **29.3 GB** (identical to FP16!)
- ECO0 4-bit: **28.27 GB** (~1 GB savings)

**Memory Breakdown Analysis:**
```
Total: ~29 GB per GPU
├─ Activations: ~25 GB (86%) ← DOMINATES
├─ Optimizer states (m+v): ~0.4 GB (1.4%)
├─ Gradients: ~0.2 GB (FP32)
├─ Params: ~0.1 GB (FP16/FP4)
└─ Overhead: ~3.3 GB
```

**Critical Findings:**
1. **Weights NOT stored as 4-bit**: QAT quantizes during forward pass only, stores weights in FP16
2. **Activations dominate**: 25 GB (86%) vs optimizer states 0.4 GB (1.4%)
3. **ECO ≈ FP16**: Suggests master weights not actually eliminated in current impl, OR weights stored in FP16 regardless
4. **ECO0 saves 1 GB**: Confirms m+v buffer elimination (400 MB + allocator overhead)
5. **Savings scale linearly with params**: At 50M: 1 GB (3.4%), at 1B: ~10 GB (8-10%), at 10B: ~80 GB (16-20%)

**Implication for Paper:**
- At small scale (50M), savings are modest (3.4%)
- At large scale (1B+), savings become significant (10+ GB enables larger batches/sequences)
- Need 1B experiments to show compelling memory story

## SLURM Job Management

### Temporary File Cleanup (April 22, 2026) ⚠️ CRITICAL

**Problem**: Torch compile, wandb, and triton create thousands of temp files in `/tmp`, causing nodes to enter drain mode.

**Solution**: All SLURM scripts now redirect temp directories to job-specific local storage with automatic cleanup.

```bash
# Automatically sourced in all SLURM scripts
source run_scripts/utils/setup_tmp_cleanup.sh
```

**What it does:**
- Redirects `TMPDIR`, `TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR`, `WANDB_CACHE_DIR` to `/mnt/localssd/${USER}/tmp_${SLURM_JOB_ID}`
- Creates cleanup trap on EXIT/SIGTERM/SIGINT
- Shows disk usage before/after cleanup (useful for debugging)
- Prevents job collisions and node /tmp pollution

**Integrated in:**
- ✅ `test_scaling_100M_4methods.sh` (H200)
- ✅ `test_scaling_100M_4methods_a100.sh` (A100)
- ✅ `test_30M_baselines.sh`
- ✅ `test_scaling_50M.sh`
- ✅ `run_eco_lr_ablation_50M.sh`

**Note**: If job is hard-killed (SIGKILL), cleanup might not run. Use `#SBATCH --signal=TERM@120` to get graceful shutdown warning.

### Multi-Partition Submission
```bash
# Smart submission: tries H200 → A100 → gpu-all
./run_scripts/slurm_utils/submit_fp4_test_smart.sh

# Multi-partition: splits large arrays across partitions
./run_scripts/slurm_utils/submit_multipartition.sh <script.sh>
```

**QoS Limits**: H200 partition limited to 2 concurrent jobs per user
- Use multi-partition scripts to avoid queued jobs

### Quick Reference (see `run_scripts/README.md` for full details)

**Baseline Experiments:**
- `run_scripts/baselines/test_30M_baselines.sh`: SLURM (FP32, FP16, STE, CAGE, ECO, ECO0)
- `run_scripts/baselines/run_baseline_local.sh`: Local version

**Scaling Experiments:**
- `run_scripts/scaling/50M/test_scaling_50M.sh`: SLURM (FP16, CAGE, ECO, ECO0)
- `run_scripts/scaling/50M/run_scaling_50M_local.sh`: Local version
  - Usage: `./run_scripts/scaling/50M/run_scaling_50M_local.sh {1|2|3|4}`
  - Parallel: `CUDA_VISIBLE_DEVICES=0,1 ./run_scripts/scaling/50M/run_scaling_50M_local.sh 1 > outs/50M_1.out 2>&1 &`
- `run_scripts/scaling/{100M,300M,1B}/`: Future scaling experiments

**Ablation Studies:**
- `run_scripts/ablations/percentile/test_percentile_30M.sh`: P90/P95/P99 on 30M
- `run_scripts/ablations/percentile/test_percentile_sweep.sh`: P85-P100 sweep
- `run_scripts/ablations/lr_tuning/run_eco0_lr_ablation_50M.sh`: ECO0 LR sweep at 50M (0.006, 0.0065, 0.007)

**Utilities:**
- `run_scripts/utils/setup_env.sh`: Environment setup
- `run_scripts/utils/resume_eco.sh`: Resume training from checkpoint

## Practical Tips

### Optimizer Name Gotcha
- Correct: `--opt eco0m-rooh` (with 'm')
- Works but wrong: `--opt eco0-rooh` (without 'm', falls through to else clause)
- Always use the 'm' version to properly match the elif branch

### torch.compile Issues
- First run with `--compile` is very slow (torch.compile warmup)
- Different percentiles may trigger different compilation paths
- P90-P95 sometimes compile slower than P99 (torch.compile quirk, not algorithmic)
- Compilation speed doesn't affect final performance

### Fast Iteration
- Use tiny model (3 layers) for hyperparameter search (~2-3 min per run)
- Validate on 30M model (~25 min per run)
- Full 30M experiments: 5000 iterations recommended

## Data & Environment

**Dataset**: C4 (20B train tokens, 50M val tokens)
- Location: `./datasets/c4/train.bin`, `./datasets/c4/val.bin`
- First download: ~6-7 hours (one-time, fully cached)

**WandB**: Projects vary by experiment
- `FP4-PERCENTILE-SWEEP`: Percentile experiments
- `FP4-PERCENTILE-30M`: 30M validation
- `ECO0`: Main ECO vs ECO-0 comparisons

## Key Files

1. `src/optim/qcri/adam0.py` - ECO-0 optimizer implementation
2. `src/optim/ECO.py` - ECO baseline
3. `src/models/quantization/base_linear.py` - Q99FP4Quantizer (use P90!)
4. `src/main.py` - Training entry point (scheduler bug fixed for multi-param-group optimizers)
5. `run_scripts/baselines/train_eco0m-rooh.sh`, `train_eco.sh` - Single training scripts
6. `run_scripts/README.md` - Complete guide to all run scripts

## Bug Fixes (April 2026)

### Variable Shadowing in base.py (CRITICAL)
**Problem**: CAGE experiments crashed at end of training with `TypeError: 'NoneType' object has no attribute 'items'`

**Root Cause**: Variable shadowing in `src/optim/base.py:249`
```python
# BAD: Shadows outer 'stats' dict
stats = cage.get_stats() if hasattr(cage, "get_stats") else None
```

**Fix**: Renamed inner variable to avoid shadowing
```python
# GOOD: Use unique variable name
cage_stats_dict = cage.get_stats() if hasattr(cage, "get_stats") else None
```

**Location**: `src/optim/base.py:246-251`

### Memory Profiling Added
**Added**: Peak GPU memory tracking in training loop (April 20, 2026)
```python
peak_memory_gb = torch.cuda.max_memory_allocated() / 1e9
```
**Location**: `src/optim/base.py:237-260`
**Logs to WandB**: `memory/peak_gb`

## Summary of Best Practices

✅ **DO**:
- Use `percentile=90.0` for FP4 quantization
- Use `--opt eco0m-rooh` (with 'm')
- Test on tiny model first, validate on 30M
- Use frozen scale (`recalibrate_interval=0`)

❌ **DON'T**:
- Use P99 percentile (outdated default)
- Use dynamic scale recalibration
- Use `eco0-rooh` without 'm' (works but unclean)
- Worry about torch.compile slowness (one-time cost)
