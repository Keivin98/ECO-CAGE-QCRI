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

## 🚀 CURRENT STATUS (April 25, 2026 - early morning)

### Recipe pivot since 2026-04-23
The default recipe shifted away from FP4 + cosine + P90 (legacy) to:
- **Quantizer**: `Q99IntQuantizer` (INT4 uniform grid) — INT4 wins at tiny/30M, ties or marginally loses at 50M, 100M pending. Confirmed clean win for CAGE at 50M (22.79 vs FP4's 23.14).
- **Scheduler**: `half-eco0` for ECO/ECO-0 (linear decay to η_max/2); `cos` for FP16/STE/CAGE. Split is along error feedback presence — 1/η coefficient in error-feedback term means cos amplifies feedback 10× late in training and destabilizes.
- **Percentile**: `99.5` for INT4 (within 0.003 val loss of P99.99 optimum; INT4 is largely percentile-insensitive). FP4 still uses P90 if anyone uses FP4.

### ✅ Complete (all numbers under new INT4 + half-eco0 + P99.5 recipe)

**30M (3B tokens, 11,444 iter, FULL LR sweeps complete):**
| Method | LR | Val PPL |
|---|---|---|
| FP16 Adam (cos) | 0.0012 | **24.76** |
| CAGE INT4 (cos) | 0.0012 | **26.03** |
| STE INT4 (cos) | 0.0012 | **26.06** (≈CAGE, confirms curvature correction useless under INT4 too) |
| **ECO-0 INT4 (half-eco0)** | **0.008** | **26.76** 🏆 (bracketed: 0.005=27.40, 0.006=27.37, 0.010=27.92) |
| ECO INT4 (half-eco0) | 0.003 | 28.75 (flat in 0.003-0.005 range) |

**Tiny (5000 iter) — scheduler & quantizer ablations done:**
- Cos vs half-eco0 vs static-eco0 sweep, INT4 vs FP4 sweep, percentile sweep all complete
- Best tiny config: INT4 + half-eco0 + LR=0.012 → 58.10 PPL

### 🏃 RUNNING NOW (April 25, ~03:00)

**SLURM (Panther H200, 16-GPU QoS):**
- 5x 50M jobs running (4x LR follow-up at LR=0.003-0.005 ECO + LR=0.004-0.005 ECO-0; 1x slow STE on broken node crirdchpxd001)
- 3x 100M tasks running (FP16, STE INT4, CAGE INT4 — all with cos scheduler)
- 6x 100M tasks pending in queue (ECO LR=0.001/0.002/0.004; ECO-0 LR=0.005/0.006/0.007)

**ETAs:**
- 50M LR follow-up done: ~+40 min from now
- 50M STE on slow node: ~+5 hr (healthy, just slow)
- 3 running 100M tasks: ~+6-7 hr
- All 100M tasks: ~+13-15 hr

### Partial 50M results so far (best of partial sweep, INT4 + half-eco0)
| Method | LR | Val PPL |
|---|---|---|
| CAGE INT4 (cos) | 0.00093 | **22.79** (beats FP4's 23.14) |
| **ECO-0 INT4** | 0.008 | 24.62 (vs FP4 era 23.89 — gap 0.73 PPL, may close with LR follow-up) |
| ECO INT4 | 0.005 | 25.80 (vs FP4 era 23.92 — gap 1.88 PPL) |
| ECO-0 INT4 | 0.010 | **DIVERGED** (37.65) — stability ceiling drops with scale |

### 🎯 Recommended LRs by Scale (current best)
| Scale | FP16/CAGE/STE (cos) | ECO (half-eco0) | ECO-0 (half-eco0) |
|-------|-----------|-----|------|
| 30M (3B tok, INT4 P99.5) | 0.0012 | **0.003** | **0.008** (bracketed) |
| 50M (5B tok, INT4 P99.5) | 0.00093 | 0.005 (provisional, follow-up at 0.003-0.004 pending) | 0.008 (provisional, follow-up at 0.004-0.005 pending) |
| 100M (10B tok, INT4 P99.5) | 0.0006 | testing 0.001-0.005 | testing 0.005-0.007 |

**Pattern:** ECO-0 LR is roughly stable (~0.008) from 30M→50M; ECO LR is in flat region 0.003-0.005. Stability ceiling drops with scale: LR=0.012 unstable at 30M, LR=0.010 diverged at 50M.

**LEGACY (FP4 + cos era, kept for reference):** 30M FP4: ECO=0.00625, ECO-0=0.01. 50M FP4: same. 100M FP4: ECO=0.005, ECO-0=0.007.

---

## Major Research Findings

### 0. Recipe consolidation (April 25, 2026)

**Default recipe for new experiments:**
- Quantizer: `Q99IntQuantizer` (uniform INT4 grid `{-7,...,7}`)
- Percentile: `99.5` (INT4 is largely percentile-insensitive; P99.5 within 0.003 val loss of empirical optimum P99.99)
- Scheduler: `half-eco0` for ECO/ECO-0, `cos` for FP16/Adam/STE/CAGE
- Iterations: scaled to match Chinchilla-ratio token budget (3B at 30M, 5B at 50M, 10B at 100M)

### 1a. Scheduler split (April 25, 2026) — NEW

**Finding**: Cosine decay vs linear half-decay preference splits along *error feedback*, not along v-accumulation:
- **Methods WITH error feedback (ECO, ECO-0)**: prefer `half-eco0` (linear decay to η_max/2). At 30M: ECO cos=32.73 vs half-eco0=30.91 (Δ=+1.82); ECO-0 wins under half-eco0 too.
- **Methods WITHOUT error feedback (Adam, STE, CAGE)**: prefer `cos` (cosine decay to η_max/10). At 30M: cos beats half-eco0 by 0.97-1.57 PPL across FP16, STE, CAGE.

**Mechanism**: The error-feedback update has a $1/\eta$ coefficient: `m ← m + (1/η)(1-1/β)·e`. Under cos decay, η shrinks 10×, so the feedback term grows 10× late in training — destabilizing precisely when the model is most quantization-sensitive. Linear half-decay caps the amplification at 2×.

**Tracking**: `EXP-C-SCHEDULER-BASELINES-30M`, `EXP-SCHEDULER-TINY-HALF`. Paper: Section 4.12.

### 1b. Quantizer comparison: INT4 vs FP4 (April 24-25, 2026) — NEW

**Finding**: INT4 (uniform grid) outperforms FP4 (non-uniform codebook) at small scale; advantage shrinks at larger scale.
- Tiny (5000 iter): INT4 wins by 6-9 PPL
- 30M (matched-iter, EXP-D): INT4 wins by 0.78 PPL
- 50M (matched-config): CAGE INT4 22.79 beats CAGE FP4 23.14 by 0.35 PPL ✅
- 50M (different schedulers): ECO-0 INT4 best so far 24.62 vs FP4-era 23.89 (gap 0.73 PPL — may close with LR follow-up)
- 100M: pending

**Mechanism**: FP4's extreme bins ($\pm4, \pm6$) need outlier clipping to be useful; once P90/P99.5 clipping is applied, FP4's non-uniform-near-zero advantage becomes counterproductive (resolution wasted on a distributional feature that's been clipped out). Uniform INT4 grid covers the typical-weight body more evenly.

**Tracking**: `EXP-D-INT-VS-FP-30M`, `EXP-50M-INT4-RERUN`, `EXP-100M-INT4-RERUN` (running). Paper: Section 4.13.

### 1c. INT4 percentile-insensitivity (April 24, 2026) — NEW

**Finding**: INT4 is nearly flat across P85-P100 (range 0.051 val loss), unlike FP4's sharp U-shape (range 0.111 with min at P90). Best at P99.99 (val loss 4.026), P99 / P99.5 / P92 within noise. **We use P99.5** as default.

**Tracking**: `EXP-PERCENTILE-INT4-TINY`. Paper: Section 4.11 (figure shows both curves normalized to their minima).

### 2. Optimal FP4 Percentile: **90th, not 99th!** (April 2026, LEGACY)

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

### Temporary File Cleanup (April 22-23, 2026) ⚠️ CRITICAL

**Problem**: Torch compile, wandb, and triton create thousands of temp files in `/tmp`, causing:
- SLURM nodes to enter drain mode (2K+ files)
- GCP instances to fill `/tmp` and crash
- Training jobs to fail with "No space left on device"

**Solution**: All scripts (SLURM + GCP) redirect temp directories to job-specific local storage with automatic cleanup.

#### SLURM Scripts (Panther)
```bash
# Automatically sourced in all SLURM scripts
source run_scripts/utils/setup_tmp_cleanup.sh
```

**What it does:**
- Redirects temp dirs to `/mnt/localssd/${USER}/tmp_${SLURM_JOB_ID}`
- Creates cleanup trap on EXIT/SIGTERM/SIGINT
- Shows disk usage before/after cleanup
- **HuggingFace tokens:** Keeps `HF_HOME` in home directory (for auth), only redirects model/dataset cache

**Integrated in:**
- ✅ All SLURM scripts in `run_scripts/scaling/` and `run_scripts/baselines/`

#### GCP Scripts (H100 Instances)
**Built-in to each script** (uses PID instead of SLURM_JOB_ID):

```bash
# Creates /mnt/localssd/${USER}/tmp_${PID}
# Redirects: TMPDIR, TORCHINDUCTOR_CACHE_DIR, TRITON_CACHE_DIR, 
#            WANDB_DIR, WANDB_CACHE_DIR, XDG_CACHE_HOME
# HuggingFace: Keeps HF_HOME=$HOME/.cache/huggingface (tokens)
#              Redirects TRANSFORMERS_CACHE, HF_DATASETS_CACHE
```

**Integrated in:**
- ✅ `run_scripts/gcp/50M_ste.sh`
- ✅ `run_scripts/gcp/100M_ste.sh`
- ✅ `run_scripts/gcp/300M_fp16_eco0.sh`

#### HuggingFace Credentials Handling ⚠️ CRITICAL

**Key insight:** HF auth tokens must remain in `$HOME/.cache/huggingface/token`, but heavy cache files (models, datasets) should go to temp.

```bash
# ✅ Correct approach (both SLURM and GCP)
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"  # Tokens accessible
export TRANSFORMERS_CACHE="$JOB_TMP/hf/transformers"   # Models to temp
export HF_DATASETS_CACHE="$JOB_TMP/hf/datasets"        # Datasets to temp

# ❌ WRONG: Would break HF auth
# export HF_HOME="$JOB_TMP/hf"  # Tokens inaccessible!
```

**What gets cleaned:**
- TorchInductor cache (~1-2 GB per job)
- Triton cache (~100-500 MB)
- WandB cache
- Downloaded HF models/datasets

**What stays persistent:**
- HuggingFace auth tokens (`~/.cache/huggingface/token`)
- WandB run history (`~/wandb/`)

**Note**: If job is hard-killed (SIGKILL), cleanup might not run. SLURM: use `#SBATCH --signal=TERM@120`. GCP: jobs usually exit cleanly.

### Multi-Partition Submission
```bash
# Smart submission: tries H200 → A100 → gpu-all
./run_scripts/slurm_utils/submit_fp4_test_smart.sh

# Multi-partition: splits large arrays across partitions
./run_scripts/slurm_utils/submit_multipartition.sh <script.sh>
```

**QoS Limits**: Two H200 account/QoS combinations available:
- **Regular** (`-A H200 -q h200_qos`): 2 concurrent jobs per user, not preemptable
- **16-GPU** (`-A H200_16GPUs -q h200_qos_16_gpus`): Up to 16 GPUs, **preemptable** by regular H200 jobs (4h grace period)
  - Use this for array sweeps with > 2 tasks
  - For long jobs, add `#SBATCH --signal=B:SIGUSR1@60` + SIGUSR1/SIGTERM traps to checkpoint before preemption

**Fallback to A100** (`-p gpu-A100 -A A100 -q a100_qos`):
- Conda env: `cage_cu128` (not `cage`)
- Useful when all H200 slots are full

### Quick Reference (see `run_scripts/README.md` for full details)

**Baseline Experiments (SLURM):**
- `run_scripts/baselines/test_30M_baselines.sh`: SLURM (FP32, FP16, STE, CAGE, ECO, ECO0)
- `run_scripts/baselines/run_baseline_local.sh`: Local version

**Scaling Experiments (SLURM):**
- `run_scripts/scaling/50M/test_scaling_50M.sh`: SLURM (FP16, CAGE, ECO, ECO0)
- `run_scripts/scaling/50M/run_scaling_50M_local.sh`: Local version
  - Usage: `./run_scripts/scaling/50M/run_scaling_50M_local.sh {1|2|3|4}`
  - Parallel: `CUDA_VISIBLE_DEVICES=0,1 ./run_scripts/scaling/50M/run_scaling_50M_local.sh 1 > outs/50M_1.out 2>&1 &`
- `run_scripts/scaling/100M/test_100M_eco0_lr_refinement.sh`: ECO-0 LR tuning (0.006, 0.008)
- `run_scripts/scaling/{100M,300M,500M}/`: Scaling experiments

**GCP Scripts (H100, No SLURM):**
- `run_scripts/gcp/50M_ste.sh`: Fill STE baseline at 50M (~6 hours, 2 GPUs)
- `run_scripts/gcp/100M_ste.sh`: Fill STE baseline at 100M (~12 hours, 2 GPUs)
- `run_scripts/gcp/300M_fp16_eco0.sh [1|2]`: 300M scaling (FP16 or ECO-0, ~30 hours, 2 GPUs)
  - Usage: `CUDA_VISIBLE_DEVICES=0,1 bash run_scripts/gcp/50M_ste.sh > logs/50M_ste.out 2>&1 &`
  - See `run_scripts/gcp/README.md` for parallel execution guide

**Ablation Studies:**
- `run_scripts/ablations/percentile/test_percentile_30M.sh`: P90/P95/P99 on 30M
- `run_scripts/ablations/percentile/test_percentile_sweep.sh`: P85-P100 sweep
- `run_scripts/ablations/lr_tuning/run_eco0_lr_ablation_50M.sh`: ECO0 LR sweep at 50M (0.006, 0.0065, 0.007)
- `run_scripts/ablations/scheduler/test_eco0_scheduler_tiny.sh`: cos vs cos-eco0 × FP4 vs INT4 × 3 LRs (12 tasks on H200_16GPUs)

**Utilities:**
- `run_scripts/utils/setup_tmp_cleanup.sh`: Temp directory cleanup (SLURM)
- `run_scripts/utils/setup_env.sh`: Environment setup
- `run_scripts/utils/resume_eco.sh`: Resume training from checkpoint

## Practical Tips

### LR Scheduler (CRITICAL — OneCycleLR behavior)

**Current default** (`--scheduler cos`): OneCycleLR with `div_factor=100, final_div_factor=0.1`
- Warmup: LR goes from `max_lr/100 → max_lr` over `pct_start` fraction
- Decay: LR goes from `max_lr → max_lr/10` using cosine annealing

**New option** (`--scheduler cos-eco0`): Constant LR after warmup
- `anneal_strategy='linear', final_div_factor=0.01`
- OneCycleLR formula: `min_lr = max_lr / (div_factor × final_div_factor) = max_lr / (100 × 0.01) = max_lr`
- Result: LR stays at `max_lr` for 90% of training (no decay)
- **Hypothesis**: ECO-0's variance from current gradients (not accumulated like Adam β2) means LR decay isn't needed
- Test script: `run_scripts/ablations/scheduler/test_eco0_scheduler_tiny.sh`

**⚠️ Bug fixed (April 23, 2026)**: `src/main.py:247` was `if` instead of `elif`, causing `cos`/`linear` to fall through to `else: raise NotImplementedError`. Now `elif`.

### Quantizer Comparison (FP vs INT)

Both follow same pipeline (`x → scale → quantize → descale`, STE gradient). Differences:

| | `Q99IntQuantizer` | `Q99FP4Quantizer` |
|---|---|---|
| Grid | Uniform `{-8..8}` | Non-uniform `{-6, -4, -3, -2, -1.5, -1, -0.75, 0, 0.5, ..., 6}` |
| Rounding | `round_at(xs, tau=0.5)` | `argmin(|xs - codebook|)` |
| Scale calibration | `scale = levels / p_val` (levels=8) | `scale = max_code / p_val` (max_code=6.0) |

Both use P90 percentile. FP4's denser-near-zero codebook better fits Gaussian-ish weight distributions.

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
