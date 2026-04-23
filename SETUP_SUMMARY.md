# Setup Summary - April 22, 2026

## ✅ Completed Tasks

### 1. CLAUDE.md Updated with 50M Results
- Added complete 50M scaling results table
- Documented ECO0 optimal config: **LR=0.01, P90**
- Added P90 vs P95 comparison (P90 wins consistently)
- Added LR sensitivity analysis (higher LR = better for ECO0)
- Documented narrowing gap to CAGE (0.98 → 0.89 perplexity)

### 2. Temp Directory Cleanup Utility Created ✅
**File:** `run_scripts/utils/setup_tmp_cleanup.sh`

**Problem Solved:**
- Torch compile, wandb, triton create thousands of temp files
- Fills `/tmp` on SLURM nodes → nodes enter drain mode
- Jobs collide when sharing temp directories

**Solution:**
- Job-isolated temp dirs: `/mnt/localssd/${USER}/tmp_${SLURM_JOB_ID}`
- Automatic cleanup on EXIT/SIGTERM/SIGINT
- Redirects: `TMPDIR`, `TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR`, `WANDB_CACHE_DIR`, `TORCH_HOME`, etc.

**Integrated in 5 scripts:**
- ✅ `run_scripts/scaling/100M/test_scaling_100M_4methods.sh`
- ✅ `run_scripts/scaling/100M/test_scaling_100M_4methods_a100.sh`
- ✅ `run_scripts/baselines/test_30M_baselines.sh`
- ✅ `run_scripts/scaling/50M/test_scaling_50M.sh`
- ✅ `run_scripts/ablations/lr_tuning/run_eco_lr_ablation_50M.sh`

### 3. ECO LR Ablation Script Created ✅
**File:** `run_scripts/ablations/lr_tuning/run_eco_lr_ablation_50M.sh`

**Purpose:** Fair comparison with ECO0

**Config:**
- Model: 50M (same as ECO0 ablation)
- LR: {0.006, 0.008, 0.01} at P90 percentile
- 3 runs (SLURM array 1-3)

**Rationale:**
- Current ECO (LR=0.00484): Loss 3.238
- ECO0 best (LR=0.01): Loss 3.221
- ECO's LR was scaled down, ECO0's optimal was unscaled
- Must verify ECO can't match ECO0 with better tuning

**To run:**
```bash
sbatch run_scripts/ablations/lr_tuning/run_eco_lr_ablation_50M.sh
```

---

## 📊 Key Findings from 50M Results

### Performance Ranking
1. 🥇 **FP16 Adam**: 3.134 loss (21.62 PPL) - baseline
2. 🥈 **CAGE 4-bit**: 3.197 loss (23.14 PPL)
3. 🥉 **ECO0 4-bit**: **3.221 loss (23.89 PPL)** ← **Our method**
4. ECO 4-bit: 3.238 loss (24.28 PPL)

### ECO0 vs Competition
- **Beats ECO** by 0.017 loss (0.7% improvement)
- **Gap to CAGE**: Only 0.024 loss (0.89 PPL)
- **Memory advantage**: 28.28 GB vs 29.29 GB (3.5% savings)
- **Gap narrowing**: 30M gap was 0.98 PPL, 50M gap is 0.89 PPL!

### P90 vs P95 (ECO0)
- **LR=0.00775**: P90 (3.242) beats P95 (3.245) by 0.003
- **LR=0.0065**: P90 (3.259) beats P95 (3.265) by 0.007
- **LR=0.006**: P90 (3.253) beats P95 (3.270) by 0.017 🔥
- **Conclusion**: P90 consistently superior, gap widens at lower LR

### LR Sensitivity (ECO0 at P90)
- 0.006 → 3.253
- 0.008 → 3.237
- 0.009 → 3.233
- **0.01 → 3.221** ✅ (best)
- **Pattern**: Higher LR = better performance for ECO0

---

## ⚠️ 100M LR Issue

### Current Situation
Your running 100M ECO0 experiment uses **LR=0.0042**

### The Problem
- Script comment: "Conservative: scaled from 0.006 at 50M"
- Calculation: `0.006 × 0.7 = 0.0042`
- **But optimal 50M LR was 0.01, not 0.006!**

### Correct Scaling
- Optimal 50M LR: **0.01**
- Scaling factor (50M→100M): 1/sqrt(2) = 0.707
- **Correct 100M LR**: 0.01 × 0.707 = **0.00707**
- **Your 0.0042 is 40% too low!**

### Recommendation
**Option A (Aggressive):** Cancel and rerun with LR=0.007  
**Option B (Conservative):** Let 0.0042 finish, then ablate {0.006, 0.007, 0.008}

**My recommendation: Option B** - finish current run, then ablate. Having the full LR curve is valuable for the paper, and 0.0042 isn't catastrophically bad.

---

## 🚀 Next Steps

### Immediate (Today)
1. ✅ CLAUDE.md updated
2. ✅ Temp cleanup utility created and integrated
3. ✅ ECO LR ablation script ready
4. 🔄 Launch ECO ablation: `sbatch run_scripts/ablations/lr_tuning/run_eco_lr_ablation_50M.sh`

### After ECO Ablation (~23 hours)
5. Analyze ECO results vs ECO0
6. Update CLAUDE.md with final 50M comparison
7. Determine if ECO improved (should still be < ECO0)

### After 100M ECO0 0.0042 Finishes
8. Create 100M ECO0 LR ablation script
9. Test LR ∈ {0.006, 0.007, 0.008} at P90
10. Compare against FP16/CAGE baselines

### Paper Preparation
11. Prepare 50M results table with error bars
12. Create scaling plot (30M → 50M → 100M)
13. Highlight P90 finding across all scales

---

## 📝 Files Modified

**Created:**
- `run_scripts/utils/setup_tmp_cleanup.sh` - Temp cleanup utility
- `run_scripts/ablations/lr_tuning/run_eco_lr_ablation_50M.sh` - ECO LR ablation

**Modified:**
- `CLAUDE.md` - Added 50M results, 100M plan, temp cleanup docs
- `run_scripts/scaling/100M/test_scaling_100M_4methods.sh` - Added cleanup source
- `run_scripts/scaling/100M/test_scaling_100M_4methods_a100.sh` - Added cleanup source
- `run_scripts/baselines/test_30M_baselines.sh` - Added cleanup source
- `run_scripts/scaling/50M/test_scaling_50M.sh` - Added cleanup source

---

## 🎯 Key Decisions Made

1. **ECO ablation**: YES - run LR sweep {0.006, 0.008, 0.01} at P90 for fair comparison
2. **100M LR=0.0042**: Let it finish, then follow up with ablation
3. **P90 confirmed**: Validated at tiny, 30M, and 50M scales - use everywhere
4. **Temp cleanup**: Applied to all active SLURM scripts to prevent node drain

---

## 🔍 Open Questions

1. Will ECO match ECO0 at LR=0.01? (Ablation will answer)
2. What's the true optimal 100M LR for ECO0? (0.006-0.008 range expected)
3. Does the CAGE gap keep narrowing at 100M? (Will know after 100M completes)
