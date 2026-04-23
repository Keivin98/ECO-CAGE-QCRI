# Experiment Status - April 21, 2026

## 🔄 Currently Running

### 50M ECO0 LR Ablation (ETA: ~2 hours)
**Jobs:** 4 configs testing LR + Percentile combinations
- Task 1: LR=0.006, P90
- Task 2: LR=0.0065, P90
- Task 3: LR=0.006, P95
- Task 4: LR=0.0065, P95

**Why:** Original ECO0 (LR=0.00775) failed at 50M - fell behind ECO in final steps (3.196 vs 3.189)

**Expected outcome:** Find optimal LR for ECO0 at 50M that beats ECO

**WandB:** ECO0-SCALING project

---

### 100M Scaling (ETA: ~10 hours)
**Job:** 277389 (4 array tasks on 2 nodes)
- Node crirdchpxd004: Task 1 (FP16) or Task 3 (ECO)
- Node crirdchpxd006: Task 2 (CAGE) or Task 4 (ECO0)

**Methods:**
1. FP16 Adam (LR=0.0006) - baseline
2. CAGE 4-bit (LR=0.0006) - QAT baseline
3. ECO 4-bit (LR=0.0031) - comparison
4. ECO0 4-bit (LR=0.0042) - ours

**Config:** 8 layers, 1024 embd, 8 heads, 10B tokens

**Expected outcomes:**
- Validate scaling trends (30M → 50M → 100M)
- Memory: ~2 GB ECO0 savings vs FP16/CAGE
- Quality: ECO0 competitive with CAGE

**Post-job cleanup:** `sbatch --dependency=afterany:277389 run_scripts/utils/cleanup_scratch_100M_slurm.sh`

---

## ✅ Completed

### 30M Baselines
| Method | Val Ppl | LR | Status |
|--------|---------|-----|--------|
| FP32 Adam | 27.49 | 0.0012 | ✅ Done |
| FP16 Adam | 27.49 | 0.0012 | ✅ Done |
| STE 4-bit | 29.07 | 0.0012 | ✅ Done |
| CAGE 4-bit | 29.08 | 0.0012 | ✅ Done |
| ECO 4-bit | 32.73 | 0.00625 | ✅ Done |
| ECO0 4-bit | 30.06 | 0.01 | ✅ Done |

**Key result:** ECO0 beats ECO by 2.67 points!

### 50M Initial Run
| Method | Val Ppl | Memory | Status |
|--------|---------|--------|--------|
| FP16 Adam | 3.074 | 29.29 GB | ✅ Done |
| CAGE 4-bit | 3.142 | 29.29 GB | ✅ Done |
| ECO 4-bit | 3.189 | 29.30 GB | ✅ Done |
| ECO0 4-bit | 3.196 | 28.27 GB | ⚠️ Failed (fell behind ECO) |

**Problem:** ECO0 ahead for 18k steps, then collapsed in final steps

---

## 📋 Next Steps (When Results Arrive)

### When 50M Ablation Finishes (~2 hours)

1. **Check WandB for final val losses:**
   ```bash
   # Compare all 4 configs
   LR=0.006, P90: ?
   LR=0.0065, P90: ?
   LR=0.006, P95: ?
   LR=0.0065, P95: ?
   ```

2. **Identify best config:**
   - Which beats ECO's 3.189?
   - Which has best val loss?

3. **Update 100M ECO0 config (if needed):**
   - If best LR differs from 0.0042, we may want to rerun 100M ECO0
   - Or accept 0.0042 as conservative choice

4. **Update paper tables:**
   - Add successful 50M ECO0 result
   - Update CLAUDE.md with findings

### When 100M Finishes (~10 hours)

1. **Extract results from WandB:**
   ```python
   # Pseudo-code
   results = {
       'FP16': {'val_ppl': XX, 'memory': XX},
       'CAGE': {'val_ppl': XX, 'memory': XX},
       'ECO': {'val_ppl': XX, 'memory': XX},
       'ECO0': {'val_ppl': XX, 'memory': XX},
   }
   ```

2. **Validate scaling predictions:**
   - Memory: Is ECO0 saving ~2 GB vs FP16/CAGE? (predicted)
   - Quality: Is ECO0 competitive with CAGE? (< 1 ppl gap)
   - Trends: Does 30M → 50M → 100M show consistent patterns?

3. **Generate figures:**
   - Training curves (loss vs iteration)
   - Scaling plot (loss vs model size)
   - Memory breakdown bar chart

4. **Update paper:**
   - Fill Table 1 (ECO comparison across scales)
   - Fill Table 2 (100M baseline comparison)
   - Update text with 100M findings

5. **Run cleanup:**
   ```bash
   # Should auto-run, but verify
   squeue -u keisufaj | grep cleanup
   ```

---

## 🎯 Key Questions to Answer

### From 50M Ablation:
- [ ] What's the optimal LR for ECO0 at 50M?
- [ ] Does P95 help (as it did at 30M)?
- [ ] Can ECO0 beat ECO's 3.189?

### From 100M Scaling:
- [ ] Do memory savings scale linearly? (1 GB @ 50M → 2 GB @ 100M)
- [ ] Does ECO0 maintain competitive quality with CAGE?
- [ ] Does ECO0 vs ECO gap persist or grow?

### For Paper:
- [ ] Is 30M + 50M + 100M enough, or do we need 1B?
- [ ] Are the scaling trends clear and convincing?
- [ ] Do we have a compelling memory story?

---

## 📊 Paper Readiness Status

### What We Have:
- ✅ 30M: Complete (6 methods)
- ✅ 50M: Initial run (4 methods, ECO0 failed)
- 🔄 50M: LR ablation (running, fixes ECO0)
- 🔄 100M: Complete suite (running)

### What We Need:
- ⏳ 50M: Successful ECO0 result (2 hours)
- ⏳ 100M: All results (10 hours)
- ❓ 1B: TBD (depends on 100M results)
- ❓ Figures: Training curves, scaling plots, memory breakdown
- ❓ Memory extraction: Get actual GB numbers from WandB for 30M

### Tables Status:
- Table 1 (ECO comparison): 30M ✅, 50M 🔄, 100M 🔄, 1B ❓
- Table 2 (30M baselines): ✅ Complete
- Table 3 (Memory breakdown): 50M ✅, 100M 🔄

---

## 💾 Scripts Ready to Use

### Analysis:
- [ ] TODO: Create WandB data extraction script
- [ ] TODO: Create figure generation scripts

### Cleanup:
- ✅ `run_scripts/utils/cleanup_scratch_100M_slurm.sh` (auto-runs after 277389)

### Next Experiments:
- ✅ `run_scripts/scaling/100M/test_scaling_100M_4methods.sh` (running)
- ❓ 1B scaling script (create if needed)

---

## 🚀 Decision Point: Do We Need 1B?

**Arguments FOR:**
- Paper emphasizes "memory savings grow with scale"
- 1B: 10 GB savings (10%) vs 100M: 2 GB (6%)
- More compelling story for large-scale training

**Arguments AGAINST:**
- Time: ~20-30 hours per run
- Risk: Hyperparameters may need tuning again
- Sufficient: 30M + 50M + 100M shows clear trend

**Recommendation:** Wait for 100M results, then decide.
- If trend is clear (linear scaling), 1B is nice-to-have
- If 100M shows issues, fix those first

---

## Summary

**Running:** 2 experiments (50M ablation, 100M scaling)
**ETA:** 2 hours (50M), 10 hours (100M)
**Next:** Extract results, update paper, decide on 1B

**Current bottleneck:** Waiting for compute 🕐
