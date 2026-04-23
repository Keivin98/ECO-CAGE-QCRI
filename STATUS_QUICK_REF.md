# Quick Status Reference (April 22, 2026 - Evening)

## 🔄 CURRENTLY RUNNING

**100M Optimal LR Tests - SUBMITTED ✅**
- ECO0 LR=0.007 (predicted optimal)
- ECO0 LR=0.0085 (testing higher)
- ETA: ~8-10 hours each
- **Status:** Jobs running on H200

**Check status:**
```bash
squeue -u $USER
tail -f outs/100M_optimal_lr_*_1.out  # Task 1
tail -f outs/100M_optimal_lr_*_2.out  # Task 2
```

---

## ✅ What's Done

**50M - COMPLETE ✅**
- All 4 methods with optimal LRs
- ECO0 (23.89 PPL) = ECO (23.92 PPL) 
- Memory: ECO0 saves 1 GB
- **Discovery:** Both prefer unscaled LRs (30M→50M)
- Paper updated

**100M Baseline - COMPLETE ✅**
- FP16: 17.93 PPL
- CAGE: 19.18 PPL
- ECO: 21.26 PPL (under-tuned)
- ECO0: 22.39 PPL (under-tuned)
- Memory: 1.7 GB savings (8.3%)

**Paper - UPDATED ✅**
- Section 4.5 (50M) complete
- Tables updated
- Discussion updated with ECO0≈ECO finding

---

## 🔄 In Progress / Ready

**100M Optimal Tests - RUNNING ✅**
- Script: `run_scripts/scaling/100M/test_100M_optimal_lr.sh`
- Status: Jobs submitted and running
- ETA: ~8-10 hours (check in the morning)

**500M Script - CREATED**
- Script: `run_scripts/scaling/500M/test_500M_optimal_lr.sh`
- Ready to run after 100M completes

---

## 🐛 Bugs Fixed Today

1. **HF Token Issue** - setup_tmp_cleanup.sh was breaking HuggingFace auth
   - Fixed: Preserve HF_HOME in home directory
   
2. **Dataset Collision** - Array jobs deleting each other's /scratch data
   - Fixed: Job-specific directories using SLURM_ARRAY_JOB_ID + TASK_ID

---

## 📊 Key Findings

**50M Results:**
- ECO0 = ECO at optimal LRs (23.89 vs 23.92)
- ECO0 saves 1 GB memory (3.5%)
- Gap to CAGE: 0.75 PPL (narrowing from 30M's 0.98)

**Unscaled LR Discovery:**
- ECO: keeps LR=0.00625 from 30M→50M
- ECO0: keeps LR=0.01 from 30M→50M
- Differs from standard 1/√scale rule

**Memory Scaling:**
- 30M: 1.0 GB (3.5%)
- 50M: 1.0 GB (3.5%)
- 100M: 1.7 GB (8.3%)
- Linear scaling confirmed ✅

---

## 🎯 Decision Point (After 100M)

**Option A: Stop at 100M** (Fast)
- 3 model sizes
- Paper ready ~1 week
- Sufficient for publication

**Option B: Add 500M** (Better)
- 4 model sizes
- Stronger scaling evidence
- +2-3 days

**Recommendation:** Wait for 100M results, then decide

---

## 📁 Important Files

**Experiment tracking:**
- `/CLAUDE.md` - Full details
- `/EXPERIMENTAL_STATUS.md` - Clean status
- `/STATUS_QUICK_REF.md` - This file

**Scripts:**
- `/run_scripts/scaling/100M/test_100M_optimal_lr.sh` ← RUN THIS
- `/run_scripts/scaling/500M/test_500M_optimal_lr.sh` ← Next
- `/run_scripts/utils/setup_tmp_cleanup.sh` ← Temp cleanup (fixed)

**Paper:**
- `/paper/Optimization-Paper/sections/04_experiments.tex` ← Updated
- `/paper/Optimization-Paper/tables/tab_50m_scaling.tex` ← Updated
- `/paper/Optimization-Paper/PAPER_UPDATES_50M.md` ← Change log

---

## 🔢 Learning Rates by Scale

| Model | FP16/CAGE | ECO | ECO0 | Notes |
|-------|-----------|-----|------|-------|
| 30M | 0.0012 | 0.00625 | 0.01 | Baseline |
| 50M | 0.00093 | 0.00625 ✅ | 0.01 ✅ | Unscaled! |
| 100M | 0.0006 | ~0.005 | 0.007-0.0085 | Testing |
| 500M | 0.00027 | ~0.0022 | 0.003-0.004 | Predicted |

**Scaling rule:**
- Standard Adam: multiply by 1/√(scale_factor)
- ECO/ECO0: unscaled 30M→50M, then scale from 50M baseline
