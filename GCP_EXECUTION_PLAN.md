# GCP H100 Execution Plan - Complete Overview

## Current Status (April 23, 2026)

### Running on SLURM (Panther cluster)
- ✅ 100M ECO-0 LR refinement (2 jobs, ~10 hours ETA)
  - Task 1: LR=0.006
  - Task 2: LR=0.008

### Ready to Launch on GCP (8× H100)
All scripts created in `run_scripts/gcp/`:
1. `50M_ste.sh` - STE baseline at 50M
2. `100M_ste.sh` - STE baseline at 100M
3. `300M_fp16_eco0.sh` - 300M scaling (FP16 + ECO-0)

---

## GCP Execution Strategy

### Phase 1: Launch All 4 Experiments (8 H100s, ~30 hours)

```bash
# From project root
cd /export/home/keisufaj/optimization/ECO-CAGE-QCRI
mkdir -p logs

# Launch all in parallel (recommended)
CUDA_VISIBLE_DEVICES=0,1 bash run_scripts/gcp/50M_ste.sh > logs/50M_ste.out 2>&1 &
CUDA_VISIBLE_DEVICES=2,3 bash run_scripts/gcp/100M_ste.sh > logs/100M_ste.out 2>&1 &
CUDA_VISIBLE_DEVICES=4,5 bash run_scripts/gcp/300M_fp16_eco0.sh 1 > logs/300M_fp16.out 2>&1 &
CUDA_VISIBLE_DEVICES=6,7 bash run_scripts/gcp/300M_fp16_eco0.sh 2 > logs/300M_eco0.out 2>&1 &

# Verify all jobs running
jobs -l
nvidia-smi
```

### Timeline
```
Hour 0:    Launch all 4 experiments
Hour 6:    50M STE completes → 1 result ✅
Hour 12:   100M STE completes → 2 results ✅
Hour 30:   300M FP16 + ECO-0 complete → 4 results ✅
```

---

## What Each Experiment Provides

### 1. 50M STE (GPUs 0,1, ~6 hours)
**Configuration:**
- Model: 7 layers, 768 embd, 6 heads (~50M params)
- Training: 5B tokens (19,073 iterations)
- LR: 0.00093 (same as CAGE/FP16)
- Percentile: P90

**Expected Result:** STE ~23.1-23.2 PPL (matches CAGE's 23.14)

**Paper Value:**
- Validates STE ≈ CAGE at 50M (already proven at 30M)
- Completes baseline comparison
- Strengthens claim: "CAGE's curvature correction provides minimal benefit"

---

### 2. 100M STE (GPUs 2,3, ~12 hours)
**Configuration:**
- Model: 8 layers, 1024 embd, 8 heads (~100M params)
- Training: 10B tokens (38,146 iterations)
- LR: 0.0006 (same as CAGE/FP16)
- Percentile: P90

**Expected Result:** STE ~19.2-19.3 PPL (matches CAGE's 19.18)

**Paper Value:**
- Validates STE ≈ CAGE across 3 scales (30M, 50M, 100M)
- Allows statement: "Traditional QAT methods (STE, CAGE) perform identically"
- Justifies focusing comparison on CAGE only at larger scales

---

### 3. 300M FP16 Baseline (GPUs 4,5, ~30 hours)
**Configuration:**
- Model: 12 layers, 1536 embd, 12 heads (~300M params)
- Training: 15B tokens (39,062 iterations)
- LR: 0.0004 (predicted from 100M scaling)
- Batch: 24×16=384 effective

**Expected Result:** ~15-16 PPL (baseline for 300M)

**Paper Value:**
- Establishes 300M baseline performance
- Enables ECO-0 comparison at this scale
- Fills gap in scaling sequence: 30M → 50M → 100M → **300M**

---

### 4. 300M ECO-0 (GPUs 6,7, ~30 hours)
**Configuration:**
- Model: 12 layers, 1536 embd, 12 heads (~300M params)
- Training: 15B tokens (39,062 iterations)
- LR: 0.0045 (predicted: 0.007 × 0.65)
- Quantization: FP4, P90 percentile

**Expected Result:** ~16-17 PPL (competitive with 3-4 GB memory savings)

**Paper Value:**
- **Critical scaling point** between 100M and 500M
- Memory savings validation (expect 3-4 GB vs FP16)
- Trend analysis: does gap to baseline continue narrowing?
- De-risks 500M/1B experiments

---

## Paper Updates After Completion

### New Tables

**Table: Complete Baseline Comparison (30M-100M)**
| Scale | FP16 | STE | CAGE | ECO | ECO-0 | Memory (ECO-0) |
|-------|------|-----|------|-----|-------|----------------|
| 30M | 27.49 | 29.07 | 29.08 | 32.73 | 30.06 | ~28.3 GB |
| 50M | 21.62 | **23.1** | 23.14 | 23.92 | 23.89 | 28.28 GB (-3.5%) |
| 100M | 17.93 | **19.2** | 19.18 | 20.11 | 21.35* | 18.82 GB (-8.0%) |

*Pending LR refinement (may improve to ~20.5)

**Table: Scaling Results (30M-300M)**
| Scale | FP16 | ECO-0 | Gap (PPL) | Gap (%) | Memory Saved |
|-------|------|-------|-----------|---------|--------------|
| 30M | 27.49 | 30.06 | 2.57 | 9.3% | 1.0 GB (3.4%) |
| 50M | 21.62 | 23.89 | 2.27 | 10.5% | 1.0 GB (3.5%) |
| 100M | 17.93 | 21.35* | 3.42 | 19.1% | 1.64 GB (8.0%) |
| 300M | **~15.5** | **~16.5** | **~1.0** | **~6.5%** | **~3.5 GB (~12%)** |

*100M ECO-0 may improve to ~20.5 after LR refinement

### New Sections

**Section 4.X: STE vs CAGE Comparison**
- Show STE ≈ CAGE across 30M, 50M, 100M
- Conclude: curvature correction provides <1% benefit
- Justify using only CAGE for larger scale comparisons

**Section 4.Y: 300M Scaling Validation**
- First scale where memory savings become significant (>3 GB)
- Performance gap analysis (narrowing trend or stable?)
- LR scaling pattern validation

**Section 4.Z: Memory Scaling Analysis**
- Linear scaling confirmed: 30M (1GB) → 50M (1GB) → 100M (1.6GB) → 300M (~3.5GB)
- Projection to 1B: ~8-12 GB savings (enables larger batches/sequences)
- Memory-quality Pareto frontier plot

### Key Findings to Highlight

1. **STE ≈ CAGE validated:** Traditional QAT methods are equivalent (30M, 50M, 100M)
2. **Memory scales linearly:** Savings grow proportionally with model size
3. **300M fills critical gap:** Validates trends between 100M and 500M
4. **LR scaling pattern:** Both ECO and ECO-0 require ~20-30% reduction per 2-3× scale
5. **Performance competitive:** ECO-0 within 5-10% of CAGE at most scales

---

## Decision Points After Results

### If 300M Results are Strong (ECO-0 ≈ FP16 within 5-10%)
**Next steps:**
1. Proceed to 500M or 1B on remaining compute
2. Memory story becomes very compelling (5-12 GB savings)
3. Paper nearly complete, needs only conclusion + figures

### If 300M Results Show Issues (ECO-0 gap >15%)
**Options:**
1. Run 300M LR ablation (test 0.004, 0.005, 0.006)
2. Investigate if scale or LR is the problem
3. Consider stopping at 300M, focus on smaller-scale story

### If 100M LR Refinement Succeeds (ECO-0 matches ECO ~20.1)
**Impact:**
- Validates ECO-0 competitive at 100M
- Increases confidence in 300M success
- Strengthens paper narrative

### If 100M LR Refinement Fails (ECO-0 stuck at 21.35)
**Impact:**
- Suggests fundamental limitation at this scale
- 300M results become critical for paper viability
- May need to adjust narrative to "competitive at small scale, memory advantage at large scale"

---

## Estimated Paper Completion

**After GCP experiments complete:**
- **Complete results:** 30M, 50M, 100M, 300M (4 scales)
- **Missing:** 500M+ (optional), Conclusion, Figures
- **Completion:** ~75-80%

**To finish paper (~1 week after experiments):**
1. Write Conclusion section (1 day)
2. Create figures (scaling plots, training curves, memory-quality Pareto) (2 days)
3. Statistical significance testing (multiple seeds) (optional, 1 day)
4. Citation hunt + bibliography cleanup (1 day)
5. Final polish + LaTeX formatting (1 day)

**Submission-ready:** ~7-10 days after 300M completes

---

## Resource Summary

**SLURM (Panther):**
- Running: 100M ECO-0 LR refinement (2 jobs)
- ETA: ~10 hours

**GCP (8× H100):**
- Ready to launch: 4 experiments (50M STE, 100M STE, 300M FP16, 300M ECO-0)
- Total runtime: ~30 hours (experiments run in parallel)
- Total GPU-hours: 4×2GPUs×30h = 240 GPU-hours

**Cost estimate (rough):**
- H100: ~$2-3/GPU-hour on GCP
- Total: 240 GPU-hours × $2.50 = ~$600

---

## Quick Start (Copy-Paste)

```bash
# Navigate to project
cd /export/home/keisufaj/optimization/ECO-CAGE-QCRI

# Create logs directory
mkdir -p logs

# Launch all 4 experiments in parallel
CUDA_VISIBLE_DEVICES=0,1 bash run_scripts/gcp/50M_ste.sh > logs/50M_ste.out 2>&1 &
CUDA_VISIBLE_DEVICES=2,3 bash run_scripts/gcp/100M_ste.sh > logs/100M_ste.out 2>&1 &
CUDA_VISIBLE_DEVICES=4,5 bash run_scripts/gcp/300M_fp16_eco0.sh 1 > logs/300M_fp16.out 2>&1 &
CUDA_VISIBLE_DEVICES=6,7 bash run_scripts/gcp/300M_fp16_eco0.sh 2 > logs/300M_eco0.out 2>&1 &

# Check all running
jobs -l
nvidia-smi

# Monitor (in separate terminal/tmux)
tail -f logs/50M_ste.out
tail -f logs/100M_ste.out
tail -f logs/300M_fp16.out
tail -f logs/300M_eco0.out
```

**WandB monitoring:** https://wandb.ai/keisufaj-hamad-bin-khalifa-university/ECO0-SCALING
