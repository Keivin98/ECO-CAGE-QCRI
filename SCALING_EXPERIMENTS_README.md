# Scaling Experiments Execution Plan

This document outlines the execution plan for scaling experiments across model sizes: 30M → 100M → 300M → 1B.

## Experiment Structure

All experiments use:
- **Quantization:** 4-bit FP4 with **P90 percentile** (optimal from ablation study)
- **Dataset:** C4
- **Iterations:** 5000
- **Warmup:** 500 steps
- **Beta1:** 0.9, **Beta2:** 0.95
- **Effective batch size:** 512 tokens (via batch_size × acc_steps)

## Phase 1: 30M Validation (In Progress)

### Step 1: Percentile Validation ⏳ Running
**Script:** `test_percentile_30M.sh`
**Status:** Jobs 277098_3 and 277098_4 running (~14 min into ~3-4 hour runs)
**Purpose:** Validate P90 percentile scales to 30M model with longer training

**Array jobs:**
1. ECO + P90
2. ECO + P95
3. ECO + P99
4. ECO0 + P90
5. ECO0 + P95
6. ECO0 + P99

**Expected outcome:** Confirm P90 > P95 > P99 for both optimizers at production scale

### Step 2: Baseline Comparison 📋 Ready
**Script:** `test_30M_baselines.sh`
**Run after:** Percentile validation completes
**Command:**
```bash
./submit_multipartition.sh test_30M_baselines.sh
```

**Array jobs:**
1. FP32 Adam (LR=0.0012)
2. FP16 Adam (LR=0.0012)
3. STE 4-bit (LR=0.0012)
4. CAGE 4-bit (LR=0.0012)
5. ECO 4-bit (LR=0.005)
6. ECO0 4-bit (LR=0.005)

**Config:** batch_size=64, acc_steps=8
**Expected runtime:** ~25 min per job × 3 batches = ~75 min total (2 concurrent)
**Purpose:** Full method comparison for paper Table 1

---

## Phase 2: Scaling to 100M 📋 Ready

**Script:** `test_scaling_100M.sh`
**Run after:** 30M baselines complete
**Command:**
```bash
./submit_multipartition.sh test_scaling_100M.sh
```

**Array jobs:**
1. FP32 Adam (LR=0.0012)
2. ECO 4-bit (LR=0.005)
3. ECO0 4-bit (LR=0.005)

**Config:** batch_size=32, acc_steps=16 (reduced batch, more accumulation)
**Model:** 8 layers, 1024 embd, 8 heads
**Expected memory:**
- FP32: ~25 GB
- ECO: ~18 GB
- ECO0: ~10 GB

**Expected runtime:** ~40-50 min per job × 2 batches = ~1.5-2 hours total
**Purpose:** Demonstrate scaling trend, confirm memory advantage grows with model size

---

## Phase 3: Scaling to 300M 📋 Ready

**Script:** `test_scaling_300M.sh`
**Run after:** 100M completes
**Command:**
```bash
./submit_multipartition.sh test_scaling_300M.sh
```

**Array jobs:**
1. FP32 Adam (LR=0.0012) ⚠️ May be tight on memory
2. ECO 4-bit (LR=0.005)
3. ECO0 4-bit (LR=0.005)

**Config:** batch_size=16, acc_steps=32 (micro-batching)
**Model:** 16 layers, 1664 embd, 13 heads
**Expected memory:**
- FP32: ~68 GB (tight!)
- ECO: ~50 GB
- ECO0: ~28 GB

**Expected runtime:** ~1-1.5 hours per job × 2 batches = ~2-3 hours total
**Purpose:** Show memory advantage at scale where FP32 barely fits

---

## Phase 4: Extreme Scaling to 1B 📋 Ready

**Script:** `test_scaling_1B.sh`
**Run after:** 300M completes (or in parallel if confident)
**Command:**
```bash
./submit_multipartition.sh test_scaling_1B.sh
```

**Array jobs:**
1. ECO 4-bit (LR=0.005) ⚠️ May OOM
2. ECO0 4-bit (LR=0.005)

**Config:** batch_size=4, acc_steps=128 (aggressive micro-batching)
**Model:** 24 layers, 2048 embd, 16 heads
**Expected memory:**
- FP32: >140 GB ❌ Won't fit
- ECO: ~135 GB ⚠️ Borderline
- ECO0: ~83 GB ✅ Comfortable

**Expected runtime:** ~2-3 hours per job × 1 batch = ~2-3 hours total (if ECO OOMs, only ECO0)
**Purpose:** **Key paper message:** "ECO0 trains 1B model where FP32 can't even fit"

---

## Timeline Estimate

| Phase | Duration | Cumulative |
|-------|----------|------------|
| 30M percentile validation (running) | ~3-4 hours | 3-4 hours |
| 30M baselines | ~1.5 hours | 4.5-5.5 hours |
| 100M scaling | ~2 hours | 6.5-7.5 hours |
| 300M scaling | ~3 hours | 9.5-10.5 hours |
| 1B scaling | ~3 hours | 12.5-13.5 hours |
| **Total wall time** | **~13 hours** | From now |

With 2 concurrent GPU allocation, most experiments run in batches. Actual wall time depends on queue availability.

---

## Execution Strategy

### Conservative (Recommended)
Run phases sequentially to validate assumptions:
1. Wait for 30M percentile → analyze → run 30M baselines
2. Wait for 30M baselines → analyze → run 100M
3. Wait for 100M → analyze → run 300M
4. Wait for 300M → analyze → run 1B

**Pros:** Catch issues early, adjust configs if needed
**Cons:** Slower total time (~13 hours wall time)

### Aggressive (If Confident)
Launch all phases immediately:
```bash
# Launch everything (will queue due to GPU limit)
./submit_multipartition.sh test_30M_baselines.sh    # Will queue
./submit_multipartition.sh test_scaling_100M.sh      # Will queue
./submit_multipartition.sh test_scaling_300M.sh      # Will queue
./submit_multipartition.sh test_scaling_1B.sh        # Will queue
```

**Pros:** Minimal human wait time
**Cons:** If configs need adjustment, waste compute

**Recommendation:** Go conservative for 30M → 100M transition, then aggressive for 100M → 300M → 1B if no issues.

---

## Expected Outputs

### WandB Logs
All runs log to WandB project: `ECO0-SCALING`

**Metrics tracked:**
- Training loss
- Validation loss
- Learning rate schedule
- Gradient norms
- (TODO) Peak GPU memory

### Result Files
Checkpoints and logs in: `./exps/`

### Paper Tables to Populate

**Table 1: 30M Baseline Comparison**
| Method | Val Loss | Memory (GB) | Throughput |
|--------|----------|-------------|------------|
| FP32 | XX.XX | XX | XX tok/s |
| FP16 | XX.XX | XX | XX tok/s |
| STE | XX.XX | XX | XX tok/s |
| CAGE | XX.XX | XX | XX tok/s |
| ECO | XX.XX | XX | XX tok/s |
| ECO0 | XX.XX | XX | XX tok/s |

**Table 2: Scaling Comparison**
| Model | Method | Val Loss | Memory (GB) | Fits? |
|-------|--------|----------|-------------|-------|
| 30M | FP32 | XX.XX | XX | ✅ |
| 30M | ECO | XX.XX | XX | ✅ |
| 30M | ECO0 | XX.XX | XX | ✅ |
| 100M | FP32 | XX.XX | XX | ✅ |
| 100M | ECO | XX.XX | XX | ✅ |
| 100M | ECO0 | XX.XX | XX | ✅ |
| 300M | FP32 | XX.XX | XX | ✅/⚠️ |
| 300M | ECO | XX.XX | XX | ✅ |
| 300M | ECO0 | XX.XX | XX | ✅ |
| 1B | FP32 | - | >140 | ❌ |
| 1B | ECO | XX.XX | ~135 | ⚠️ |
| 1B | ECO0 | XX.XX | ~83 | ✅ |

---

## Memory Profiling TODO

Add to training code:
```python
import torch

# After each training step
peak_memory = torch.cuda.max_memory_allocated() / 1e9  # GB
log_to_wandb({"peak_memory_gb": peak_memory})
```

This will populate the "Memory (GB)" columns in the tables above.

---

## Post-Experiment Analysis Checklist

After all experiments complete:

- [ ] Download all WandB runs
- [ ] Extract final validation losses
- [ ] Extract peak memory usage
- [ ] Generate training curve plots (loss vs iterations)
- [ ] Generate scaling plot (loss vs model size)
- [ ] Generate memory comparison plot
- [ ] Fill all "XX.XX" placeholders in paper
- [ ] Write results section narrative
- [ ] Write conclusion
- [ ] Add missing citations
- [ ] Generate final figures

**ETA to submission-ready paper:** ~1 day of writing after all experiments complete.
