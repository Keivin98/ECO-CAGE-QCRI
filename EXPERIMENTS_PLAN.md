# Experiments Plan for ECO0 Paper

## Current Status (April 2026)

### ✅ Completed
1. **P90 Percentile Ablation (Tiny Model)**
   - Tested: P85, P90, P92, P93, P95, P96, P97, P98, P99, P99.5, P99.9, P99.99, P100
   - Result: P90 optimal (4.976 vs 5.068 for P99, 1.8% improvement)
   - Model: 3 layers, 128 embd, 2 heads, 1000 iterations
   - Script: `test_percentile_fine.sh`

### ⏳ Running Now
2. **30M Percentile Validation**
   - Testing: P90, P95, P99 for both ECO and ECO0
   - Model: 30M (6 layers, 640 embd, 5 heads)
   - Iterations: 5000
   - Script: `test_percentile_30M.sh`
   - Jobs: 277XXX (2 concurrent, 6 total, ~25 min each)
   - Purpose: Validate P90 finding scales to production model size

### 📋 Queued (Ready to Run)
3. **30M Baseline Comparison**
   - Methods: FP32, FP16, STE, CAGE, ECO, ECO0
   - All quantized use P90 percentile
   - Script: `test_30M_baselines.sh` ✨ **NEW**
   - Array: 6 jobs (can run 2 concurrent = 3 batches × 25 min ≈ 75 min)

## Experiment Scripts

### Test Scripts
| Script | Purpose | Model | Methods | Status |
|--------|---------|-------|---------|--------|
| `test_percentile_sweep.sh` | Initial percentile sweep | Tiny | ECO0 only | ✅ Done |
| `test_percentile_fine.sh` | Fine-grained sweep around P90 | Tiny | ECO0 only | ✅ Done |
| `test_percentile_30M.sh` | Validate P90 on 30M | 30M | ECO + ECO0 | ⏳ Running |
| `test_30M_baselines.sh` | Full baseline comparison | 30M | 6 methods | 📋 Ready |
| `test_scaling_100M.sh` | Scaling: 100M model | 100M | FP32 + ECO + ECO0 | 📋 Ready |
| `test_scaling_300M.sh` | Scaling: 300M model | 300M | FP32 + ECO + ECO0 | 📋 Ready |
| `test_scaling_1B.sh` | Scaling: 1B model | 1B | ECO + ECO0 | 📋 Ready |
| `test_fp4_scale_fix.sh` | Scale recalibration test | Tiny | ECO + ECO0 | ✅ Done (rejected) |

### Training Scripts (Production)
| Script | Purpose | Optimizer |
|--------|---------|-----------|
| `train_eco0m-rooh.sh` | ECO0 training | eco0m-rooh |
| `train_eco.sh` | ECO training | eco |
| `train.sh` | General training | configurable |

### Submission Helpers
| Script | Purpose |
|--------|---------|
| `submit_fp4_test_smart.sh` | Try H200 → A100 → gpu-all |
| `submit_multipartition.sh` | Split array across partitions |

## Baseline Methods

### 1. FP32 Adam (Upper Bound)
- **Purpose:** Best possible quality baseline
- **Config:** No quantization, full precision
- **Memory:** ~18 bytes/param (4B params + 4B master + 4B grads + 4B m + 4B v)
- **Command:** `--opt adamw --w-quant NoQuantizer`

### 2. FP16 Adam (Mixed Precision Baseline)
- **Purpose:** Standard industry practice
- **Config:** FP16 compute, FP32 optimizer
- **Memory:** ~14 bytes/param (2B params + 4B grads + 4B m + 4B v)
- **Command:** `--opt adamw --w-bits 16`

### 3. Simple STE (Naive QAT)
- **Purpose:** 4-bit QAT without error feedback
- **Config:** 4-bit weights, master weights, full optimizer, no error feedback
- **Memory:** ~16.5 bytes/param (0.5B params + 4B master + 4B grads + 4B m + 4B v)
- **Command:** `--opt adamw --w-quant Q99FP4Quantizer`

### 4. CAGE (Curvature-Aware QAT)
- **Purpose:** State-of-art QAT with curvature correction
- **Config:** 4-bit weights, master weights, full optimizer, curvature correction
- **Memory:** ~16.5 bytes/param (same as STE + curvature overhead)
- **Command:** `--opt adamw --use-cage True --w-quant Q99FP4Quantizer`
- **Note:** Can work with any quantizer, not just Hadamard

### 5. ECO (Error Feedback in Momentum)
- **Purpose:** Master weight elimination baseline
- **Config:** 4-bit weights, no master weights, full optimizer, error feedback in m
- **Memory:** ~12.5 bytes/param (0.5B params + 4B grads + 4B m + 4B v)
- **LR:** 0.005 (higher than Adam due to error feedback)
- **Command:** `--opt eco --w-quant Q99FP4Quantizer --lr 0.005`

### 6. ECO0 (Our Method)
- **Purpose:** Eliminate both master weights AND optimizer states
- **Config:** 4-bit weights, no master weights, no optimizer states, error feedback in grads
- **Memory:** ~4.5 bytes/param (0.5B params + 4B grads)
- **LR:** 0.005 (higher than Adam due to gradient decay + error feedback)
- **Command:** `--opt eco0m-rooh --w-quant Q99FP4Quantizer --lr 0.005`

## Important: Learning Rate Differences

**Standard methods (FP32/FP16/STE/CAGE):** LR = 0.0012
- Use standard Adam momentum scaling

**Error feedback methods (ECO/ECO0):** LR = 0.005 (~4× higher)
- ECO: Error feedback in momentum changes effective step size
- ECO0: Gradient decay (grad = β₁ × grad_old + g_new) absorbs (1-β₁) factor into LR
- Both need higher LR to compensate for different momentum dynamics

## Key Findings So Far

### P90 Percentile (Tiny Model)
- **Finding:** FP4 quantization requires 90th percentile, not 99th
- **Improvement:** 1.8% (5.068 → 4.976 val loss)
- **Reason:** FP4's non-uniform codebook is sensitive to outliers
- **Status:** ✅ Added to paper (Method section + Ablation study)

### Scale Recalibration (Tiny Model)
- **Finding:** Dynamic scale recalibration does NOT help
- **Result:** Frozen scale is optimal
- **Status:** ✅ Tested and rejected, documented in CLAUDE.md

## Next Steps After Current Jobs Finish

### Step 1: Analyze 30M Validation Results
- Does P90 still win at 30M scale?
- Does ECO0 ≥ ECO at 30M?
- If yes → proceed to baselines
- If no → investigate why

### Step 2: Run 30M Baseline Comparison
```bash
./submit_multipartition.sh test_30M_baselines.sh
```
- 6 methods × 30M model × 5000 iterations
- 2 concurrent jobs → 3 batches → ~75 minutes total
- Generates: Main comparison table for paper

### Step 3: Memory Profiling
- Add memory tracking to training loop
- Log: peak memory, params, grads, optimizer states, activations
- Generate: Memory breakdown table

### Step 4: Generate Figures
- Training curves (convergence)
- Scaling plot (if we run 100M)
- Memory vs performance tradeoff
- Throughput comparison

### Step 5: Fill Paper
- Complete tables with actual numbers
- Write results section
- Write conclusion
- Add citations
- Generate figures

## Timeline Estimate

| Task | Duration | Dependencies |
|------|----------|--------------|
| 30M validation (running) | ~30 min | None |
| Analyze results | ~15 min | 30M validation |
| 30M baselines | ~75 min | Analysis |
| Memory profiling | ~30 min | Baselines |
| Generate figures | ~1 hour | All data |
| Fill paper | ~2 hours | Figures ready |
| Citations & polish | ~1 hour | Paper draft |
| **Total** | **~6 hours** | From now |

## Paper Status

### Complete Sections
- ✅ Abstract (needs numbers)
- ✅ Introduction (needs numbers)
- ✅ Related Work (needs citations)
- ✅ Method (complete with P90)
- ✅ Experiments setup + ablation

### Incomplete Sections
- ❌ Results (empty placeholder)
- ❌ Conclusion (empty placeholder)
- ❌ Figures (all placeholders)
- ❌ Tables (all "XX.XX")

### Critical TODOs
- Fill ~20 "XX" placeholders with actual numbers
- Add ~20 missing BibTeX citations
- Generate 4-5 figures from WandB
- Write 2-3 pages of results
- Write 1 page conclusion

## Submission Target

**Conference:** NeurIPS 2026 (hypothetical)
**Page Limit:** 9 pages (currently ~5-6 estimated)
**Current Status:** Method complete, experiments 20% done
**To Submission:** ~1 week of work remaining
