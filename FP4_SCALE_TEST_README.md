# FP4 Scale Recalibration Test

## What This Tests

This experiment diagnoses whether **frozen scale calibration** is causing ECO to outperform ECO-0 with Q99FP4Quantizer.

## The Hypothesis

**Problem**: Q99FP4Quantizer calibrates scale once at initialization (on random weights) and never updates during training.

- As training progresses, weight distribution changes
- But scale stays frozen → quantization becomes increasingly inaccurate
- Non-uniform FP4 codebook is very sensitive to scale mismatch

**Why ECO might handle this better**:
- ECO stores explicit momentum that accumulates error corrections over time
- Acts as a low-pass filter smoothing out scale mismatch effects

**Why ECO-0 struggles**:
- Stateless variance computed from current gradient only
- More sensitive to per-step quantization quality
- No historical smoothing to compensate for bad quantization

## The Fix

Added `recalibrate_interval` parameter to Q99FP4Quantizer:
- `recalibrate_interval=0`: Original behavior (frozen scale)
- `recalibrate_interval=N`: Recalibrate scale every N calls to `hard_quantize()`

## Test Matrix

| Run | Optimizer | Scale Mode | Config |
|-----|-----------|------------|--------|
| 1   | ECO       | Frozen     | `recalibrate_interval=0` (baseline) |
| 2   | ECO       | Dynamic    | `recalibrate_interval=100` |
| 3   | ECO-0     | Frozen     | `recalibrate_interval=0` (baseline) |
| 4   | ECO-0     | Dynamic    | `recalibrate_interval=100` |

## Expected Results

### If scale drift IS the issue:

**ECO**: Minor improvement with dynamic scale
```
Frozen:  ~3.28 val loss
Dynamic: ~3.26 val loss (small improvement)
```

**ECO-0**: **Major improvement** with dynamic scale
```
Frozen:  ~3.35 val loss (current poor performance)
Dynamic: ~3.25 val loss (back to winning!)
```

### If scale drift is NOT the issue:

No significant difference between frozen and dynamic scale for either optimizer.

## How to Run

```bash
sbatch test_fp4_scale_fix.sh
```

Monitor in WandB project: `FP4-SCALE-DEBUG`

## What to Look For in WandB

1. **Validation loss curves**: Does dynamic scale help?
2. **Training stability**: Are there fewer spikes with dynamic scale?
3. **Final val loss ranking**: Does ECO-0 + dynamic scale beat ECO + frozen scale?

## Interpreting Results

### Scenario A: Dynamic scale fixes ECO-0
✅ **Scale drift confirmed as root cause**
- Use `recalibrate_interval=100` for FP4 in production
- ECO-0 wins with proper scale management
- Paper contribution: "ECO-0 + dynamic scale recalibration"

### Scenario B: Dynamic scale helps both equally
⚠️ **Scale drift is a factor, but not the only issue**
- FP4 codebook quality might also be a problem
- Consider better codebook design (symmetric, denser)
- ECO might still have inherent advantage with FP4

### Scenario C: No improvement with dynamic scale
❌ **Scale drift is not the issue**
- Problem lies elsewhere (codebook design, error feedback interaction, etc.)
- Need deeper investigation into FP4 vs Int differences

## Next Steps Based on Results

If scale fix works → Run full 30M sweep with dynamic scale FP4
If scale fix doesn't work → Investigate codebook asymmetry or switch to better FP4 format
If partial fix → Combine dynamic scale + improved codebook

## Quick Check Commands

```bash
# Check if job is running
squeue -u kisufaj

# Monitor output in real-time
tail -f fp4_scale_test_*.out

# Check for errors
tail -f fp4_scale_test_*.err
```
