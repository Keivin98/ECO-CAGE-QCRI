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

```bash
# ECO-0 (primary)
bash train_eco0m-rooh.sh --model-size-prefix=30M --lr=0.0012

# ECO baseline
bash train_eco.sh --model-size-prefix=30M --lr=0.0012

# Hyperparameter sweeps
bash run_slurm_eco_b13.sh {1|2|3|4}
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

### 3. ECO vs ECO-0 Performance

**With optimal settings (P90 percentile)**:
- ECO-0 consistently outperforms ECO on tiny model
- Validation on 30M model in progress

## SLURM Job Management

### Multi-Partition Submission
```bash
# Smart submission: tries H200 → A100 → gpu-all
./submit_fp4_test_smart.sh

# Multi-partition: splits large arrays across partitions
./submit_multipartition.sh <script.sh>
```

**QoS Limits**: H200 partition limited to 2 concurrent jobs per user
- Use multi-partition scripts to avoid queued jobs

### Available Scripts
- `test_percentile_30M.sh`: Validates P90 on 30M model (ECO + ECO-0, P90/P95/P99)
- `test_percentile_fine.sh`: Fine-grained percentile sweep (tiny model)
- `test_percentile_sweep.sh`: Broad percentile sweep (P85-P100)

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
5. `train_eco0m-rooh.sh`, `train_eco.sh` - Training scripts

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
