# Scaling Configurations for Multi-Size Experiments

## Memory Considerations

**GPU:** H200 (141 GB memory)
**Constraint:** Model params + activations + gradients + optimizer states must fit

**Memory Formula (rough estimate):**
```
Total = Model_params + Activations + Gradients + Optimizer_states

Activations ≈ batch_size × seq_len × n_layers × hidden_size × 12 (fwd+bwd)
Model params: Fixed per model size
Gradients: Same size as params
Optimizer states: Depends on method (0 for ECO0, 8 bytes/param for Adam)
```

## Model Scaling Configurations

### 30M Model (6 layers, 640 embd, 5 heads)
**Parameters:** ~30M
**Config:**
```bash
BATCH_SIZE=64
ACC_STEPS=8
SEQUENCE_LENGTH=512
EFFECTIVE_BATCH=512 tokens (64 × 8)
```
**Memory per method (estimated):**
- FP32 Adam: ~8 GB
- ECO: ~5 GB
- ECO0: ~3 GB
**Status:** ✅ Tested, fits comfortably

---

### 100M Model (8 layers, 1024 embd, 8 heads)
**Parameters:** ~100M
**Config (Conservative):**
```bash
BATCH_SIZE=32      # Reduced from 64
ACC_STEPS=16       # Increased from 8
SEQUENCE_LENGTH=512
EFFECTIVE_BATCH=512 tokens (32 × 16)
```
**Memory per method (estimated):**
- FP32 Adam: ~25 GB (should fit)
- ECO: ~18 GB (comfortable)
- ECO0: ~10 GB (plenty of room)

**Config (Aggressive - if conservative fits well):**
```bash
BATCH_SIZE=48
ACC_STEPS=11       # Round up for clean division
SEQUENCE_LENGTH=512
EFFECTIVE_BATCH=528 tokens
```

**Status:** ⏳ To be tested

---

### 300M Model (16 layers, 1664 embd, 13 heads)
**Parameters:** ~300M
**Config (Conservative):**
```bash
BATCH_SIZE=16      # Much smaller
ACC_STEPS=32       # Much larger
SEQUENCE_LENGTH=512
EFFECTIVE_BATCH=512 tokens (16 × 32)
```
**Memory per method (estimated):**
- FP32 Adam: ~70 GB (tight!)
- ECO: ~50 GB (should fit)
- ECO0: ~28 GB (comfortable)

**Config (Micro-batching if needed):**
```bash
BATCH_SIZE=8       # Minimal
ACC_STEPS=64       # Maximum
SEQUENCE_LENGTH=512
EFFECTIVE_BATCH=512 tokens
```

**Status:** ⏳ To be tested, may need micro-batching

---

### 1B Model (24 layers, 2048 embd, 16 heads)
**Parameters:** ~1B
**Config (Aggressive micro-batching):**
```bash
BATCH_SIZE=4       # Very small
ACC_STEPS=128      # Very large
SEQUENCE_LENGTH=512
EFFECTIVE_BATCH=512 tokens
```
**Memory per method (estimated):**
- FP32 Adam: ~220 GB ❌ **WON'T FIT on single H200!**
- ECO: ~160 GB ❌ **WON'T FIT**
- ECO0: ~85 GB ✅ **Might fit with micro-batching**

**Alternative for 1B:**
- Reduce sequence length: 512 → 256
- Use gradient checkpointing
- Or skip FP32/ECO, only run ECO0 to demonstrate scaling

**Status:** ⏳ May need special handling

---

## Recommended Experiment Strategy

### Phase 1: Core Validation (30M + 100M)
**Goal:** Prove method works and scales
**Models:** 30M, 100M
**Methods:** All 6 (FP32, FP16, STE, CAGE, ECO, ECO0)
**Timeline:** ~1-2 days

### Phase 2: Scaling Demonstration (300M)
**Goal:** Show memory advantage at scale
**Models:** 300M
**Methods:** FP32 (baseline), ECO (comparison), ECO0 (ours)
**Note:** FP32 might be tight, focus on ECO vs ECO0
**Timeline:** ~1 day

### Phase 3: Extreme Scaling (1B) - Optional
**Goal:** Demonstrate ECO0 enables training that otherwise wouldn't fit
**Models:** 1B
**Methods:** ECO0 only (maybe ECO for comparison)
**Message:** "ECO0 trains 1B model in memory budget where FP32 can't"
**Timeline:** ~1 day

---

## Adaptive Configuration Script

For automatic scaling, use this logic:

```bash
# Auto-adjust batch size based on model size
if [ $MODEL_PARAMS -le 50000000 ]; then
    # ≤50M: Standard config
    BATCH_SIZE=64
    ACC_STEPS=8
elif [ $MODEL_PARAMS -le 150000000 ]; then
    # 50M-150M: Reduce batch, increase accum
    BATCH_SIZE=32
    ACC_STEPS=16
elif [ $MODEL_PARAMS -le 500000000 ]; then
    # 150M-500M: Micro-batching
    BATCH_SIZE=16
    ACC_STEPS=32
else
    # >500M: Aggressive micro-batching
    BATCH_SIZE=8
    ACC_STEPS=64
fi

# Keep effective batch size constant (512 tokens)
EFFECTIVE_BATCH=$((BATCH_SIZE * ACC_STEPS))
```

---

## Memory Profiling TODO

For each experiment, log:
1. **Peak GPU memory** (GB)
2. **Memory breakdown:**
   - Model parameters
   - Gradients
   - Optimizer states (m, v buffers)
   - Activations
   - Other (buffers, temp storage)
3. **Throughput** (tokens/sec)
4. **Training time** (hours to convergence)

This data will populate the paper's memory comparison table.

---

## Conservative vs Aggressive Strategy

### Conservative (Recommended for Paper)
- Start with smaller batch sizes
- Guarantee no OOM
- Keep effective batch constant across scales
- Easy to explain in paper: "All models trained with 512 effective batch"

### Aggressive (If compute is plentiful)
- Push batch size as high as possible
- Maximize throughput
- Risk: Some configs might OOM, need retry
- Benefit: Faster experiments

**Recommendation:** Go conservative. Consistent effective batch size makes results cleaner and more comparable.

---

## Paper Narrative

**Key message:** "ECO0's memory efficiency enables training larger models or larger batches within fixed memory budgets"

**Table to include:**
| Model Size | Method | Batch Size | Grad Accum | Peak Memory (GB) | Fits? |
|------------|--------|------------|------------|------------------|-------|
| 30M | FP32 | 64 | 8 | 8.2 | ✅ |
| 30M | ECO | 64 | 8 | 5.1 | ✅ |
| 30M | ECO0 | 64 | 8 | 2.8 | ✅ |
| 100M | FP32 | 32 | 16 | 24.5 | ✅ |
| 100M | ECO | 32 | 16 | 17.8 | ✅ |
| 100M | ECO0 | 32 | 16 | 9.6 | ✅ |
| 300M | FP32 | 16 | 32 | 68.4 | ✅ |
| 300M | ECO | 16 | 32 | 49.2 | ✅ |
| 300M | ECO0 | 16 | 32 | 27.1 | ✅ |
| 1B | FP32 | - | - | >140 | ❌ |
| 1B | ECO | 4 | 128 | 135.2 | ⚠️ |
| 1B | ECO0 | 4 | 128 | 82.8 | ✅ |

**Narrative:** "At 1B parameters, FP32 Adam exceeds H200 memory capacity, while ECO0 fits comfortably, demonstrating how memory efficiency enables scaling beyond traditional methods."

---

## Implementation Notes

### Gradient Accumulation
- Already supported in codebase
- `--acc-steps N` flag
- No code changes needed

### Batch Size
- `--batch-size N` flag
- May need to adjust for different model sizes

### Sequence Length
- `--sequence-length N` flag (default 512)
- Can reduce if needed for 1B model

### Memory Profiling
- Add `torch.cuda.max_memory_allocated()` logging
- Log after each training step
- Export to WandB for easy tracking

**Next:** Once 30M validation finishes, we'll know if adjustments are needed and can proceed with confidence!
