# 100M ECO-0 Learning Rate Analysis & Recommendation

## Current Results Summary

| Method | LR | Val Loss | Perplexity | Gap to Best |
|--------|-----|----------|------------|-------------|
| **ECO** (best) | **0.005** | **3.020** | **20.11** | **baseline** |
| ECO | 0.0031 | 3.075 | 21.26 | +1.15 PPL |
| ECO-0 | 0.007 | 3.078 | 21.35 | +1.24 PPL |
| ECO-0 | 0.0085 | 3.087 | 21.41 | +1.30 PPL |
| ECO-0 | 0.0042 | 3.130 | 22.39 | +2.28 PPL |

**Key Observations:**
1. ECO-0 at LR=0.007 achieves 21.35 PPL (current best for ECO-0)
2. ECO-0 at LR=0.0085 is **worse** (21.41 PPL) → optimum is ≤ 0.007
3. Gap to ECO: 1.24 PPL (20.11 vs 21.35)
4. At 50M, ECO-0 ≈ ECO (23.89 vs 23.92), so this gap is concerning

## Learning Rate Scaling Pattern Analysis

### ECO-0 Across Scales:
| Scale | Optimal LR | Scaling Factor from Previous |
|-------|------------|------------------------------|
| 30M | 0.01 | baseline |
| 50M (1.67×) | 0.01 | 1.0× (unscaled!) |
| 100M (2×) | 0.007? | 0.7× |

**Expected from sqrt scaling (50M→100M):** 0.01 / √2 = 0.00707 ≈ **0.007** ✓

### ECO Across Scales:
| Scale | Optimal LR | Scaling Factor from Previous |
|-------|------------|------------------------------|
| 30M | 0.00625 | baseline |
| 50M (1.67×) | 0.00625 | 1.0× (unscaled!) |
| 100M (2×) | 0.005 | 0.8× |

**Pattern:** Both methods show unscaled LRs from 30M→50M, then ~20-30% reduction for 50M→100M

## Hypothesis: Can ECO-0 Close the Gap?

**Evidence that further LR tuning may help:**

1. **50M precedent:** With optimal LRs, ECO-0 matched ECO (0.03 PPL difference)
2. **Limited search space:** Only tested 3 LRs at 100M (0.0042, 0.007, 0.0085)
3. **Monotonic improvement:** 0.0042 → 0.007 improved, but 0.007 → 0.0085 degraded
4. **ECO's optimal (0.005) is untested for ECO-0:** Gap between our tests

## LR Search Space Analysis

```
   Tested        Untested      Tested       Tested
     ↓              ↓            ↓            ↓
[0.0042]────[0.005]──[0.006]──[0.007]────[0.0085]
 22.39 PPL   ECO     ?        21.35 PPL   21.41 PPL
                    opt

           ↑──────────────────────↑
         Critical gap to explore
```

**Promising region:** 0.005 - 0.007 (between ECO's optimal and our current best)

## Recommended Experiments

### Priority 1: Test Lower LRs (Most Likely to Improve)

**Option A: Conservative 2-point search**
- **LR=0.006** - Midpoint between ECO's 0.005 and our 0.007
- **LR=0.0065** - Split the difference 0.006-0.007

**Rationale:**
- 0.0085 being worse suggests optimum is ≤ 0.007
- 0.006 is exactly between ECO (0.005) and ECO-0 current best (0.007)
- 0.0065 covers the upper half of the promising range

### Option B: More aggressive 3-point search
- **LR=0.0055** - Closer to ECO's optimal
- **LR=0.006** - Middle ground
- **LR=0.0065** - Upper range

**Rationale:**
- Tests if ECO-0 benefits from being closer to ECO's LR
- More thorough coverage of the promising region
- Higher computational cost (3 runs vs 2)

### Option C: Single best guess
- **LR=0.006** only

**Rationale:**
- Midpoint between ECO (0.005) and current best (0.007)
- If budget is limited, most likely to improve
- Can follow up based on result

## Expected Outcomes

### Best Case (LR=0.006 or 0.0065)
- ECO-0 achieves ~20.1-20.5 PPL, matching or approaching ECO's 20.11
- Validates that ECO-0 can compete with ECO at 100M scale
- Confirms memory reduction (8%) comes with minimal quality tradeoff

### Neutral Case
- ECO-0 achieves ~20.8-21.2 PPL (marginal improvement over 21.35)
- Suggests 100M scale may be challenging for ECO-0
- Still valuable to document the tradeoff

### Worst Case
- No improvement over 21.35 PPL
- Suggests ECO-0 has fundamental limitation at this scale
- Would need to investigate alternative approaches or accept quality gap

## Memory vs Quality Tradeoff Analysis

Current state at 100M:
- **ECO:** 20.11 PPL, 20.46 GB
- **ECO-0:** 21.35 PPL, 18.82 GB

**Tradeoff:** 8.0% memory reduction for 6.2% quality degradation

For comparison at 50M:
- **ECO:** 23.92 PPL, 29.30 GB  
- **ECO-0:** 23.89 PPL, 28.28 GB

**Tradeoff:** 3.5% memory reduction for 0.1% quality **improvement**

The 100M gap is concerning compared to 50M parity. **LR tuning is critical.**

## Recommendation

**Primary recommendation:** Test **LR ∈ {0.006, 0.0065}** (Option A)

**Reasoning:**
1. Covers the most promising region (between ECO's 0.005 and our 0.007)
2. Two experiments is manageable (2×10 hours ≈ 1 day)
3. If LR=0.006 substantially improves, we know to go lower
4. If LR=0.0065 is better, optimum is right at current best
5. If both fail to improve, we can conclude 21.35 is near optimal for ECO-0

**Next steps after results:**
- If 0.006 > 0.0065 > 0.007: test 0.0055 (go lower)
- If 0.0065 ≈ 0.007 > 0.006: optimum is around 0.0065-0.007 (done)
- If 0.006 is best: test 0.0055 to find lower bound
- If none improve: accept 21.35 as optimal, proceed to 500M

## Script to Run

```bash
# Create 100M ECO-0 LR refinement script
sbatch run_scripts/scaling/100M/test_100M_eco0_lr_refinement.sh

# Should test:
# - Task 1: ECO-0 LR=0.006 P90
# - Task 2: ECO-0 LR=0.0065 P90
```

Each experiment: ~38,146 iterations, ~8-10 hours on H200/A100
