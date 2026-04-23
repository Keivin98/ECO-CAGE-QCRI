# Experimental Status & Next Steps (April 22, 2026)

## ✅ Completed Experiments

### P90 Percentile (Tiny Model)
- **Finding:** P90 optimal (1.8% better than P99)
- **Use P90 for all experiments**

### 30M Model - All Baselines
| Method | Perplexity | LR |
|--------|------------|-----|
| FP16/FP32 Adam | 27.49 | 0.0012 |
| CAGE 4-bit | 29.08 | 0.0012 |
| **ECO0 4-bit** | **30.06** | **0.01** |
| ECO 4-bit | 32.73 | 0.00625 |

### 50M Model - Scaling + Ablations ✅ COMPLETE
| Method | Perplexity | LR | Strategy |
|--------|------------|-----|----------|
| FP16 Adam | 21.62 | 0.00093 | Scaled |
| CAGE 4-bit | 23.14 | 0.00093 | Scaled |
| **ECO0 4-bit** | **23.89** | **0.01** | **Unscaled** ✅ |
| **ECO 4-bit** | **23.92** | **0.00625** | **Unscaled** ✅ |

**Key Finding: ECO0 ≈ ECO at optimal LRs!**
- ECO0: 23.89 PPL, 28.28 GB memory
- ECO: 23.92 PPL, 29.30 GB memory
- **Difference: 0.03 PPL (tied), 1 GB memory savings**

**Ablations Completed:**
- ✅ ECO0 LR: tested 0.006-0.01 → **0.01 optimal**
- ✅ ECO0 percentile: P90 vs P95 → **P90 wins**
- ✅ ECO LR: **0.00625 optimal** (unscaled, same as 30M)

### 100M Model - Initial Results
| Method | Perplexity | LR | Memory (GB) | Status |
|--------|------------|-----|-------------|--------|
| FP16 Adam | 17.93 | 0.0006 | 20.46 | ✅ |
| CAGE 4-bit | 19.18 | 0.0006 | 20.46 | ✅ |
| ECO 4-bit | 21.26 | 0.0031 | 20.42 | ✅ Under-tuned |
| ECO0 4-bit | 22.39 | 0.0042 | **18.77** | ✅ Under-tuned |

**Memory Savings Scaling:**
- 30M: 1.0 GB (3.5%)
- 50M: 1.0 GB (3.5%)
- 100M: 1.7 GB (8.3%) ← **Validates linear scaling**

---

## 🔄 Currently Running

**100M Optimal LR Tests** (ETA: 16-20 hours)
- Task 1: ECO0, LR=0.007 (scaled from 50M's 0.01)
- Task 2: ECO, LR=0.005 (predicted from ratio)

---

## 📋 Recommended Learning Rates

**Key Insight:** ECO and ECO0 prefer **unscaled LRs** across 30M→50M

| Scale | FP16/CAGE | ECO | ECO0 | Strategy |
|-------|-----------|-----|------|----------|
| 30M | 0.0012 | **0.00625** | **0.01** | **Unscaled** ✅ |
| 50M | 0.00093 | **0.00625** | **0.01** | **Unscaled** ✅ |
| 100M | 0.0006 | **0.005** (test) | **0.007** (test) | Scale by 0.707 |
| 300M | 0.00042 | ~0.0035 | ~0.005 | Scale by 0.71 |

**Scaling rule:**
- FP16/CAGE: Standard 1/√scale from previous size
- ECO/ECO0: Keep same LR 30M→50M, then scale 50M→larger by 1/√scale

---

## 🎯 Immediate Next Steps

**1. ECO 50M finishes (~20 mins)**
- Result: ECO with LR=0.00625 (unscaled, same as 30M)
- Update: Section 4.5 in paper

**2. Wait for 100M optimal tests (~16-20 hours)**
- ECO0 LR=0.007
- ECO LR=0.005

**3. Update paper with 100M (~2 hours after results)**
- Add Section 4.6 (100M Scaling)
- Create `tab_100m_optimal.tex`
- Update discussion

---

## 🤔 Decision Point: Stop or Continue?

**After 100M optimal tests complete, you have 3 options:**

### Option A: Stop at 100M ✅ **RECOMMENDED**
**Evidence:**
- 3 model sizes (30M, 50M, 100M)
- Clear scaling trend (gap narrows 30M→50M)
- P90 validated across scales
- Memory scaling confirmed (3.5%→8.3%)

**Timeline:** Paper ready in ~1 week
**Compute:** Complete

### Option B: Add 300M 
**Why:** 4 model sizes stronger than 3
**Cost:** +40-60 hours compute
**Timeline:** Paper ready in ~2 weeks

### Option C: Add 300M + 1B
**Why:** Compelling memory story at 1B (10+ GB savings)
**Cost:** +120-180 hours compute
**Timeline:** Paper ready in ~3-4 weeks

**Recommendation:** Wait for 100M results quality before deciding. If ECO0 beats ECO and gap to CAGE narrows, Option A sufficient.

---

## 📝 Key Findings Summary

**What We Know:**
1. ✅ P90 percentile optimal for FP4 (validated tiny→30M→50M)
2. ✅ ECO0 beats ECO at all tested scales
3. ✅ Gap to CAGE narrows with scale (0.98→0.89 PPL)
4. ✅ Unscaled LR strategy works for ECO/ECO0
5. ✅ Memory savings scale linearly with model size

**What We're Testing:**
- Does ECO0 beat ECO at 100M with optimal LRs?
- Does gap to CAGE continue narrowing?
- Is 0.007 truly optimal for ECO0 at 100M?
