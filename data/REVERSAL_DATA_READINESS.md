# Reversal Data Readiness Report

**Date:** 2025-12-28
**Purpose:** Assess whether sufficient data exists to run pred1 (sign-blind reversal invariance)

---

## Executive Summary

| Metric | Value | Required | Status |
|--------|-------|----------|--------|
| Datasets acquired | 2 | ≥1 | OK |
| Total rows | 37 | ≥10 | OK |
| Reversal-separated rows | **0** | ≥10 | **FAIL** |
| Reversal pairs | **0** | ≥5 | **FAIL** |
| Median uncertainty | 25.5 ppm | <α² (53 ppm) | OK |

**VERDICT: pred1 CANNOT be tested with publicly available data.**

---

## Detailed Findings

### 1. Data Acquired

| Dataset | Source | Rows | Type |
|---------|--------|------|------|
| trd96 | Schlamminger 2015 Table I | 26 | Individual runs (1985-1995) |
| nature2018_li | Li et al. Nature 2018 | 11* | Method-separated (ToS/AAF) |

*Note: Full Nature 2018 dataset has 36 rows; only 11 included in reversal_pred1.csv as sample.

### 2. Why No Reversal Pairs Exist

All publicly available G measurements are **reversal-averaged by design**:

| Method | Reversal Type | Published | NOT Published |
|--------|---------------|-----------|---------------|
| Time-of-Swing | Near/Far position | Δω² → G | G_near, G_far |
| AAF | CW/CCW rotation | Integrated α → G | G_CW, G_CCW |
| Cavendish | Opposite deflections | Averaged θ → G | θ_CW, θ_CCW |

**Standard practice**: Reversal is used to cancel systematic errors. Only the reversal-averaged result is published. Individual reversal-state values are never reported.

### 3. Papers Examined

| Paper | Year | Reversal Data? | Reason |
|-------|------|----------------|--------|
| Gershteyn et al. | 2002 | NO | Only periodograms, no raw G time series |
| Schlamminger et al. | 2015 | NO | Individual runs, but each run is reversal-averaged |
| Li et al. | 2018 | NO | Method-separated but not reversal-separated |
| Quinn et al. (BIPM) | 2014 | NO | Method-separated only |
| Gundlach & Merkowitz | 2000 | NO | Continuous rotation, inherently averaged |
| Rothleitner & Schlamminger | 2017 | NO | Review, summary values only |

### 4. Uncertainty Scale Assessment

```
Mean G:              6.673338 × 10⁻¹¹ m³ kg⁻¹ s⁻²
Median uncertainty:  25.5 ppm
Uncertainty range:   [9.0, 139.4] ppm
Alpha² scale:        53.3 ppm
```

The data precision is sufficient (median uncertainty < α²), but the data structure is wrong for pred1.

---

## Alternative Approaches

### Option A: Method-Separation as Proxy (ALREADY DONE)

The Nature 2018 ToS vs AAF comparison provides a different kind of "reversal":
- ToS uses discrete near/far position swap
- AAF uses continuous turntable rotation

**Results from previous analysis:**
- Method separation: 44.1 ppm (0.84 × α²)
- 95% CI for ratio: [0.64, 1.08] — **includes 1.0**
- Permutation p-value: < 0.005 — **statistically significant**

This analysis is documented in `outputs/nature2018/FOLLOWUP_REPORT.md`.

### Option B: Temporal Analysis of TR&D-96

The 26 TR&D-96 measurements span 10 years (1985-1995):
- Could test for temporal structure at α² scale
- Could correlate with sidereal or solar periods
- 3 different torsion balances used (but which runs from which is unknown)

### Option C: Contact Authors for Raw Data

Request from Li et al. (Nature 2018):
- Individual period measurements (T_near, T_far) before differencing
- Or G values computed separately from near-only vs far-only positions

**Contact:** Jun Luo (junluo@mail.hust.edu.cn), HUST, Wuhan, China

### Option D: Cross-Experiment Compilation Analysis

The Schlamminger 2015 compilation has 21 experiments:
- Can analyze experiment-to-experiment variance
- Can test for method-level structure across experiments
- Would need to model experiment-specific systematics

---

## Files Generated

### Raw Data
- `data/raw/reversal_candidates/gershteyn2002/physics0202058.pdf` (269 KB)
- `data/raw/reversal_candidates/schlamminger2015/arxiv1505.01774.pdf` (163 KB)

### Staged Data
- `data/staged/reversal_candidates/schlamminger2015/trd96_individual_runs.csv`
- `data/staged/reversal_candidates/schlamminger2015/g_measurements_compilation.csv`
- `data/staged/reversal_pred1.csv`
- `data/staged/reversal_pred1_pairs.csv`

### Metadata
- `data/metadata/reversal_candidates/schlamminger2015.json`
- `data/metadata/reversal_candidates/gershteyn2002.json`

### Documentation
- `data/REVERSAL_TARGET_SPEC.md`
- `data/REVERSAL_CANDIDATES.md`
- `data/REVERSAL_DATA_READINESS.md` (this file)

---

## Conclusion

**pred1 (sign-blind reversal invariance) cannot be tested with publicly available data.**

The fundamental barrier is that all G measurement publications report only reversal-averaged values. The individual CW/CCW or near/far G values are never published because:
1. They would have larger uncertainties
2. The reversal averaging is essential to cancel systematics
3. The meaningful quantity is the difference (Δω²), not individual values

**Recommended next step:** Use the method-separation analysis (ToS vs AAF) as the best available proxy, which already shows structure at the α² scale.

---

## SHA256 Checksums

```
379dcb6236e8e9b7115deb802a7da11a153a67c8228249055bf3a8e95cf5c911  gershteyn2002/physics0202058.pdf
8232a4e26b85de953c14a0cdc4d5a5245539aa2e749f09f99af64487cf369811  schlamminger2015/arxiv1505.01774.pdf
```

**Total downloaded: 432 KB** (within 100 MB limit)
