# Reversal-Separated G Data Candidates

**Date:** 2025-12-28
**Purpose:** Identify public datasets with reversal-separated or configuration-separated G values

---

## Search Summary

After extensive searching, **no publicly available dataset was found with true reversal-separated G values** (CW/CCW or near/far reported separately). All precision G experiments:
1. Use reversal as a systematic-cancellation technique
2. Report only the reversal-averaged final G value
3. Do not publish G_near vs G_far or G_CW vs G_CCW separately

---

## Candidate Papers Evaluated

### 1. Gershteyn et al. 2002 (arXiv:physics/0202058)
- **Title:** Experimental evidence that the gravitational constant varies with orientation
- **URL:** https://arxiv.org/abs/physics/0202058
- **Method:** Dynamic torsion balance, 7 months of continuous measurement
- **Claims:** G varies 0.054% with sidereal period (23.89 hr)
- **Reversal-separated?** NO
- **Data available:** Only periodograms and summary statistics; raw G time-series NOT published
- **File size:** 269 KB (PDF)
- **Status:** UNUSABLE - no individual G values published

### 2. Schlamminger et al. 2015 (arXiv:1505.01774)
- **Title:** Recent measurements of the gravitational constant as a function of time
- **URL:** https://arxiv.org/abs/1505.01774
- **Method:** Compilation of G measurements 1980-2015
- **Reversal-separated?** NO (but has **individual run data**)
- **Data available:**
  - **Table I: 26 TR&D-96 measurements (1985-1995)** with dates and type-A uncertainties
  - **Table II: 21 precision G measurements** with dates and methods
- **File size:** 163 KB (PDF)
- **Status:** USABLE as configuration-separated proxy (different torsion balances)

### 3. Nature 2018 Li et al. (already staged)
- **Title:** Measurements of the gravitational constant using two independent methods
- **URL:** https://www.nature.com/articles/s41586-018-0431-5
- **Reversal-separated?** NO (data is reversal-averaged)
- **Data available:** 36 individual G values (7 ToS + 29 AAF)
- **Status:** Already staged; method-separated but not reversal-separated

### 4. Rothleitner & Schlamminger 2017 Review (PMC8195032)
- **Title:** Measurements of the Newtonian constant of gravitation, G
- **Reversal-separated?** NO
- **Data available:** Table II has period values (T_far, ΔT) but NOT individual near/far G values
- **Status:** UNUSABLE for pred1

### 5. BIPM 2014 (Quinn et al.)
- **DOI:** 10.1098/rsta.2014.0032
- **Reversal-separated?** NO
- **Data available:** Method-separated (Cavendish vs servo) but not reversal-separated
- **Status:** Summary values only

---

## Selected Candidates for Extraction

| Rank | Candidate | Data Type | Rows | Usable for |
|------|-----------|-----------|------|------------|
| 1 | Schlamminger 2015 Table I | Individual G runs | 26 | Temporal analysis |
| 2 | Schlamminger 2015 Table II | Summary G values | 21 | Cross-experiment comparison |
| 3 | Nature 2018 (already have) | Method-separated G | 36 | Method reversal proxy |

---

## Why True Reversal-Separated Data Doesn't Exist Publicly

The standard practice in all precision G experiments:

1. **Time-of-Swing (ToS):**
   - Measure period T_near and T_far
   - Compute Δω² = (2π/T_near)² - (2π/T_far)²
   - Derive G from Δω² only
   - **Never publish G_near or G_far separately**

2. **Angular Acceleration Feedback (AAF):**
   - Continuous turntable rotation averages over many cycles
   - **Reversal averaging is built into the method**

3. **Cavendish/Deflection methods:**
   - Measure deflection in CW and CCW configurations
   - Report only the average
   - **Individual CW/CCW G values not published**

---

## Recommended Next Steps

### Option A: Use TR&D-96 as a Proxy
The 26 individual G measurements from Karagioz & Izmailov (TR&D-96) span 10 years with:
- Three different torsion balances
- Individual dates and uncertainties
- Can test for temporal/configuration structure

### Option B: Use Method-Separation as Proxy
The Nature 2018 ToS vs AAF comparison (already analyzed) provides:
- Method separation ≈ 0.84 × α²
- 95% CI includes 1.0 (consistent with α² scale)

### Option C: Contact Authors
Request raw data from Li et al. (Nature 2018) or other experimenters.

---

## Files Downloaded

| File | Size | SHA256 (first 16 chars) |
|------|------|-------------------------|
| gershteyn2002/physics0202058.pdf | 269 KB | (pending) |
| schlamminger2015/arxiv1505.01774.pdf | 163 KB | (pending) |

**Total downloaded: 432 KB** (well under 100 MB limit)
