# Search for Reversal-Separated G Measurement Data

**Date:** 2025-12-28
**Purpose:** Find datasets where G values are published separately for each reversal direction

---

## Search Queries Performed

1. `gravitational constant G measurement reversal configuration separate clockwise counterclockwise source mass positions published data`
2. `"gravitational constant" measurement "clockwise" "counterclockwise" separate G values table data`
3. `Gundlach Merkowitz 2000 gravitational constant G angular acceleration separate rotation directions data`
4. `BIPM gravitational constant G measurement individual runs separate configurations published table data 2014`
5. `"gravitational constant" "near position" "far position" separate G values measurement data table`

---

## Candidate Papers Examined

### 1. Gundlach & Merkowitz (2000) - UWash
- **DOI:** 10.1103/PhysRevLett.85.2869
- **Method:** Angular acceleration feedback with continuous rotation
- **Result:** G = (6.674215 ± 0.000092) × 10⁻¹¹
- **Reversal-separated data?** NO
- **Reason:** Uses continuous attractor rotation; G derived from integrated acceleration, not separate CW/CCW values

### 2. BIPM (2001, 2014)
- **Reference:** Phil. Trans. R. Soc. A (2014)
- **Method:** Electrostatic compensation + deflection modes
- **Result:** Two G values from different methods, not different reversal directions
- **Reversal-separated data?** NO
- **Reason:** Reports method-separated values, not position-reversal-separated values

### 3. Li et al. Nature (2018) - Already staged
- **DOI:** 10.1038/s41586-018-0431-5
- **Methods:** ToS and AAF
- **Reversal-separated data?** NO
- **Reason:** Source Data provides G values derived from Δω² (already averages near/far)

### 4. Rothleitner & Schlamminger Review (2017)
- **DOI:** 10.1063/1.5004406 (Rev. Sci. Instrum.)
- **Content:** Comprehensive review of G measurements
- **Tables:** Parameters of experiments (period values, not separate G values)
- **Reversal-separated data?** NO
- **Reason:** Review article; summarizes final G values, not raw reversal-direction data

---

## Why Reversal-Separated Data is Not Published

The standard practice in G measurements is:

1. **ToS method:**
   - Measure period T_near at source-mass "near" position
   - Measure period T_far at source-mass "far" position
   - Compute Δω² = (2π/T_near)² - (2π/T_far)²
   - Derive G from Δω² (proportional to gravitational gradient)
   - **The reversal is inherent to the method; only Δω² → G is reported**

2. **AAF method:**
   - Continuous turntable rotation
   - Angular acceleration is integrated over many rotation cycles
   - **Reversal averaging is built into the measurement**

No standard precision G experiment publishes "G if we only used near" vs "G if we only used far" because:
- Such values would have much larger uncertainty
- The reversal averaging is essential to cancel systematic effects
- The meaningful quantity is the difference (Δω²), not the individual values

---

## Conclusion

**No publicly available dataset was found with reversal-direction-separated G values.**

All precision G measurements use reversal (position swap or rotation) as a fundamental technique to cancel systematics. The published results are always reversal-averaged.

---

## Recommended Next Steps for pred1

To test pred1 (sign-blind reversal invariance), the options are:

### Option 1: Contact Authors for Raw Data
Request period-by-period or run-by-run data from Li et al. (2018):
- Near-position period time series
- Far-position period time series
- Or G values computed separately from near-only vs far-only data

**Contact:** Jun Luo (junluo@mail.hust.edu.cn), HUST, Wuhan, China

### Option 2: Use Method Separation as Proxy
The ToS vs AAF comparison provides a different kind of "reversal":
- ToS uses position swap
- AAF uses continuous rotation
- These probe systematics differently

This is already captured in the m3fit and m3_indication analyses.

### Option 3: Accept pred1 as Untestable with Public Data
Document that pred1 requires data granularity not available in published sources.

---

## Files Searched (not downloaded)

- arXiv:gr-qc/0006043 (Gundlach & Merkowitz abstract only)
- PMC8294353 (commentary, no supplementary data)
- PMC8290936 (review, no raw reversal data)

**Total additional downloads: 0 bytes** (stayed within limit)
