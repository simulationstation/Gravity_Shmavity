# Nature 2018 Li et al. Attachment Inventory for Pre-Averaging Observables

**Date:** 2025-12-28
**Purpose:** Identify pre-averaging observables (near/far periods) for reversal-style testing

---

## File Summary

| File | Size | Sheets | Contains Pre-Avg Data? |
|------|------|--------|------------------------|
| source_data_extdata_fig5.xlsx | 15.6 KB | c | **YES** - Near/Far periods |
| source_data_fig2.xlsx | 4.2 MB | a,b,c,d,e,f | **YES** - Near/Far periods (a,b) |
| source_data_extdata_fig4.xlsx | 15.0 KB | c | No - Temperature/Acceleration |
| source_data_extdata_fig2.xlsx | 36.7 KB | c | No - Amplitude time series |
| source_data_fig3.xlsx | 10.3 KB | c | No - Historical G values |
| supplementary_data_suppfig1.xlsx | 1.2 MB | c | No - Power spectral density |
| supplementary_information.pdf | 2.7 MB | - | Background text only |

---

## Candidate Pre-Averaging Tables

### 1. source_data_extdata_fig5.xlsx - Sheet 'c'

**Description:** Period measurements at near and far source mass positions under different voltage conditions.

**Columns:**
- `Unnamed: 0` - Condition label (Ground, 0.1V, -0.1V)
- `Time (day)` - Time of near measurement
- `Period at near position (s)` - Period when source mass is near
- `Time (day).1` - Time of far measurement
- `Period at far position (s)` - Period when source mass is far

**Data Structure:**
- 27 rows total
- Near and far measured at different times (~3 days apart)
- Groups by voltage condition: Ground (rows 0-9, 22-26), 0.1V (rows 11-14), -0.1V (rows 16-20)
- Separator rows with '--' between conditions

**Pairing Strategy:** Same-row pairing (HIGH confidence - explicit paired columns)

**Sample Data:**
```
Condition  Time_near  T_near(s)   Time_far   T_far(s)
Ground     7.499      431.48941   10.500     433.19044
Ground     13.501     431.48898   16.500     433.18997
```

### 2. source_data_fig2.xlsx - Sheet 'a' (Fiber 1)

**Description:** Period time series for ToS measurement with Fiber 1.

**Columns:**
- `Time (day)` - Measurement time
- `Period at near position (s)` - Period when source mass is near
- `Period at far position (s)` - Period when source mass is far

**Data Structure:**
- 20 rows: rows 0-9 are near measurements, rows 10-19 are far measurements
- Near times: ~1, 7, 13, 19, 25, 31, 37, 43, 49, 55 days
- Far times: ~4, 10, 16, 22, 28, 34, 40, 46, 52, 58 days
- Measurements interleaved ~3 days apart

**Pairing Strategy:** Temporal proximity pairing (HIGH confidence - consistent 3-day offset)

### 3. source_data_fig2.xlsx - Sheet 'b' (Fiber 2)

**Description:** Period time series for ToS measurement with Fiber 2.

**Same structure as Sheet 'a' but with different fiber (different base period ~432.78s vs ~431.49s)**

**Pairing Strategy:** Temporal proximity pairing (HIGH confidence - consistent 3-day offset)

---

## Non-Candidate Tables (No Near/Far Separation)

### source_data_fig2.xlsx - Sheet 'c'
G values for 7 fibers (already averaged)

### source_data_fig2.xlsx - Sheet 'd'
Power spectral density data

### source_data_fig2.xlsx - Sheet 'e'
Acceleration time series (AAF method - no discrete reversal states)

### source_data_fig2.xlsx - Sheet 'f'
AAF G values by configuration (already averaged)

### source_data_extdata_fig4.xlsx
Temperature vs acceleration correlation data

### source_data_extdata_fig2.xlsx
Amplitude drift for coated/uncoated fibers

### source_data_fig3.xlsx
Historical G measurements compilation

### supplementary_data_suppfig1.xlsx
Power spectral density before/after compensation

---

## Summary of Reversal-Style Observables Found

| Source | Type | N Pairs | Confidence |
|--------|------|---------|------------|
| extdata_fig5 sheet c | T_near, T_far (same row) | ~24 | HIGH |
| fig2 sheet a | T_near, T_far (temporal) | 10 | HIGH |
| fig2 sheet b | T_near, T_far (temporal) | 10 | HIGH |
| **Total** | | **~44** | |

---

## Key Finding

**The Nature 2018 attachments DO contain pre-averaging observables:**
- Period at near position (T_near)
- Period at far position (T_far)

These are the direct measurements BEFORE computing Δω² = (2π/T_near)² - (2π/T_far)².

This allows testing pred1 at the **observable level**: Does |T_near - T_far| behave consistently across conditions?
