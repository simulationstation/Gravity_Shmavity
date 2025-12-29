# Nature 2018 Li et al. Data Acquisition Report

**Date:** 2025-12-28
**Source:** Li Q et al., Nature 560, 582-588 (2018)
**DOI:** 10.1038/s41586-018-0431-5

---

## Discovered Attachment Links

| ID | Label | Size | Extension |
|----|-------|------|-----------|
| MOESM1 | Supplementary_Information | 2647 KB | pdf |
| MOESM2 | Supplementary_Data_SuppFig1 | 1125 KB | xlsx |
| MOESM3 | Source_Data_Fig2 | 4071 KB | xlsx |
| MOESM4 | Source_Data_Fig3 | 10 KB | xlsx |
| MOESM5 | Source_Data_ExtData_Fig2 | 35 KB | xlsx |
| MOESM6 | Source_Data_ExtData_Fig4 | 14 KB | xlsx |
| MOESM7 | Source_Data_ExtData_Fig5 | 15 KB | xlsx |

**Total download size:** 7.74 MB (well under 200 MB limit)

---

## What Was Downloaded

All 7 attachments were successfully downloaded from `static-content.springer.com`:

1. **Supplementary Information (PDF)**: 2.65 MB
   - Experimental method details
   - Supplementary Tables 1-4 (not yet extracted)
   - Alignment and positioning procedures

2. **Source Data Fig. 2 (XLSX)**: 4.07 MB - **KEY DATA**
   - Sheet 'a': ToS period time series (apparatus 1)
   - Sheet 'b': ToS period time series (apparatus 2)
   - Sheet 'c': **ToS G values per fiber (7 measurements)**
   - Sheet 'd': Power spectral density data
   - Sheet 'e': Acceleration time series
   - Sheet 'f': **AAF G values per run (29 measurements)**

3. **Source Data Fig. 3 (XLSX)**: 10 KB
   - Historical G compilation (17 experiments)

4. **Source Data Extended Data Fig. 2-5 (XLSX)**: ~65 KB total
   - Fiber decay curves
   - Temperature vs acceleration
   - Period near/far with electrostatic tests

---

## Staged Data Summary

### Primary Granular Dataset

| File | Rows | Description |
|------|------|-------------|
| g_measurements_nature2018_granular.csv | **36** | All individual G measurements |
| g_configs_nature2018_granular.csv | **10** | Configuration metadata |

### Method Breakdown

| Method | Measurements | Configs | Notes |
|--------|--------------|---------|-------|
| time_of_swing | 7 | 7 fibers | Fibers 1-7 |
| angular_acceleration_feedback | 29 | 3 groups | AAF-I (4), AAF-II (10), AAF-III (15) |

### G Value Statistics

| Metric | ToS | AAF |
|--------|-----|-----|
| Count | 7 | 29 |
| Mean (×10⁻¹¹) | 6.674182 | 6.674480 |
| Std (×10⁻¹¹) | 0.000095 | 0.000088 |
| vs CODATA | -17.7 ppm | +26.9 ppm |
| Method difference | **44.6 ppm** (≈ α²) |

---

## Reversal Pair Analysis for M3 Tests

### What We Have

- **36 individual G measurements** with uncertainties
- All measurements have σ (12-33 ppm relative)
- Two distinct methods (ToS, AAF) that can be compared

### What's Missing for pred1 (Reversal Invariance)

**The Source Data provides G values that are already reversal-averaged.**

In the ToS method:
- Each G value is derived from Δω² = ω²_far - ω²_near
- The near/far reversal is internal to each measurement
- No separate "G from near only" vs "G from far only" available

In the AAF method:
- Each G value uses continuous turntable rotation
- The angular modulation averages over reversal
- No separate "clockwise" vs "counter-clockwise" G values

### Reversal Pairs Found

**Count: 0** (at the G-value level)

The Source Data does not include:
- G values computed separately for each source mass position
- G values computed separately for each rotation direction

### Alternative for pred1

The ExtData Fig 5 data provides **24 near/far period pairs**, but these are electrostatic sensitivity tests, not independent G determinations. They show:
- Period at near position: ~431.49 s
- Period at far position: ~433.19 s
- Period difference: 1.70113 ± 0.0001 s

This demonstrates the reversal mechanism works, but doesn't give separate G values.

---

## Impact on M3 Tests

| Test | Status | Explanation |
|------|--------|-------------|
| **pred1** (reversal invariance) | **INCONCLUSIVE** | No un-averaged reversal pairs in Source Data |
| **pred2** (topology scaling) | **TESTABLE** | Can use fiber identity or AAF group as proxy |
| **pred3** (symmetry nulls) | **TESTABLE** | Can compare within-group residuals |
| **pred4** (threshold) | **TESTABLE** | Quality proxy available (uncertainty) |
| **pred5** (cross-domain) | **TESTABLE** | ToS vs AAF method comparison |
| **m3fit** | **TESTABLE** | 36 measurements with config variation |

---

## Next Steps for True Reversal Data

If reversal-resolved G values are required, the options are:

### Option 1: Request from Authors

The paper's corresponding author is:
- **Jun Luo** (junluo@mail.hust.edu.cn)
- HUST, Wuhan, China

Draft email:

> Subject: Request for reversal-resolved G values from Nature 2018 data
>
> Dear Professor Luo,
>
> I am analyzing the G measurement data from your Nature 2018 paper
> (doi:10.1038/s41586-018-0431-5) for systematic pattern tests.
>
> The published Source Data provides G values that are already averaged
> over source-mass position reversal. For our analysis, we would benefit
> from G values computed separately for:
> - Near vs far source positions (ToS method)
> - Individual turntable angles (AAF method)
>
> Would it be possible to share these intermediate values, or can you
> confirm whether such data exist?
>
> Thank you for your consideration.

### Option 2: Derive from Period Data

The period time series in sheets 'a' and 'b' only contain near-position
periods. The far-position periods are not in the Source Data (columns show
NaN). ExtData Fig 5 has both but those are electrostatic tests.

---

## Files Created

```
data/
├── raw/nature2018_li/attachments/
│   ├── supplementary_information.pdf (2.65 MB)
│   ├── supplementary_data_suppfig1.xlsx (1.13 MB)
│   ├── source_data_fig2.xlsx (4.07 MB)
│   ├── source_data_fig3.xlsx (10 KB)
│   ├── source_data_extdata_fig2.xlsx (36 KB)
│   ├── source_data_extdata_fig4.xlsx (15 KB)
│   └── source_data_extdata_fig5.xlsx (15 KB)
├── staged/nature2018_li/
│   ├── fig2_a_tos_period_series1.csv
│   ├── fig2_b_tos_period_series2.csv
│   ├── fig2_c_tos_g_by_fiber.csv
│   ├── fig2_f_aaf_g_by_run.csv
│   ├── fig3_c_historical_g.csv
│   ├── extdata_fig4_temp_accel.csv
│   └── extdata_fig5_period_nearfar.csv
├── staged/
│   ├── g_measurements_nature2018_granular.csv (36 rows)
│   └── g_configs_nature2018_granular.csv (10 rows)
├── metadata/
│   └── nature2018_li.json
└── DATA_NOTE.md
```

---

## Summary

**We downloaded the most granular publicly available data from Li et al. Nature 2018.**

- **36 individual G measurements** (7 ToS + 29 AAF)
- **100% with uncertainties** (12-33 ppm)
- **ToS-AAF method difference**: 44.6 ppm ≈ α² (notable)

**Limitation for pred1**: The published Source Data does not include
reversal-resolved G values. Each measurement is already averaged over
the near/far or rotation reversal direction.

For true reversal pair analysis, author contact or alternative datasets
would be needed.
