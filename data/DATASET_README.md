# G Measurement Datasets for M3 Ladder Analysis

**Date fetched:** 2025-12-28
**Total size:** ~5.5 MB (well under 50 MB limit)

## Summary

This directory contains high-quality gravitational constant (G) measurement data for M3 (α² ≈ 53 ppm) ladder analysis, including **within-experiment individual run data** from the Nature 2018 Li et al. paper.

## What Was Fetched

### 1. Nature 2018 Li et al. Source Data (NEW - Within-Experiment Structure)
- **Source:** Nature 560, 582-588 (2018)
- **DOI:** 10.1038/s41586-018-0431-5
- **Access:** Direct download from static-content.springer.com
- **Downloaded:** 6 XLSX source data files (5.15 MB total)
- **Content:**
  - **ToS method:** 7 individual fiber measurements + weighted average
  - **AAF method:** 29 individual runs (4 AAF-I, 10 AAF-II, 15 AAF-III) + weighted average
  - Period time series (40 measurements at near/far positions)
  - Historical G compilation (17 experiments)

### 2. NSR Compilation (Cross-Experiment Context)
- **Source:** National Science Review, Vol 7, Issue 12, 2020
- **DOI:** 10.1093/nsr/nwaa165
- **Access:** Open-access via PMC (PMC8290936)
- **Content:** 13 individual G measurements from 2000-2018

### 3. CODATA Recommended Values
- **Source:** NIST / CODATA Task Group
- **Content:** Historical CODATA recommended values 1973-2022
- **Current value:** G = 6.67430(15) × 10⁻¹¹ m³ kg⁻¹ s⁻²

## Files

```
data/
├── raw/
│   ├── nature2018_li_source_data/
│   │   ├── source_data_moesm2.xlsx   # Power spectral density (1.1 MB)
│   │   ├── source_data_moesm3.xlsx   # Main data: G values, periods (4.2 MB)
│   │   ├── source_data_moesm4.xlsx   # Historical compilation (10 KB)
│   │   ├── source_data_moesm5.xlsx   # Fiber amplitude data (37 KB)
│   │   ├── source_data_moesm6.xlsx   # Temperature vs accel (15 KB)
│   │   └── source_data_moesm7.xlsx   # Ground vs underground (16 KB)
│   └── g_compilation/
│       ├── nsr_2021_g_measurements.csv
│       └── codata_recommended_values.csv
├── staged/
│   ├── g_measurements_minimal.csv         # Unified (52 measurements)
│   ├── g_configs_minimal.csv              # Unified (26 configs)
│   ├── nature2018_li_tos_individual.csv   # ToS method details
│   ├── nature2018_li_aaf_individual.csv   # AAF method details
│   ├── nature2018_li_period_timeseries.csv
│   └── nature2018_li_historical_compilation.csv
├── metadata/
│   ├── nature2018_li_source_data.json     # Full provenance
│   ├── nsr_compilation.json
│   └── codata_recommended.json
└── DATASET_README.md
```

## Validation Results

### Nature 2018 Within-Experiment Data
```
Total measurements: 38 (8 ToS + 30 AAF)

Time-of-Swing method (7 individual + 1 average):
  G range: 6.674017 - 6.674274 (×10⁻¹¹)
  Within-method scatter: 14.2 ppm
  Mean deviation from CODATA: -17.7 ppm

Angular Acceleration Feedback (29 individual + 1 average):
  G range: 6.674302 - 6.674607 (×10⁻¹¹)
  Within-method scatter: 13.2 ppm
  Mean deviation from CODATA: +26.9 ppm

ToS - AAF difference: 44.9 ppm
```

### Cross-Experiment Summary
```
N experiments (NSR compilation): 13
Inter-experiment scatter: 168.4 ppm
Range: -358 to +193 ppm from CODATA
```

## Relevance to M3 (α² ~ 53 ppm) Check

| Metric | Value | vs α² (53 ppm) | Interpretation |
|--------|-------|----------------|----------------|
| α² | 53.3 ppm | - | M3 target scale |
| ToS within-method scatter | 14.2 ppm | << α² | **Can probe M3** |
| AAF within-method scatter | 13.2 ppm | << α² | **Can probe M3** |
| ToS-AAF difference | 44.9 ppm | ~ α² | **Suspicious!** |
| Best single precision | 12 ppm | < α² | Sensitive |
| Inter-experiment scatter | 168 ppm | >> α² | Masks signal |

**Key insight:** The within-experiment scatter (~13-14 ppm) is well below the α² scale (53 ppm), meaning the individual run data CAN probe M3-level effects. Critically, the ToS-AAF method difference (45 ppm) is remarkably close to α², warranting investigation.

## Data Structure for M3 Analysis

### g_measurements_minimal.csv Columns
| Column | Description |
|--------|-------------|
| dataset_id | `nature2018_li`, `nsr_compilation`, or `codata_2022` |
| method_id | `time_of_swing`, `angular_acceleration_feedback`, etc. |
| config_id | Unique identifier (e.g., `TOS_fiber_1`, `AAF_AAF-II`) |
| timestamp_or_run_index | Run number or year |
| G_value | G in SI units (m³ kg⁻¹ s⁻²) |
| G_sigma | 1σ uncertainty in SI units |
| G_value_1e11 | G in units of 10⁻¹¹ |
| G_sigma_1e11 | Uncertainty in units of 10⁻¹¹ |
| uncertainty_ppm | Relative uncertainty |
| notes | Provenance and flags |

### Configuration Groups (nature2018_li)
- `TOS_fiber_1` through `TOS_fiber_7`: Individual fiber measurements
- `TOS_weighted_average`: Published ToS result
- `AAF_AAF-I` (4 runs), `AAF_AAF-II` (10 runs), `AAF_AAF-III` (15 runs)
- `AAF_weighted_average`: Published AAF result

## Caveats

1. **Period data not yet linked to G:** The period time series could enable configuration-resolved analysis
2. **Fiber identity not fully characterized:** ToS fibers may have different systematic properties
3. **AAF run grouping unclear:** AAF-I, II, III may represent different experimental conditions
4. **No timestamps:** Run indices, not actual times

## References

1. Li Q, et al. (2018). Measurements of the gravitational constant using two independent methods. Nature 560, 582-588. DOI: 10.1038/s41586-018-0431-5

2. Li Q, et al. (2020). Precision measurement of the Newtonian gravitational constant. National Science Review 7(12), 1803-1817. DOI: 10.1093/nsr/nwaa165

3. Tiesinga E, et al. (2024). CODATA Recommended Values of the Fundamental Physical Constants: 2022. arXiv:2409.03787
