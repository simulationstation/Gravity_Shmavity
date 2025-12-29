# Reversal-Separated G Data Target Specification

**Date:** 2025-12-28
**Purpose:** Define what constitutes "usable" reversal-separated data for pred1 testing

---

## Definition of "Reversal-Separated"

A dataset is **reversal-separated** if it provides G values (or pre-averaging observables)
that are reported SEPARATELY for each reversal configuration, rather than as a single
reversal-averaged result.

### Acceptable Reversal Types

| Type | Description | Example Labels |
|------|-------------|----------------|
| Rotational | CW vs CCW turntable/attractor rotation | CW, CCW, +, - |
| Positional | Near vs Far source-mass positions | near, far, N, F, A, B |
| Swapped | Source masses physically swapped | config_A, config_B, swapped |
| Sign flip | Polarity reversal in feedback systems | +dir, -dir, forward, reverse |

---

## Minimum Required Columns

| Column | Required? | Description |
|--------|-----------|-------------|
| dataset_id | YES | Unique identifier for the paper/experiment |
| config_id | YES | Apparatus/configuration identifier |
| reversal_flag | YES | Indicator of reversal state (0/1 or label) |
| run_id or point_index | YES | Unique identifier for the measurement |
| G_value or observable | YES | The measured value (G or period/frequency) |
| G_sigma or uncertainty | PREFERRED | Measurement uncertainty |
| units | YES | Units of the value |
| notes | PREFERRED | Provenance (table name, figure reference) |

---

## Examples of Usable Data

### Example 1: CW/CCW Rotation (AAF method)
```
run_id, reversal_flag, G_value_1e11, G_sigma_1e11
1, CW, 6.67428, 0.00012
1, CCW, 6.67431, 0.00011
2, CW, 6.67425, 0.00013
2, CCW, 6.67429, 0.00012
```

### Example 2: Near/Far Position (ToS method)
```
fiber_id, position, period_s, period_sigma_s
1, near, 532.1847, 0.0003
1, far, 530.9123, 0.0003
2, near, 532.1851, 0.0004
2, far, 530.9119, 0.0004
```

### Example 3: Swapped Masses
```
run_id, config, G_value_1e11
1, masses_standard, 6.67421
1, masses_swapped, 6.67419
2, masses_standard, 6.67423
2, masses_swapped, 6.67420
```

---

## What We CANNOT Use

- Single reversal-averaged G value only
- Δω² or ΔG without the individual components
- "Final G" tables without reversal breakdown
- Period differences without individual periods

---

## Pairing Rules

For pred1 testing, we need to form **reversal pairs**:

1. Same `config_id` (same apparatus configuration)
2. Same `run_id` or temporal proximity (consecutive measurements)
3. Opposite `reversal_flag` values
4. Both values have the same units

A valid pair allows computing:
- `delta = value_A - value_B`
- `abs_delta = |delta|`
- `sigma_delta = sqrt(sigma_A^2 + sigma_B^2)` if uncertainties exist

---

## Minimum Viable Dataset

- At least 5 reversal pairs from at least 1 experiment
- Uncertainties preferred but not strictly required
- Direct G values preferred over derived observables
