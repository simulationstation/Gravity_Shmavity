# Mverse Ladder

Reproducible research codebase to test a layered substrate hypothesis (M0..M10) with a fixed alpha ladder of couplings.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .

ladder simulate --preset gravity_synth --out out/synth_grav
ladder fit --data out/synth_grav/data.csv --max_k 10 --out out/fit_grav

ladder simulate --preset timeseries_synth --out out/synth_ts
ladder fit --data out/synth_ts/data.npz --max_k 10 --out out/fit_ts

ladder calibrate --preset gravity_synth --alpha 0.05 --out out/calib
ladder end-to-end --preset gravity_synth --max_k 10 --out out/e2e
```

## Data formats

### Gravity configuration CSV
Required columns:
- `y`: observed residual or fractional deviation
- `temp`: temperature proxy
- `geom`: geometry proxy
- `config_0..config_{d-1}`: configuration embedding features

### Time series NPZ
Arrays:
- `t`: time array
- `ch1`, `ch2`: channel arrays

## Reproducibility
All simulation and fitting routines accept a `seed` parameter. Plots and reports are saved in the output directories.
