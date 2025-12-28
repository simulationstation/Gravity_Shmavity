# M3 Squared Tests

This repository implements five falsifiable M3 “squared-coupling” prediction tests plus a minimal M3 (α²) gravity residual model-fit harness. The tests check internal consistency of an α²-scaled hypothesis against nuisance alternatives; they are **not** evidence of new physics.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .

m3test validate-data --staged data/staged
m3test pred1 --staged data/staged --out outputs/pred1
m3test pred2 --staged data/staged --out outputs/pred2
m3test pred3 --staged data/staged --out outputs/pred3
m3test pred4 --staged data/staged --out outputs/pred4
m3test pred5 --staged data/staged --out outputs/pred5
m3test m3fit --staged data/staged --out outputs/m3fit
m3test all --staged data/staged --out outputs/all
```

Each command writes:
- `results.json`
- `report.md`
- `figures/*.png`

All figures are generated at runtime into `outputs/`.

## Staged data

The CLI expects the staged CSVs in `data/staged/`:
- `g_measurements_minimal.csv`
- `g_configs_minimal.csv`

Optional mapping files:
- `data/staged/topology_map.csv` with columns `config_id,topology_proxy`
- `data/staged/proxy_map.csv` with columns `config_id,quality_proxy`

## Tests

1. **Sign-blind intensity law** (`pred1`): checks reversal invariance using paired magnitudes with an equivalence tolerance tied to α².
2. **Topology-squared scaling** (`pred2`): compares linear vs. T² fits and reports the quadratic coefficient relative to α².
3. **Symmetry null suppression** (`pred3`): compares raw vs. nulled amplitudes and reports suppression factors.
4. **Threshold turn-on** (`pred4`): compares smooth fits against a sigmoid-squared threshold model for quality/coherence proxies.
5. **Cross-domain α² anchor** (`pred5`): compares dataset-scale clustering near α² with a permutation baseline.

## Minimal M3 gravity harness

`m3fit` compares a nuisance-only regression against a model with a single α²-scaled coupling term. It is an identifiability test, not a full GR replacement.
