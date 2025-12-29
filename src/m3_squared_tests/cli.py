"""CLI entrypoints for M3 squared-coupling tests."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from m3_squared_tests.io.load_staged import load_staged
from m3_squared_tests.io.schema import validate_schema
from m3_squared_tests.testsuites.m3_gravity_fit import run_m3_fit
from m3_squared_tests.testsuites.pred1_sign_blind import run_pred1
from m3_squared_tests.testsuites.pred2_topology_sq import run_pred2
from m3_squared_tests.testsuites.pred3_symmetry_null import run_pred3
from m3_squared_tests.testsuites.pred4_threshold import run_pred4
from m3_squared_tests.testsuites.pred5_cross_domain_anchor import run_pred5

app = typer.Typer(add_completion=False)


def _write_validation(out_dir: Path, result) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(result, indent=2))


@app.command("validate-data")
def validate_data(staged: Path = typer.Option(..., help="Path to staged data directory")) -> None:
    data = load_staged(staged)
    result = validate_schema(data.measurements, data.configs)
    payload = {"ok": result.ok, "errors": result.errors}
    _write_validation(Path("outputs") / "validate", payload)
    if not result.ok:
        raise typer.Exit(code=1)


@app.command("pred1")
def pred1(
    staged: Path = typer.Option(..., help="Path to staged data directory"),
    out: Path = typer.Option(..., help="Output directory"),
) -> None:
    data = load_staged(staged)
    run_pred1(data, out)


@app.command("pred2")
def pred2(
    staged: Path = typer.Option(..., help="Path to staged data directory"),
    out: Path = typer.Option(..., help="Output directory"),
    topology_map: Optional[Path] = typer.Option(None, help="Optional topology map CSV"),
) -> None:
    data = load_staged(staged)
    run_pred2(data, out, topology_map=topology_map)


@app.command("pred3")
def pred3(
    staged: Path = typer.Option(..., help="Path to staged data directory"),
    out: Path = typer.Option(..., help="Output directory"),
) -> None:
    data = load_staged(staged)
    run_pred3(data, out)


@app.command("pred4")
def pred4(
    staged: Path = typer.Option(..., help="Path to staged data directory"),
    out: Path = typer.Option(..., help="Output directory"),
    proxy_map: Optional[Path] = typer.Option(None, help="Optional quality proxy map CSV"),
) -> None:
    data = load_staged(staged)
    run_pred4(data, out, proxy_map=proxy_map)


@app.command("pred5")
def pred5(
    staged: Path = typer.Option(..., help="Path to staged data directory"),
    out: Path = typer.Option(..., help="Output directory"),
) -> None:
    data = load_staged(staged)
    run_pred5(data, out)


@app.command("m3fit")
def m3fit(
    staged: Path = typer.Option(..., help="Path to staged data directory"),
    out: Path = typer.Option(..., help="Output directory"),
) -> None:
    data = load_staged(staged)
    run_m3_fit(data, out)


@app.command("all")
def run_all(
    staged: Path = typer.Option(..., help="Path to staged data directory"),
    out: Path = typer.Option(..., help="Output directory"),
    topology_map: Optional[Path] = typer.Option(None, help="Optional topology map CSV"),
    proxy_map: Optional[Path] = typer.Option(None, help="Optional quality proxy map CSV"),
) -> None:
    data = load_staged(staged)
    base = Path(out)
    run_pred1(data, base / "pred1")
    run_pred2(data, base / "pred2", topology_map=topology_map)
    run_pred3(data, base / "pred3")
    run_pred4(data, base / "pred4", proxy_map=proxy_map)
    run_pred5(data, base / "pred5")
    run_m3_fit(data, base / "m3fit")
