import subprocess
from pathlib import Path


def test_cli_simulate_and_fit(tmp_path: Path):
    simulate_out = tmp_path / "synth"
    fit_out = tmp_path / "fit"
    subprocess.check_call(["python", "cli.py", "simulate", "--preset", "gravity_synth", "--out", str(simulate_out)])
    subprocess.check_call(
        [
            "python",
            "cli.py",
            "fit",
            "--data",
            str(simulate_out / "data.csv"),
            "--max-k",
            "3",
            "--out",
            str(fit_out),
        ]
    )
    assert (fit_out / "report.md").exists()
