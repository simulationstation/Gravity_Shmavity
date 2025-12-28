from pathlib import Path

from typer.testing import CliRunner

from m3_squared_tests.cli import app


runner = CliRunner()


def test_cli_pred1_smoke(tmp_path: Path):
    result = runner.invoke(
        app,
        [
            "pred1",
            "--staged",
            "data/staged",
            "--out",
            str(tmp_path / "pred1"),
        ],
    )
    assert result.exit_code == 0
