from pathlib import Path

import pandas as pd

from m3_squared_tests.io.load_staged import StagedData
from m3_squared_tests.testsuites.pred3_symmetry_null import run_pred3


def test_pred3_suppression(tmp_path: Path):
    measurements = pd.DataFrame(
        {
            "dataset_id": ["d1", "d1"],
            "config_id": ["c1", "c2"],
            "G_value_1e11": [6.0, 6.0001],
        }
    )
    configs = pd.DataFrame(
        {
            "dataset_id": ["d1", "d1"],
            "config_id": ["c1", "c2"],
            "method_type": ["m", "m"],
            "geometry_mode": ["g", "g"],
        }
    )
    merged = measurements.merge(configs, on=["dataset_id", "config_id"], how="left")
    data = StagedData(measurements=measurements, configs=configs, merged=merged)

    result = run_pred3(data, tmp_path)
    assert result["status"] == "ok"
    assert result["median_suppression"] > 0
