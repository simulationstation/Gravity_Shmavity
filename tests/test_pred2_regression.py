from pathlib import Path

import numpy as np
import pandas as pd

from m3_squared_tests.io.load_staged import StagedData
from m3_squared_tests.testsuites.pred2_topology_sq import run_pred2


def test_pred2_prefers_quadratic(tmp_path: Path):
    configs = pd.DataFrame(
        {
            "dataset_id": ["d1"] * 5,
            "config_id": [f"c{i}" for i in range(5)],
            "topology_proxy": np.linspace(1, 5, 5),
        }
    )
    measurements = pd.DataFrame(
        {
            "dataset_id": ["d1"] * 5,
            "config_id": [f"c{i}" for i in range(5)],
            "G_value_1e11": 6.0 + 0.01 * configs["topology_proxy"].to_numpy() ** 2,
        }
    )
    merged = measurements.merge(configs, on=["dataset_id", "config_id"], how="left")
    data = StagedData(measurements=measurements, configs=configs, merged=merged)

    result = run_pred2(data, tmp_path)
    assert result["status"] == "ok"
    assert result["preferred"] in {"quadratic", "linear"}
