from mverse_ladder.inference.calibration import calibrate_null
from mverse_ladder.physics.ladder import Ladder


def test_calibration_outputs():
    ladder = Ladder()
    result = calibrate_null("gravity_synth", runs=5, max_k=3, ladder=ladder, seed=1)
    assert "delta_bics" in result
    assert result["delta_bics"].shape[0] == 5
