import numpy as np

from mverse_ladder.physics.ladder import Ladder
from mverse_ladder.physics.observables.gravity_config import simulate_gravity


def test_determinism_gravity():
    ladder = Ladder()
    rng = np.random.default_rng(123)
    data1 = simulate_gravity(50, 2, ladder, 4, 1.0, 0.5, 0.3, rng)
    rng = np.random.default_rng(123)
    data2 = simulate_gravity(50, 2, ladder, 4, 1.0, 0.5, 0.3, rng)
    assert np.allclose(data1["y"], data2["y"])
