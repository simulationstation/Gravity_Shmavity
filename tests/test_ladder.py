import numpy as np

from mverse_ladder.physics.ladder import Ladder, DEFAULT_ALPHA


def test_ladder_eps():
    ladder = Ladder(alpha=DEFAULT_ALPHA)
    assert np.isclose(ladder.epsilon(3), DEFAULT_ALPHA ** 2)
    assert np.isclose(ladder.epsilon(10), DEFAULT_ALPHA ** 16)


def test_ladder_sequence_include_m1():
    ladder = Ladder(include_m1=True)
    seq = list(ladder.ladder_sequence(3))
    assert len(seq) == 3
