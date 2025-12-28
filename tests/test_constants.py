from m3_squared_tests import constants


def test_alpha2_ppm():
    ppm = constants.alpha2_ppm()
    assert 50 < ppm < 60


def test_alpha4():
    assert constants.alpha4() == constants.ALPHA2 * constants.ALPHA2
