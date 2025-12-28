from mverse_ladder.inference.model_selection import ModelResult, stop_rule


def test_stop_rule():
    results = [
        ModelResult(k=0, bic=100, loglik=-50),
        ModelResult(k=1, bic=80, loglik=-40),
        ModelResult(k=2, bic=78, loglik=-39),
    ]
    assert stop_rule(results, threshold=5) == 1
