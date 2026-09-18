from stats import mean_tokens_per_second

def test_normal():
    assert mean_tokens_per_second([10.0, 20.0, 30.0]) == 20.0

def test_empty_returns_zero():
    assert mean_tokens_per_second([]) == 0.0

def test_single():
    assert mean_tokens_per_second([7.5]) == 7.5
