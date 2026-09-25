from model import weight_gb

def test_8b_at_16bit():
    assert abs(weight_gb(8_000_000_000, 2) - 16.0) < 1e-9

def test_360m_at_16bit():
    assert abs(weight_gb(361_821_120, 2) - 0.72364224) < 1e-9
