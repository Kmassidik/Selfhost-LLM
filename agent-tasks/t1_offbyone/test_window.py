from window import last_n

def test_basic():
    assert last_n([1, 2, 3, 4, 5], 2) == [4, 5]

def test_all():
    assert last_n([1, 2, 3], 3) == [1, 2, 3]

def test_one():
    assert last_n(["a", "b", "c"], 1) == ["c"]
