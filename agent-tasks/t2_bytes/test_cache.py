from cache import cache_bytes

def test_llama3_8b():
    # 32 layers, 8 kv heads, head dim 128, 16-bit, one token = 128 KB
    assert cache_bytes(32, 8, 128, 2, 1) == 131072

def test_scales_with_tokens():
    assert cache_bytes(32, 8, 128, 2, 10) == 1310720

def test_half_precision_halves_it():
    a = cache_bytes(32, 8, 128, 2, 4)
    b = cache_bytes(32, 8, 128, 1, 4)
    assert a == 2 * b
