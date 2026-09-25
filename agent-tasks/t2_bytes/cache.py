def cache_bytes(layers, kv_heads, head_dim, dtype_bytes, tokens):
    """Bytes held by a key-value cache.

    Two values are stored per token per head: a key and a value.
    """
    return layers * kv_heads * head_dim * dtype_bytes * tokens
