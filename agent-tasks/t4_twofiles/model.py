from units import to_gb

def weight_gb(params, bytes_per_param):
    """Size of the weights in gigabytes."""
    return to_gb(params * bytes_per_param)
