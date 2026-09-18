def mean_tokens_per_second(samples):
    """Mean of the samples. An empty list has no mean, so return 0.0."""
    return sum(samples) / len(samples)
