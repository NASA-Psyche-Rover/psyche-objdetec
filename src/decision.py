def should_proceed(cluster_density, threshold=0.15):
    """
    Decide whether to proceed or stop based on cluster density.
    """
    if cluster_density > threshold:
        return "STOP"
    else:
        return "PROCEED"
