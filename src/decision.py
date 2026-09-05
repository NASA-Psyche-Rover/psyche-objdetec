RISK_THRESHOLD = 1.0
PROXIMITY_THRESHOLD = 0.6
DENSITY_THRESHOLD = 0.35


def should_proceed(
    risk_score,
    object_proximity=0.0,
    cluster_density=0.0,
    risk_threshold=RISK_THRESHOLD,
    proximity_threshold=PROXIMITY_THRESHOLD,
    density_threshold=DENSITY_THRESHOLD,
):
    """
    Navigation gate combining terrain risk, nearest-object proximity, and
    obstacle cluster density.

    risk_score       : from TerrainAnalyzer.get_risk_assessment(); >= risk_threshold
                        means a terrain hazard (slope, roughness, or drop-off)
                        crossed its own calibrated threshold.
    object_proximity : from utils.estimate_object_proximity(); depth-sampled
                        closeness of the nearest detected object. >= proximity_threshold
                        means an object is actually close, as opposed to
                        cluster_density which only measures 2D frame coverage
                        regardless of distance (a nearby pebble and a distant
                        boulder can score the same there).
    cluster_density  : from utils.compute_cluster_density(); fraction of the frame
                        covered by detected-object bounding boxes.

    Returns "STOP" (terrain hazard or a close object), "CAUTION" (dense but
    distant obstacle cluster), or "PROCEED".
    """
    if risk_score >= risk_threshold:
        return "STOP"
    if object_proximity >= proximity_threshold:
        return "STOP"
    if cluster_density >= density_threshold:
        return "CAUTION"
    return "PROCEED"
