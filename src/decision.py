def should_proceed(terrain_risk, threshold=0.15):
    """
    Decide whether to proceed or stop based on terrain risk.

    terrain_risk is the slope value returned by TerrainAnalyzer.get_risk_assessment;
    higher means steeper / more dangerous terrain.
    """
    if terrain_risk > threshold:
        return "STOP"
    else:
        return "PROCEED"
