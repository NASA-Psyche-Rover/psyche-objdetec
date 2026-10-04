RISK_THRESHOLD = 1.0
PROXIMITY_THRESHOLD = 0.6

# Metric path (OAK-D Lite stereo depth). Distances are along the ground from
# the camera to the nearest hazard in the rover's driving corridor.
STOP_DISTANCE_M = 0.30
CAUTION_DISTANCE_M = 1.00

# Drops and slopes get more room than objects: the rover has to stop with
# all of itself still on solid ground.
TERRAIN_STOP_DISTANCE_M = 0.50
TERRAIN_CAUTION_DISTANCE_M = 1.50

TERRAIN_NAMES = {"drop": "DROP AHEAD", "slope": "STEEP SLOPE AHEAD", "unseen": "NO GROUND AHEAD"}

# Below this fraction of valid depth straight ahead, the camera has lost
# sight of whatever is in front of it (see DecisionLatch).
MIN_AHEAD_DEPTH_FRAC = 0.10


def should_proceed(
    risk_score,
    object_proximity=0.0,
    risk_threshold=RISK_THRESHOLD,
    proximity_threshold=PROXIMITY_THRESHOLD,
):
    """
    Navigation gate for the MiDaS (relative-depth) path: terrain risk plus
    nearest-object proximity.

    risk_score       : from TerrainAnalyzer.get_risk_assessment(); >= risk_threshold
                        means a terrain hazard (slope, roughness, or drop-off)
                        crossed its own calibrated threshold.
    object_proximity : from utils.estimate_object_proximity(); depth-sampled
                        closeness of the nearest detected object. >= proximity_threshold
                        means an object is actually close.

    Returns "STOP" (terrain hazard or a close object) or "PROCEED".

    There is deliberately no 2D "how much of the frame is covered by boxes"
    term: frame coverage can't tell a nearby pebble from a distant boulder,
    so it is not a distance and not a reason to stop or slow a rover.
    """
    if risk_score >= risk_threshold:
        return "STOP"
    if object_proximity >= proximity_threshold:
        return "STOP"
    return "PROCEED"


def decide_metric(
    nearest_object_m,
    object_label="object",
    terrain_kind=None,
    terrain_m=None,
):
    """
    Navigation gate for the metric (OAK-D Lite) path. Everything is a real
    distance along the ground, inside the rover's driving corridor.

    nearest_object_m    : nearest rock/obstacle, meters, or None.
    object_label        : what it is, for the reason text.
    terrain_kind        : from metric_terrain.analyze(): "drop" (table edge,
                          crater rim), "slope" (too steep to drive down),
                          "unseen" (the ground ahead returns no depth, so it
                          can't be vouched for), or None.
    terrain_m           : distance to where the ground stops.

    Returns (decision, reason). Objects use STOP_DISTANCE_M /
    CAUTION_DISTANCE_M; terrain uses the longer TERRAIN_* distances, since
    driving off an edge isn't a bump. The reason names the hazard
    ("DROP AHEAD 0.42 m") for the UI.
    """
    hazards = []   # (rank, distance, reason)
    if nearest_object_m is not None:
        rank = 2 if nearest_object_m < STOP_DISTANCE_M else 1 if nearest_object_m < CAUTION_DISTANCE_M else 0
        hazards.append((rank, nearest_object_m, f"{object_label} {nearest_object_m:.2f} m"))
    if terrain_kind is not None and terrain_m is not None:
        rank = 2 if terrain_m < TERRAIN_STOP_DISTANCE_M else 1 if terrain_m < TERRAIN_CAUTION_DISTANCE_M else 0
        name = TERRAIN_NAMES[terrain_kind]
        hazards.append((rank, terrain_m, f"{name} {terrain_m:.2f} m"))

    hazards = [hz for hz in hazards if hz[0] > 0]
    if not hazards:
        return "PROCEED", "path clear"
    rank, _, reason = min(hazards, key=lambda hz: (-hz[0], hz[1]))   # worst first, then nearest
    return ("STOP" if rank == 2 else "CAUTION"), reason


class DecisionLatch:
    """Keeps the decision from flapping. A change only takes effect once the
    new answer has held for consecutive perception steps: `escalate_frames`
    to get more cautious, `release_frames` to relax. Both are short -- a
    perception step is ~60-100 ms, so 2 and 3 steps respond in roughly
    0.15-0.3 s while still ignoring a single bad depth frame.

    It also covers the stereo blind zone. Under ~18 cm the OAK-D Lite returns
    no depth at all, so an obstacle that keeps approaching *disappears* from
    the distance signal exactly when it matters most. If the last confirmed
    hazard was inside `blind_hold_m` and the view straight ahead then loses
    its depth, STOP is held instead of released.
    """

    _RANK = {"PROCEED": 0, "CAUTION": 1, "STOP": 2}

    def __init__(self, escalate_frames=2, release_frames=3, blind_hold_m=0.45,
                 min_ahead_depth_frac=MIN_AHEAD_DEPTH_FRAC):
        self.escalate_frames = escalate_frames
        self.release_frames = release_frames
        self.blind_hold_m = blind_hold_m
        self.min_ahead_depth_frac = min_ahead_depth_frac
        self.decision = "PROCEED"
        self.reason = "path clear"
        self._pending = None
        self._pending_frames = 0
        self._last_hazard_m = None

    def update(self, decision, reason, nearest_hazard_m, ahead_valid_frac):
        blind = ahead_valid_frac < self.min_ahead_depth_frac
        if (nearest_hazard_m is None and blind and self.decision == "STOP"
                and self._last_hazard_m is not None and self._last_hazard_m < self.blind_hold_m):
            decision, reason = "STOP", f"too close to measure (last {self._last_hazard_m:.2f} m)"

        if decision == self.decision:
            self.reason = reason
            self._pending, self._pending_frames = None, 0
        else:
            if decision != self._pending:
                self._pending, self._pending_frames = decision, 0
            self._pending_frames += 1
            needed = (self.escalate_frames if self._RANK[decision] > self._RANK[self.decision]
                      else self.release_frames)
            if self._pending_frames >= needed:
                self.decision, self.reason = decision, reason
                self._pending, self._pending_frames = None, 0

        if self.decision == "STOP" and nearest_hazard_m is not None:
            self._last_hazard_m = nearest_hazard_m
        elif self.decision != "STOP":
            self._last_hazard_m = None
        return self.decision, self.reason
