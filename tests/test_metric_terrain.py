"""
Synthetic-scene tests for src/metric_terrain.py's drop / slope detection and
src/decision.py's decide_metric.

Each scene is ray-cast from a camera `CAM_H` above level ground, pitched
`PITCH_DEG` down -- the rover's mounting -- into a metric depth image, so the
expected answer (where the ground ends, how steeply) is known exactly.
"""

import numpy as np
import pytest

from src import metric_terrain as mt
from src.decision import decide_metric

W, H = 640, 480
K = np.array([[460.0, 0, 320.0], [0, 460.0, 240.0], [0, 0, 1.0]])
CAM_H = 0.15
PITCH_DEG = 20.0


def render(surface_height, noise_m=0.0, seed=0):
    """Depth image (meters, NaN = no return) of the terrain y = surface_height(forward),
    where forward is horizontal distance ahead of the camera and height is
    relative to the ground under the rover (0 = level, negative = below)."""
    pitch = np.radians(PITCH_DEG)
    up = np.array([0.0, -np.cos(pitch), -np.sin(pitch)])       # world up, in camera frame
    fwd = np.array([0.0, -np.sin(pitch), np.cos(pitch)])       # horizontal forward, in camera frame

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u, dtype=float)], axis=-1)
    r_up, r_fwd = rays @ up, rays @ fwd

    # March each ray until it goes below the terrain.
    depth = np.full((H, W), np.nan)
    for t in np.arange(0.1, 8.0, 0.005):
        above_ground = CAM_H + t * r_up                         # ray point's height over the level datum
        hit = np.isnan(depth) & (above_ground <= surface_height(t * r_fwd)) & (r_fwd > 0)
        depth[hit] = t
    depth[depth < mt.MIN_RANGE_M] = np.nan
    if noise_m:
        depth += np.random.default_rng(seed).normal(0, noise_m, depth.shape) * depth ** 2
    return depth.astype(np.float32)


def flat(f):
    return np.zeros_like(f)


def approach(surface, **kwargs):
    """Analyze `surface` the way the rover meets it: after driving up on level
    ground, so the tracker already knows where the ground is."""
    tracker = mt.GroundTracker()
    mt.analyze(render(flat), K, tracker)
    return mt.analyze(render(surface, **kwargs), K, tracker)


def table_edge(edge_m, drop_m=0.75):
    return lambda f: np.where(f < edge_m, 0.0, -drop_m)


def downslope(start_m, angle_deg):
    return lambda f: np.where(f < start_m, 0.0, -(f - start_m) * np.tan(np.radians(angle_deg)))


def test_flat_ground_has_no_hazard():
    result = mt.analyze(render(flat, noise_m=0.003), K)
    assert result.ground_found
    assert result.camera_height_m == pytest.approx(CAM_H, abs=0.02)
    assert result.hazard_kind is None
    assert result.ground_visible_frac > 0.9
    assert not [o for o in result.obstacles if o.in_corridor]


@pytest.mark.parametrize("edge_m", [0.45, 0.8, 1.2])
def test_table_edge_is_a_drop_at_the_edge(edge_m):
    result = mt.analyze(render(table_edge(edge_m), noise_m=0.003), K)
    assert result.ground_found
    assert result.camera_height_m == pytest.approx(CAM_H, abs=0.02)   # the table, not the floor below
    assert result.hazard_kind == "drop"
    assert result.hazard_m == pytest.approx(edge_m, abs=0.08)


def test_drop_still_found_when_tracker_holds_the_plane():
    # Right at the edge there is too little table left in view to fit; the
    # tracker's plane from the approach is what keeps the drop detectable.
    tracker = mt.GroundTracker()
    mt.analyze(render(table_edge(1.0)), K, tracker)
    result = mt.analyze(render(table_edge(0.30)), K, tracker)
    assert result.ground_found
    assert result.hazard_kind == "drop"
    assert result.hazard_m == pytest.approx(0.30, abs=0.08)


def test_steep_downslope_in_view_is_a_slope_hazard():
    # Close enough that the camera looks down onto the slope face itself.
    result = approach(downslope(0.3, 23.0), noise_m=0.002)
    assert result.hazard_kind == "slope"
    assert result.hazard_m == pytest.approx(0.3, abs=0.25)
    assert 20 <= result.hazard_angle_deg < 45


def test_slope_hidden_below_the_line_of_sight_reads_as_a_drop():
    # From 15 cm up, a 32 degree downslope starting 0.6 m out falls away
    # faster than the camera's line of sight: its face is hidden exactly
    # like a cliff's, so it must be treated as one.
    crater = lambda f: np.maximum(downslope(0.6, 32.0)(f), -0.4)
    result = approach(crater, noise_m=0.002)
    assert result.hazard_kind == "drop"
    assert result.hazard_m == pytest.approx(0.6, abs=0.1)


def test_ground_that_returns_no_depth_is_flagged_unseen():
    # A pit with no bottom in range: nothing comes back beyond the rim.
    result = approach(downslope(0.6, 80.0))
    assert result.hazard_kind == "unseen"
    assert result.hazard_m == pytest.approx(0.6, abs=0.1)


def test_gentle_downslope_is_drivable():
    result = mt.analyze(render(downslope(0.6, 8.0), noise_m=0.002), K)
    assert result.hazard_kind is None


def test_rock_on_flat_ground_is_an_obstacle_not_a_drop():
    rock = lambda f: np.where((f > 0.7) & (f < 0.9), 0.15, 0.0)
    result = mt.analyze(render(rock, noise_m=0.002), K)
    assert result.hazard_kind is None
    assert result.nearest_obstacle_m == pytest.approx(0.7, abs=0.1)


def test_decide_metric():
    assert decide_metric(None) == ("PROCEED", "path clear")
    assert decide_metric(5.0, "rock")[0] == "PROCEED"            # far objects never stop the rover
    assert decide_metric(0.6, "rock") == ("CAUTION", "rock 0.60 m")
    assert decide_metric(0.2, "rock") == ("STOP", "rock 0.20 m")
    assert decide_metric(None, terrain_kind="drop", terrain_m=1.2) == ("CAUTION", "DROP AHEAD 1.20 m")
    assert decide_metric(None, terrain_kind="drop", terrain_m=0.4) == ("STOP", "DROP AHEAD 0.40 m")
    assert decide_metric(None, terrain_kind="slope", terrain_m=0.4) == ("STOP", "STEEP SLOPE AHEAD 0.40 m")
    # a STOP outranks a nearer CAUTION
    assert decide_metric(0.6, "rock", terrain_kind="drop", terrain_m=0.45)[1] == "DROP AHEAD 0.45 m"
    assert decide_metric(None, terrain_kind="unseen", terrain_m=0.9) == ("CAUTION", "NO GROUND AHEAD 0.90 m")
