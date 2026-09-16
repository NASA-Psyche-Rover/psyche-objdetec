"""
Tests for src/global_map.py. Fully synthetic, no MiDaS, no live pipeline.

The end-to-end test deliberately runs the REAL src.depth_to_cloud.depth_to_cloud
and src.traversability_grid.cloud_to_grid (not a mocked/hand-built grid) so the
whole camera-pixel -> camera-frame -> plane-grid -> sensor-frame -> world-frame
chain is actually exercised, per the task's explicit instruction not to mock
the grid in that test.

Camera/world axis convention used by every pose in this file (documented here
once, reused everywhere): the pipeline's camera/sensor frame is X=right,
Y=down, Z=forward (see src/depth_to_cloud.py's back-projection). A flat ground
plane in that frame has *constant* Y (camera height above ground) and *varying*
X/Z -- so a Pose that's a pure identity rotation would map the plane's real
spatial variation (X, Z) onto world (x, y) but drop world z = sensor Z, i.e.
drop the one axis that actually varies most usefully among "forward distance",
which flattens the interesting geometry. Every pose here instead applies a
fixed camera->world axis remap (world_x = sensor_X, world_y = sensor_Z,
world_z = -sensor_Y) before any additional yaw, so top-down world (x, y)
actually reflects the ground plane's real horizontal spread.
"""

import json

import numpy as np
import pytest

from src.depth_to_cloud import depth_to_cloud
from src.global_map import GlobalMap
from src.nav_types import make_se3, transform_point
from src.traversability_grid import (
    _fit_ground_plane,
    _orient_up,
    _plane_basis,
    _project_to_plane,
    cloud_to_grid,
)

RESOLUTION = 0.05


def _K(fx=100.0, fy=100.0, cx=100.0, cy=100.0):
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


# Fixed camera-frame -> world-axis remap (see module docstring): world_x =
# sensor_X, world_y = sensor_Z, world_z = -sensor_Y.
_R_CONV = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, -1.0, 0.0],
])


def _yaw(theta):
    """Rotation about the camera's own vertical (Y) axis -- the natural
    "yaw" axis for this camera-frame convention."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c],
    ])


def _ground_depth(K, img_h, img_w, v_lo, v_hi, h_cam, block=None, hole=None):
    """
    Builds a depth image whose real-geometry back-projection (via the REAL
    depth_to_cloud) is an exact flat ground plane at camera-frame Y = h_cam
    (below the camera, since "up" = -Y), for image rows [v_lo, v_hi).
    Pixels outside that row range are NaN (out of frame / not part of this
    synthetic scene) and are filtered out by `_cloud_from_depth` below --
    same as a real depth sensor's invalid-pixel handling.

    block: optional {"v_lo","v_hi","u_lo","u_hi","height"} -- a raised
        block region, at camera-frame Y = h_cam - height (i.e. `height`
        meters above the ground plane).
    hole: optional {"v_lo","v_hi","u_lo","u_hi"} -- a missing-return
        (occlusion/crater) region, forced to NaN.

    The derivation: for a ground point at target height Y on a ray through
    pixel (u, v), the pinhole model Y = (v - cy) * Z / fy solves to
    Z = Y * fy / (v - cy); X follows from the standard X = (u - cx) * Z / fx.
    Solving for depth this way (rather than picking a depth directly)
    guarantees the back-projected point lands at the *exact* target Y for
    every pixel in the region, regardless of v.
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    vv, uu = np.meshgrid(np.arange(img_h), np.arange(img_w), indexing="ij")
    denom = (vv - cy).astype(np.float64)

    depth = np.full((img_h, img_w), np.nan, dtype=np.float64)
    row_mask = (vv >= v_lo) & (vv < v_hi)
    with np.errstate(divide="ignore", invalid="ignore"):
        z_ground = h_cam * fy / denom
    depth[row_mask] = z_ground[row_mask]

    if block is not None:
        y_block = h_cam - block["height"]
        with np.errstate(divide="ignore", invalid="ignore"):
            z_block = y_block * fy / denom
        bmask = (
            (vv >= block["v_lo"]) & (vv < block["v_hi"])
            & (uu >= block["u_lo"]) & (uu < block["u_hi"])
        )
        depth[bmask] = z_block[bmask]

    if hole is not None:
        hmask = (
            (vv >= hole["v_lo"]) & (vv < hole["v_hi"])
            & (uu >= hole["u_lo"]) & (uu < hole["u_hi"])
        )
        depth[hmask] = np.nan

    return depth.astype(np.float32)


def _cloud_from_depth(depth, K, stamp=None):
    """Real depth_to_cloud, then drop NaN (invalid-pixel) rows -- same
    filtering a real depth sensor's cloud producer would need."""
    cloud = depth_to_cloud(depth, K, scale=1.0, is_metric=True)
    valid = ~np.isnan(np.asarray(cloud)).any(axis=1)
    cloud = cloud[valid]  # boolean-index preserves the PointCloud subclass + attrs
    if stamp is not None:
        cloud.stamp = stamp
    return cloud


def _expected_world_cell(analytic_sensor_point, ground_only_points, grid, pose, gmap):
    """Independently re-derive which world cell a known camera-frame point
    should land in, using the real grid's own published origin/resolution/
    grid_to_sensor (outputs of the real cloud_to_grid call) plus the real
    pose -- without calling GlobalMap.integrate(). Mirrors
    tests/test_traversability_grid.py's _expected_cell pattern, extended
    through grid_to_sensor + pose."""
    normal, centroid = _fit_ground_plane(ground_only_points)
    normal = _orient_up(normal)
    basis1, basis2 = _plane_basis(normal)
    _, gx, gy = _project_to_plane(np.array([analytic_sensor_point]), centroid, normal, basis1, basis2)

    col = int(np.floor((gx[0] - grid.origin[0]) / grid.resolution))
    row = int(np.floor((gy[0] - grid.origin[1]) / grid.resolution))

    # Grid cells are flat (z=0 in-plane): the world position for this cell
    # is derived from its (col, row) index, not the analytic point's true
    # (off-plane) height -- exactly what GlobalMap.integrate does. NOTE:
    # grid_to_sensor's translation already bakes in grid.origin (see
    # cloud_to_grid), so the point fed in here must be the cell-local
    # (col/row * resolution) coordinate, NOT offset by origin again -- see
    # the same note in src/global_map.py's integrate().
    flat_x = col * grid.resolution
    flat_y = row * grid.resolution
    sensor_pt = transform_point(grid.grid_to_sensor, (flat_x, flat_y, 0.0))
    world_pt = transform_point(pose, sensor_pt)

    erow = int(np.floor((world_pt[1] - gmap.world_origin[1]) / gmap.resolution))
    ecol = int(np.floor((world_pt[0] - gmap.world_origin[0]) / gmap.resolution))
    return row, col, erow, ecol


def test_end_to_end_block_lands_at_correct_world_cell():
    K = _K()
    H_CAM = 1.0
    BLOCK_HEIGHT = 0.15
    block = {"v_lo": 160, "v_hi": 175, "u_lo": 90, "u_hi": 110, "height": BLOCK_HEIGHT}

    depth = _ground_depth(K, img_h=200, img_w=200, v_lo=140, v_hi=190, h_cam=H_CAM, block=block)
    cloud = _cloud_from_depth(depth, K, stamp=42.0)

    grid = cloud_to_grid(cloud, resolution=RESOLUTION)
    assert grid.grid_to_sensor is not None

    theta = np.deg2rad(30.0)
    R = _R_CONV @ _yaw(theta)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)  # sanity: still a rotation
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-6)
    t = np.array([5.0, 2.0, -3.0])
    pose = make_se3(R, t)

    gmap = GlobalMap(resolution=RESOLUTION, width_m=30.0, height_m=30.0, world_origin=(-15.0, -15.0))
    gmap.integrate(pose, grid)
    assert gmap.last_stamp == 42.0

    # Ground-only points (same formula as _ground_depth, no block) to
    # independently refit the same plane cloud_to_grid converged to.
    v_c, u_c = 167.5, 100.0  # geometric center of the block region
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    y_block = H_CAM - BLOCK_HEIGHT
    z_block = y_block * fy / (v_c - cy)
    x_block = (u_c - cx) * z_block / fx
    analytic_block_point = np.array([x_block, y_block, z_block])

    ground_only = _cloud_from_depth(
        _ground_depth(K, img_h=200, img_w=200, v_lo=140, v_hi=190, h_cam=H_CAM), K
    )

    row, col, erow, ecol = _expected_world_cell(analytic_block_point, np.asarray(ground_only), grid, pose, gmap)
    assert grid.data[row, col] == 100  # sanity: this is really the blocked cell

    occ = gmap.occupancy()
    assert 0 <= erow < occ.shape[0] and 0 <= ecol < occ.shape[1]
    assert occ[erow, ecol] == 100


def test_overlapping_grids_accumulate_without_smear():
    K = _K()
    depth = _ground_depth(K, img_h=200, img_w=200, v_lo=140, v_hi=190, h_cam=1.0)
    cloud = _cloud_from_depth(depth, K, stamp=1.0)
    grid = cloud_to_grid(cloud, resolution=RESOLUTION)

    pose_a = make_se3(_R_CONV, (0.0, 0.0, 0.0))
    pose_b = make_se3(_R_CONV, (0.3, 0.0, 0.0))  # a different pose: shifted 0.3 m sideways

    map_a = GlobalMap(resolution=RESOLUTION, width_m=20.0, height_m=20.0, world_origin=(-10.0, -10.0))
    map_a.integrate(pose_a, grid)

    map_b = GlobalMap(resolution=RESOLUTION, width_m=20.0, height_m=20.0, world_origin=(-10.0, -10.0))
    map_b.integrate(pose_b, grid)

    combined = GlobalMap(resolution=RESOLUTION, width_m=20.0, height_m=20.0, world_origin=(-10.0, -10.0))
    combined.integrate(pose_a, grid)
    combined.integrate(pose_b, grid)

    overlap = map_a.observed & map_b.observed
    only_a = map_a.observed & ~map_b.observed
    assert np.any(overlap), "test setup should produce a genuine overlap region"
    assert np.any(only_a), "test setup should also produce a non-overlapping region"

    # Overlap region: log-odds accumulate (sum of both single-frame
    # contributions), not overwritten by the second integrate() call.
    assert np.allclose(
        combined.log_odds[overlap], map_a.log_odds[overlap] + map_b.log_odds[overlap]
    )

    # Non-overlap region touched only by pose_a: unaffected by pose_b's
    # integration -- no smear into cells the second frame never actually saw.
    assert np.allclose(combined.log_odds[only_a], map_a.log_odds[only_a])


def test_dropoff_cell_logs_anomaly_at_correct_world_coordinate():
    K = _K()
    hole = {"v_lo": 165, "v_hi": 172, "u_lo": 95, "u_hi": 105}
    depth = _ground_depth(K, img_h=200, img_w=200, v_lo=140, v_hi=190, h_cam=1.0, hole=hole)
    cloud = _cloud_from_depth(depth, K, stamp=7.0)
    grid = cloud_to_grid(cloud, resolution=RESOLUTION)

    pose = make_se3(_R_CONV, (0.0, 0.0, 0.0))
    gmap = GlobalMap(resolution=RESOLUTION, width_m=20.0, height_m=20.0, world_origin=(-10.0, -10.0))
    gmap.integrate(pose, grid)

    dropoffs = [a for a in gmap.anomalies if a.type == "drop_off"]
    assert dropoffs, "expected at least one drop_off anomaly from the hole region"
    assert all(a.stamp == 7.0 for a in dropoffs)

    # The hole's center pixel, back-projected via the same ground formula
    # (no NaN there in a hole-free version), should be near a logged anomaly.
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    v_c, u_c = 168.5, 100.0
    z = 1.0 * fy / (v_c - cy)
    x = (u_c - cx) * z / fx
    sensor_pt = np.array([x, 1.0, z])
    world_pt = transform_point(pose, sensor_pt)

    dists = [np.hypot(a.world_xy[0] - world_pt[0], a.world_xy[1] - world_pt[1]) for a in dropoffs]
    assert min(dists) < 0.5  # within one hole-width of the hole's center


def test_save_load_round_trips_map_and_anomalies(tmp_path):
    K = _K()
    hole = {"v_lo": 165, "v_hi": 172, "u_lo": 95, "u_hi": 105}
    depth = _ground_depth(K, img_h=200, img_w=200, v_lo=140, v_hi=190, h_cam=1.0, hole=hole)
    cloud = _cloud_from_depth(depth, K, stamp=99.0)
    grid = cloud_to_grid(cloud, resolution=RESOLUTION)

    pose = make_se3(_R_CONV, (1.0, 0.5, 0.0))
    gmap = GlobalMap(resolution=RESOLUTION, width_m=20.0, height_m=20.0, world_origin=(-10.0, -10.0),
                      known_classes={0, 1, 2})
    gmap.integrate(pose, grid)
    assert len(gmap.anomalies) > 0

    path = tmp_path / "roundtrip_map"
    gmap.save(path)

    loaded = GlobalMap.load(path)

    assert loaded.resolution == gmap.resolution
    assert loaded.width_m == gmap.width_m
    assert loaded.height_m == gmap.height_m
    assert loaded.world_origin == gmap.world_origin
    assert loaded.known_classes == gmap.known_classes
    assert loaded.last_stamp == gmap.last_stamp

    assert np.array_equal(loaded.log_odds, gmap.log_odds)
    assert np.array_equal(loaded.observed, gmap.observed)
    assert np.array_equal(loaded.occupancy(), gmap.occupancy())

    assert len(loaded.anomalies) == len(gmap.anomalies)
    for a, b in zip(loaded.anomalies, gmap.anomalies):
        assert a.type == b.type
        assert a.stamp == b.stamp
        assert np.allclose(a.world_xy, b.world_xy)
        # JSON has no tuple type, so list vs. tuple differs after a round
        # trip even though the meta content is identical -- normalize.
        assert json.loads(json.dumps(a.meta)) == json.loads(json.dumps(b.meta))
