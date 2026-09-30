"""
Tests for src/traversability_grid.py. Fully synthetic metric clouds only --
no MiDaS, no live pipeline. Uses the module's own plane-fit/projection
helpers to compute expected cell indices, so assertions check actual grid
semantics rather than hardcoded coordinates that would be fragile to the
(arbitrary but deterministic, for a flat noise-free plane) choice of
in-plane basis vectors.
"""

import numpy as np
import pytest

from src.depth_to_cloud import PointCloud
from src.nav_types import (
    REASON_NONE,
    REASON_NO_RETURN,
    REASON_OBSTACLE,
    REASON_ROUGHNESS,
    REASON_SLOPE,
    TraversabilityGrid,
    reason_names,
    transform_point,
)
from src.traversability_grid import (
    HEIGHT_NOISE_TOL,
    OBSTACLE_HEIGHT,
    ROUGHNESS_TOL,
    _fit_ground_plane,
    _orient_up,
    _plane_basis,
    _project_to_plane,
    cloud_to_grid,
)

RESOLUTION = 0.05


def _expected_cell(point, ground_pts, resolution=RESOLUTION):
    """Replicate cloud_to_grid's projection/binning for a single point, to
    find which (row, col) it's expected to land in -- using the ground-only
    points to compute the same centroid/basis the RANSAC fit converges to
    for a flat, noise-free synthetic ground plane."""
    normal, centroid = _fit_ground_plane(ground_pts)
    normal = _orient_up(normal)
    basis1, basis2 = _plane_basis(normal)
    _, gx_all, gy_all = _project_to_plane(ground_pts, centroid, normal, basis1, basis2)
    min_gx, min_gy = float(gx_all.min()), float(gy_all.min())

    _, gx, gy = _project_to_plane(np.array([point]), centroid, normal, basis1, basis2)
    col = int((gx[0] - min_gx) / resolution)
    row = int((gy[0] - min_gy) / resolution)
    return row, col


def _flat_ground(x_range, z_range, step=0.1, y=0.0):
    xs = np.arange(*x_range, step)
    zs = np.arange(*z_range, step)
    xx, zz = np.meshgrid(xs, zs)
    pts = np.stack([xx.ravel(), np.full(xx.size, y), zz.ravel()], axis=-1)
    return pts.astype(np.float64)


def test_raised_block_reads_blocked_and_surrounding_ground_reads_free():
    # Ground sampled denser than the grid resolution (0.02 < 0.05) so every
    # cell gets a real return -- matching a real depth sensor's per-pixel
    # density, unlike a sparse synthetic lattice that would leave natural
    # sampling gaps a naive reader might mistake for occlusion.
    ground = _flat_ground((-1.0, 1.0), (1.0, 3.0), step=0.02, y=0.0)

    # A block raised 0.15 m "up" -- since up = -Y (see traversability_grid's
    # _orient_up docstring), that's Y = -0.15 -- well above OBSTACLE_HEIGHT.
    block = _flat_ground((-0.15, 0.16), (1.85, 2.16), step=0.02, y=-0.15)

    all_pts = np.vstack([ground, block])
    cloud = PointCloud(all_pts, is_metric=True)

    grid = cloud_to_grid(cloud, resolution=RESOLUTION)

    assert grid.data.dtype == np.int8
    assert set(np.unique(grid.data)) <= {-1, 0, 100}

    block_center = np.array([0.0, -0.15, 2.0])
    row, col = _expected_cell(block_center, ground)
    assert grid.data[row, col] == 100

    # A ground point comfortably away from the block should read free, and
    # so should its immediate neighborhood.
    far_ground = np.array([0.8, 0.0, 1.2])
    frow, fcol = _expected_cell(far_ground, ground)
    assert grid.data[frow, fcol] == 0
    neighborhood = grid.data[max(frow - 1, 0):frow + 2, max(fcol - 1, 0):fcol + 2]
    assert np.all((neighborhood == 0) | (neighborhood == -1))
    assert np.any(neighborhood == 0)


def test_hole_in_ground_reads_blocked_not_free_or_unknown():
    ground = _flat_ground((-1.0, 1.0), (1.0, 3.0), step=0.02, y=0.0)

    # Carve a hole: drop any ground point falling inside a small X/Z window,
    # simulating missing returns (occlusion / crater / negative obstacle).
    # Half-width 0.1 m ~= 2 cells at resolution 0.05, well within
    # OCCLUSION_SEARCH_RADIUS_CELLS (3) of the hole's surrounding ground.
    hole_center = np.array([0.0, 0.0, 2.0])
    in_hole = (np.abs(ground[:, 0] - hole_center[0]) < 0.1) & (np.abs(ground[:, 2] - hole_center[2]) < 0.1)
    ground_with_hole = ground[~in_hole]

    cloud = PointCloud(ground_with_hole, is_metric=True)
    grid = cloud_to_grid(cloud, resolution=RESOLUTION)

    row, col = _expected_cell(hole_center, ground_with_hole)
    assert grid.data[row, col] == 100  # occluded/crater, not free (0) or unknown (-1)

    # Ground well away from the hole is still free.
    far_ground = np.array([-0.8, 0.0, 1.2])
    frow, fcol = _expected_cell(far_ground, ground_with_hole)
    assert grid.data[frow, fcol] == 0


def test_non_metric_cloud_raises_value_error():
    ground = _flat_ground((-1.0, 1.0), (1.0, 3.0), step=0.1, y=0.0)
    cloud = PointCloud(ground, is_metric=False)

    with pytest.raises(ValueError):
        cloud_to_grid(cloud)


def test_plain_array_without_is_metric_attr_raises_value_error():
    ground = _flat_ground((-1.0, 1.0), (1.0, 3.0), step=0.1, y=0.0).astype(np.float32)

    with pytest.raises(ValueError):
        cloud_to_grid(ground)


def test_stamp_propagates_from_cloud_to_grid():
    ground = _flat_ground((-1.0, 1.0), (1.0, 3.0), step=0.1, y=0.0)
    cloud = PointCloud(ground, is_metric=True)
    cloud.stamp = 12345.0

    grid = cloud_to_grid(cloud, resolution=RESOLUTION)
    assert grid.stamp == 12345.0


def test_stamp_defaults_to_none_when_absent():
    ground = _flat_ground((-1.0, 1.0), (1.0, 3.0), step=0.1, y=0.0)
    cloud = PointCloud(ground, is_metric=True)

    grid = cloud_to_grid(cloud, resolution=RESOLUTION)
    assert grid.stamp is None


def test_labels_tag_blocked_obstacle_cells():
    ground = _flat_ground((-1.0, 1.0), (1.0, 3.0), step=0.1, y=0.0)
    block = _flat_ground((-0.15, 0.16), (1.85, 2.16), step=0.05, y=-0.15)
    all_pts = np.vstack([ground, block])
    cloud = PointCloud(all_pts, is_metric=True)

    labels = np.concatenate([
        np.zeros(len(ground), dtype=int),   # class 0 = ground
        np.full(len(block), 7, dtype=int),  # class 7 = rock
    ])

    grid = cloud_to_grid(cloud, resolution=RESOLUTION, labels=labels)

    block_center = np.array([0.0, -0.15, 2.0])
    row, col = _expected_cell(block_center, ground)
    assert grid.data[row, col] == 100
    assert grid.labels.get((row, col)) == 7


def test_grid_to_sensor_places_known_cell_on_fitted_plane():
    ground = _flat_ground((-1.0, 1.0), (1.0, 3.0), step=0.02, y=0.0)
    cloud = PointCloud(ground, is_metric=True)
    grid = cloud_to_grid(cloud, resolution=RESOLUTION)

    assert grid.grid_to_sensor.shape == (4, 4)
    assert grid.grid_to_sensor.dtype == np.float32

    # A known cell (col=3, row=5) -> grid-metric (x, y, 0) -> apply
    # grid_to_sensor -> should land back on the fitted ground plane
    # (Y == 0, since the synthetic ground plane here *is* Y=0) at the
    # sensor-frame X/Z implied by the grid's own origin + basis.
    col, row = 3, 5
    grid_point = np.array([col * RESOLUTION, row * RESOLUTION, 0.0])
    sensor_point = transform_point(grid.grid_to_sensor, grid_point)

    # It must sit on the plane the module itself fit: recompute that plane
    # independently from the ground points and check the point's distance
    # to it is ~0.
    normal, centroid = _fit_ground_plane(ground)
    normal = _orient_up(normal)
    plane_dist = abs(np.dot(sensor_point - centroid, normal))
    assert plane_dist < 1e-4

    # And it should match the expected sensor-frame point derived directly
    # from origin + basis (independent of grid_to_sensor's own internals).
    basis1, basis2 = _plane_basis(normal)
    expected = centroid + (grid.origin[0] + col * RESOLUTION) * basis1 \
        + (grid.origin[1] + row * RESOLUTION) * basis2
    assert np.allclose(sensor_point, expected, atol=1e-5)


def test_returns_traversability_grid_dataclass():
    ground = _flat_ground((-1.0, 1.0), (1.0, 3.0), step=0.1, y=0.0)
    cloud = PointCloud(ground, is_metric=True)
    grid = cloud_to_grid(cloud, resolution=RESOLUTION)

    assert isinstance(grid, TraversabilityGrid)
    assert grid.resolution == RESOLUTION
    assert isinstance(grid.origin, tuple) and len(grid.origin) == 2


# -- Slope / roughness criteria -----------------------------------------
#
# The suite these join passed against an earlier revision whose "slope" check
# actually measured within-cell height spread and thresholded it at
# resolution * tan(SLOPE_THRESHOLD_DEG) -- 0.023 m at 5 cm cells, below the
# plane fit's own noise. Nothing here exercised slope at all, so the defect
# was invisible. These tests pin the criterion the module now implements:
# grade of the ground surface measured *between* cells over
# SLOPE_BASELINE_CELLS, with roughness as a separate within-cell criterion.


def _ramp(x_range, z_range, angle_deg, step=0.02, z_hinge=None):
    """Flat in X, tilted in Z by `angle_deg`, hinged at `z_hinge` so the ramp
    rises out of the y=0 plane rather than replacing it. Up is -Y (see
    _orient_up), so rising means Y going negative."""
    xs = np.arange(*x_range, step)
    zs = np.arange(*z_range, step)
    xx, zz = np.meshgrid(xs, zs)
    hinge = z_hinge if z_hinge is not None else z_range[0]
    rise = np.maximum(zz - hinge, 0.0) * np.tan(np.radians(angle_deg))
    pts = np.stack([xx.ravel(), -rise.ravel(), zz.ravel()], axis=-1)
    return pts.astype(np.float64)


def _cell_value_at(grid, point, ground_pts):
    row, col = _expected_cell(point, ground_pts)
    return grid.data[row, col]


def test_flat_ground_is_not_slope_blocked():
    ground = _flat_ground((-1.0, 1.0), (1.0, 3.0), step=0.02, y=0.0)
    grid = cloud_to_grid(PointCloud(ground, is_metric=True), resolution=RESOLUTION)

    # The interior of a perfectly flat plane must be entirely free -- no cell
    # blocked by slope or roughness. Trim a SLOPE_BASELINE_CELLS-wide margin,
    # where cells legitimately read -1/100 from the occlusion halo at the
    # data hull rather than from terrain.
    m = 3
    interior = grid.data[m:-m, m:-m]
    assert np.all(interior == 0), (
        f"{np.count_nonzero(interior != 0)} of {interior.size} flat interior cells "
        f"are not free"
    )


def test_gentle_grade_below_threshold_reads_free():
    # 10 deg, well under SLOPE_THRESHOLD_DEG (25). Rise over the 2-cell
    # baseline is 0.10 * tan(10 deg) = 0.018 m -- deliberately also under
    # HEIGHT_NOISE_TOL, so this pins both gates at once.
    ramp = _ramp((-1.0, 1.0), (1.0, 2.0), angle_deg=10.0)
    grid = cloud_to_grid(PointCloud(ramp, is_metric=True), resolution=RESOLUTION)

    m = 3
    interior = grid.data[m:-m, m:-m]
    assert np.all(interior == 0)


def test_steep_grade_above_threshold_reads_blocked():
    # A 35 deg ramp rising out of an otherwise flat scene. The flat part
    # dominates the cloud, so RANSAC fits y=0 and the ramp reads as relief
    # against it -- the regime this test is about (see the module docstring's
    # note on uniform-grade scenes, which this is deliberately not).
    flat = _flat_ground((-1.0, 1.0), (1.0, 2.5), step=0.02, y=0.0)
    ramp = _ramp((-1.0, 1.0), (2.5, 2.7), angle_deg=35.0, step=0.02, z_hinge=2.5)
    cloud = PointCloud(np.vstack([flat, ramp]), is_metric=True)
    grid = cloud_to_grid(cloud, resolution=RESOLUTION)

    # Sample partway up the ramp, where the rise off the fitted plane is
    # above HEIGHT_NOISE_TOL but still below OBSTACLE_HEIGHT -- so this is
    # the slope criterion firing, not the obstacle-height one.
    z = 2.60
    rise = (z - 2.5) * np.tan(np.radians(35.0))
    assert HEIGHT_NOISE_TOL < rise < OBSTACLE_HEIGHT, (
        f"test setup broken: rise {rise:.3f} m is not in the slope-only band"
    )
    assert _cell_value_at(grid, np.array([0.0, -rise, z]), flat) == 100


def test_slope_does_not_inflate_obstacles_into_surrounding_ground():
    # Regression: measuring slope on a height that includes obstacle returns
    # blocks a ring of clear ground around every obstacle, because the cell
    # under the obstacle reads high against its flat neighbors. Ground right
    # beside a block must stay free.
    ground = _flat_ground((-1.0, 1.0), (1.0, 3.0), step=0.02, y=0.0)
    block = _flat_ground((-0.10, 0.11), (1.95, 2.16), step=0.02, y=-0.20)
    cloud = PointCloud(np.vstack([ground, block]), is_metric=True)
    grid = cloud_to_grid(cloud, resolution=RESOLUTION)

    # 0.30 m clear of the block edge: about 6 cells, well beyond the 2-cell
    # slope baseline, and beyond the 3-cell occlusion halo too.
    beside = np.array([0.41, 0.0, 2.05])
    assert _cell_value_at(grid, beside, ground) == 0


def test_rough_cell_blocks_on_within_cell_spread_not_slope():
    # A patch whose returns vary within each cell by more than ROUGHNESS_TOL
    # but whose mean stays level, so there is no between-cell grade. Only the
    # roughness criterion can catch this.
    ground = _flat_ground((-1.0, 1.0), (1.0, 3.0), step=0.02, y=0.0)
    rng = np.random.default_rng(0)
    patch_mask = (np.abs(ground[:, 0]) < 0.15) & (np.abs(ground[:, 2] - 2.0) < 0.15)
    rough = ground.copy()
    spread = ROUGHNESS_TOL * 1.5
    rough[patch_mask, 1] = rng.choice([-spread / 2, spread / 2], size=patch_mask.sum())

    grid = cloud_to_grid(PointCloud(rough, is_metric=True), resolution=RESOLUTION)
    assert _cell_value_at(grid, np.array([0.0, 0.0, 2.0]), ground) == 100


def test_sensor_noise_on_flat_ground_does_not_block():
    # Plane-fit-scale jitter of +/- 0.015 m: within-cell spread runs to about
    # 0.03 m, over the old criterion's 0.023 m threshold, so this is the case
    # that used to block flat ground wholesale. It stays free now for two
    # reasons -- slope is computed from per-cell *means*, which average the
    # jitter out, and ROUGHNESS_TOL (0.06) sits above this spread rather than
    # inside it.
    #
    # Note this does not exercise HEIGHT_NOISE_TOL, which is inert at the
    # default resolution/baseline/angle by construction (see its comment in
    # the module); it guards a tuned-down configuration, not this one.
    rng = np.random.default_rng(1)
    ground = _flat_ground((-1.0, 1.0), (1.0, 3.0), step=0.02, y=0.0)
    ground[:, 1] += rng.uniform(-0.015, 0.015, size=len(ground))

    grid = cloud_to_grid(PointCloud(ground, is_metric=True), resolution=RESOLUTION)
    m = 3
    interior = grid.data[m:-m, m:-m]
    blocked = np.count_nonzero(interior == 100)
    assert blocked == 0, f"{blocked} of {interior.size} noisy-but-flat cells blocked"


# -- Reason codes -------------------------------------------------------


def test_reasons_is_parallel_to_data_and_zero_where_not_blocked():
    ground = _flat_ground((-1.0, 1.0), (1.0, 3.0), step=0.02, y=0.0)
    block = _flat_ground((-0.15, 0.16), (1.85, 2.16), step=0.02, y=-0.20)
    cloud = PointCloud(np.vstack([ground, block]), is_metric=True)
    grid = cloud_to_grid(cloud, resolution=RESOLUTION)

    assert grid.reasons is not None
    assert grid.reasons.shape == grid.data.shape
    assert grid.reasons.dtype == np.uint8

    # The two channels must agree exactly: a reason set iff the cell is
    # blocked. A cell blocked with no reason is unexplained; a cell with a
    # reason but not blocked is a stale flag (what a consumer editing `data`
    # without updating `reasons` leaves behind).
    assert np.all((grid.reasons != REASON_NONE) == (grid.data == 100))


def test_obstacle_and_no_return_get_distinct_reasons():
    ground = _flat_ground((-1.0, 1.0), (1.0, 3.0), step=0.02, y=0.0)
    block = _flat_ground((-0.15, 0.16), (1.85, 2.16), step=0.02, y=-0.20)

    # Carve a hole elsewhere, well away from the block.
    hole_center = np.array([-0.6, 0.0, 1.4])
    in_hole = (np.abs(ground[:, 0] - hole_center[0]) < 0.1) & (
        np.abs(ground[:, 2] - hole_center[2]) < 0.1
    )
    cloud = PointCloud(np.vstack([ground[~in_hole], block]), is_metric=True)
    grid = cloud_to_grid(cloud, resolution=RESOLUTION)

    brow, bcol = _expected_cell(np.array([0.0, -0.20, 2.0]), ground)
    hrow, hcol = _expected_cell(hole_center, ground)

    # Both read 100 in `data` -- the reason channel is the only thing that
    # tells a crater apart from a rock, which is the point of having it.
    assert grid.data[brow, bcol] == 100
    assert grid.data[hrow, hcol] == 100

    assert grid.reasons[brow, bcol] & REASON_OBSTACLE
    assert not grid.reasons[brow, bcol] & REASON_NO_RETURN

    assert grid.reasons[hrow, hcol] & REASON_NO_RETURN
    assert not grid.reasons[hrow, hcol] & REASON_OBSTACLE


def test_slope_and_roughness_reasons_are_reported_separately():
    # A steep ramp out of a flat scene: slope, not roughness.
    flat = _flat_ground((-1.0, 1.0), (1.0, 2.5), step=0.02, y=0.0)
    ramp = _ramp((-1.0, 1.0), (2.5, 2.7), angle_deg=35.0, step=0.02, z_hinge=2.5)
    grid = cloud_to_grid(PointCloud(np.vstack([flat, ramp]), is_metric=True),
                         resolution=RESOLUTION)
    rise = (2.60 - 2.5) * np.tan(np.radians(35.0))
    row, col = _expected_cell(np.array([0.0, -rise, 2.60]), flat)
    # Slope alone: a smooth ramp's within-cell spread is
    # resolution * tan(35 deg) = 0.035 m, under ROUGHNESS_TOL (0.06), so the
    # roughness flag must stay clear here.
    assert grid.reasons[row, col] & REASON_SLOPE
    assert not grid.reasons[row, col] & REASON_ROUGHNESS

    # A rough patch: roughness fires, and slope fires with it. That is not a
    # defect to assert against -- scattering returns to +/- spread/2 leaves
    # each cell's *mean* somewhere inside that band depending on which side
    # its few points fell, so adjacent means genuinely differ by enough to
    # read as a grade. Real rubble behaves the same way. The channel is a
    # bitmask precisely so this shows up as "slope and roughness" instead of
    # one criterion silently shadowing the other.
    ground = _flat_ground((-1.0, 1.0), (1.0, 3.0), step=0.02, y=0.0)
    rng = np.random.default_rng(0)
    patch = (np.abs(ground[:, 0]) < 0.15) & (np.abs(ground[:, 2] - 2.0) < 0.15)
    rough = ground.copy()
    spread = ROUGHNESS_TOL * 1.5
    rough[patch, 1] = rng.choice([-spread / 2, spread / 2], size=patch.sum())
    grid = cloud_to_grid(PointCloud(rough, is_metric=True), resolution=RESOLUTION)
    row, col = _expected_cell(np.array([0.0, 0.0, 2.0]), ground)
    assert grid.reasons[row, col] & REASON_ROUGHNESS
    assert reason_names(grid.reasons[row, col]) == ["slope", "roughness"]


def test_reason_names_decodes_a_combined_mask():
    assert reason_names(REASON_NONE) == []
    assert reason_names(REASON_SLOPE) == ["slope"]
    assert reason_names(REASON_SLOPE | REASON_ROUGHNESS) == ["slope", "roughness"]
    assert reason_names(REASON_OBSTACLE | REASON_NO_RETURN) == ["obstacle", "no_return"]
