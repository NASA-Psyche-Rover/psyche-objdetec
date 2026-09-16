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
from src.nav_types import TraversabilityGrid, transform_point
from src.traversability_grid import (
    GROUND_TOL,
    OBSTACLE_HEIGHT,
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
