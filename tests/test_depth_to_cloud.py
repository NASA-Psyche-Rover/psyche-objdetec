"""
Geometry tests for src/depth_to_cloud.py.

Deliberately synthetic-only: a known K and a known synthetic metric depth
map (a fronto-parallel plane, plus one hand-picked pixel), so the expected
(X, Y, Z) can be computed by hand and asserted exactly. No MiDaS, no real
camera, no live pipeline involved -- this validates the back-projection
math is correct-by-test ahead of the OAK-D Lite stereo path plugging in
real K + is_metric=True.
"""

import numpy as np
import pytest

from src.depth_to_cloud import PointCloud, depth_to_cloud

# Synthetic intrinsics: fx = fy = 100, principal point at (50, 50).
FX = FY = 100.0
CX = CY = 50.0
K = np.array([
    [FX, 0.0, CX],
    [0.0, FY, CY],
    [0.0, 0.0, 1.0],
])

H, W = 100, 100
PLANE_Z = 2.0  # meters


def _index(u, v, w=W):
    """Pixel (u, v) -> flat row index, matching depth_to_cloud's row-major flatten."""
    return v * w + u


def test_fronto_parallel_plane_all_points_share_z():
    depth = np.full((H, W), PLANE_Z, dtype=np.float32)
    cloud = depth_to_cloud(depth, K, scale=1.0, is_metric=True)

    assert cloud.shape == (H * W, 3)
    assert cloud.dtype == np.float32
    assert np.allclose(cloud[:, 2], PLANE_Z)


def test_principal_point_backprojects_to_zero_xy():
    depth = np.full((H, W), PLANE_Z, dtype=np.float32)
    cloud = depth_to_cloud(depth, K, scale=1.0, is_metric=True)

    center = cloud[_index(int(CX), int(CY))]
    assert np.allclose(center, [0.0, 0.0, PLANE_Z])


def test_known_offset_pixel_exact_coords():
    # u=60, v=50 -> X = (60-50)*2.0/100 = 0.2, Y = (50-50)*2.0/100 = 0.0, Z = 2.0
    depth = np.full((H, W), PLANE_Z, dtype=np.float32)
    cloud = depth_to_cloud(depth, K, scale=1.0, is_metric=True)

    p = cloud[_index(60, 50)]
    assert np.allclose(p, [0.2, 0.0, 2.0], atol=1e-6)

    # u=50, v=70 -> X = 0.0, Y = (70-50)*2.0/100 = 0.4, Z = 2.0
    p2 = cloud[_index(50, 70)]
    assert np.allclose(p2, [0.0, 0.4, 2.0], atol=1e-6)


def test_scale_applies_before_backprojection():
    # Raw depth of 200 "units" with scale=0.01 -> Z = 2.0 m, matching the
    # exact-plane case above -- same expected X/Y/Z.
    depth = np.full((H, W), 200.0, dtype=np.float32)
    cloud = depth_to_cloud(depth, K, scale=0.01, is_metric=True)

    p = cloud[_index(60, 50)]
    assert np.allclose(p, [0.2, 0.0, 2.0], atol=1e-6)


def test_single_known_point_in_otherwise_zero_depth_map():
    depth = np.zeros((H, W), dtype=np.float32)
    depth[30, 40] = 5.0  # v=30, u=40
    cloud = depth_to_cloud(depth, K, scale=1.0, is_metric=True)

    expected_x = (40 - CX) * 5.0 / FX  # -0.5
    expected_y = (30 - CY) * 5.0 / FY  # -1.0
    p = cloud[_index(40, 30)]
    assert np.allclose(p, [expected_x, expected_y, 5.0], atol=1e-6)

    # Every other point still sits at Z=0 (depth=0), X=Y=0.
    other = cloud[_index(0, 0)]
    assert np.allclose(other, [0.0, 0.0, 0.0])


@pytest.mark.parametrize("is_metric", [True, False])
def test_is_metric_flag_carried_through_to_output(is_metric):
    depth = np.full((H, W), PLANE_Z, dtype=np.float32)
    cloud = depth_to_cloud(depth, K, scale=1.0, is_metric=is_metric)

    assert isinstance(cloud, PointCloud)
    assert cloud.is_metric is is_metric


def test_output_is_plain_ndarray_compatible():
    depth = np.full((H, W), PLANE_Z, dtype=np.float32)
    cloud = depth_to_cloud(depth, K, scale=1.0, is_metric=True)

    assert isinstance(cloud, np.ndarray)
    assert cloud.shape[1] == 3
