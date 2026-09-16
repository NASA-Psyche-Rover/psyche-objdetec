"""
Depth map -> camera-frame point cloud, via pinhole back-projection.

This is deliberately backend-agnostic: it takes a depth map and a camera
intrinsics matrix K and does the geometry, nothing else. It does NOT know
or care whether `depth` came from MiDaS (relative, unitless) or a metric
stereo sensor -- that's exactly why `is_metric` is a caller-supplied flag
rather than something this function infers.

Do not feed live MiDaS output into this today: MiDaS produces per-frame,
unitless relative depth (see src/terrain_risk.py), not meters, so its
output does not satisfy this function's `is_metric=True` contract. This
module is validated against synthetic metric depth + synthetic K only
(see tests/test_depth_to_cloud.py) so the geometry is correct-by-test
ahead of the OAK-D Lite stereo path, which will supply real K and
`is_metric=True` once the camera is in hand.
"""

import numpy as np

from src.nav_types import PointCloud


def depth_to_cloud(depth, K, scale=1.0, is_metric=False):
    """
    Back-project a depth map into a camera-frame point cloud.

    Args:
        depth: (H, W) array. Depth values in whatever unit the source
            produces; multiplied by `scale` to get meters (Z = depth * scale).
        K: (3, 3) camera intrinsics matrix [[fx, 0, cx], [0, fy, cy], [0, 0, 1]].
        scale: scalar multiplier converting `depth` units to meters.
        is_metric: whether `depth` (after `scale`) is true metric depth
            (e.g. OAK-D Lite stereo) vs. relative/unitless (e.g. MiDaS).
            Carried through onto the returned PointCloud's `.is_metric`
            attribute so downstream consumers can tell the two apart.

    Returns:
        PointCloud: (N, 3) float32 array, meters, camera frame, N = H * W,
        row-major flattened (pixel (u, v) at index v * W + u). Also carries
        an `.is_metric` attribute set to the `is_metric` argument.

    Back-projection (pinhole model):
        Z = depth * scale
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
    """
    depth = np.asarray(depth, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    Z = depth * scale
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    points = np.stack([X, Y, Z], axis=-1).reshape(-1, 3).astype(np.float32)
    return PointCloud(points, is_metric=is_metric)
