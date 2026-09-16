"""
Metric point cloud -> top-down TraversabilityGrid, via RANSAC ground-plane
fitting (numpy only -- no Open3D/PCL).

This is a new, standalone module: it does not import or modify main.py,
src/terrain_risk.py, or anything in the live pipeline. It operates purely
on src.depth_to_cloud.PointCloud instances and is only meaningful for
metric clouds (see `cloud_to_grid`'s is_metric check below) -- i.e. the
future OAK-D Lite stereo path, not the current MiDaS relative-depth path.

`TraversabilityGrid` (including its `stamp`, `labels`, and `grid_to_sensor`
fields) now lives in src/nav_types.py, the single source of truth for all
four pipeline contract types -- see docs/CONTRACTS.md, which documents all
of them as of this revision.

Frame note: this module has no Pose input (see docs/CONTRACTS.md `Pose`),
so it grids directly in the plane-projected frame of the input cloud
(basis vectors spanning the fitted ground plane), not a world frame.
`origin` is the (x, y) coordinate of cell [0, 0] in that plane-local
frame, and `grid_to_sensor` (see `cloud_to_grid`) is the SE3 mapping
grid-metric coordinates back into that same sensor frame. Transforming
into a world frame via a separate `Pose` (sensor pose in world) is a
caller concern, same as CONTRACTS.md leaves camera->world generally.

Classification concept (slope / roughness / drop-off) is borrowed from
src/terrain_risk.py's taxonomy for continuity, but none of its threshold
*values* are reused -- those are tuned against MiDaS's unitless disparity
output and don't apply to a metric point cloud. GROUND_TOL, OBSTACLE_HEIGHT,
and SLOPE_THRESHOLD_DEG below are new constants in real metric/angular units.
"""

import numpy as np

from src.nav_types import TraversabilityGrid, make_se3

# -- New metric-space constants (meters / degrees). Not copied from
# src/terrain_risk.py -- its slope/roughness/drop thresholds are tuned
# against MiDaS's unitless, per-frame-normalized disparity output and
# have no valid conversion into real units.
GROUND_TOL = 0.03          # meters: points within this of the fitted plane count as ground
OBSTACLE_HEIGHT = 0.10     # meters: points above the plane by more than this are a blocking obstacle
SLOPE_THRESHOLD_DEG = 25.0  # degrees: local rise/run beyond this makes a cell untraversable

# RANSAC plane-fit parameters (not part of the metric contract above, just
# fitting hyperparameters).
RANSAC_ITERATIONS = 300
RANSAC_DIST_THRESHOLD = 0.02  # meters, inlier distance to the candidate plane

# How many cells to search around an empty cell for nearby returns before
# calling it "never observed" (-1) rather than "occluded" (100). Sized to
# cover a small crater/ledge gap, not just single missing-pixel dropouts.
OCCLUSION_SEARCH_RADIUS_CELLS = 3


def _fit_ground_plane(points, n_iterations=RANSAC_ITERATIONS,
                       dist_threshold=RANSAC_DIST_THRESHOLD, rng=None):
    """RANSAC plane fit, numpy only. Returns (normal, centroid) of the
    largest inlier set, refined with an SVD least-squares fit over the
    inliers. `normal` is unit length but not yet oriented "up" -- see
    `_orient_up`."""
    rng = rng or np.random.default_rng()
    n = points.shape[0]
    if n < 3:
        raise ValueError("Need at least 3 points to fit a ground plane.")

    best_count = -1
    best_inliers = None

    for _ in range(n_iterations):
        idx = rng.choice(n, size=3, replace=False)
        p0, p1, p2 = points[idx]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm < 1e-8:
            continue  # degenerate (near-collinear) sample, skip
        normal = normal / norm
        dist = np.abs((points - p0) @ normal)
        inliers = dist < dist_threshold
        count = int(np.count_nonzero(inliers))
        if count > best_count:
            best_count = count
            best_inliers = inliers

    if best_inliers is None or best_count < 3:
        raise ValueError("RANSAC failed to fit a ground plane (degenerate or too few points).")

    inlier_pts = points[best_inliers]
    centroid = inlier_pts.mean(axis=0)
    _, _, vt = np.linalg.svd(inlier_pts - centroid)
    normal = vt[-1]
    normal = normal / np.linalg.norm(normal)
    return normal, centroid


def _orient_up(normal):
    """Flip `normal` if needed so it points "up": in this pipeline's
    camera-frame convention Y increases downward (see src/depth_to_cloud.py's
    back-projection: Y = (v - cy) * Z / fy grows with image row v, which
    grows downward), so "up" is the -Y direction."""
    if np.dot(normal, np.array([0.0, -1.0, 0.0])) < 0:
        normal = -normal
    return normal


def _plane_basis(normal):
    """Two unit vectors spanning the plane perpendicular to `normal`,
    used as the top-down grid's local (x, y) axes."""
    ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(normal, ref)) > 0.9:
        ref = np.array([0.0, 0.0, 1.0])
    basis1 = np.cross(normal, ref)
    basis1 = basis1 / np.linalg.norm(basis1)
    basis2 = np.cross(normal, basis1)
    basis2 = basis2 / np.linalg.norm(basis2)
    return basis1, basis2


def _project_to_plane(points, centroid, normal, basis1, basis2):
    """Per point: signed height above the plane (along `normal`) and the
    two in-plane (top-down) coordinates."""
    rel = points - centroid
    height = rel @ normal
    gx = rel @ basis1
    gy = rel @ basis2
    return height, gx, gy


def _dilate(mask, radius):
    """Manual boolean dilation (no scipy dependency) by `radius` cells,
    4/8-connected per step."""
    out = mask.copy()
    for _ in range(radius):
        padded = np.pad(out, 1, mode="constant", constant_values=False)
        neighbors = np.zeros_like(out)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                neighbors |= padded[1 + dr:1 + dr + out.shape[0], 1 + dc:1 + dc + out.shape[1]]
        out = out | neighbors
    return out


def cloud_to_grid(cloud, resolution=0.05, labels=None):
    """
    Project a metric point cloud into a top-down TraversabilityGrid.

    Args:
        cloud: src.depth_to_cloud.PointCloud (or any (N, 3) float array with
            an `is_metric` attribute) with `is_metric is True`. Raises
            ValueError otherwise -- a grid built from a non-metric (e.g.
            raw MiDaS) cloud has no consistent units and is geometrically
            meaningless.
        resolution: meters per cell.
        labels: optional (N,) array of per-point class ids, same length/
            order as `cloud`. If given, blocked cells caused by an actual
            obstacle (height > OBSTACLE_HEIGHT) get tagged with the
            majority class id of the points that triggered them, in
            `grid.labels`.

    Returns:
        TraversabilityGrid with `stamp` propagated from `cloud.stamp`
        (None if the cloud doesn't carry one), and `grid_to_sensor` set to
        the SE3 (sensor <- grid) transform built from the fitted plane's
        basis + this grid's origin (see src/nav_types.py `TraversabilityGrid`).
    """
    if not getattr(cloud, "is_metric", False):
        raise ValueError(
            "cloud_to_grid requires a metric point cloud (cloud.is_metric == True). "
            "A non-metric (e.g. raw MiDaS relative-depth) cloud has no consistent "
            "real-world units, so a grid built from it would be geometrically meaningless."
        )

    points = np.asarray(cloud, dtype=np.float64)
    stamp = getattr(cloud, "stamp", None)

    if points.shape[0] < 3:
        raise ValueError("Need at least 3 points to build a traversability grid.")

    normal, centroid = _fit_ground_plane(points)
    normal = _orient_up(normal)
    basis1, basis2 = _plane_basis(normal)
    height, gx, gy = _project_to_plane(points, centroid, normal, basis1, basis2)

    min_gx, max_gx = float(gx.min()), float(gx.max())
    min_gy, max_gy = float(gy.min()), float(gy.max())

    W = int(np.floor((max_gx - min_gx) / resolution)) + 1
    H = int(np.floor((max_gy - min_gy) / resolution)) + 1

    col = np.clip(((gx - min_gx) / resolution).astype(int), 0, W - 1)
    row = np.clip(((gy - min_gy) / resolution).astype(int), 0, H - 1)
    flat_idx = row * W + col

    counts = np.zeros(H * W, dtype=int)
    np.add.at(counts, flat_idx, 1)
    max_h = np.full(H * W, -np.inf)
    np.maximum.at(max_h, flat_idx, height)
    min_h = np.full(H * W, np.inf)
    np.minimum.at(min_h, flat_idx, height)

    counts = counts.reshape(H, W)
    max_h = max_h.reshape(H, W)
    min_h = min_h.reshape(H, W)
    observed = counts > 0

    # Slope check: local rise/run within a cell vs. SLOPE_THRESHOLD_DEG.
    rise_limit = resolution * np.tan(np.radians(SLOPE_THRESHOLD_DEG))
    height_range = np.where(observed, max_h - min_h, 0.0)
    slope_blocked = observed & (height_range > rise_limit)

    obstacle_blocked = observed & (max_h > OBSTACLE_HEIGHT)

    # Occlusion / negative-obstacle: empty cell, but near cells that *were*
    # observed -- i.e. inside the sensor's scanned footprint, just with no
    # return here (a crater, ledge, or something blocking the ground return).
    # Empty cells with no nearby observations at all are genuinely unscanned.
    footprint = _dilate(observed, OCCLUSION_SEARCH_RADIUS_CELLS)
    occluded = (~observed) & footprint

    data = np.full((H, W), -1, dtype=np.int8)
    data[observed] = 0
    data[obstacle_blocked] = 100
    data[slope_blocked] = 100
    data[occluded] = 100

    label_map = {}
    if labels is not None:
        labels_arr = np.asarray(labels)
        if labels_arr.shape[0] != points.shape[0]:
            raise ValueError(
                f"labels length ({labels_arr.shape[0]}) must match cloud length ({points.shape[0]})."
            )
        obstacle_pts = height > OBSTACLE_HEIGHT
        per_cell_labels = {}
        for r, c, lbl in zip(row[obstacle_pts], col[obstacle_pts], labels_arr[obstacle_pts]):
            key = (int(r), int(c))
            per_cell_labels.setdefault(key, []).append(lbl)
        for key, lbls in per_cell_labels.items():
            values, counts_ = np.unique(lbls, return_counts=True)
            label_map[key] = values[np.argmax(counts_)]

    # grid_to_sensor (SE3, sensor <- grid): rotation columns are the plane's
    # own basis (basis1 -> grid x, basis2 -> grid y, normal -> grid z), so a
    # grid-metric point (x, y, 0) rotates onto the fitted plane in the
    # sensor frame; translation places grid cell (0, 0) (i.e. (min_gx,
    # min_gy) in the centroid-relative plane frame) at its true sensor-frame
    # position. basis1 x basis2 == normal by construction (see _plane_basis),
    # so this is a proper right-handed rotation.
    rotation = np.stack([basis1, basis2, normal], axis=1)
    translation = centroid + min_gx * basis1 + min_gy * basis2
    grid_to_sensor = make_se3(rotation, translation)

    return TraversabilityGrid(
        data=data,
        resolution=resolution,
        origin=(min_gx, min_gy),
        stamp=stamp,
        labels=label_map,
        grid_to_sensor=grid_to_sensor,
    )
