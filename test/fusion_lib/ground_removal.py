"""
Ground plane removal via RANSAC. Skip this step and your fusion pipeline will
happily report the floor as an obstacle cluster.

Implemented from scratch on numpy (no Open3D/PCL dependency) so it's easy to
read and tune the two knobs that actually matter for your terrain:
  distance_threshold -- how bumpy is "still the ground"? Loosen this for
                        rougher mock-Psyche terrain, tighten it for a flat lab floor.
  max_iterations      -- more iterations = more likely to find the true best
                        plane, at the cost of runtime. 200 is a reasonable start
                        on a Jetson.
"""
import numpy as np


def fit_plane_ransac(points, distance_threshold=0.05, max_iterations=200, seed=42):
    """Returns ((normal, d), inlier_mask) for the plane normal . p + d = 0
    with the most inliers, or (None, None) if no valid plane was found."""
    n = points.shape[0]
    if n < 3:
        return None, None

    rng = np.random.default_rng(seed)
    best_inliers = None
    best_plane = None

    for _ in range(max_iterations):
        idx = rng.choice(n, size=3, replace=False)
        p1, p2, p3 = points[idx]
        normal = np.cross(p2 - p1, p3 - p1)
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            continue  # degenerate sample (three near-collinear points)
        normal = normal / norm
        d = -normal.dot(p1)

        dist = np.abs(points @ normal + d)
        inliers = dist < distance_threshold

        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers = inliers
            best_plane = (normal, d)

    return best_plane, best_inliers


def remove_ground_plane(points, distance_threshold=0.05, max_iterations=200):
    """
    points: (N,3) LiDAR points

    Returns:
      obstacle_points: (M,3) points NOT on the fitted ground plane
      ground_mask:     (N,) bool, True where the point WAS classified as ground
    """
    plane, inliers = fit_plane_ransac(points, distance_threshold, max_iterations)
    if plane is None:
        return points, np.zeros(len(points), dtype=bool)
    return points[~inliers], inliers
