"""
No hardware required. Validates the whole pipeline against a fake scene
before you touch the real rover: a flat ground plane, a "rock" cluster
sitting 2m ahead and 20cm to the right, and a YOLOv6-style box roughly where
that rock should land in the image.

Run:
    python scripts/test_synthetic_fusion.py

What to check in the output:
  - the median fused distance should land close to 2.0m (the rock's true
    forward distance) -- if it's way off, check your sign conventions on R/t
  - ground removal should classify ~400 points as ground and leave ~40-60
    as obstacle points
  - clustering should find exactly 1 real cluster (the rock); the random
    noise points should mostly show up as unclustered (-1)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fusion_lib.projection import project_lidar_to_image
from fusion_lib.matching import Detection2D, match_points_to_detections
from fusion_lib.ground_removal import remove_ground_plane
from fusion_lib.clustering import cluster_obstacle_points
from fusion_lib.bbox3d import fit_axis_aligned_bbox
from fusion_lib.occupancy_grid import build_occupancy_grid

# --- Rough OAK-D-Lite-like intrinsics at a 640x480 working resolution ---
# (swap for the real values from scripts/get_oak_calibration.py once you have them)
IMG_W, IMG_H = 640, 480
FX = FY = 450.0
CX, CY = IMG_W / 2, IMG_H / 2
K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]])
DIST = np.zeros(5)

# --- Rough extrinsics: LiDAR mounted 5cm above and 3cm behind the camera lens ---
# (swap for the real R, t from your calibration tool once you have them)
R = np.eye(3)
T = np.array([0.0, 0.05, -0.03])

# LiDAR frame convention used here: x = right, y = down, z = forward
GROUND_HEIGHT = 0.15   # LiDAR mounted ~15cm above the ground plane
ROCK_CENTER = np.array([0.2, 0.0, 2.0])  # 20cm right, 2m ahead


def make_synthetic_scene(seed=0):
    rng = np.random.default_rng(seed)
    ground = np.column_stack([
        rng.uniform(-2, 2, 400),
        np.full(400, GROUND_HEIGHT),
        rng.uniform(0.3, 4.0, 400),
    ])
    rock = ROCK_CENTER + rng.normal(0, 0.05, size=(40, 3))
    noise = rng.uniform(-3, 3, size=(20, 3))
    return np.vstack([ground, rock, noise])


def main():
    points = make_synthetic_scene()

    # --- 1. project + object-level fusion ---
    pixels, valid, cam_pts = project_lidar_to_image(points, R, T, K, DIST, IMG_W, IMG_H)
    print(f"[projection] {valid.sum()} / {len(points)} points landed inside the image")

    # In practice this box comes straight from your YOLOv6 output for this frame.
    detections = [Detection2D(x1=280, y1=180, x2=420, y2=340, cls="rock", confidence=0.9)]
    fused = match_points_to_detections(points, pixels, valid, detections)
    print("\n[object-level fusion]")
    for f in fused:
        print(f"  [{f.detection.cls}] median range={f.distance_m:.2f}m "
              f"(std={f.distance_std:.2f}, n_points={f.num_points}) "
              f"-- true distance was {np.linalg.norm(ROCK_CENTER):.2f}m")
    if not fused:
        print("  no detections matched -- check bounding box coords / calibration")

    # --- 2. full 3D perception: ground removal -> clustering -> 3D boxes ---
    obstacle_points, ground_mask = remove_ground_plane(points, distance_threshold=0.05)
    print(f"\n[ground removal] {ground_mask.sum()} ground points removed, "
          f"{len(obstacle_points)} obstacle points remain")

    labels = cluster_obstacle_points(obstacle_points, eps=0.2, min_samples=5)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"\n[clustering] found {n_clusters} cluster(s)")
    for cluster_id in sorted(set(labels)):
        if cluster_id == -1:
            continue
        cluster_pts = obstacle_points[labels == cluster_id]
        center, size = fit_axis_aligned_bbox(cluster_pts)
        print(f"  cluster {cluster_id}: center={center.round(2)}, size={size.round(2)}, "
              f"n_points={len(cluster_pts)}")

    # --- 3. occupancy grid ---
    grid = build_occupancy_grid(obstacle_points[:, [0, 2]], grid_size=0.1,
                                 x_range=(-3, 3), y_range=(0, 5))
    print(f"\n[occupancy grid] shape={grid.shape}, occupied cells={(grid > 0).sum()}")

    # --- sanity plot: top-down view ---
    plt.figure(figsize=(6, 6))
    plt.scatter(points[ground_mask][:, 0], points[ground_mask][:, 2],
                s=4, label="ground (removed)")
    plt.scatter(obstacle_points[:, 0], obstacle_points[:, 2],
                s=10, label="obstacle points")
    plt.scatter([ROCK_CENTER[0]], [ROCK_CENTER[2]], marker="x", s=120,
                color="red", label="true rock center")
    plt.xlabel("x (m, right +)")
    plt.ylabel("z (m, forward)")
    plt.title("Synthetic fusion test -- top-down view")
    plt.legend()
    plt.gca().set_aspect("equal")
    out_path = Path(__file__).resolve().parent / "synthetic_test_output.png"
    plt.savefig(out_path, dpi=120)
    print(f"\nSaved top-down plot to {out_path}")


if __name__ == "__main__":
    main()
