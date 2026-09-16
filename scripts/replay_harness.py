"""
Synthetic-metric replay harness: a scripted trajectory through a
ground-truth world scene, run through the REAL cloud_to_grid ->
GlobalMap.integrate -> planner.plan pipeline, with no MiDaS and no
hardware involved.

Why synthetic-metric, not MiDaS: MiDaS produces relative, unitless depth
(see src/terrain_risk.py) -- it can never satisfy cloud_to_grid's
is_metric=True requirement, so it cannot exercise this pipeline honestly.
Before the OAK-D Lite is in hand, the only honest end-to-end test is a
synthetic scene with KNOWN metric ground truth: known world obstacle
positions, a known scripted trajectory, and an assertion that what comes
out the other end (the accumulated occupancy grid) matches that ground
truth. On day 3, when OAK-D Lite metric depth is available, the only
change is *where the per-frame PointCloud comes from* (a real depth frame
back-projected via depth_to_cloud instead of a synthetic camera-frame
crop) -- this harness's loop (cloud -> cloud_to_grid -> integrate -> plan)
does not change at all.

This also means depth_to_cloud.py's pinhole back-projection isn't
exercised here on purpose: with the ground-truth scene already in 3D,
rendering it to a depth image and back-projecting it would just be a
lossy round trip through pixel space for no benefit. Skipping straight to
"known 3D points in the camera frame, cropped to a frustum" is the
lower-noise way to validate everything downstream of a cloud (cloud_to_grid,
GlobalMap, planner) without also re-testing depth_to_cloud's geometry,
which tests/test_depth_to_cloud.py already covers directly.

Frame convention (read this before touching any pose math here):
  - Camera/sensor frame (matches src/depth_to_cloud.py and
    src/traversability_grid.py throughout this codebase): X = right,
    Y = down, Z = forward (optical convention). "Up" is -Y.
  - World frame here: X, Y horizontal (ground plane), Z = up. This is
    the convention docs/CONTRACTS.md's `Pose` assumes generically and
    that GlobalMap.integrate expects a Pose to map into (it only ever
    reads world x, y -- see GlobalMap.integrate's "drop z").
  - Because those two conventions don't share an axis meaning 1:1 (camera
    "forward" is Z; world "forward-ish, when facing the +y direction" is
    Y), every pose in this harness composes a FIXED camera->world axis
    remap with the trajectory's yaw:
        world_x = sensor_X, world_y = sensor_Z, world_z = -sensor_Y
    (as the 3x3 matrix `_AXIS_REMAP` below), THEN rotates by the robot's
    heading (yaw about world Z) and translates by the robot's world (x,
    y) position. Getting this composition backwards (e.g. applying an
    identity-rotation pose) silently collapses the ground plane's real
    spatial variation onto a single world row, since sensor Y (which
    varies only with camera height, not position) would leak into world y
    instead of sensor Z (which is what actually varies across the ground
    plane) -- exactly the class of bug this harness's test is built to catch.
"""

import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # so `python scripts/replay_harness.py` finds src/

from src.global_map import GlobalMap
from src.nav_types import PointCloud, make_se3, transform_point
from src.planner import plan
from src.traversability_grid import (
    _fit_ground_plane,
    _orient_up,
    _plane_basis,
    _project_to_plane,
    cloud_to_grid,
)

ROOT = Path(__file__).resolve().parent.parent

# -- Fixed camera(sensor)-frame -> world-axis remap (see module docstring).
_AXIS_REMAP = np.array([
    [1.0, 0.0, 0.0],   # world_x =  sensor_X
    [0.0, 0.0, 1.0],   # world_y =  sensor_Z
    [0.0, -1.0, 0.0],  # world_z = -sensor_Y
])


def _yaw(theta):
    """Rotation about world Z (up)."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ])


def make_pose(x, y, theta, camera_height):
    """World <- sensor Pose for a robot at world (x, y), heading `theta`
    (radians, yaw about world Z), with the camera mounted at
    `camera_height` meters above the ground (world Z)."""
    rotation = _yaw(theta) @ _AXIS_REMAP
    translation = (x, y, camera_height)
    return make_se3(rotation, translation)


# ---------------------------------------------------------------------------
# Ground-truth scene
# ---------------------------------------------------------------------------

def make_ground_truth_scene(
    ground_bounds=(-2.5, 2.5, -0.5, 10.0),
    ground_spacing=0.025,
    obstacles=None,
):
    """
    Builds a flat world-Z=0 ground plane plus N obstacle columns, at KNOWN
    world (x, y, size, height).

    `ground_bounds` must comfortably exceed the farthest point any
    trajectory pose's frustum can reach (see make_trajectory + max_range) --
    running out of ground-truth scene *inside* a frame's simulated range
    reads exactly like a real depth sensor hitting a dropoff/edge-of-world,
    which would confound this harness's obstacle-position assertions with
    an artifact of the scene being too small rather than the trajectory's
    last pose (y=6, range up to 3m => must see out to y=9) actually being
    near real obstacles.

    Returns:
        dict with:
          "points": (M, 3) float64 world-frame points (ground + obstacles).
          "obstacles": the resolved obstacle list (defaults filled in).
          "ground_bounds": (xmin, xmax, ymin, ymax) as given.
    """
    if obstacles is None:
        obstacles = [
            {"x": 0.4, "y": 2.2, "size": 0.3, "height": 0.3},
            {"x": -0.5, "y": 4.3, "size": 0.3, "height": 0.3},
        ]

    xmin, xmax, ymin, ymax = ground_bounds
    xs = np.arange(xmin, xmax, ground_spacing)
    ys = np.arange(ymin, ymax, ground_spacing)
    gx, gy = np.meshgrid(xs, ys)
    ground_points = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], axis=-1)

    obstacle_point_list = []
    for obs in obstacles:
        half = obs["size"] / 2.0
        ox = np.arange(obs["x"] - half, obs["x"] + half, ground_spacing)
        oy = np.arange(obs["y"] - half, obs["y"] + half, ground_spacing)
        oz = np.arange(ground_spacing, obs["height"], ground_spacing)  # skip z=0 (that's the ground)
        if oz.size == 0:
            oz = np.array([obs["height"]])
        oxx, oyy, ozz = np.meshgrid(ox, oy, oz)
        obstacle_point_list.append(np.stack([oxx.ravel(), oyy.ravel(), ozz.ravel()], axis=-1))

    all_points = np.vstack([ground_points] + obstacle_point_list)
    return {"points": all_points, "obstacles": obstacles, "ground_bounds": ground_bounds}


# ---------------------------------------------------------------------------
# Scripted trajectory
# ---------------------------------------------------------------------------

def make_trajectory(camera_height=1.0):
    """List of (x, y, theta_deg) robot poses: forward translation along
    world +y with a few nonzero-yaw frames mixed in (not a pure straight
    line -- exercising rotation, not just translation, is the point)."""
    xy_theta_deg = [
        (0.0, 0.0, 0.0),
        (0.0, 1.0, 5.0),
        (0.1, 2.0, -5.0),
        (0.0, 3.0, 8.0),
        (0.0, 4.0, 0.0),
        (0.0, 5.0, -8.0),
        (0.0, 6.0, 0.0),
    ]
    poses = [make_pose(x, y, math.radians(theta_deg), camera_height) for x, y, theta_deg in xy_theta_deg]
    return xy_theta_deg, poses


# ---------------------------------------------------------------------------
# Per-frame sensing simulation
# ---------------------------------------------------------------------------

def world_to_camera(world_points, pose):
    """Inverse pose: world points -> camera/sensor frame. For an SE3
    Pose = [R | t], the inverse is [R.T | -R.T @ t]; for an (N, 3) array
    of world points this is (points - t) @ R (see module docstring for
    why (points - t) @ R equals R.T @ (p - t) per point)."""
    pose = np.asarray(pose, dtype=np.float64)
    R, t = pose[:3, :3], pose[:3, 3]
    return (world_points - t) @ R


def crop_to_frustum(camera_points, fov_deg=70.0, max_range=3.0, min_range=0.05):
    """Boolean mask: points in front of the camera (Z in [min_range,
    max_range]) and within the horizontal field of view (angle off the
    camera's forward/Z axis, in the X-Z plane)."""
    x, z = camera_points[:, 0], camera_points[:, 2]
    in_range = (z >= min_range) & (z <= max_range)
    angle = np.degrees(np.arctan2(np.abs(x), np.maximum(z, 1e-9)))
    in_fov = angle <= (fov_deg / 2.0)
    return in_range & in_fov


def _in_frustum(sensor_points, fov_deg, max_range, min_range):
    x, z = sensor_points[:, 0], sensor_points[:, 2]
    in_range = (z >= min_range) & (z <= max_range)
    angle = np.degrees(np.arctan2(np.abs(x), np.maximum(z, 1e-9)))
    return in_range & (angle <= fov_deg / 2.0)


def _drop_frustum_boundary_artifacts(grid, cloud, fov_deg, max_range, min_range):
    """
    cloud_to_grid grids over the RECTANGULAR bounding box of the (plane-
    projected) point cloud. A single camera frame's visible points aren't
    rectangular, though -- an angular FOV crop produces a wedge -- so the
    bounding box has corners/edges the wedge never actually reached. Cells
    in that gap have no real points, sit near cells that DO (the wedge's
    edge), and so cloud_to_grid's occlusion heuristic (a real point of
    src/traversability_grid.py's design: "empty cell near observed cells
    -> probably a crater/occlusion, not just unscanned") reads them as
    occluded/blocked. That's the correct call for a genuine gap INSIDE a
    sensor's coverage; it's a false positive for the wedge-vs-bounding-box
    gap, which is an artifact of this harness's crop shape, not the scene.

    This reclassifies exactly that gap back to unknown (-1): a "blocked,
    no real point observed there" cell only survives as blocked if its
    reconstructed sensor-frame position is genuinely inside this frame's
    simulated frustum (a true occlusion); one that falls outside it is
    reclassified as unknown. Cells with real supporting points (obstacle
    or slope detections) are untouched regardless.

    A related edge case gets the same treatment: cells right at the LAST
    ring of the true frustum boundary (max range or max FOV angle) have
    nothing beyond them to compare against, so "no return past here" is
    genuinely ambiguous between "occluded" and "simply the sensor's own
    limit" -- this shrinks the trusted-as-occlusion region by a couple of
    cells' worth of range/angle margin so that ring reads unknown too,
    rather than confidently blocked. A real depth sensor has this exact
    same ambiguity at its own max range, so unlike the wedge-vs-box
    artifact above, this margin IS something a real sensor integration
    would still want.
    """
    # Must read `cloud` itself (float32, per src/nav_types.py PointCloud),
    # not a separately-kept float64 copy of the same points -- cloud_to_grid
    # binned points at float32 precision, and re-deriving "observed" at
    # float64 precision can disagree right at a cell boundary (a point that
    # floors into cell N at float64 can floor into N-1 or N+1 once rounded
    # to float32), which silently breaks the "candidates = ~observed" test
    # below for exactly the single-cell edge cases this function exists to
    # catch.
    pts = np.asarray(cloud, dtype=np.float64)
    normal, centroid = _fit_ground_plane(pts)
    normal = _orient_up(normal)
    basis1, basis2 = _plane_basis(normal)
    _, gx, gy = _project_to_plane(pts, centroid, normal, basis1, basis2)

    min_gx, min_gy = grid.origin
    res = grid.resolution
    height, width = grid.data.shape
    col = np.clip(((gx - min_gx) / res).astype(int), 0, width - 1)
    row = np.clip(((gy - min_gy) / res).astype(int), 0, height - 1)
    observed = np.zeros((height, width), dtype=bool)
    observed[row, col] = True

    trusted_max_range = max_range - 3 * res
    trusted_fov_deg = fov_deg - 5.0

    data = grid.data.copy()
    candidates = np.argwhere((data == 100) & ~observed)
    for r, c in candidates:
        if (int(r), int(c)) in grid.labels:
            continue  # has real supporting points -- not a boundary artifact
        gxp = c * res  # cell-local coordinate -- see src/global_map.py's integrate() note
        gyp = r * res
        sensor_pt = transform_point(grid.grid_to_sensor, (gxp, gyp, 0.0))
        if not _in_frustum(sensor_pt[np.newaxis, :], trusted_fov_deg, trusted_max_range, min_range)[0]:
            data[r, c] = -1

    return replace(grid, data=data)


# ---------------------------------------------------------------------------
# Main replay loop
# ---------------------------------------------------------------------------

def run_simulation(
    resolution=0.05,
    world_origin=(-3.0, -1.0),
    width_m=6.0,
    height_m=8.0,
    fov_deg=70.0,
    max_range=3.0,
    plan_every=2,
    start_cell=None,
    goal_cell=None,
    camera_height=1.0,
):
    """
    Runs the full synthetic replay: for each scripted pose, crops the
    ground-truth scene to that pose's camera frame + frustum, builds a
    REAL PointCloud -> cloud_to_grid -> GlobalMap.integrate, and every
    `plan_every` frames calls the REAL planner.plan() on the
    accumulated occupancy grid so far.

    Returns a dict with everything a caller (a test, or __main__'s
    visualization) needs: the scene, trajectory, GlobalMap, and the
    planned paths seen along the way.
    """
    scene = make_ground_truth_scene()
    xy_theta_deg, poses = make_trajectory(camera_height=camera_height)

    global_map = GlobalMap(resolution=resolution, width_m=width_m, height_m=height_m, world_origin=world_origin)

    def _world_to_cell(wx, wy):
        col = int(np.floor((wx - global_map.world_origin[0]) / global_map.resolution))
        row = int(np.floor((wy - global_map.world_origin[1]) / global_map.resolution))
        return row, col

    if start_cell is None:
        start_cell = _world_to_cell(xy_theta_deg[0][0], xy_theta_deg[0][1])
    if goal_cell is None:
        goal_cell = _world_to_cell(xy_theta_deg[-1][0], xy_theta_deg[-1][1])

    planned_paths = []  # list of (frame_idx, path)

    for frame_idx, pose in enumerate(poses):
        cam_points = world_to_camera(scene["points"], pose)
        visible = crop_to_frustum(cam_points, fov_deg=fov_deg, max_range=max_range)
        visible_points = cam_points[visible]

        if visible_points.shape[0] < 3:
            continue  # nothing usable seen this frame -- skip, same as a real empty-cloud frame

        cloud = PointCloud(visible_points, is_metric=True, stamp=float(frame_idx))
        grid = cloud_to_grid(cloud, resolution=resolution)
        grid = _drop_frustum_boundary_artifacts(grid, cloud, fov_deg, max_range, min_range=0.05)
        global_map.integrate(pose, grid)

        if frame_idx % plan_every == 0:
            path = plan(global_map.occupancy(), start_cell, goal_cell)
            planned_paths.append((frame_idx, path))

    final_path = plan(global_map.occupancy(), start_cell, goal_cell)

    return {
        "scene": scene,
        "trajectory_xy_theta": xy_theta_deg,
        "poses": poses,
        "global_map": global_map,
        "start_cell": start_cell,
        "goal_cell": goal_cell,
        "planned_paths": planned_paths,
        "final_path": final_path,
    }


# ---------------------------------------------------------------------------
# Visualization (uses cv2 -- already a core dependency, see requirements.txt
# -- rather than adding matplotlib just for this harness)
# ---------------------------------------------------------------------------

def save_visualization(sim_result, out_path, scale=4):
    import cv2

    global_map = sim_result["global_map"]
    occ = global_map.occupancy()
    h, w = occ.shape

    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[occ == -1] = (60, 60, 60)      # unknown: dark gray
    img[occ == 0] = (225, 225, 225)    # free: light gray
    img[occ == 100] = (0, 0, 0)        # blocked: black

    img = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)

    def world_to_px(wx, wy):
        col = (wx - global_map.world_origin[0]) / global_map.resolution
        row = (wy - global_map.world_origin[1]) / global_map.resolution
        px = int(col * scale)
        py = int((h - 1 - row) * scale)  # flip so +y (forward) points up in the image
        return px, py

    # Trajectory (blue).
    traj_px = [world_to_px(x, y) for x, y, _ in sim_result["trajectory_xy_theta"]]
    for p0, p1 in zip(traj_px, traj_px[1:]):
        cv2.line(img, p0, p1, (255, 128, 0), 2)
    for p in traj_px:
        cv2.circle(img, p, 3, (255, 128, 0), -1)

    # Latest planned path (green).
    for row, col in sim_result["final_path"]:
        wx = global_map.world_origin[0] + (col + 0.5) * global_map.resolution
        wy = global_map.world_origin[1] + (row + 0.5) * global_map.resolution
        cv2.circle(img, world_to_px(wx, wy), 2, (0, 200, 0), -1)

    # Ground-truth obstacles (yellow outline) -- so the PNG itself shows
    # whether blocked cells actually line up with ground truth.
    for obs in sim_result["scene"]["obstacles"]:
        half = obs["size"] / 2.0
        corners = [world_to_px(obs["x"] - half, obs["y"] - half), world_to_px(obs["x"] + half, obs["y"] + half)]
        cv2.rectangle(img, corners[0], corners[1], (0, 255, 255), 1)

    # Logged anomalies (red dots).
    for a in global_map.anomalies:
        cv2.circle(img, world_to_px(*a.world_xy), 2, (0, 0, 255), -1)

    out_path = str(out_path)
    cv2.imwrite(out_path, img)
    return out_path


if __name__ == "__main__":
    result = run_simulation()
    out_file = ROOT / "replay_harness_output.png"
    saved = save_visualization(result, out_file)
    print(f"[replay_harness] frames: {len(result['poses'])}, "
          f"anomalies: {len(result['global_map'].anomalies)}, "
          f"final path length: {len(result['final_path'])}")
    print(f"[replay_harness] saved visualization -> {saved}")
