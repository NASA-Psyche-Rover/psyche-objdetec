# Psyche rover: OAK-D-Lite + Livox Mid-360S fusion

Implements the pipeline discussed for the EPICS Psyche rover: project LiDAR
points onto the camera image, match them to YOLOv6 detections for
object-level distance, and separately build a full 3D perception pipeline
(ground removal -> clustering -> 3D boxes -> occupancy grid).

## Layout

```
fusion_lib/            # sensor-agnostic math -- no ROS, no hardware needed
  calibration.py        load/save K, dist, R, t
  projection.py          project LiDAR points onto the camera image (cv2.projectPoints)
  matching.py            object-level fusion: points-in-box -> median distance
  ground_removal.py      RANSAC plane fit, numpy-only
  clustering.py          DBSCAN over non-ground points
  bbox3d.py               axis-aligned + PCA-oriented 3D box fitting
  occupancy_grid.py      points -> 2D occupancy grid (ROS OccupancyGrid convention)

scripts/
  test_synthetic_fusion.py   run this FIRST, no hardware required
  get_oak_calibration.py     run on the Jetson to pull real camera intrinsics

ros2_node/
  fusion_node.py         template ROS2 node wiring real topics into fusion_lib

calibration/
  extrinsics_template.yaml   placeholder R/t -- replace after calibrating
```

## Step 1: validate the math (no hardware needed)

```
pip install -r requirements.txt
python scripts/test_synthetic_fusion.py
```

This builds a fake scene (a ground plane + a "rock" 2m ahead) and runs the
whole pipeline against it. Check that:
- the fused median distance lands close to the true simulated distance
- ground removal strips out most of the synthetic ground points
- clustering finds exactly one real cluster (the rock)

Already found one real issue this way: matching against raw points (before
ground removal) can pull in background points that happen to share a similar
pixel column, inflating the distance spread. `ros2_node/fusion_node.py`
already runs ground removal before matching for this reason -- worth keeping
that order if you extend the pipeline further.

## Step 2: get real calibration

**Camera intrinsics** (K, distortion) -- run on the Jetson with the
OAK-D-Lite plugged in:
```
pip install depthai
python scripts/get_oak_calibration.py
```

**LiDAR -> camera extrinsics (R, t)** -- this is the one that needs a real
calibration tool, done once the sensors are in their FINAL mounted position:
- Targetless (no checkerboard needed): `hku-mars/livox_camera_calib` on GitHub
- Board-based (official Livox method): `Livox-SDK/livox_camera_lidar_calibration`

Save the result into `calibration/extrinsics.yaml` following the same format
as `extrinsics_template.yaml`.

**Re-run this any time the mount is touched** -- reprinted bracket,
re-seated sensor, anything. Stale R/t fails silently: detections just drift
off without an error.

## Step 3: wire in real hardware

`ros2_node/fusion_node.py` is a template, not finished code -- it assumes:
- your DepthAI pipeline publishes YOLOv6 detections as a
  `vision_msgs/Detection2DArray` on some topic (update `DETECTIONS_TOPIC`
  to match whatever you actually publish)
- `livox_ros_driver2` is running and publishing `/livox/lidar`

Things that will need adjusting once real data is flowing:
- `GRID_SIZE` / `X_RANGE` / `Y_RANGE` in `fusion_node.py`, tuned to your
  actual test course dimensions
- `eps` in `cluster_obstacle_points` -- start at 0.2m, tune against your
  real obstacles' size (see Research Question #1's Test 3/4 obstacle course)
- `distance_threshold` in `remove_ground_plane` -- loosen for rougher
  mock-Psyche terrain, tighten for a flat lab floor

## A note on unmatched detections

`match_points_to_detections` silently drops a YOLOv6 box if zero LiDAR
points land inside it, rather than guessing a distance. Worth logging those
separately during testing -- a detection with no LiDAR support at all is
either a false positive, or a sign your extrinsics/timing sync is off.
