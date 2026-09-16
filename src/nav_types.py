"""
Single source of truth for the navigation/mapping pipeline's shared data
contracts -- see docs/CONTRACTS.md for the narrative description of each
type and why it exists (the ROS `nav_msgs/OccupancyGrid` parallel for
`TraversabilityGrid`, the camera-frame-vs-world-frame distinction between
`PointCloud` and `Pose`, etc).

Previously `PointCloud` lived in src/depth_to_cloud.py and
`TraversabilityGrid` lived in src/traversability_grid.py -- each module
defined only the type it needed. This module consolidates all four so
future modules (planning, mapping accumulation, etc.) import one place
instead of reaching into whichever module happened to define a type first.
depth_to_cloud.py and traversability_grid.py now import from here; no
behavior changed in either.
"""

from dataclasses import dataclass, field

import numpy as np


class PointCloud(np.ndarray):
    """(N, 3) float32 ndarray, meters, camera frame.

    Plain ndarray subclass so it's usable anywhere an (N, 3) float32 array
    is expected, but carries `is_metric` and `stamp` through the pipeline:
      - is_metric: whether these are true metric coordinates (e.g. OAK-D
        Lite stereo) vs. relative/unitless (e.g. back-projected MiDaS
        disparity scaled by a guessed constant). Downstream geometry (e.g.
        traversability_grid.cloud_to_grid) requires is_metric is True.
      - stamp: opaque timestamp/sequence marker from the producing frame,
        propagated through derived products (e.g. TraversabilityGrid.stamp)
        so consumers can correlate a grid back to the cloud/frame it came
        from. None if the producer didn't set one.
    """

    def __new__(cls, points, is_metric=False, stamp=None):
        obj = np.asarray(points, dtype=np.float32).view(cls)
        obj.is_metric = is_metric
        obj.stamp = stamp
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.is_metric = getattr(obj, "is_metric", False)
        self.stamp = getattr(obj, "stamp", None)


@dataclass
class TraversabilityGrid:
    """(H, W) occupancy-style grid -- mirrors ROS's `nav_msgs/OccupancyGrid`
    cell semantics and value range intentionally, so this type can be
    swapped for a real `OccupancyGrid` message later with a thin adapter
    instead of a redesign.
    """

    data: np.ndarray       # (H, W) int8:  0 = free, 100 = blocked, -1 = unknown
    resolution: float      # meters per cell
    origin: tuple           # (x, y) coordinate of cell [0, 0]'s lower-left corner,
                             # in whatever frame the producer's points were in
                             # (e.g. the plane-projected sensor frame for
                             # traversability_grid.cloud_to_grid -- see grid_to_sensor)
    stamp: float = None     # propagated from the source PointCloud.stamp; None if unset
    labels: dict = field(default_factory=dict)
        # {(row, col): class_id} -- optional per-cell semantic tag for
        # blocked cells, populated only when a producer is given per-point
        # labels (e.g. cloud_to_grid's `labels` argument). Empty dict, not
        # None, when unused, so consumers can always do `grid.labels.get(...)`.
    grid_to_sensor: np.ndarray = None
        # (4, 4) float32 SE3 Pose (sensor <- grid): maps a point expressed
        # in grid-metric coordinates (x = col * resolution, y = row *
        # resolution, z = 0, i.e. already offset by `origin`) into the 3D
        # sensor frame the source cloud was in. None if the producer didn't
        # compute one (e.g. a grid not derived from a plane fit).


def make_se3(rotation, translation):
    """Compose a (3, 3) rotation matrix and (3,) translation vector into a
    (4, 4) float32 SE3 `Pose` (see docs/CONTRACTS.md `Pose`)."""
    rotation = np.asarray(rotation, dtype=np.float32)
    translation = np.asarray(translation, dtype=np.float32)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = rotation
    pose[:3, 3] = translation
    return pose


def transform_point(pose, point):
    """Apply a (4, 4) SE3 `Pose` to a (3,) point: returns the (3,)
    transformed point (homogeneous coordinate handled internally)."""
    point = np.asarray(point, dtype=np.float64)
    homogeneous = np.append(point, 1.0)
    result = np.asarray(pose, dtype=np.float64) @ homogeneous
    return result[:3]


@dataclass
class Detection:
    """A single object detection, image frame."""

    xyxy: tuple      # (x1, y1, x2, y2) pixel coords, in the frame the detector ran on
    class_id: int
    conf: float      # detection confidence, [0, 1]
