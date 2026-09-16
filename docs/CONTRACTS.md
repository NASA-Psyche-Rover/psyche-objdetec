# Data Contracts

Shared types for the navigation/mapping pipeline. These are the boundaries
between modules (camera → perception → mapping → planning): every module that
produces or consumes point clouds, occupancy grids, poses, or detections
should import and use these types rather than inventing ad hoc tuples/dicts,
so pipeline stages stay swappable (e.g. MiDaS depth → OAK-D Lite stereo depth,
or a future planner) without renegotiating shapes/units at every boundary.

No behavior in the existing pipeline (`main.py`, `src/terrain_risk.py`,
`src/detect.py`, etc) changes with the introduction of these types — they're
the target types for the new nav-interfaces modules (`src/depth_to_cloud.py`,
`src/traversability_grid.py`) to adopt.

**`src/nav_types.py` is the single source of truth for all four types below.**
Import from there, not from whichever module happens to produce a value of
that type (e.g. `from src.nav_types import PointCloud`, not
`from src.depth_to_cloud import PointCloud`, even though `depth_to_cloud`
happens to construct `PointCloud` instances).

## `PointCloud`

```python
np.ndarray  # shape (N, 3), dtype float32, units: meters, frame: camera
```

Implemented as a thin `np.ndarray` subclass (not a dataclass — it needs to
behave as a real ndarray everywhere one is expected), carrying two extra
attributes:

- `is_metric: bool` — whether these are true metric coordinates (e.g. OAK-D
  Lite stereo) vs. relative/unitless (e.g. MiDaS disparity back-projected
  with a guessed scale). Consumers that need real units (e.g.
  `traversability_grid.cloud_to_grid`) must check this and reject
  `is_metric=False` clouds rather than silently treating them as metric.
- `stamp` — opaque timestamp/sequence marker from the producing frame, or
  `None` if the producer didn't set one. Propagated through derived
  products (e.g. `TraversabilityGrid.stamp`) so a downstream consumer can
  correlate a grid back to the frame it came from.

Other notes:
- Each row is `[x, y, z]` in the camera's own frame (not world frame) —
  see `Pose` for the transform into world coordinates.
- `N` may be 0 (empty cloud is valid and should be handled by consumers).
- No implicit ordering/structure (i.e. not assumed to be a structured/range
  image) unless a producer documents otherwise.

## `TraversabilityGrid`

```python
@dataclass
class TraversabilityGrid:
    data: np.ndarray       # shape (H, W), dtype int8
                            #   0   = free
                            #   100 = blocked
                            #   -1  = unknown
    resolution: float      # meters per cell
    origin: tuple[float, float]  # (x, y) coordinate of cell [0, 0]'s lower-left corner
    stamp: float = None     # propagated from the source PointCloud.stamp; None if unset
    labels: dict = field(default_factory=dict)
        # {(row, col): class_id} — optional per-cell semantic tag for blocked
        # cells, populated only when a producer is given per-point labels
        # (e.g. cloud_to_grid's `labels` argument). Empty dict when unused,
        # never None, so consumers can always do `grid.labels.get(...)`.
    grid_to_sensor: np.ndarray = None
        # (4, 4) float32 SE3 Pose (sensor <- grid): maps a point expressed in
        # grid-metric coordinates (x = col * resolution, y = row * resolution,
        # z = 0, already offset by `origin`) into the 3D sensor frame the
        # source cloud was in. None if the producer didn't compute one.
```

- Mirrors ROS's `nav_msgs/OccupancyGrid` cell semantics and value range
  intentionally, so this type can be swapped for a real `OccupancyGrid`
  message later with a thin adapter instead of a redesign.
- `data[row, col]` — row indexes along grid Y, col along grid X, consistent
  with `nav_msgs/OccupancyGrid`'s row-major layout.
- `origin` is **not necessarily world-frame** — it's in whatever frame the
  producer's points were in. `traversability_grid.cloud_to_grid` has no
  `Pose` input, so it grids directly in the sensor-relative, plane-projected
  frame of the input cloud; `origin` there is that plane-local frame's `(x,
  y)` coordinate of cell `[0, 0]`, and `grid_to_sensor` is how a consumer
  gets back to the 3D sensor frame from a grid cell. Transforming further
  into a world frame (via a separate `Pose`, the sensor's pose in world) is
  a caller concern.

## `Pose`

```python
np.ndarray  # shape (4, 4), dtype float32, SE(3), world <- sensor
```

- A homogeneous transform mapping points in the sensor's frame into the
  world frame: `p_world = Pose @ p_sensor_homogeneous`.
- Must be a valid SE(3) member (orthonormal 3x3 rotation block, no scale/
  shear) — producers are responsible for this invariant, consumers may
  assume it.
- `src/nav_types.py` implements `Pose` as a light newtype, not a class:
  plain `(4, 4)` float32 arrays, plus two helper functions —
  `make_se3(rotation, translation)` to build one from a 3x3 rotation +
  translation vector, and `transform_point(pose, point)` to apply one to a
  3D point. `TraversabilityGrid.grid_to_sensor` is a `Pose` in this sense,
  just sensor←grid instead of world←sensor.

## `Detection`

```python
@dataclass
class Detection:
    xyxy: tuple[int, int, int, int]  # (x1, y1, x2, y2) pixel coords, image frame
    class_id: int
    conf: float                      # detection confidence, [0, 1]
```

- `xyxy` is in the frame the detector actually ran on (i.e. consumers that
  need original-resolution coordinates are responsible for rescaling, the
  same way `main.py` currently rescales boxes from a downscaled detection
  pass back to `TARGET_WIDTH`/`TARGET_HEIGHT`).
