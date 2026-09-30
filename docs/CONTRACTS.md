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

## Frame Conventions

Every transform, axis convention, and composition rule the pipeline relies
on. The types below (`PointCloud`, `Pose`, `TraversabilityGrid`) each name a
frame; this section defines what those names mean.

### Naming

Write transforms as **`A_from_B`**: the transform that takes a point
expressed in frame `B` and produces it in frame `A`.

```python
p_world = world_from_camera @ p_camera_homogeneous
```

Read it as cancellation — `world_from_camera @ camera_from_grid` leaves
`world_from_grid` — which makes a mis-ordered composition visible at the call
site instead of at runtime. Existing fields use the equivalent arrow form in
their comments (`Pose` is `world <- sensor`, `grid_to_sensor` is
`sensor <- grid`); both notations mean the same thing, but prefer `A_from_B`
for new identifiers, since there the direction survives into the variable
name rather than living only in a comment.

### Axis conventions per frame

**Camera frame** — `+X` right, `+Y` **down**, `+Z` forward out of the lens.

This is not a free choice; it is fixed by the back-projection in
`src/depth_to_cloud.py`:

```python
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
```

`v` is the image row, which grows **downward**, so `Y` grows downward with
it. Every `PointCloud` produced by that function is in this frame.

**The consequence: "up" is `-Y`, not `+Z`.** This is load-bearing.
`traversability_grid._orient_up()` hardcodes it when it decides which way to
flip the fitted ground-plane normal. A cloud that arrives in any other
convention — notably a lidar cloud, which is conventionally `+Z` up — will
have its ground normal flipped, which silently inverts obstacle height
(`max_h > OBSTACLE_HEIGHT`) and therefore inverts free vs. blocked across the
whole grid. There is no runtime check for this. Either rotate such a cloud
into the camera convention before calling `cloud_to_grid`, or generalize
`_orient_up` to take the expected up-axis as an argument.

**Grid frame** — `x` along `basis1`, `y` along `basis2`, `z` along the fitted
plane normal. Right-handed by construction (`_plane_basis` builds
`basis2 = normal × basis1`, so `basis1 × basis2 == normal`), and `z` points
up out of the ground plane because `_orient_up` has already run. A grid cell
lies at `z = 0`.

**World frame** — *not* fixed by this repo. It is whatever the pose source
publishes. The pipeline requires only that it is consistent across calls to
`GlobalMap.integrate()`. In particular, do not assume world `+Z` is up, and
do not assume world shares the camera's `-Y`-up convention.

### Composing with the extrinsic

A pose source rarely reports the camera's pose directly. It reports the pose
of the body/base frame, or of whichever sensor it is tracking. Get to the
camera by composing with the fixed mounting extrinsic:

```
world_from_camera = world_from_body @ body_from_camera
```

`body_from_camera` is a calibration constant — it changes only when the
camera is physically remounted. It is **not currently stored anywhere in this
repo**; supplying it is the caller's responsibility, the same way supplying
`K` is.

### Which frame `integrate()` actually wants

`GlobalMap.integrate(pose, grid)` documents `pose` as `world <- sensor`.
"Sensor" there means **the frame the source point cloud was in** — because
`integrate` composes `pose` with `grid.grid_to_sensor`, and
`cloud_to_grid` built that transform from the cloud's own coordinates. On the
OAK-D stereo path, that frame is the **camera**. So `integrate` wants
`world_from_camera`.

This is the most likely place to lose a day. A LIO/SLAM stack typically
publishes `world_from_lidar` or `world_from_body`, not `world_from_camera`.
Passing one of those straight into `integrate()` is wrong unless that sensor
and the camera are physically coincident — which they are not. It will not
raise; it will just place every observation at an offset pose, and the
log-odds accumulation will smear obstacles across the map in a way that looks
like bad depth rather than a bad transform. Compose first:

```
world_from_camera = world_from_lidar @ lidar_from_camera
```

If map quality degrades as the rover moves but single frames look correct,
suspect this before suspecting the perception stack.

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
    reasons: np.ndarray = None
        # (H, W) uint8 bitmask of REASON_* flags — why each blocked cell is
        # blocked. REASON_NONE wherever `data` is not 100. None if the
        # producer didn't compute one.
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
### Blocking reasons

`data` answers *whether* a cell is drivable; `reasons` answers *why not*.

```python
REASON_NONE      = 0       # not blocked (free or unknown)
REASON_OBSTACLE  = 1 << 0  # positive obstacle above the ground surface
REASON_SLOPE     = 1 << 1  # ground grade too steep
REASON_ROUGHNESS = 1 << 2  # within-cell height spread too high
REASON_NO_RETURN = 1 << 3  # no return, inside the region the sensor did observe
```

- It is a **bitmask**, because a cell can be untraversable for several
  reasons at once (rubble on a grade trips slope and roughness both), and
  which combinations fire is what you want visible when tuning thresholds
  against real terrain. Test with `reasons & REASON_SLOPE`, never
  `reasons == REASON_SLOPE`.
- The flags live **beside `data`, not inside it**. `data` stays within
  `nav_msgs/OccupancyGrid`'s 0/100/-1 range, so the adapter to a real
  `OccupancyGrid` message stays thin and this channel rides alongside it
  rather than blocking the swap.
- Values are part of the contract. **Append new flags; never renumber
  existing ones** — a saved map or anomaly log written by an older revision
  would otherwise decode into the wrong reasons. uint8 leaves room for eight
  total.
- `reasons` may be `None` (a producer that doesn't compute one), so consumers
  must handle that. `GlobalMap.integrate` is the worked example: with the
  channel present a drop-off anomaly means `REASON_NO_RETURN` and nothing
  else, and without it the code falls back to an over-reporting label
  heuristic and says so.
- A consumer that **edits `data`** must keep `reasons` in step. Demoting a
  cell to `-1` while leaving `REASON_NO_RETURN` set on it leaves a cell that
  is not blocked but still reads as a crater to anything scanning the reason
  channel. `nav_types.reason_names(mask)` decodes a value into flag names for
  logs and anomaly metadata.

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
  world frame: `p_world = Pose @ p_sensor_homogeneous`. In the `A_from_B`
  naming of **Frame Conventions** above, this is `world_from_sensor` — and
  see that section for what "sensor" resolves to at each call site, and for
  how to compose one from a pose source that reports a different frame.
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
