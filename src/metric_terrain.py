"""
Metric perception for the OAK-D Lite path: real stereo depth (meters) ->
distance to whatever is in the rover's way, plus the distance heatmap.

This is the metric counterpart to src/terrain_risk.py. That module works on
MiDaS's *relative*, per-frame-normalized depth, which has no units -- it can
say "this looks close" but never "that rock is 28 cm away". Everything here
runs on the OAK-D's stereo depth aligned to the RGB frame (see
src/camera_stream.py OakDLiteCamera), so every number is a real distance.

The camera sits low on the rover, pitched slightly down at the ground ahead.
analyze() fits that ground and reports what departs from it, in both
directions, all deliberately conservative about what counts as evidence --
stereo depth on textureless or repetitive surfaces produces scattered wrong
matches, and a rover that stops for those never moves:

  obstacles      connected blobs standing above the ground (rocks)
  terrain hazard where the ground ahead stops: a "drop" (table edge, crater
                 rim), a "slope" too steep to drive down, or "unseen" --
                 ground that should be in view but returns no depth, which
                 is what a drop looks like when its bottom is too far or too
                 dark to measure
  ahead_m        fallback only, when no ground can be found: nearest surface
                 around the optical axis

Frames: camera frame throughout (X right, Y down, Z forward -- see
docs/CONTRACTS.md "Frame Conventions"). The camera is assumed mounted
roughly upright (within MAX_GROUND_TILT_DEG).
"""

from dataclasses import dataclass, field

import cv2
import numpy as np

from src.depth_to_cloud import depth_to_cloud

# -- Sensor limits (OAK-D Lite, 640x400 mono pair, extended disparity) -------
MIN_RANGE_M = 0.18   # below this the stereo pair can't match -> no depth at all
MAX_RANGE_M = 10.0   # beyond this a stereo reading is noise; treated as no depth

# Depth noise grows with the square of range (~4 mm at 1 m, ~15 mm at 2 m,
# ~35 mm at 3 m for this baseline). Ground-relative geometry -- which compares
# heights of a few cm -- is only trusted inside this range.
GROUND_MAX_RANGE_M = 2.5

# -- Rover geometry -----------------------------------------------------------
CORRIDOR_HALF_WIDTH_M = 0.30  # half the swept width of the rover + margin
ROCK_MIN_HEIGHT_M = 0.06      # anything standing taller than this above the ground is an obstacle
OVERHEAD_CLEARANCE_M = 0.80   # ...unless it is above this (the rover drives under it)

# -- Ground fit ---------------------------------------------------------------
MAX_GROUND_TILT_DEG = 40.0    # ground normal must be within this of camera "up" (-Y)
MIN_CAMERA_HEIGHT_M = 0.03
MAX_CAMERA_HEIGHT_M = 2.0
GROUND_RANSAC_ITERATIONS = 150
GROUND_INLIER_DIST_M = 0.025
MIN_GROUND_INLIERS = 400      # subsampled points; and...
MIN_GROUND_INLIER_FRAC = 0.20 # ...this share of the near points, or it isn't "the ground"

# -- Evidence thresholds ------------------------------------------------------
AHEAD_MIN_VALID_FRAC = 0.30   # the straight-ahead window needs this much depth to be believed

# -- Drops and slopes ---------------------------------------------------------
# A point counts as "below the ground" once it is this far under the fitted
# plane. The allowance grows with distance because a small error in the
# plane's tilt is a large height error far away (1 degree = 7 cm at 4 m).
DROP_MIN_DEPTH_M = 0.08
DROP_GRADE_TOL = 0.06         # extra allowance, meters per meter of forward distance
MIN_DROP_POINTS = 25          # subsampled points needed below ground in the corridor
DROP_ANGLE_DEG = 45.0         # falls away steeper than this -> "drop"
SLOPE_MAX_DEG = 20.0          # ...steeper than this -> "slope"; gentler is drivable
DROP_VISIBLE_SPAN_M = 1.0     # surface seen within this of the edge tells us the steepness
GROUND_CHECK_NEAR_M = 0.25    # stretch of ground ahead whose visibility is monitored
GROUND_CHECK_FAR_M = 1.50
GROUND_BIN_M = 0.10           # ...in steps of this
GROUND_SEEN_MIN_FRAC = 0.20   # a step with less depth coverage than this is "empty"
GROUND_EMPTY_BINS = 3         # this many empty steps in a row = the ground ends there
MIN_BLOB_AREA_PX = 30         # obstacle blob size, in subsampled pixels (~480 full-res px)

STRIDE = 4                    # subsample depth by this before back-projecting


@dataclass
class Obstacle:
    """A blob of points standing above the ground -- found from geometry
    alone, so it exists whether or not YOLO recognises what it is."""
    xyxy: tuple          # box in full-resolution image pixels
    distance_m: float    # forward distance to its nearest face
    height_m: float      # how far it stands above the ground
    lateral_m: float     # + = right of heading
    in_corridor: bool


@dataclass
class TerrainResult:
    ground_found: bool
    ground_held: bool = False            # plane carried over from earlier frames, not fitted in this one
    camera_height_m: float = None
    hazard_kind: str = None              # "drop" | "slope" | "unseen" (ground gives no depth) | None
    hazard_m: float = None               # forward distance to where the ground stops
    hazard_angle_deg: float = None       # how steeply it falls away, if that could be measured
    ground_visible_frac: float = None    # of the ground just ahead; None if that ground isn't in view
    ahead_m: float = None                # no-ground fallback: nearest surface straight ahead of the lens
    ahead_valid_frac: float = 0.0        # how much of the straight-ahead window has depth at all
    obstacles: list = field(default_factory=list)   # blobs above the ground, nearest first
    # plane + axes, for projecting other things (e.g. YOLO boxes) into rover frame
    normal: np.ndarray = None
    centroid: np.ndarray = None
    forward: np.ndarray = None
    right: np.ndarray = None

    @property
    def nearest_obstacle_m(self):
        """Nearest above-ground blob inside the driving corridor, or None."""
        dists = [o.distance_m for o in self.obstacles if o.in_corridor]
        return min(dists) if dists else None


# ---------------------------------------------------------------------------
# Depth conditioning
# ---------------------------------------------------------------------------

def clean_depth(depth_mm, prev_m=None, alpha=0.5):
    """uint16 millimeters -> float32 meters, NaN where there is no valid
    depth. A 5x5 median removes stereo speckle (the on-device median filter
    is unavailable with extended disparity + subpixel), and a light temporal
    blend with the previous frame steadies the reading -- but only where the
    two frames already agree to within 10%, so a rock that actually moved
    closer is never smoothed back toward where it used to be."""
    d = cv2.medianBlur(depth_mm, 5).astype(np.float32) / 1000.0
    d[(d < MIN_RANGE_M) | (d > MAX_RANGE_M) | (depth_mm == 0)] = np.nan
    if prev_m is not None and prev_m.shape == d.shape:
        agree = np.abs(d - prev_m) < 0.10 * d   # False wherever either is NaN
        d = np.where(agree, alpha * d + (1 - alpha) * prev_m, d)
    return d


# ---------------------------------------------------------------------------
# Ground plane
# ---------------------------------------------------------------------------

def fit_ground_plane(points, rng=None):
    """RANSAC plane fit restricted to planes that could be the ground: roughly
    level relative to the camera and below it. Returns (normal, centroid) with
    `normal` unit length and pointing up, or None if no such plane has enough
    support (camera facing a wall, ground not in view, depth too sparse)."""
    rng = rng or np.random.default_rng()
    n = points.shape[0]
    if n < MIN_GROUND_INLIERS:
        return None
    up = np.array([0.0, -1.0, 0.0])
    cos_tilt = np.cos(np.radians(MAX_GROUND_TILT_DEG))

    idx = rng.integers(0, n, size=(GROUND_RANSAC_ITERATIONS, 3))
    p0, p1, p2 = points[idx[:, 0]], points[idx[:, 1]], points[idx[:, 2]]
    normals = np.cross(p1 - p0, p2 - p0)
    norms = np.linalg.norm(normals, axis=1)
    ok = norms > 1e-8
    normals[ok] /= norms[ok, None]
    normals[normals @ up < 0] *= -1
    cam_height = -np.einsum("ij,ij->i", p0, normals)   # camera origin above the plane
    ok &= (normals @ up > cos_tilt) & (cam_height > MIN_CAMERA_HEIGHT_M) & (cam_height < MAX_CAMERA_HEIGHT_M)
    if not ok.any():
        return None

    # Score every surviving candidate at once against a subsample of the
    # cloud (one matrix product instead of a Python loop over candidates --
    # this runs every frame), then count the winner's inliers on all points.
    normals, origins = normals[ok], p0[ok]
    sample = points[:: max(1, n // 1500)]
    offsets = np.einsum("ij,ij->i", origins, normals)
    counts = (np.abs(sample @ normals.T - offsets) < GROUND_INLIER_DIST_M).sum(axis=0)
    # Of the well-supported candidates, take the one closest beneath the
    # camera. At a table edge both the table and the floor below it are
    # level planes under the camera; the rover is standing on the nearer one.
    strong = counts >= 0.5 * counts.max()
    best = int(np.flatnonzero(strong)[np.argmin(cam_height[ok][strong])])
    best_inliers = np.abs((points - origins[best]) @ normals[best]) < GROUND_INLIER_DIST_M
    best_count = int(best_inliers.sum())
    if best_count < max(MIN_GROUND_INLIERS, MIN_GROUND_INLIER_FRAC * n):
        return None

    inlier_pts = points[best_inliers]
    centroid = inlier_pts.mean(axis=0)
    _, _, vt = np.linalg.svd(inlier_pts - centroid, full_matrices=False)
    normal = vt[-1] / np.linalg.norm(vt[-1])
    if normal @ up < 0:
        normal = -normal
    return normal, centroid


class GroundTracker:
    """Remembers the ground plane between frames. The camera is bolted to the
    rover, so its height above the ground and its tilt change slowly; a
    per-frame fit that suddenly disagrees (the floor below a table edge
    instead of the table, a wall, nothing at all) is ignored and the last
    good plane is used instead. That matters most exactly when a drop is
    close: there is little real ground left in view to fit, and the plane
    from a moment ago is what says the ground *should* still be there."""

    def __init__(self, hold_frames=45, blend=0.3):
        self.hold_frames, self.blend = hold_frames, blend
        self.normal = None
        self.height = None
        self.held = False
        self._missed = 0

    def update(self, fit):
        """fit: (normal, centroid) from fit_ground_plane(), or None."""
        self.held = True
        if fit is not None:
            normal, centroid = fit
            height = float(-centroid @ normal)
            consistent = (self.normal is not None
                          and abs(height - self.height) < 0.3 * self.height
                          and normal @ self.normal > np.cos(np.radians(12)))
            if self.normal is None or self._missed > self.hold_frames:
                self.normal, self.height, self._missed, self.held = normal, height, 0, False
                return
            if consistent:
                n = (1 - self.blend) * self.normal + self.blend * normal
                self.normal = n / np.linalg.norm(n)
                self.height = (1 - self.blend) * self.height + self.blend * height
                self._missed, self.held = 0, False
                return
        self._missed += 1
        if self._missed > self.hold_frames and fit is None:
            self.normal = self.height = None

    @property
    def plane(self):
        """(normal, point on plane directly beneath the camera), or None."""
        if self.normal is None:
            return None
        return self.normal, -self.height * self.normal


# ---------------------------------------------------------------------------
# Terrain + obstacle analysis
# ---------------------------------------------------------------------------

def analyze(depth_m, K, tracker=None):
    """Geometric pass on one cleaned depth frame (meters, NaN = invalid,
    aligned to the RGB image K belongs to). Pass the same GroundTracker every
    frame to keep the ground plane stable. Returns a TerrainResult."""
    h, w = depth_m.shape
    result = TerrainResult(ground_found=False)

    sub = depth_m[::STRIDE, ::STRIDE]
    sh, sw = sub.shape
    K_s = K.copy()
    K_s[:2, :] /= STRIDE
    # Back-project unit depth once to get each pixel's ray; points = ray * depth.
    rays = np.asarray(depth_to_cloud(np.ones((sh, sw)), K_s, is_metric=True), dtype=np.float64)
    valid = np.isfinite(sub).reshape(-1)
    pts = rays * np.nan_to_num(sub).reshape(-1, 1)
    near = valid & (pts[:, 2] < GROUND_MAX_RANGE_M)

    fit = fit_ground_plane(pts[near])
    if tracker is not None:
        tracker.update(fit)
        plane = tracker.plane
        result.ground_held = plane is not None and tracker.held
    else:
        plane = fit

    if plane is None:
        # No ground reference at all: fall back to whatever is nearest around
        # the optical axis. The 10th percentile needs >= 3% of the window to
        # agree (10% of >= 30% valid), which scattered wrong matches don't reach.
        window = depth_m[h // 3: 2 * h // 3, w // 3: 2 * w // 3]
        vals = window[np.isfinite(window)]
        result.ahead_valid_frac = vals.size / window.size
        if result.ahead_valid_frac >= AHEAD_MIN_VALID_FRAC:
            result.ahead_m = float(np.percentile(vals, 10))
        return result

    normal, centroid = plane
    fwd = np.array([0.0, 0.0, 1.0])
    fwd = fwd - (fwd @ normal) * normal
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, normal)
    cam_h = float(-centroid @ normal)

    height = (pts - centroid) @ normal
    forward = pts @ fwd
    lateral = pts @ right
    in_corridor = np.abs(lateral) < CORRIDOR_HALF_WIDTH_M

    result.ground_found = True
    result.normal, result.centroid, result.forward, result.right = normal, centroid, fwd, right
    result.camera_height_m = cam_h

    # -- Obstacles: blobs standing above the ground
    standing = near & (height > ROCK_MIN_HEIGHT_M) & (height < OVERHEAD_CLEARANCE_M)
    mask = standing.reshape(sh, sw).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n_blobs, blob_ids, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    flat_ids = blob_ids.reshape(-1)
    for b in range(1, n_blobs):
        x, y, bw, bh, area = stats[b]
        if area < MIN_BLOB_AREA_PX:
            continue
        sel = flat_ids == b
        lat = lateral[sel]
        result.obstacles.append(Obstacle(
            xyxy=(int(x * STRIDE), int(y * STRIDE), int((x + bw) * STRIDE), int((y + bh) * STRIDE)),
            # median: the blob's own distance, not its nearest stray point
            distance_m=float(np.median(forward[sel])),
            height_m=float(np.percentile(height[sel], 95)),
            lateral_m=float(np.median(lat)),
            in_corridor=bool((np.abs(lat) < CORRIDOR_HALF_WIDTH_M).mean() >= 0.25),
        ))
    result.obstacles.sort(key=lambda o: o.distance_m)

    # -- Where does the ground stop? Two kinds of evidence, both along the
    # rays that *should* land on the ground in the corridor ahead.
    tol = DROP_MIN_DEPTH_M + DROP_GRADE_TOL * forward

    # (1) Measured: points in the path that lie below the plane. Each was seen
    # along a ray that passed through where the ground should have been;
    # where that ray crosses the plane is at or beyond the edge, so the
    # nearest crossings mark the edge itself.
    # No range cap: the floor below a table edge is first visible many meters out.
    # "In the path" is judged where the ray crosses the ground plane, not
    # where the point is: the floor seen past a table edge is meters away
    # and spreads far wider than the corridor it was seen through.
    shrink = cam_h / np.maximum(cam_h - height, 1e-6)      # crossing = point * shrink, along the ray
    below = valid & (forward > 0) & (height < -tol) & (np.abs(lateral * shrink) < CORRIDOR_HALF_WIDTH_M)
    if below.sum() >= MIN_DROP_POINTS:
        f, hgt = forward[below], height[below]
        crossing = f * shrink[below]
        edge = float(np.percentile(crossing, 10))
        # How steeply does it fall away? Only answerable from surface seen
        # close behind the edge. From a low camera a real drop hides its own
        # face: the first thing visible is the bottom, far beyond the edge.
        close = f < edge + DROP_VISIBLE_SPAN_M
        if close.sum() >= MIN_DROP_POINTS // 2:
            angle = float(np.median(np.degrees(np.arctan2(-hgt[close], np.maximum(f[close] - edge, 1e-3)))))
            kind = "drop" if angle >= DROP_ANGLE_DEG else "slope" if angle >= SLOPE_MAX_DEG else None
        else:
            angle, kind = None, "drop"
        if kind is not None:
            result.hazard_kind, result.hazard_m, result.hazard_angle_deg = kind, edge, angle

    # (2) Unmeasured: a stretch of ground that should be in view but returns
    # no depth at all, right across the corridor -- a drop whose bottom is too
    # far, too dark, or out of the stereo pair's reach. Walk outward in
    # GROUND_BIN_M steps; the ground "ends" at the first step where almost no
    # ray came back, provided the steps after it are just as empty.
    down = rays @ normal                       # < 0 for rays heading toward the ground
    t = -cam_h / np.minimum(down, -1e-6)
    hit_f, hit_l = (rays @ fwd) * t, (rays @ right) * t
    expected = ((down < -1e-6) & (hit_f > GROUND_CHECK_NEAR_M) & (hit_f < GROUND_CHECK_FAR_M)
                & (np.abs(hit_l) < CORRIDOR_HALF_WIDTH_M))
    if expected.sum() >= 50:
        seen = valid & (height > -tol)          # ground, or something standing on it
        result.ground_visible_frac = float(seen[expected].mean())
        n_bins = int(round((GROUND_CHECK_FAR_M - GROUND_CHECK_NEAR_M) / GROUND_BIN_M))
        bins = ((hit_f[expected] - GROUND_CHECK_NEAR_M) / GROUND_BIN_M).astype(int).clip(0, n_bins - 1)
        total = np.bincount(bins, minlength=n_bins)
        got = np.bincount(bins, weights=seen[expected], minlength=n_bins)
        in_view = total >= 15
        empty = in_view & (got < GROUND_SEEN_MIN_FRAC * total)
        for b in range(n_bins - GROUND_EMPTY_BINS + 1):
            run = slice(b, b + GROUND_EMPTY_BINS)
            if empty[run].all():
                end_m = GROUND_CHECK_NEAR_M + b * GROUND_BIN_M
                if result.hazard_kind is None or end_m < result.hazard_m - 2 * GROUND_BIN_M:
                    result.hazard_kind, result.hazard_m, result.hazard_angle_deg = "unseen", end_m, None
                break
    return result


def edge_line_px(terrain, K, distance_m):
    """Image endpoints of the line across the driving corridor at
    `distance_m` ahead on the ground plane -- for drawing where a drop or
    slope begins. Returns ((x1, y1), (x2, y2)) or None if it is behind the
    camera."""
    foot = terrain.centroid
    ends = []
    for side in (-CORRIDOR_HALF_WIDTH_M, CORRIDOR_HALF_WIDTH_M):
        p = foot + distance_m * terrain.forward + side * terrain.right
        if p[2] <= 0.05:
            return None
        ends.append((int(K[0, 0] * p[0] / p[2] + K[0, 2]), int(K[1, 1] * p[1] / p[2] + K[1, 2])))
    return tuple(ends)


def box_distance(depth_m, K, xyxy, terrain=None):
    """Distance to a detected object (YOLO box, full-res pixels) from the
    aligned metric depth. Returns (distance_m, in_corridor), or (None, False)
    if the box has no usable depth -- outside the stereo overlap, closer than
    MIN_RANGE_M, or a textureless surface.

    Samples the central 60% of the box (the edges are mostly background) and
    takes the median of the valid depths there, and only if at least a fifth
    of that region has depth. A low percentile would be "the near face", but
    it also latches onto anything in *front* of a distant object that happens
    to fall inside its box -- reporting a person across the room as 25 cm
    away because a table edge crosses their box."""
    x1, y1, x2, y2 = xyxy
    h, w = depth_m.shape
    mx, my = (x2 - x1) * 0.2, (y2 - y1) * 0.2
    xa, xb = int(max(x1 + mx, 0)), int(min(x2 - mx, w))
    ya, yb = int(max(y1 + my, 0)), int(min(y2 - my, h))
    if xb <= xa or yb <= ya:
        return None, False
    region = depth_m[ya:yb, xa:xb]
    vals = region[np.isfinite(region)]
    if vals.size < max(10, 0.2 * region.size):
        return None, False
    z = float(np.median(vals))

    # Lateral extent of the box at that depth, to decide if it is in the path.
    fx, cx, fy, cy = K[0, 0], K[0, 2], K[1, 1], K[1, 2]
    left_m, right_m = (x1 - cx) * z / fx, (x2 - cx) * z / fx
    in_corridor = left_m < CORRIDOR_HALF_WIDTH_M and right_m > -CORRIDOR_HALF_WIDTH_M

    if terrain is not None and terrain.ground_found:
        # Report along-the-ground distance, consistent with the geometric obstacles.
        p = np.array([((x1 + x2) / 2 - cx) * z / fx, ((y1 + y2) / 2 - cy) * z / fy, z])
        z = float(p @ terrain.forward)
    return z, in_corridor


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def fill_for_display(depth_m, size=(320, 240), levels=6):
    """Display-only: a dense, smoothly blended version of the stereo depth.
    Stereo leaves holes wherever it can't match (blank walls, glare); this
    fills them by push-pull -- average the valid pixels down an image
    pyramid, then come back up, keeping real measurements wherever they
    exist and taking the coarser level's estimate only where they don't.
    A few milliseconds, versus ~0.5 s for a MiDaS pass on this hardware.

    Filled regions are interpolated from the depth around them, not
    measured, which is why no distance calculation uses this."""
    valid = np.isfinite(depth_m).astype(np.float32)
    total = np.nan_to_num(depth_m).astype(np.float32)
    weight = cv2.resize(valid, size, interpolation=cv2.INTER_AREA)
    total = cv2.resize(total, size, interpolation=cv2.INTER_AREA)

    pyramid = [(total, weight)]
    for _ in range(levels):
        total, weight = cv2.pyrDown(total), cv2.pyrDown(weight)
        pyramid.append((total, weight))

    total, weight = pyramid[-1]
    filled = total / np.maximum(weight, 1e-4)
    if weight.max() < 1e-3:
        return np.full((size[1], size[0]), np.nan, np.float32)   # no depth anywhere
    for total, weight in reversed(pyramid[:-1]):
        up = cv2.pyrUp(filled, dstsize=(total.shape[1], total.shape[0]))
        trust = np.clip(weight * 2.0, 0, 1)   # mostly-valid neighbourhoods keep their own value
        filled = trust * (total / np.maximum(weight, 1e-4)) + (1 - trust) * up
    return cv2.GaussianBlur(filled, (0, 0), 2.0)


def depth_heatmap(depth_m, size, max_range_m=4.0):
    """Metric distance heatmap on a fixed scale in meters, so a colour means
    the same distance in every frame. VIRIDIS, yellow = close, purple = far,
    smoothly blended like the MiDaS inset in main.py -- but in real meters."""
    w, h = size
    depth_m = fill_for_display(depth_m, size)
    cmap = cv2.COLORMAP_VIRIDIS
    norm = np.clip((np.nan_to_num(depth_m, nan=max_range_m) - MIN_RANGE_M) / (max_range_m - MIN_RANGE_M), 0, 1)
    img = cv2.applyColorMap(((1.0 - norm) * 255).astype(np.uint8), cmap)
    img[~np.isfinite(depth_m)] = 0

    # colour bar with meter ticks along the bottom
    bar = cv2.applyColorMap(np.linspace(255, 0, w - 20).astype(np.uint8)[None, :], cmap)
    img[h - 14:h - 6, 10:w - 10] = bar
    for m in (0.3, 1.0, 2.0, 3.0, max_range_m):
        x = 10 + int((m - MIN_RANGE_M) / (max_range_m - MIN_RANGE_M) * (w - 21))
        cv2.line(img, (x, h - 18), (x, h - 6), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, f"{m:g}" if m < max_range_m else f"{m:g}+", (min(max(x - 8, 2), w - 24), h - 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, "Distance (m)", (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return img
