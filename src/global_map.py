"""
World-frame occupancy accumulation: repeated TraversabilityGrid observations,
each taken from a possibly-different sensor pose, fused into one persistent
GlobalMap via log-odds updates.

New, standalone module -- does not import or modify main.py or anything in
the live pipeline. Builds on src/nav_types.py (Pose, TraversabilityGrid) and
src/traversability_grid.py's cloud_to_grid output.

Pose is always an explicit argument to `integrate()` -- this module never
fetches, estimates, or assumes a pose source (no odometry, no localization).
Wiring a real pose source (wheel odometry, VIO, etc) is a caller concern.

Log-odds, not raw 0/100/-1: a single-frame TraversabilityGrid is a noisy,
momentary read. Storing raw thresholded values would mean the *last*
observation of a cell always wins -- one bad frame (mis-fit plane, spurious
occlusion) permanently corrupts that cell. Log-odds accumulation lets
repeated free observations progressively overturn a stale "blocked" cell
(and vice versa), the standard occupancy-grid-mapping approach (this is the
same value range/update convention ROS's map_server-style mapping uses).

Anomaly provenance limitation (flagged, not fixed here -- fixing it means
extending traversability_grid.py's TraversabilityGrid.data beyond a plain
int8 occupancy value, which the "new module only" scope for this change
doesn't cover): TraversabilityGrid.data doesn't record *why* a cell reads
100 (obstacle vs. local slope vs. no-return/occlusion) -- only `labels`
(populated for obstacle-height points only) distinguishes anything. This
module's drop-off/no-return anomaly heuristic is therefore: a blocked cell
with no entry in `grid.labels` is logged as a possible no-return/crater
anomaly. That's correct for the occlusion and unlabeled-obstacle cases, but
will also catch slope-blocked cells that happen to carry no label -- a
false positive this module can't currently distinguish from a true drop-off.
"""

import json
from dataclasses import dataclass, field

import numpy as np

from src.nav_types import transform_point


@dataclass
class Anomaly:
    """One logged anomaly, keyed by its true WORLD coordinate (not a grid
    index -- anomalies are meaningful even if they fall outside the map's
    fixed extent)."""

    world_xy: tuple      # (x, y) world-frame coordinate
    type: str            # "drop_off" | "unrecognized_class"
    stamp: float = None  # propagated from the source grid.stamp
    meta: dict = field(default_factory=dict)


class GlobalMap:
    """Persistent world-frame occupancy grid, built from repeated
    TraversabilityGrid observations at (possibly) different sensor poses.

    Internally stores float log-odds, not the ternary 0/100/-1 the
    TraversabilityGrid/OccupancyGrid contract uses -- see module docstring.
    `occupancy()` converts back to that contract's representation on
    demand.
    """

    # Standard log-odds occupancy-grid-mapping constants. Asymmetric
    # occupied/free magnitudes (bigger push toward "occupied" per hit than
    # the pull toward "free" per miss) bias the map toward caution, which
    # matches should_proceed()'s existing STOP-biased conservatism in
    # src/decision.py -- an unrelated module, but a deliberately consistent
    # design choice, not a coincidence.
    LOG_ODDS_OCCUPIED = 0.85
    LOG_ODDS_FREE = -0.4
    LOG_ODDS_MIN = -2.0
    LOG_ODDS_MAX = 3.5
    LOG_ODDS_PRIOR = 0.0

    # Probability thresholds occupancy() applies to convert log-odds back to
    # 0/100/-1. The gap between them (0.4-0.6) is deliberately non-trivial:
    # cells whose confidence hasn't cleared either bar read -1 (unknown)
    # rather than guessing, same as a cell that's never been touched at all.
    OCCUPIED_PROB_THRESHOLD = 0.6
    FREE_PROB_THRESHOLD = 0.4

    def __init__(self, resolution, width_m, height_m, world_origin=(0.0, 0.0), known_classes=None):
        """
        Args:
            resolution: meters per world-grid cell.
            width_m, height_m: fixed world-grid extent in meters. The map
                does not grow -- observations landing outside this extent
                still get anomaly-logged (anomalies are keyed by world
                coordinate, independent of the grid array) but don't update
                the occupancy array.
            world_origin: (x, y) world coordinate of world-grid cell [0, 0]'s
                lower-left corner (same convention as
                TraversabilityGrid.origin -- see docs/CONTRACTS.md).
            known_classes: optional iterable of class ids considered
                "recognized" obstacle semantics. If given, any
                `grid.labels` entry integrate() sees that isn't in this set
                gets logged as an "unrecognized_class" anomaly. If None
                (default), that check is skipped entirely -- there's no
                fixed obstacle taxonomy anywhere else in this codebase yet
                (YOLO currently runs on generic COCO classes, per README),
                so there's nothing meaningful to validate labels against
                until a caller supplies one.
        """
        self.resolution = float(resolution)
        self.width_m = float(width_m)
        self.height_m = float(height_m)
        self.world_origin = (float(world_origin[0]), float(world_origin[1]))
        self.known_classes = set(known_classes) if known_classes is not None else None

        self.width_cells = max(1, int(round(self.width_m / self.resolution)))
        self.height_cells = max(1, int(round(self.height_m / self.resolution)))

        self.log_odds = np.full((self.height_cells, self.width_cells), self.LOG_ODDS_PRIOR, dtype=np.float64)
        self.observed = np.zeros((self.height_cells, self.width_cells), dtype=bool)

        self.anomalies = []  # list[Anomaly]
        self.last_stamp = None

    # -- coordinate helpers --------------------------------------------

    def _world_to_cell(self, wx, wy):
        col = int(np.floor((wx - self.world_origin[0]) / self.resolution))
        row = int(np.floor((wy - self.world_origin[1]) / self.resolution))
        return row, col

    def _in_bounds(self, row, col):
        return 0 <= row < self.height_cells and 0 <= col < self.width_cells

    # -- integration ------------------------------------------------------

    def integrate(self, pose, grid):
        """
        Fuse one TraversabilityGrid observation into the map.

        Args:
            pose: (4, 4) SE3 Pose, world <- sensor -- see docs/CONTRACTS.md
                `Pose`. Always required; never inferred.
            grid: TraversabilityGrid to integrate. Requires
                `grid.grid_to_sensor` (set, e.g., by
                `traversability_grid.cloud_to_grid`) to place cells in 3D
                before applying `pose`.
        """
        if grid.grid_to_sensor is None:
            raise ValueError(
                "integrate() requires grid.grid_to_sensor to place grid cells in 3D "
                "before applying pose -- this grid doesn't have one set."
            )

        stamp = grid.stamp
        if stamp is not None:
            self.last_stamp = stamp

        rows, cols = np.nonzero(grid.data != -1)  # skip unknown cells entirely

        for r, c in zip(rows.tolist(), cols.tolist()):
            # NOTE: no "+ grid.origin[...]" here -- grid.grid_to_sensor's
            # translation (built in cloud_to_grid) already bakes origin in
            # (translation = centroid + min_gx*basis1 + min_gy*basis2), so
            # the point fed into transform_point must be the cell-local
            # (col*resolution, row*resolution) coordinate, not offset by
            # origin again. Adding origin here double-counts it and shifts
            # every integrated cell by roughly `origin` itself -- see
            # tests/test_global_map.py and tests/test_replay_harness.py,
            # which independently re-derive expected world positions and
            # would fail if this regresses.
            gx = c * grid.resolution
            gy = r * grid.resolution

            sensor_pt = transform_point(grid.grid_to_sensor, (gx, gy, 0.0))
            world_pt = transform_point(pose, sensor_pt)
            wx, wy = float(world_pt[0]), float(world_pt[1])  # drop z

            cell_value = int(grid.data[r, c])
            label = grid.labels.get((r, c))

            wrow, wcol = self._world_to_cell(wx, wy)
            if self._in_bounds(wrow, wcol):
                delta = self.LOG_ODDS_OCCUPIED if cell_value == 100 else self.LOG_ODDS_FREE
                self.log_odds[wrow, wcol] = np.clip(
                    self.log_odds[wrow, wcol] + delta, self.LOG_ODDS_MIN, self.LOG_ODDS_MAX
                )
                self.observed[wrow, wcol] = True

            # Anomaly logging is independent of map bounds/array updates --
            # it's a list keyed by world coordinate, not a grid cell.
            if cell_value == 100 and label is None:
                self.anomalies.append(Anomaly(
                    world_xy=(wx, wy), type="drop_off", stamp=stamp,
                    meta={"sensor_cell": (r, c)},
                ))
            if self.known_classes is not None and label is not None and label not in self.known_classes:
                self.anomalies.append(Anomaly(
                    world_xy=(wx, wy), type="unrecognized_class", stamp=stamp,
                    meta={"class_id": int(label), "sensor_cell": (r, c)},
                ))

    # -- readout ------------------------------------------------------

    def occupancy(self):
        """Threshold the internal log-odds map back into the
        TraversabilityGrid/OccupancyGrid contract's int8 representation:
        0 = free, 100 = blocked, -1 = unknown (never observed, or observed
        but not confidently resolved either way)."""
        prob = 1.0 / (1.0 + np.exp(-self.log_odds))
        data = np.full(self.log_odds.shape, -1, dtype=np.int8)
        data[self.observed & (prob >= self.OCCUPIED_PROB_THRESHOLD)] = 100
        data[self.observed & (prob <= self.FREE_PROB_THRESHOLD)] = 0
        return data

    # -- persistence ------------------------------------------------------

    def save(self, path):
        """Write the map to `{path}.npy` (log-odds + observed mask, stacked
        so a single array round-trips full internal state) plus
        `{path}.json` (resolution/origin/extent/known_classes) and
        `{path}.anomalies.json` (the anomaly log)."""
        path = str(path)

        stacked = np.stack([self.log_odds, self.observed.astype(np.float64)], axis=0)
        np.save(f"{path}.npy", stacked)

        meta = {
            "resolution": self.resolution,
            "width_m": self.width_m,
            "height_m": self.height_m,
            "world_origin": list(self.world_origin),
            "known_classes": sorted(self.known_classes) if self.known_classes is not None else None,
            "last_stamp": self.last_stamp,
        }
        with open(f"{path}.json", "w") as f:
            json.dump(meta, f, indent=2)

        anomalies = [
            {"world_xy": list(a.world_xy), "type": a.type, "stamp": a.stamp, "meta": a.meta}
            for a in self.anomalies
        ]
        with open(f"{path}.anomalies.json", "w") as f:
            json.dump(anomalies, f, indent=2)

    @classmethod
    def load(cls, path):
        """Inverse of `save()`: reconstructs a GlobalMap (log-odds,
        observed mask, and anomaly log all restored) from `{path}.npy` /
        `{path}.json` / `{path}.anomalies.json`."""
        path = str(path)

        with open(f"{path}.json") as f:
            meta = json.load(f)

        known_classes = set(meta["known_classes"]) if meta.get("known_classes") is not None else None
        obj = cls(
            resolution=meta["resolution"],
            width_m=meta["width_m"],
            height_m=meta["height_m"],
            world_origin=tuple(meta["world_origin"]),
            known_classes=known_classes,
        )

        stacked = np.load(f"{path}.npy")
        obj.log_odds = stacked[0]
        obj.observed = stacked[1].astype(bool)
        obj.last_stamp = meta.get("last_stamp")

        with open(f"{path}.anomalies.json") as f:
            raw_anomalies = json.load(f)
        obj.anomalies = [
            Anomaly(world_xy=tuple(a["world_xy"]), type=a["type"], stamp=a["stamp"], meta=a["meta"])
            for a in raw_anomalies
        ]

        return obj
