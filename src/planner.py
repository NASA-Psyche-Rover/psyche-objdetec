"""
Global path planning over a GlobalMap occupancy grid (A*), plus a thin
reactive-veto layer that guards the planner's next step with the existing
live decision layer.

New, standalone module -- does not import or modify main.py or anything in
the live pipeline. Builds on src/global_map.py's `occupancy()` output (the
TraversabilityGrid/OccupancyGrid int8 contract -- see docs/CONTRACTS.md)
and src/decision.py's `should_proceed()`, unchanged and unwrapped-in-logic:
this module never reimplements or duplicates should_proceed's thresholds,
it only calls it and reacts to the string it returns.

Division of responsibility, deliberately: the planner answers "what's the
best global route to the goal, given everything mapped so far" -- a slow,
whole-map computation. `safety_gate` answers "is it safe to take the very
next step of that route, right now" -- a fast, single-frame check driven by
the same live terrain_risk/object_proximity/cluster_density signals
main.py already computes every frame. The planner has no visibility into
those live signals (it only sees the accumulated map) and the gate has no
visibility into the route (it only sees one waypoint) -- that separation is
intentional, not an oversight: a stale global plan should never be trusted
to override a fresh STOP from the live sensors.
"""

import heapq
import math
from dataclasses import dataclass, field

import numpy as np

from src.decision import should_proceed

SQRT2 = math.sqrt(2.0)

# 8-connected offsets: (delta_row, delta_col, step_cost).
_NEIGHBORS = [
    (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
    (-1, -1, SQRT2), (-1, 1, SQRT2), (1, -1, SQRT2), (1, 1, SQRT2),
]


def _octile_heuristic(a, b):
    """Admissible/consistent for 8-connected grids with diagonal cost
    sqrt(2): the true remaining cost is at least this (cell weights are
    always >= 1, so scaling by them only ever increases the real cost)."""
    dr, dc = abs(a[0] - b[0]), abs(a[1] - b[1])
    return (SQRT2 - 1) * min(dr, dc) + max(dr, dc)


def plan(occupancy, start_cell, goal_cell, unknown_cost=5.0):
    """
    A* over a world occupancy grid (int8, `TraversabilityGrid`/
    `nav_msgs.OccupancyGrid` cell semantics -- see docs/CONTRACTS.md and
    src/global_map.py `GlobalMap.occupancy()`).

    Args:
        occupancy: (H, W) int8 array. 100 = impassable (never expanded),
            0 = normal cost, -1 = traversable but expensive (see
            `unknown_cost`) -- lets the planner cross unexplored terrain
            toward a goal when nothing better is known, while preferring
            an already-confirmed-free route if one exists.
        start_cell, goal_cell: (row, col) grid indices.
        unknown_cost: per-step cost multiplier for entering an unknown (-1)
            cell (a straight step into one costs `unknown_cost`, a diagonal
            step costs `unknown_cost * sqrt(2)` -- same diagonal/straight
            ratio as free cells, just scaled up).

    Returns:
        list of (row, col) waypoint cells from start to goal, inclusive of
        both endpoints, in traversal order. Empty list if no path exists
        (unreachable goal, or start/goal itself out of bounds or blocked)
        -- never raises for an unreachable goal.
    """
    occupancy = np.asarray(occupancy)
    height, width = occupancy.shape
    start = (int(start_cell[0]), int(start_cell[1]))
    goal = (int(goal_cell[0]), int(goal_cell[1]))

    def in_bounds(cell):
        r, c = cell
        return 0 <= r < height and 0 <= c < width

    def passable(cell):
        return occupancy[cell[0], cell[1]] != 100

    def cell_weight(cell):
        return unknown_cost if occupancy[cell[0], cell[1]] == -1 else 1.0

    if not in_bounds(start) or not in_bounds(goal) or not passable(start) or not passable(goal):
        return []
    if start == goal:
        return [start]

    g_score = {start: 0.0}
    came_from = {}
    open_heap = [(_octile_heuristic(start, goal), 0.0, start)]
    closed = set()

    while open_heap:
        _, g, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == goal:
            path = [current]
            while path[-1] in came_from:
                path.append(came_from[path[-1]])
            path.reverse()
            return path
        closed.add(current)

        for dr, dc, step in _NEIGHBORS:
            neighbor = (current[0] + dr, current[1] + dc)
            if not in_bounds(neighbor) or neighbor in closed or not passable(neighbor):
                continue
            if dr != 0 and dc != 0:
                # No cutting diagonally through a blocked orthogonal corner.
                if not passable((current[0] + dr, current[1])) or not passable((current[0], current[1] + dc)):
                    continue

            tentative = g + step * cell_weight(neighbor)
            if tentative < g_score.get(neighbor, math.inf):
                g_score[neighbor] = tentative
                came_from[neighbor] = current
                heapq.heappush(open_heap, (tentative + _octile_heuristic(neighbor, goal), tentative, neighbor))

    return []  # goal unreachable


def cells_to_world(cells, global_map):
    """Convert a list of (row, col) grid cells into world (x, y) coordinates
    using `global_map.resolution` / `global_map.world_origin`. Uses each
    cell's center (not its lower-left corner) -- a waypoint is somewhere to
    drive *to*, and a cell's center is the natural target, whereas `origin`
    (per docs/CONTRACTS.md) is deliberately the corner, for grid indexing."""
    resolution = global_map.resolution
    origin_x, origin_y = global_map.world_origin
    return [
        (origin_x + (col + 0.5) * resolution, origin_y + (row + 0.5) * resolution)
        for row, col in cells
    ]


@dataclass
class GateResult:
    """Outcome of `safety_gate`: what the rover should actually do about
    the planner's proposed next waypoint, right now."""

    action: str            # "PROCEED" | "HALT"
    decision: str           # should_proceed()'s raw output, unchanged -- "PROCEED" | "CAUTION" | "STOP"
    waypoint: object = None  # the input waypoint, passed through if action == "PROCEED"; None if halted
    replan: bool = False     # True if the caller should trigger a fresh plan() call
    meta: dict = field(default_factory=dict)


def safety_gate(next_waypoint, terrain_risk, object_proximity=0.0, cluster_density=0.0, **should_proceed_kwargs):
    """
    Reactive veto over the planner's proposed next step, using the SAME
    live risk signals main.py already computes every frame (terrain_risk
    from TerrainAnalyzer, object_proximity/cluster_density from
    src/utils.py). Calls the existing `should_proceed()` unmodified and
    unduplicated -- this function contains no risk-threshold logic of its
    own, only the policy for what to do with should_proceed's answer:

      STOP    -> override the plan: halt in place, flag that a replan is
                 needed (the map that produced this route may itself be
                 stale/wrong given whatever just triggered STOP).
      CAUTION -> pass the waypoint through (the planner's route still
                 stands), but mark the result so a caller can slow down,
                 log, or otherwise react without discarding the plan.
      PROCEED -> pass the waypoint through unmarked.

    Any extra keyword args are forwarded to should_proceed() unchanged
    (e.g. custom thresholds), so this gate never hardcodes a threshold
    should_proceed already owns.
    """
    decision = should_proceed(terrain_risk, object_proximity, cluster_density, **should_proceed_kwargs)

    if decision == "STOP":
        return GateResult(action="HALT", decision=decision, waypoint=None, replan=True)
    if decision == "CAUTION":
        return GateResult(action="PROCEED", decision=decision, waypoint=next_waypoint, replan=False,
                           meta={"caution": True})
    return GateResult(action="PROCEED", decision=decision, waypoint=next_waypoint, replan=False)
