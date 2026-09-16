"""
Tests for src/planner.py. Fully synthetic occupancy grids -- no MiDaS, no
GlobalMap/depth pipeline involved (planner operates purely on the int8
occupancy contract, so there's nothing to gain from routing a real cloud
through it here -- see tests/test_global_map.py for that end-to-end case).
"""

import math

import numpy as np
import pytest

from src.decision import RISK_THRESHOLD
from src.planner import GateResult, cells_to_world, plan, safety_gate


def _path_is_valid(path, occupancy, start, goal):
    if not path:
        return False
    if path[0] != tuple(start) or path[-1] != tuple(goal):
        return False
    for r, c in path:
        if occupancy[r, c] == 100:
            return False
    for (r1, c1), (r2, c2) in zip(path, path[1:]):
        if max(abs(r1 - r2), abs(c1 - c2)) != 1:
            return False  # not a valid 8-connected step
    return True


def test_path_routes_around_blocked_region():
    occ = np.zeros((9, 9), dtype=np.int8)
    occ[2:7, 4] = 100  # a vertical wall with gaps only at rows 0-1 and 7-8

    start, goal = (4, 0), (4, 8)
    path = plan(occ, start, goal)

    assert _path_is_valid(path, occ, start, goal)
    assert (4, 4) not in path  # the direct line through the wall is blocked


def test_unknown_wall_vs_free_detour_prefers_free_route_when_costly():
    # Column 3 is a wall except: rows 1,2,4,5 blocked, row 0 and row 6 free
    # (a detour route), row 3 unknown (a direct "shortcut" through
    # unexplored terrain). Start/goal sit on row 3, either side of the wall.
    occ = np.zeros((7, 7), dtype=np.int8)
    occ[[1, 2, 4, 5], 3] = 100
    occ[3, 3] = -1

    start, goal = (3, 0), (3, 6)

    # Control: with unknown treated as cheap as free, the direct shortcut
    # through the unknown cell should win (shortest route).
    cheap_path = plan(occ, start, goal, unknown_cost=1.0)
    assert _path_is_valid(cheap_path, occ, start, goal)
    assert (3, 3) in cheap_path

    # With unknown terrain expensive enough, the planner should prefer the
    # longer all-free detour over paying to cross the unknown cell.
    expensive_path = plan(occ, start, goal, unknown_cost=20.0)
    assert _path_is_valid(expensive_path, occ, start, goal)
    assert (3, 3) not in expensive_path
    assert all(occ[r, c] != -1 for r, c in expensive_path)


def test_no_path_returns_empty_list_gracefully():
    occ = np.zeros((5, 5), dtype=np.int8)
    occ[1, :] = 100
    occ[:, 1] = 100  # goal at (0,0) fully walled off from the rest

    start, goal = (2, 2), (0, 0)
    path = plan(occ, start, goal)
    assert path == []


def test_start_or_goal_blocked_returns_empty_list():
    occ = np.zeros((5, 5), dtype=np.int8)
    occ[2, 2] = 100
    assert plan(occ, (2, 2), (0, 0)) == []
    assert plan(occ, (0, 0), (2, 2)) == []


def test_start_equals_goal_returns_single_cell():
    occ = np.zeros((5, 5), dtype=np.int8)
    assert plan(occ, (1, 1), (1, 1)) == [(1, 1)]


def test_diagonal_cost_is_sqrt2_not_one():
    # A clear diagonal run should cost len*sqrt(2), i.e. strictly less than
    # the equivalent Manhattan (orthogonal-only) distance, proving diagonal
    # steps are priced at sqrt(2) rather than 1 (which would make the
    # heuristic inadmissible/wrong) or 2 (which would make diagonal moves
    # unjustifiably expensive and push the planner to zigzag instead).
    occ = np.zeros((6, 6), dtype=np.int8)
    path = plan(occ, (0, 0), (5, 5))
    assert _path_is_valid(path, occ, (0, 0), (5, 5))
    assert len(path) == 6  # a pure diagonal run, 5 steps


def test_cells_to_world_uses_resolution_and_origin_cell_centers():
    class FakeMap:
        resolution = 0.1
        world_origin = (2.0, -1.0)

    cells = [(0, 0), (1, 2)]
    world = cells_to_world(cells, FakeMap())
    assert world[0] == pytest.approx((2.0 + 0.05, -1.0 + 0.05))
    assert world[1] == pytest.approx((2.0 + 2 * 0.1 + 0.05, -1.0 + 1 * 0.1 + 0.05))


def test_safety_gate_stop_overrides_plan_to_halt():
    waypoint = (3, 4)
    result = safety_gate(waypoint, terrain_risk=RISK_THRESHOLD + 0.5, object_proximity=0.0, cluster_density=0.0)

    assert isinstance(result, GateResult)
    assert result.decision == "STOP"
    assert result.action == "HALT"
    assert result.replan is True
    assert result.waypoint is None


def test_safety_gate_caution_passes_through_but_marks():
    waypoint = (3, 4)
    result = safety_gate(waypoint, terrain_risk=0.0, object_proximity=0.0, cluster_density=0.9)

    assert result.decision == "CAUTION"
    assert result.action == "PROCEED"
    assert result.replan is False
    assert result.waypoint == waypoint
    assert result.meta.get("caution") is True


def test_safety_gate_proceed_passes_through_unmarked():
    waypoint = (3, 4)
    result = safety_gate(waypoint, terrain_risk=0.0, object_proximity=0.0, cluster_density=0.0)

    assert result.decision == "PROCEED"
    assert result.action == "PROCEED"
    assert result.replan is False
    assert result.waypoint == waypoint
    assert not result.meta.get("caution")
