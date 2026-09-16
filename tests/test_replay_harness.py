"""
The true system regression test: runs scripts/replay_harness.py's full
synthetic-metric loop (real cloud_to_grid, real GlobalMap.integrate, real
planner.plan -- no mocks) and checks the output against KNOWN ground
truth. If the camera<->world frame convention documented in
replay_harness.py is ever wrong, obstacles land in the wrong world cells
and the first assertion below fails -- that's the point of this test.
"""

import numpy as np

from scripts.replay_harness import run_simulation

TOLERANCE_M = 0.3  # obstacle half-size (0.15 m) + grid quantization + margin


def _blocked_world_centers(global_map):
    occ = global_map.occupancy()
    rows, cols = np.nonzero(occ == 100)
    xs = global_map.world_origin[0] + (cols + 0.5) * global_map.resolution
    ys = global_map.world_origin[1] + (rows + 0.5) * global_map.resolution
    return np.stack([xs, ys], axis=-1)


def test_final_occupancy_matches_ground_truth_obstacle_positions():
    result = run_simulation()
    global_map = result["global_map"]
    obstacles = result["scene"]["obstacles"]

    blocked = _blocked_world_centers(global_map)
    assert blocked.shape[0] > 0, "expected at least some blocked cells from the two obstacles"

    # Every known obstacle has a blocked cell near it -- this is the
    # geometry sanity check: if the axis convention were wrong (e.g. an
    # identity-rotation pose, collapsing world y onto sensor Y instead of
    # sensor Z), obstacles would never accumulate at their true world (x,
    # y) and this would fail.
    for obs in obstacles:
        center = np.array([obs["x"], obs["y"]])
        dists = np.linalg.norm(blocked - center, axis=1)
        assert dists.min() < TOLERANCE_M, (
            f"no blocked cell found near ground-truth obstacle at {center}; "
            f"closest blocked cell was {dists.min():.3f} m away -- "
            f"check the camera<->world frame convention"
        )

    # No blocked cell far from every known obstacle -- catches the
    # opposite failure mode (obstacles smeared/duplicated elsewhere, or
    # flat ground misclassified as blocked).
    obstacle_centers = np.array([[o["x"], o["y"]] for o in obstacles])
    for bx, by in blocked:
        dists = np.linalg.norm(obstacle_centers - np.array([bx, by]), axis=1)
        assert dists.min() < TOLERANCE_M, (
            f"blocked cell at world ({bx:.3f}, {by:.3f}) is not near any known obstacle "
            f"(closest is {dists.min():.3f} m away) -- spurious/misplaced occupancy"
        )


def test_final_occupancy_reflects_open_ground_as_free():
    result = run_simulation()
    global_map = result["global_map"]
    occ = global_map.occupancy()
    assert np.any(occ == 0), "expected some confidently free ground cells in the accumulated map"


def test_path_found_to_goal_on_accumulated_map():
    result = run_simulation()
    path = result["final_path"]

    assert path, "expected a path from start to goal on the accumulated map"
    assert path[0] == tuple(result["start_cell"])
    assert path[-1] == tuple(result["goal_cell"])

    occ = result["global_map"].occupancy()
    assert all(occ[r, c] != 100 for r, c in path)


def test_periodic_planning_ran_and_eventually_found_a_path():
    result = run_simulation(plan_every=2)
    assert result["planned_paths"], "expected at least one periodic plan() call"
    # Not every early frame (map still mostly unknown) is guaranteed to
    # find a path, but by the end of the trajectory one should exist.
    assert any(path for _, path in result["planned_paths"])
