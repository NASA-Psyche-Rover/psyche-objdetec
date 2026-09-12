"""
Discretizes obstacle points into a 2D grid your path planner can route
around, following the same 0/100/-1 convention ROS2's nav_msgs/OccupancyGrid
uses (0 = free, 100 = occupied, -1 = unknown) so this drops straight into
that message type later if you go the ROS2 route.
"""
import numpy as np


def build_occupancy_grid(points_2d, grid_size=0.1, x_range=(-3, 3), y_range=(0, 5)):
    """
    points_2d: (N,2) array of (x, y) obstacle points in the rover's ground
               plane (e.g. LiDAR x and z if z is "forward")
    grid_size: cell size in meters
    x_range, y_range: extent of the map in meters

    Returns a 2D int8 numpy array, rows = y_range, cols = x_range,
    0 = free, 100 = occupied.
    """
    nx = int((x_range[1] - x_range[0]) / grid_size)
    ny = int((y_range[1] - y_range[0]) / grid_size)
    grid = np.zeros((ny, nx), dtype=np.int8)

    for x, y in points_2d:
        if x_range[0] <= x < x_range[1] and y_range[0] <= y < y_range[1]:
            ix = int((x - x_range[0]) / grid_size)
            iy = int((y - y_range[0]) / grid_size)
            grid[iy, ix] = 100
    return grid
