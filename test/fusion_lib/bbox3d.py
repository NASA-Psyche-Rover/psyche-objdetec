"""
Turns a cluster of points into an actual object footprint: where it is, and
how big it is. This is the difference between "a rock was detected somewhere"
and "there's a rock at this exact position, this wide, this tall" -- the
thing your occupancy grid / path planner actually needs.
"""
import numpy as np


def fit_axis_aligned_bbox(points):
    """Simplest version: box aligned to the rover's own x/y/z axes.
    Returns (center, size), each a (3,) array."""
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    return (mins + maxs) / 2, maxs - mins


def fit_oriented_bbox(points):
    """
    A tighter box aligned to the object's own principal axes via PCA, useful
    once axis-aligned boxes start looking noticeably too loose on elongated
    or diagonally-oriented rocks/ridges.

    Returns (center, size, rotation) where rotation's columns are the
    object's local axes expressed in the rover's frame.
    """
    mean = points.mean(axis=0)
    centered = points - mean
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    rotation = eigvecs[:, order]

    rotated = centered @ rotation
    mins, maxs = rotated.min(axis=0), rotated.max(axis=0)
    size = maxs - mins
    center_local = (mins + maxs) / 2
    center = mean + rotation @ center_local
    return center, size, rotation
