"""
Groups the leftover (non-ground) points into distinct physical objects.
DBSCAN is used instead of k-means because you don't know how many obstacles
are in the scene ahead of time.

eps is the main knob: it's the max distance (meters) between two points for
them to count as "the same object." Too small and one rock gets split into
several fake clusters; too large and two separate rocks merge into one.
Start around 0.15-0.25m for small terrain features and tune against your
actual mock-Psyche obstacles.
"""
import numpy as np
from sklearn.cluster import DBSCAN


def cluster_obstacle_points(points, eps=0.2, min_samples=5):
    """
    points: (N,3) obstacle points (post ground-removal)

    Returns labels: (N,) int array. -1 means "noise" (not part of any
    cluster) -- typically stray points you should just ignore.
    """
    if len(points) == 0:
        return np.array([], dtype=int)
    return DBSCAN(eps=eps, min_samples=min_samples).fit(points).labels_
