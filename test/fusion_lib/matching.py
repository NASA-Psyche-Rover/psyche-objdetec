"""
Object-level fusion: for each YOLOv6 box, find the LiDAR points that landed
inside it and turn them into one distance estimate for that detection.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class Detection2D:
    x1: float
    y1: float
    x2: float
    y2: float
    cls: str
    confidence: float


@dataclass
class FusedDetection:
    detection: Detection2D
    distance_m: float      # median range of matched points -- robust to stray points
    distance_std: float    # spread of matched points; a rough reliability signal
    num_points: int


def match_points_to_detections(points_lidar, pixels, valid_mask, detections):
    """
    points_lidar: (N,3) the same LiDAR points passed to project_lidar_to_image
    pixels, valid_mask: outputs of project_lidar_to_image
    detections: list[Detection2D], straight from your YOLOv6 output for this frame

    Returns list[FusedDetection]. A detection with zero matched points is
    dropped rather than given a fake distance -- that itself is a signal
    worth logging (see README's "unmatched detections" note).
    """
    ranges = np.linalg.norm(points_lidar, axis=1)
    fused = []
    for det in detections:
        in_box = (
            valid_mask
            & (pixels[:, 0] >= det.x1) & (pixels[:, 0] <= det.x2)
            & (pixels[:, 1] >= det.y1) & (pixels[:, 1] <= det.y2)
        )
        matched_ranges = ranges[in_box]
        if matched_ranges.size == 0:
            continue
        fused.append(FusedDetection(
            detection=det,
            distance_m=float(np.median(matched_ranges)),
            distance_std=float(np.std(matched_ranges)),
            num_points=int(matched_ranges.size),
        ))
    return fused
