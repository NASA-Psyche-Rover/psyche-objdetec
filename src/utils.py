import cv2
import numpy as np

def compute_cluster_density(boxes, frame_area):
    """
    Estimate density as total object area / frame area.
    """
    if len(boxes) == 0:
        return 0
    total_area = sum([(x2 - x1) * (y2 - y1) for (x1, y1, x2, y2) in boxes])
    return total_area / frame_area

def draw_alert(frame, text, color=(0, 0, 255)):
    """
    Draw alert text on the frame.
    """
    cv2.putText(frame, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
