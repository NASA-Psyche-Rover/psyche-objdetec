import cv2
import numpy as npparent
from pathlib import Path

SAMPLE_IMAGES_DIR = Path(__file__).resolve().parent .parent/ "data" / "sample_images"

def draw_test_button(frame, padding=20):
    """
    Draw a 'Test' button in the top-right corner. Returns (x1, y1, x2, y2) for hit testing.
    """
    h, w = frame.shape[:2]
    btn_w, btn_h = 80, 36
    x1, y1 = w - btn_w - padding, padding
    x2, y2 = x1 + btn_w, y1 + btn_h
    cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 120, 200), -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 150, 255), 2)
    (tw, th), _ = cv2.getTextSize("Test", cv2.QT_FONT_NORMAL, 0.6, 2)
    tx = x1 + (btn_w - tw) // 2
    ty = y1 + (btn_h + th) // 2 - 2
    cv2.putText(frame, "Test", (tx, ty), cv2.QT_FONT_NORMAL, 0.6, (255, 255, 255), 2)
    return (x1, y1, x2, y2)

def get_sample_images():
    """Return list of image paths from data/sample_images/. Supports jpg, jpeg, png, bmp."""
    if not SAMPLE_IMAGES_DIR.exists():
        print(f"No sample images directory found at {SAMPLE_IMAGES_DIR}")
        return []
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    paths = sorted([
        str(p) for p in SAMPLE_IMAGES_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    ])
    print(f"Found {len(paths)} sample images in {SAMPLE_IMAGES_DIR}")
    return paths

def box_closeness_from_depth(depth_map, box, frame_shape):
    """
    Return normalized closeness (0-1) for one box from a depth map.
    Higher = closer to camera. depth_map is resized to frame_shape if needed.
    """
    if depth_map is None or depth_map.size == 0:
        return None
    h, w = frame_shape[:2]
    if depth_map.shape[0] != h or depth_map.shape[1] != w:
        depth_map = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_LINEAR)
    x1, y1, x2, y2 = box
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    roi = depth_map[y1:y2, x1:x2]
    d_max = float(npparent.max(depth_map))
    if d_max <= 0:
        return None
    return float(npparent.mean(roi)) / d_max

def compute_cluster_density(boxes, frame_area):
    """
    Camera-distance-based risk: closer objects appear larger in the image.
    Uses the largest detection's area as a fraction of frame area (closest object
    dominates). Returns value in [0, 1]; higher = something is close = higher risk.
    """
    if len(boxes) == 0:
        return 0.0
    max_box_area = max(
        (x2 - x1) * (y2 - y1)
        for (x1, y1, x2, y2) in boxes
    )
    return max_box_area / frame_area

def draw_alert(frame, text, color=(0, 0, 255), line_height=35):
    """
    Draw alert text on the frame. If text is a list, draws each item on its own line.
    """
    lines = text if isinstance(text, list) else [text]
    for i, line in enumerate(lines):
        y = 50 + i * line_height
        cv2.putText(frame, str(line), (30, y), cv2.QT_FONT_NORMAL, 0.8, color, 2)
