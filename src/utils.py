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
    (tw, th), _ = cv2.getTextSize("Test", cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    tx = x1 + (btn_w - tw) // 2
    ty = y1 + (btn_h + th) // 2 - 2
    cv2.putText(frame, "Test", (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
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

def compute_cluster_density(boxes, frame_area):
    """
    Estimate density as total object area / frame area.
    """
    if len(boxes) == 0:
        return 0
    total_area = sum([(x2 - x1) * (y2 - y1) for (x1, y1, x2, y2) in boxes])
    return total_area / frame_area

def draw_alert(frame, text, color=(0, 0, 255), line_height=35):
    """
    Draw alert text on the frame. If text is a list, draws each item on its own line.
    """
    lines = text if isinstance(text, list) else [text]
    for i, line in enumerate(lines):
        y = 50 + i * line_height
        cv2.putText(frame, str(line), (30, y), cv2.QT_FONT_NORMAL, 0.8, color, 2)
