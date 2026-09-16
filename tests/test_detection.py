"""
Runs the real ObjectDetector (real YOLO inference, no mocking) against
data/sample_images/ and asserts it returns well-formed Detection objects
(see src/nav_types.py). This is the module that becomes the ROS2 detection
publisher later, so it must never depend on cv2 display/window state to
produce results -- this test only calls .detect(), never .imshow()/HUD code.
"""

from pathlib import Path

import cv2
import pytest

from src.detection import ObjectDetector
from src.nav_types import Detection

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_images"


def _sample_image_paths():
    if not SAMPLE_DIR.is_dir():
        return []
    return sorted(str(p) for p in SAMPLE_DIR.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))


@pytest.fixture(scope="module")
def detector():
    return ObjectDetector("models/best.pt")


@pytest.mark.parametrize("image_path", _sample_image_paths())
def test_detect_returns_well_formed_detections(detector, image_path):
    frame = cv2.imread(image_path)
    assert frame is not None, f"failed to load {image_path}"

    detections = detector.detect(frame)
    assert isinstance(detections, list)

    h, w = frame.shape[:2]
    for d in detections:
        assert isinstance(d, Detection)

        x1, y1, x2, y2 = d.xyxy
        assert all(isinstance(v, int) for v in d.xyxy)
        assert 0 <= x1 < x2 <= w
        assert 0 <= y1 < y2 <= h

        assert isinstance(d.class_id, int)
        assert d.class_id >= 0

        assert isinstance(d.conf, float)
        assert 0.0 <= d.conf <= 1.0


def test_class_name_returns_a_string(detector):
    frame = cv2.imread(_sample_image_paths()[0])
    detections = detector.detect(frame)
    for d in detections:
        name = detector.class_name(d.class_id)
        assert isinstance(name, str) and name


def test_falls_back_to_pretrained_weights_when_model_path_missing(tmp_path):
    missing = tmp_path / "does_not_exist.pt"
    fallback_detector = ObjectDetector(str(missing))
    assert fallback_detector.model is not None

    frame = cv2.imread(_sample_image_paths()[0])
    detections = fallback_detector.detect(frame)
    assert isinstance(detections, list)
