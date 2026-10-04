"""
YOLOv8 object detection, decoupled from any drawing/display concerns.

This replaces src/detect.py's Detector: same model loading + graceful
best.pt -> yolov8n.pt fallback, but returns List[Detection] (see
src/nav_types.py) instead of (boxes, labels, results) tuples baked for
main.py's OpenCV HUD. Keeping this module free of cv2 (or any other
display library) is the point -- it's meant to become the ROS2 detection
publisher later, where there's no HUD to draw into at all.

main.py still wants human-readable class names for its on-screen labels,
but a display string isn't part of the Detection contract (xyxy/class_id/
conf only, see docs/CONTRACTS.md) -- so name lookup lives here as a
separate `class_name()` method, not on Detection itself.
"""

from pathlib import Path

from ultralytics import YOLO

from src.nav_types import Detection

DEFAULT_MODEL_PATH = "models/yolov8n.pt"

# Rock detectors, best first. best.pt is the custom-trained asteroid model
# (notebooks/train_yolov8.ipynb); rocks_yoloworld.pt is an open-vocabulary
# YOLO-World model with its classes fixed to rock/boulder/stone (built by
# scripts/build_rock_model.py -- no training data needed). The pretrained
# COCO yolov8n.pt has no rock class at all, so it is a last resort that
# detects generic objects only.
ROCK_MODEL_PATHS = ("models/best.pt", "models/rocks_yoloworld.pt")


class ObjectDetector:
    """Wraps an Ultralytics YOLO model for rover obstacle detection.

    Falls back to the pretrained YOLOv8n weights if model_path doesn't exist
    or is an empty placeholder (e.g. models/best.pt before the asteroid
    dataset has been trained on -- see notebooks/train_yolov8.ipynb).
    """

    def __init__(self, model_path="models/best.pt", conf=0.4):
        candidates = (model_path,) if isinstance(model_path, (str, Path)) else tuple(model_path)
        self.model_path = None
        for candidate in candidates:
            path = Path(candidate)
            if path.exists() and path.stat().st_size > 0:
                self.model_path = str(path)
                break
        if self.model_path is None:
            print(f"[ObjectDetector] {', '.join(map(str, candidates))} missing or empty, "
                  f"falling back to {DEFAULT_MODEL_PATH}")
            self.model_path = DEFAULT_MODEL_PATH
        self.model = YOLO(self.model_path)
        self.is_rock_model = self.model_path != DEFAULT_MODEL_PATH
        self.conf = conf

    def detect(self, frame):
        """
        Runs YOLO inference on a frame.

        Returns:
            list[Detection]: xyxy in the frame's own pixel coordinates
            (see docs/CONTRACTS.md `Detection` -- rescaling to a different
            resolution is a caller concern), class_id as the model's raw
            integer class index, conf as the detection confidence.
        """
        results = self.model.predict(frame, conf=self.conf, verbose=False)
        detections = []
        for r in results:
            for i, box in enumerate(r.boxes.xyxy):
                x1, y1, x2, y2 = box.tolist()
                detections.append(Detection(
                    xyxy=(int(x1), int(y1), int(x2), int(y2)),
                    class_id=int(r.boxes.cls[i].item()),
                    conf=float(r.boxes.conf[i].item()),
                ))
        return detections

    def class_name(self, class_id):
        """Human-readable class name for a class_id (e.g. COCO names for
        the pretrained fallback), for callers that want a display label."""
        return self.model.names.get(class_id, f"class_{class_id}")
