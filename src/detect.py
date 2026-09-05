from pathlib import Path
from ultralytics import YOLO

DEFAULT_MODEL_PATH = "models/yolov8n.pt"


class Detector:
    """Wraps an Ultralytics YOLO model for rover obstacle detection.

    Falls back to the pretrained YOLOv8n weights if model_path doesn't exist
    or is an empty placeholder (e.g. models/best.pt before the asteroid
    dataset has been trained on — see notebooks/train_yolov8.ipynb).
    """

    def __init__(self, model_path="models/best.pt", conf=0.4):
        path = Path(model_path)
        if path.exists() and path.stat().st_size > 0:
            self.model = YOLO(str(path))
        else:
            print(f"[Detector] '{model_path}' missing or empty, falling back to {DEFAULT_MODEL_PATH}")
            self.model = YOLO(DEFAULT_MODEL_PATH)
        self.conf = conf

    def detect_objects(self, frame):
        """
        Runs YOLO inference on a frame.
        Returns (boxes, labels, results) where:
          - boxes: [(x1, y1, x2, y2), ...]
          - labels: [str, ...] class names for each detection
        """
        results = self.model.predict(frame, conf=self.conf, verbose=False)
        boxes = []
        labels = []
        for r in results:
            names = r.names
            for i, box in enumerate(r.boxes.xyxy):
                x1, y1, x2, y2 = box.tolist()
                boxes.append((int(x1), int(y1), int(x2), int(y2)))
                cls_id = int(r.boxes.cls[i].item())
                labels.append(names.get(cls_id, f"class_{cls_id}"))
        return boxes, labels, results
