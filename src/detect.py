from ultralytics import YOLO
import cv2

class Detector:
    def __init__(self, model_path="models/best.pt", conf=0.4):
        self.model = YOLO("models/yolov8n.pt")  # small pretrained model
        #self.model = YOLO(model_path)
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
