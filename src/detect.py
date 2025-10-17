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
        Returns list of bounding boxes [(x1, y1, x2, y2), ...]
        """
        results = self.model.predict(frame, conf=self.conf, verbose=False)
        boxes = []
        for r in results:
            for box in r.boxes.xyxy:
                x1, y1, x2, y2 = box.tolist()
                boxes.append((int(x1), int(y1), int(x2), int(y2)))
        return boxes, results
