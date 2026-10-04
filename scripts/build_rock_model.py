"""
Build models/rocks_yoloworld.pt: a YOLO detector that only looks for rocks,
without needing a labelled rock dataset.

YOLO-World is open-vocabulary -- its classes are set from text rather than
fixed at training time. This fixes them to rock / boulder / stone and saves
the result as an ordinary .pt that src/detection.py's ObjectDetector loads
like any other YOLO model (see ROCK_MODEL_PATHS there). Once saved, the text
encoder is no longer needed at runtime.

One-time setup, needs network access:
    pip install git+https://github.com/ultralytics/CLIP.git
    python scripts/build_rock_model.py

A model fine-tuned on real asteroid/analog imagery (notebooks/train_yolov8.ipynb
-> models/best.pt) will beat this and takes priority if present.
"""

from pathlib import Path

from ultralytics import YOLOWorld

ROCK_CLASSES = ["rock", "boulder", "stone"]
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

if __name__ == "__main__":
    model = YOLOWorld(str(MODELS_DIR / "yolov8s-worldv2.pt"))  # downloaded on first use
    model.set_classes(ROCK_CLASSES)
    out = MODELS_DIR / "rocks_yoloworld.pt"
    model.save(str(out))
    print(f"Saved {out} with classes {ROCK_CLASSES}")
