"""
OAK-D Lite object detection + real depth-sensing test.

Runs YOLO detection (via the existing Detector class) on the OAK-D Lite's RGB
feed, and uses the OAK-D's own stereo depth (real millimeters) to report the
actual distance to each detected object -- no MiDaS, no relative depth, no
terrain risk pipeline. This is a focused test of detection + real depth fused
together, using src/camera_stream.py's OakDLiteCamera and the new
estimate_object_distances_mm() helper.

Requires: OAK-D Lite plugged in via USB-C, and src/camera_stream.py already
updated with the DepthAI v3 OakDLiteCamera + distance-sensing functions.
"""

import time
import cv2

from src.camera_stream import OakDLiteCamera, estimate_object_distances_mm
from src.detect import Detector

print("DEBUG: imports done")

# ---- Config ----
TARGET_WIDTH, TARGET_HEIGHT = 640, 480
DETECT_EVERY = 2          # run YOLO every N frames
STOP_DISTANCE_M = 1.0     # nearest object closer than this -> STOP
CAUTION_DISTANCE_M = 2.5  # nearest object closer than this -> CAUTION
DECISION_SMOOTHING = 0.3  # 0 = instant/flickery, 1 = never changes; EMA on nearest_m


def draw_hud(frame, decision, nearest_m, fps, num_boxes):
    color = {
        "STOP": (0, 0, 255),
        "CAUTION": (0, 200, 255),
        "PROCEED": (0, 255, 0),
    }.get(decision, (255, 255, 255))

    lines = [
        f"Decision: {decision}",
        f"Nearest object: {nearest_m:.2f} m" if nearest_m is not None else "Nearest object: no depth data",
        f"Objects detected: {num_boxes}",
        f"FPS: {fps:.1f}",
    ]
    for i, line in enumerate(lines):
        y = 30 + i * 26
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def main():
    print("DEBUG: main() started")

    print("DEBUG: about to open OAK-D Lite")
    try:
        oak_cam = OakDLiteCamera(rgb_size=(TARGET_WIDTH, TARGET_HEIGHT), fps=30)
        print("DEBUG: OAK-D Lite opened successfully")
    except Exception as e:
        print(f"ERROR: OAK-D Lite not available: {e}")
        print("This test requires the OAK-D Lite connected via USB-C -- exiting.")
        return

    print("DEBUG: about to load detector")
    detector = Detector("models/best.pt")
    print("DEBUG: detector loaded")

    cv2.namedWindow("OAK-D Detection + Depth")

    boxes, labels = [], []
    frame_count = 0
    t0 = time.time()
    fps = 0.0
    smoothed_nearest_m = None

    print("DEBUG: entering main loop")
    loop_count = 0

    while True:
        loop_count += 1

        frame, depth_frame = oak_cam.read()
        if frame is None:
            if loop_count <= 20:
                print("DEBUG: waiting for first frame...")
            # Avoid a tight busy-spin while no frame is ready yet -- this was
            # the main source of flicker/CPU thrash, since tryGet() is
            # non-blocking and returns instantly with nothing to show.
            cv2.waitKey(1)
            continue

        frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
        frame_count += 1

        # -------- Detection (every DETECT_EVERY frames, on a half-res copy for speed) --------
        if frame_count % DETECT_EVERY == 0:
            small_w, small_h = TARGET_WIDTH // 2, TARGET_HEIGHT // 2
            small = cv2.resize(frame, (small_w, small_h))
            boxes_small, labels, _ = detector.detect_objects(small)

            sx, sy = TARGET_WIDTH / small_w, TARGET_HEIGHT / small_h
            boxes = [
                (int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy))
                for (x1, y1, x2, y2) in boxes_small
            ]

        # -------- Real depth per object (OAK-D stereo, millimeters) --------
        distances_m = []
        if depth_frame is not None and boxes:
            distances_mm = estimate_object_distances_mm(
                boxes, depth_frame, (TARGET_WIDTH, TARGET_HEIGHT)
            )
            distances_m = [d / 1000.0 if d is not None else None for d in distances_mm]

        valid_distances = [d for d in distances_m if d is not None]
        raw_nearest_m = min(valid_distances) if valid_distances else None

        # Smooth the nearest-distance signal with an EMA so the decision
        # doesn't flap between STOP/CAUTION/PROCEED on single noisy frames
        # right at a threshold boundary. Missing readings don't reset it --
        # they just leave the smoothed value where it was.
        if raw_nearest_m is not None:
            if smoothed_nearest_m is None:
                smoothed_nearest_m = raw_nearest_m
            else:
                smoothed_nearest_m = (
                    DECISION_SMOOTHING * smoothed_nearest_m
                    + (1 - DECISION_SMOOTHING) * raw_nearest_m
                )

        if smoothed_nearest_m is None:
            decision = "..."
        elif smoothed_nearest_m < STOP_DISTANCE_M:
            decision = "STOP"
        elif smoothed_nearest_m < CAUTION_DISTANCE_M:
            decision = "CAUTION"
        else:
            decision = "PROCEED"

        # -------- Drawing --------
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            label = labels[i] if i < len(labels) else "?"
            dist = distances_m[i] if i < len(distances_m) else None
            box_color = (0, 255, 0)
            if dist is not None and dist < STOP_DISTANCE_M:
                box_color = (0, 0, 255)
            elif dist is not None and dist < CAUTION_DISTANCE_M:
                box_color = (0, 200, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            dist_text = f"{dist:.2f} m" if dist is not None else "no depth"
            cv2.putText(frame, f"{label} - {dist_text}", (x1, max(y1 - 8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

        t1 = time.time()
        dt = t1 - t0
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)
        t0 = t1

        draw_hud(frame, decision, smoothed_nearest_m, fps, len(boxes))

        cv2.imshow("OAK-D Detection + Depth", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("DEBUG: 'q' pressed, exiting")
            break

    oak_cam.close()
    cv2.destroyAllWindows()
    print("DEBUG: done")


if __name__ == "__main__":
    main()