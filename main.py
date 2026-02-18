import time
import cv2

from src.camera_stream import get_video_source
from src.detect import Detector
from src.utils import compute_cluster_density, draw_alert
from src.decision import should_proceed


def main():
    cap = get_video_source(0)

    # Pi 5 smooth settings (camera)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 60)

    detector = Detector("models/best.pt")

    frame_area = None
    frame_count = 0
    boxes = []
    decision = "..."
    cluster_density = 0.0

    # FPS counter
    t0 = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize once for display
        frame = cv2.resize(frame, (640, 480))

        if frame_area is None:
            frame_area = frame.shape[0] * frame.shape[1]

        frame_count += 1

        # Run detection every N frames (2 = faster updates, 3 = smoother)
        if frame_count % 2 == 0:
            # Smaller input to YOLO for speed
            yolo_frame = cv2.resize(frame, (320, 240))
            boxes_small, _ = detector.detect_objects(yolo_frame)

            # Scale boxes back up to 640x480
            sx = 640 / 320
            sy = 480 / 240
            boxes = [(int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)) for (x1, y1, x2, y2) in boxes_small]

            cluster_density = compute_cluster_density(boxes, frame_area)
            decision = should_proceed(cluster_density)

        # Draw boxes (latest)
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        draw_alert(frame, f"Decision: {decision}")

        # FPS overlay
        t1 = time.time()
        dt = t1 - t0
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)
        t0 = t1
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Psyche Vision Nav", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
