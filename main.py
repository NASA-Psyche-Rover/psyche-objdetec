import time
import cv2

from src.camera_stream import get_video_source
from src.detect import Detector
from src.utils import compute_cluster_density, draw_alert, draw_test_button, get_sample_images
from src.decision import should_proceed

def show_sample_images(detector):
    """Show sample images in a viewer."""
    paths = get_sample_images()
    cv2.namedWindow("Sample Images")

    if not paths:
        import numpy as np
        placeholder = np.zeros((300, 500, 3), dtype=np.uint8)
        placeholder[:] = (40, 40, 40)
        cv2.putText(placeholder, "Add images to data/sample_images/", (40, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Sample Images", placeholder)
        cv2.waitKey(2000)
        cv2.destroyWindow("Sample Images")
        return

    idx = 0
    while True:
        img = cv2.imread(paths[idx])
        if img is None:
            idx = (idx + 1) % len(paths)
            continue

        boxes, labels, _ = detector.detect_objects(img)

        for (x1, y1, x2, y2), label in zip(boxes, labels):
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.putText(img,
                    f"Image {idx+1}/{len(paths)} - Space: next | B/Esc: exit",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2)

        cv2.imshow("Sample Images", img)

        key = cv2.waitKey(100) & 0xFF
        if key in (ord("q"), ord("b"), 27):
            break
        elif key == ord(" "):
            idx = (idx + 1) % len(paths)

    cv2.destroyWindow("Sample Images")


def main():
    # Try webcam first
    try:
        cap = get_video_source(0)
        use_webcam = True

        # Pi 5 optimized settings
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 60)

    except IOError:
        paths = get_sample_images()
        if not paths:
            print("No webcam and no images in data/sample_images/")
            return
        cap = None
        use_webcam = False
        sample_idx = 0

    detector = Detector("models/best.pt")

    frame_area = None
    frame_count = 0
    boxes = []
    labels = []
    decision = "..."
    cluster_density = 0.0

    # FPS tracking
    t0 = time.time()
    fps = 0.0

    ui_state = {
        "test_clicked": False,
        "button_rect": None,
        "frame_size": None
    }

    def on_mouse(event, x, y, _flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        rect = ui_state["button_rect"]
        fsize = ui_state["frame_size"]
        if rect is None or fsize is None:
            return

        fh, fw = fsize
        x1, y1, x2, y2 = rect

        try:
            win = cv2.getWindowImageRect("Psyche Vision Nav")
            sx, sy = x * fw / win[2], y * fh / win[3]
        except cv2.error:
            sx, sy = x, y

        if x1 <= sx <= x2 and y1 <= sy <= y2:
            ui_state["test_clicked"] = True

    cv2.namedWindow("Psyche Vision Nav")
    cv2.setMouseCallback("Psyche Vision Nav", on_mouse)

    while True:

        # -------- Frame Acquisition --------
        if use_webcam:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (640, 480))
        else:
            paths = get_sample_images()
            frame = cv2.imread(paths[sample_idx])
            if frame is None:
                sample_idx = (sample_idx + 1) % len(paths)
                continue

        if frame_area is None:
            frame_area = frame.shape[0] * frame.shape[1]

        frame_count += 1

        # -------- Detection --------
        if use_webcam:
            # Run detection every 2 frames for speed
            if frame_count % 2 == 0:
                small = cv2.resize(frame, (320, 240))
                boxes_small, labels, _ = detector.detect_objects(small)

                sx, sy = 640 / 320, 480 / 240
                boxes = [
                    (int(x1 * sx), int(y1 * sy),
                     int(x2 * sx), int(y2 * sy))
                    for (x1, y1, x2, y2) in boxes_small
                ]

                cluster_density = compute_cluster_density(boxes, frame_area)
                decision = should_proceed(cluster_density)
        else:
            boxes, labels, _ = detector.detect_objects(frame)
            cluster_density = compute_cluster_density(boxes, frame_area)
            decision = should_proceed(cluster_density)

        # -------- Drawing --------
        for (x1, y1, x2, y2), label in zip(boxes, labels):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # FPS
        t1 = time.time()
        dt = t1 - t0
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)
        t0 = t1

        # HUD
        alert_lines = [
            f"Decision: {decision}",
            f"FPS: {fps:.1f}"
        ]

        if labels:
            alert_lines.append("Objects: " + ", ".join(sorted(set(labels))))

        alert_lines.append("T: sample images | Q: quit")
        if not use_webcam:
            alert_lines.append("Space: next image")

        draw_alert(frame, alert_lines)

        ui_state["button_rect"] = draw_test_button(frame)
        ui_state["frame_size"] = frame.shape[:2]

        cv2.imshow("Psyche Vision Nav", frame)

        # -------- Controls --------
        key = cv2.waitKey(1) & 0xFF

        if key == ord("t") or ui_state["test_clicked"]:
            ui_state["test_clicked"] = False
            show_sample_images(detector)

        elif key == ord("q"):
            break

        elif not use_webcam and key == ord(" "):
            sample_idx = (sample_idx + 1) % len(get_sample_images())

    if use_webcam and cap:
        cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
