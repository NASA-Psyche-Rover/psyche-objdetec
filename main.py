import time
import cv2

from src.camera_stream import get_video_source
from src.detect import Detector
from src.utils import compute_cluster_density, draw_alert, draw_test_button, get_sample_images
from src.decision import should_proceed


def main():
    cap = get_video_source(0)

    # Pi 5 smooth settings (camera)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 60)
def show_sample_images(detector):
    """Show sample images in a viewer. Cycle with space/arrows, close with Escape or 'b'."""
    paths = get_sample_images()
    cv2.namedWindow("Sample Images")
    if not paths:
        import numpy as np
        placeholder = np.zeros((300, 500, 3), dtype=np.uint8)
        placeholder[:] = (40, 40, 40)
        cv2.putText(placeholder, "Add images to data/sample_images/", (50, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(placeholder, "(jpg, png, bmp)", (50, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
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
            cv2.putText(img, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(img, f"Image {idx + 1}/{len(paths)} - Space: next, B/Esc: back", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("Sample Images", img)
        k = cv2.waitKey(100) & 0xFF
        if k == ord("q") or k == 27:  # Esc
            break
        if k == ord("b"):
            break
        if k == ord(" ") or k == 83:  # Space or Right
            idx = (idx + 1) % len(paths)
        elif k == 81:  # Left
            idx = (idx - 1) % len(paths)
    cv2.destroyWindow("Sample Images")

def main():
    try:
        cap = get_video_source(0)  # 0 = webcam; can change to "data/sample_video.mp4"
        use_webcam = True
    except IOError:
        paths = get_sample_images()
        if not paths:
            print("No webcam and no images in data/sample_images/. Add images or fix webcam.")
            return
        cap = None
        use_webcam = False
        sample_idx = [0]

    detector = Detector("models/best.pt")

    frame_area = None
    frame_count = 0
    boxes = []
    decision = "..."
    cluster_density = 0.0

    # FPS counter
    t0 = time.time()
    fps = 0.0

    # Shared state for mouse callback (coords are in window space, we store scale for mapping)
    ui_state = {"test_clicked": False, "button_rect": None, "frame_size": None}

    def on_mouse(event, x, y, _flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        rect = ui_state.get("button_rect")
        fsize = ui_state.get("frame_size")
        if rect is None or fsize is None:
            return
        fh, fw = fsize
        x1, y1, x2, y2 = rect
        # Map window coords to image coords (OpenCV returns window coords)
        try:
            win = cv2.getWindowImageRect("Psyche Vision Nav")
            if win[2] > 0 and win[3] > 0:
                sx, sy = x * fw / win[2], y * fh / win[3]
                if x1 <= sx <= x2 and y1 <= sy <= y2:
                    ui_state["test_clicked"] = True
        except cv2.error:
            if x1 <= x <= x2 and y1 <= y <= y2:
                ui_state["test_clicked"] = True

    cv2.namedWindow("Psyche Vision Nav")
    cv2.setMouseCallback("Psyche Vision Nav", on_mouse, None)

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
        if use_webcam:
            ret, frame = cap.read()
            if not ret:
                break
        else:
            paths = get_sample_images()
            frame = cv2.imread(paths[sample_idx[0]])
            if frame is None:
                sample_idx[0] = (sample_idx[0] + 1) % len(paths)
                continue
            ret = True
        if frame_area is None:
            frame_area = frame.shape[0] * frame.shape[1]

        # Run detection to get boxes and labels
        boxes, labels, _ = detector.detect_objects(frame)
        cluster_density = compute_cluster_density(boxes, frame_area)
        decision = should_proceed(cluster_density)

        # Draw boxes and labels on each box
        for (x1, y1, x2, y2), label in zip(boxes, labels):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Draw HUD: decision + detected object labels
        alert_lines = [f"Decision: {decision}"]
        if labels:
            unique_labels = sorted(set(labels))
            alert_lines.append(f"Objects Detected: {', '.join(unique_labels)}")
        alert_lines.append("T: sample images | Q: quit")
        if not use_webcam:
            alert_lines.append("Space: next image")
        draw_alert(frame, alert_lines)

        # Draw Test button and store rect for hit testing
        ui_state["button_rect"] = draw_test_button(frame)
        ui_state["frame_size"] = frame.shape[:2]
        cv2.imshow("Psyche Vision Nav", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("t") or ui_state["test_clicked"]:
            ui_state["test_clicked"] = False
            show_sample_images(detector)
        elif key == ord("q"):
            break
        elif use_webcam and key == ord(" "):
            pass  # no-op for webcam
        elif not use_webcam and key == ord(" "):
            sample_idx[0] = (sample_idx[0] + 1) % len(get_sample_images())

    if use_webcam and cap:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
