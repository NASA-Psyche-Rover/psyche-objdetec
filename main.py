import cv2
from src.camera_stream import get_video_source
from src.detect import Detector
from src.utils import compute_cluster_density, draw_alert
from src.decision import should_proceed

def main():
    cap = get_video_source(0)  # 0 = webcam; can change to "data/sample_video.mp4"
    detector = Detector("models/best.pt")
    frame_area = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_area is None:
            frame_area = frame.shape[0] * frame.shape[1]

        boxes, _ = detector.detect_objects(frame)
        cluster_density = compute_cluster_density(boxes, frame_area)
        decision = should_proceed(cluster_density)

        # Draw boxes
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

        draw_alert(frame, f"Decision: {decision}")
        cv2.imshow("Psyche Vision Nav", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
