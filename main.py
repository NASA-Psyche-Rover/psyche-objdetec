import time
import threading
import cv2

from src.detect import Detector
from src.utils import draw_alert
from src.proximity_stop import ProximityStopper, ProximityStopConfig


# -----------------------
# Motor integration layer
# -----------------------
class MotorAdapter:
    """
    Wraps movement.py into a consistent interface:
      - forward()
      - stop()
      - cleanup()
    Works even if movement.py exposes different function names.
    """

    def __init__(self):
        self.enabled = False
        self._obj = None
        self._mode = None  # "class" or "func"

        mod = None
        try:
            import movement as mod  # movement.py in project root
        except Exception:
            try:
                from src import movement as mod  # movement.py inside src/
            except Exception:
                mod = None

        if mod is None:
            print("[MOTORS] movement.py not found. Running VISION ONLY.")
            return

        # 1) Preferred: MotorController class
        if hasattr(mod, "MotorController"):
            try:
                self._obj = mod.MotorController()
                self._mode = "class"
                self.enabled = True
                print("[MOTORS] Using movement.MotorController()")
                return
            except Exception as e:
                print(f"[MOTORS] MotorController init failed: {e}. Running VISION ONLY.")
                return

        # 2) Function-style module
        # Try to detect usable functions
        forward_fn = getattr(mod, "forward", None)
        stop_fn = getattr(mod, "stop", None) or getattr(mod, "stop_all", None)
        cleanup_fn = getattr(mod, "cleanup", None)

        # If they don't have forward(), we can emulate it using left/right_forward()
        if forward_fn is None and hasattr(mod, "left_forward") and hasattr(mod, "right_forward"):
            def forward_fn():
                mod.left_forward()
                mod.right_forward()

        # If they don't have cleanup(), we may have GPIO.cleanup
        if cleanup_fn is None and hasattr(mod, "GPIO"):
            def cleanup_fn():
                try:
                    mod.GPIO.cleanup()
                except Exception:
                    pass

        if callable(stop_fn) and callable(forward_fn):
            self._obj = {"forward": forward_fn, "stop": stop_fn, "cleanup": cleanup_fn}
            self._mode = "func"
            self.enabled = True
            print("[MOTORS] Using functions from movement.py")
            # Make sure we start safe
            try:
                stop_fn()
            except Exception:
                pass
            return

        print("[MOTORS] movement.py found but no usable motor interface. Running VISION ONLY.")

    def forward(self):
        if not self.enabled:
            return
        if self._mode == "class":
            self._obj.forward()
        else:
            self._obj["forward"]()

    def stop(self):
        if not self.enabled:
            return
        if self._mode == "class":
            self._obj.stop()
        else:
            self._obj["stop"]()

    def cleanup(self):
        if not self.enabled:
            return
        try:
            if self._mode == "class":
                if hasattr(self._obj, "cleanup"):
                    self._obj.cleanup()
            else:
                if callable(self._obj.get("cleanup")):
                    self._obj["cleanup"]()
        except Exception:
            pass


# -----------------------
# Smooth camera thread
# -----------------------
class CameraThread:
    def __init__(self, index=0, width=640, height=480, fps=60):
        self.cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        self.lock = threading.Lock()
        self.latest = None
        self.running = True
        self.t = threading.Thread(target=self._reader, daemon=True)
        self.t.start()

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            with self.lock:
                self.latest = frame

    def read_latest(self):
        with self.lock:
            if self.latest is None:
                return False, None
            return True, self.latest.copy()

    def release(self):
        self.running = False
        try:
            self.t.join(timeout=1.0)
        except Exception:
            pass
        self.cap.release()


# -----------------------
# Main
# -----------------------
def main():
    # ========= Performance knobs =========
    DISPLAY_W, DISPLAY_H = 640, 480
    YOLO_W, YOLO_H = 320, 240

    # UI will be ~30–60 FPS smooth. YOLO runs at this rate.
    YOLO_HZ = 10  # try 8–15

    # ========= Stop-only-when-close =========
    stopper = ProximityStopper(ProximityStopConfig(
        stop_threshold=0.80,
        resume_threshold=0.65,
        stop_confirm_frames=2,
        resume_confirm_frames=2
    ))

    # ========= Init =========
    motors = MotorAdapter()
    if motors.enabled:
        motors.stop()

    cam = CameraThread(index=0, width=DISPLAY_W, height=DISPLAY_H, fps=60)
    detector = Detector("models/best.pt")

    frame_area = DISPLAY_W * DISPLAY_H

    # Cached inference state
    boxes_scaled = []
    labels_cached = []
    decision = "PROCEED"
    close_cov = 0.0

    # Only send motor updates when state changes
    last_motor_state = None  # "forward" or "stop"

    # UI FPS
    ui_t0 = time.time()
    ui_fps = 0.0

    # YOLO scheduler
    next_yolo_t = 0.0

    try:
        while True:
            ok, frame = cam.read_latest()
            if not ok or frame is None:
                time.sleep(0.01)
                continue

            frame = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))
            now = time.time()

            # -------- YOLO (timed) --------
            if now >= next_yolo_t:
                next_yolo_t = now + (1.0 / float(YOLO_HZ))

                yolo_frame = cv2.resize(frame, (YOLO_W, YOLO_H))

                try:
                    boxes_small, labels, _ = detector.detect_objects(yolo_frame)
                except Exception:
                    boxes_small, _ = detector.detect_objects(yolo_frame)
                    labels = None

                if labels is None:
                    labels = ["obj"] * len(boxes_small)

                sx = DISPLAY_W / YOLO_W
                sy = DISPLAY_H / YOLO_H
                boxes_scaled = [
                    (int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy))
                    for (x1, y1, x2, y2) in boxes_small
                ]
                labels_cached = labels

                decision, close_cov, _ = stopper.update(boxes_scaled, labels_cached, frame_area)

                # -------- Motors (start/stop only) --------
                if motors.enabled:
                    if decision == "STOP":
                        if last_motor_state != "stop":
                            motors.stop()
                            last_motor_state = "stop"
                    else:
                        if last_motor_state != "forward":
                            motors.forward()
                            last_motor_state = "forward"

            # -------- Draw (fast) --------
            for (x1, y1, x2, y2) in boxes_scaled:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # UI FPS
            ui_t1 = time.time()
            dt = ui_t1 - ui_t0
            if dt > 0:
                ui_fps = 0.9 * ui_fps + 0.1 * (1.0 / dt)
            ui_t0 = ui_t1

            cov_pct = close_cov * 100.0
            extra = f"{decision} | close={cov_pct:.0f}% | UI FPS={ui_fps:.0f} | YOLO Hz={YOLO_HZ}"
            if not motors.enabled:
                extra += " | MOTORS=OFF"
            draw_alert(frame, extra)

            cv2.imshow("Psyche Vision Nav", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

    finally:
        # Always safe-stop the rover
        try:
            motors.stop()
            motors.cleanup()
        except Exception:
            pass

        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
