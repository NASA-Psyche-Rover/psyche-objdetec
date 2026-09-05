"""
Standalone terrain-risk demo/benchmark: webcam -> TerrainAnalyzer -> HUD, with
no object detection or navigation stack involved. Useful for isolating depth
inference performance (FPS, frame-drop rate, inference latency) from the rest
of the pipeline, and for a quick live demo of the risk visualization on its
own.

Depth inference runs in a background thread so the display loop never blocks
on a MiDaS pass -- the main loop always shows the latest available result.

Run from the repo root:
    python scripts/terrain_demo.py
"""

import queue
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.terrain_risk import TerrainAnalyzer

FRAME_SCALE = 0.5


def main():
    analyzer = TerrainAnalyzer()

    frame_queue = queue.Queue(maxsize=1)
    result_queue = queue.Queue(maxsize=1)

    frames_read = 0
    frames_dropped = 0
    latencies = []

    def worker():
        while True:
            frame = frame_queue.get()
            if frame is None:
                break
            t0 = time.perf_counter()
            result = analyzer.get_risk_assessment(frame)
            latencies.append((time.perf_counter() - t0) * 1000)
            if result_queue.full():
                try:
                    result_queue.get_nowait()
                except queue.Empty:
                    pass
            result_queue.put(result)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam.")

    last_result = None
    prev_time = time.time()
    fps = 0.0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames_read += 1
            small = cv2.resize(frame, (0, 0), fx=FRAME_SCALE, fy=FRAME_SCALE)

            try:
                frame_queue.put_nowait(small)
            except queue.Full:
                frames_dropped += 1

            try:
                last_result = result_queue.get_nowait()
            except queue.Empty:
                pass
            if last_result is None:
                continue

            depth_map, status, risk_score, color = last_result
            depth_vis = cv2.applyColorMap((depth_map * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)

            display = small.copy()
            cv2.putText(display, f"{status}  risk={risk_score:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
            prev_time = now
            cv2.putText(display, f"FPS: {fps:.1f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            if frames_read:
                drop_pct = frames_dropped / frames_read * 100
                cv2.putText(display, f"drops: {frames_dropped}/{frames_read} ({drop_pct:.1f}%)",
                            (10, display.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 255), 1)

            cv2.imshow("terrain risk demo", np.hstack([display, depth_vis]))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        frame_queue.put(None)
        thread.join(timeout=5)
        cap.release()
        cv2.destroyAllWindows()

        print("\n" + "=" * 55)
        print("  TERRAIN DEMO BENCHMARK SUMMARY")
        print("=" * 55)
        print(f"  Frames read     : {frames_read}")
        print(f"  Frames dropped  : {frames_dropped}")
        if frames_read:
            print(f"  Drop rate       : {frames_dropped / frames_read * 100:.2f}%")
        if latencies:
            lats = np.array(latencies)
            print(f"  Inference mean  : {lats.mean():.1f} ms")
            print(f"  Inference P95   : {np.percentile(lats, 95):.1f} ms")
        print("=" * 55)


if __name__ == "__main__":
    main()
