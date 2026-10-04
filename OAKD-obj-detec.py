"""
OAK-D Lite rover autonomy: rock detection + metric distance + terrain drops.

The camera sits low on the rover, pitched slightly down at the ground ahead.
Everything is measured with the OAK-D Lite's own stereo depth (aligned to the
RGB image), as real distances along the ground, inside the corridor the rover
will actually sweep:

  Objects   YOLO boxes (median stereo depth inside the box) and blobs
            standing above the ground, whether or not YOLO knows what they are
              nearest < STOP_DISTANCE_M     -> STOP
              nearest < CAUTION_DISTANCE_M  -> CAUTION

  Terrain   where the ground ahead stops: a drop (table edge, crater rim), a
            slope too steep to drive down, or ground that returns no depth
              edge < TERRAIN_STOP_DISTANCE_M     -> STOP
              edge < TERRAIN_CAUTION_DISTANCE_M  -> CAUTION

  otherwise -> PROCEED

Distance is the only thing that decides. An object across the room is boxed
and labelled with its distance but cannot stop the rover. There is no
cluster-density term: how much of the frame boxes cover is not a distance.

Display (one window): camera view with boxes + distances on the left, with
the drop/slope edge drawn on the ground and named ("DROP AHEAD"); on the
right the metric distance heatmap (fixed scale, meters) above a status panel.

Threads: the main loop only shows camera frames, so the feed stays at camera
rate. Detection + distance + decision + heatmap run in a background thread
that always works on the newest frame; the display draws its latest results.

Run from the repo root:   python OAKD-obj-detec.py
  --snapshot PATH   also write the display to PATH about once a second
  --no-display      run without opening a window (use with --snapshot)
Keys: q quits.

Requires the OAK-D Lite on USB. See src/metric_terrain.py for the geometry
and src/decision.py (decide_metric, DecisionLatch) for the decision rule.
"""

import os

# Must be set before numpy is imported. Left alone, the BLAS library behind
# numpy starts a busy-waiting thread per core, which fights the display loop
# and the detector for CPU -- measured 2-5x slower perception on a 6-core
# Jetson. The arrays here are small; one thread is faster.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import threading
import time
import traceback

import cv2
import numpy as np
import torch

from src import metric_terrain as mt
from src.camera_stream import OakDLiteCamera
from src.decision import (
    CAUTION_DISTANCE_M, STOP_DISTANCE_M, TERRAIN_CAUTION_DISTANCE_M, TERRAIN_NAMES,
    TERRAIN_STOP_DISTANCE_M, DecisionLatch, decide_metric,
)
from src.detection import ROCK_MODEL_PATHS, ObjectDetector

# ---- Config ----
TARGET_WIDTH, TARGET_HEIGHT = 640, 480
ROCK_CONF = 0.25    # rocks are low-contrast against regolith; favour recall,
                    # the geometric check backs up anything this lets through
PANEL_W = 320       # width of the right-hand heatmap/status column

DECISION_COLORS = {"STOP": (0, 0, 255), "CAUTION": (0, 200, 255), "PROCEED": (0, 255, 0)}


def distance_color(dist_m):
    if dist_m is None:
        return (200, 200, 200)
    if dist_m < STOP_DISTANCE_M:
        return DECISION_COLORS["STOP"]
    if dist_m < CAUTION_DISTANCE_M:
        return DECISION_COLORS["CAUTION"]
    return DECISION_COLORS["PROCEED"]


def put(img, text, org, scale, color, thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def terrain_color(terrain):
    if terrain.hazard_kind is None:
        return DECISION_COLORS["PROCEED"]
    return DECISION_COLORS["STOP" if terrain.hazard_m < TERRAIN_STOP_DISTANCE_M else "CAUTION"]


def status_panel(size, decision, reason, hazard_m, hazard_source, terrain, items, rates, detector_note):
    """Text panel under the heatmap: the decision, what caused it, the
    terrain ahead, and every tracked thing with its distance -- so a STOP can
    always be traced to a specific distance on screen."""
    w, h = size
    panel = np.full((h, w, 3), 25, np.uint8)
    color = DECISION_COLORS[decision]
    cv2.rectangle(panel, (0, 0), (w, 52), color, -1)
    put(panel, decision, (10, 26), 0.8, (0, 0, 0), 2)
    put(panel, reason, (10, 45), 0.45, (0, 0, 0))

    y = 72
    if not terrain.ground_found:
        put(panel, "Terrain: ground not found", (10, y), 0.45, DECISION_COLORS["CAUTION"])
    elif terrain.hazard_kind is None:
        put(panel, f"Terrain: clear (camera {terrain.camera_height_m:.2f} m up)", (10, y), 0.42,
            DECISION_COLORS["PROCEED"])
    else:
        put(panel, f"Terrain: {TERRAIN_NAMES[terrain.hazard_kind]} {terrain.hazard_m:.2f} m", (10, y), 0.45,
            terrain_color(terrain))
    y += 20
    put(panel, f"Nearest in path: {hazard_source} {hazard_m:.2f} m" if hazard_m is not None
        else "Nearest in path: no objects", (10, y), 0.42, (255, 255, 255))
    y += 22
    put(panel, f"{detector_note}:", (10, y), 0.4, (170, 170, 170))
    for label, dist, in_path in items[:4]:
        y += 18
        text = f"{label}  {dist:.2f} m" if dist is not None else f"{label}  no depth"
        put(panel, text + ("" if in_path else "  (not in path)"), (18, y), 0.42,
            distance_color(dist) if in_path else (150, 150, 150))
    put(panel, f"stop: object < {STOP_DISTANCE_M:.2f} m, drop < {TERRAIN_STOP_DISTANCE_M:.2f} m",
        (10, h - 26), 0.36, (150, 150, 150))
    fps, detect_s = rates
    put(panel, f"video {fps:.0f} fps   detection every {detect_s * 1000:.0f} ms",
        (10, h - 10), 0.36, (150, 150, 150))
    return panel


class Worker(threading.Thread):
    """Background loop that always processes the newest input and publishes
    its latest output. Input and output are single attributes swapped by
    plain assignment, so the display thread never waits on a worker and a
    worker never queues up stale frames."""

    def __init__(self, step):
        super().__init__(daemon=True)
        self._step = step
        self._input = None
        self.output = None
        self.failed = False
        self.step_s = 0.0   # smoothed time per step
        self._stop = threading.Event()

    def submit(self, item):
        self._input = item

    def stop(self):
        self._stop.set()

    def run(self):
        last = None
        while not self._stop.is_set():
            item = self._input
            if item is None or item is last:
                time.sleep(0.002)
                continue
            last = item
            try:
                t = time.time()
                self.output = self._step(item)
                took = time.time() - t
                self.step_s = 0.8 * self.step_s + 0.2 * took if self.step_s else took
            except Exception:
                traceback.print_exc()
                self.failed = True
                return


class Perception:
    """One perception step: stereo depth -> geometry, YOLO -> boxes with
    distances, nearest thing in the path -> decision."""

    def __init__(self, detector, K, heatmap_size):
        self.detector, self.K, self.heatmap_size = detector, K, heatmap_size
        self.heatmap = None
        self.latch = DecisionLatch()
        self.ground = mt.GroundTracker()
        self.depth_m = None
        self.shown_hazard_m = None

    def __call__(self, item):
        frame, depth_mm = item
        K = self.K
        self.depth_m = depth_m = mt.clean_depth(depth_mm, self.depth_m)
        terrain = mt.analyze(depth_m, K, self.ground)

        rocks = []   # (xyxy, label, distance_m or None, in_corridor)
        for det in self.detector.detect(frame):
            dist, in_corridor = mt.box_distance(depth_m, K, det.xyxy, terrain)
            rocks.append((det.xyxy, self.detector.class_name(det.class_id), dist, in_corridor))

        # Nearest thing in the path, from any source
        candidates = [(d, label) for _, label, d, in_corr in rocks if d is not None and in_corr]
        if terrain.ahead_m is not None:
            candidates.append((terrain.ahead_m, "straight ahead"))
        if terrain.nearest_obstacle_m is not None:
            candidates.append((terrain.nearest_obstacle_m, "obstacle"))
        hazard_m, hazard_source = min(candidates) if candidates else (None, None)

        decision, reason = decide_metric(hazard_m, hazard_source, terrain.hazard_kind, terrain.hazard_m)
        decision, reason = self.latch.update(decision, reason, hazard_m, terrain.ahead_valid_frac)

        # Steady the displayed number (the decision above uses the raw one).
        shown = self.shown_hazard_m
        if hazard_m is None:
            shown = None
        elif shown is None or abs(hazard_m - shown) > 0.25 * shown:
            shown = hazard_m
        else:
            shown = 0.8 * shown + 0.2 * hazard_m
        self.shown_hazard_m = shown

        self.heatmap = mt.depth_heatmap(depth_m, self.heatmap_size)

        return dict(terrain=terrain, rocks=rocks, decision=decision, reason=reason,
                    hazard_m=shown, hazard_source=hazard_source, heatmap=self.heatmap)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--snapshot", help="write the display to this image file about once a second")
    parser.add_argument("--no-display", action="store_true", help="don't open a window")
    args = parser.parse_args()

    torch.set_num_threads(2)
    cv2.setNumThreads(2)

    try:
        oak_cam = OakDLiteCamera(rgb_size=(TARGET_WIDTH, TARGET_HEIGHT), fps=30)
    except Exception as e:
        print(f"ERROR: OAK-D Lite not available: {e}")
        print("This needs the OAK-D Lite connected via USB-C -- exiting.")
        return
    K = oak_cam.K
    print(f"OAK-D Lite open. Objects: STOP < {STOP_DISTANCE_M:.2f} m, CAUTION < {CAUTION_DISTANCE_M:.2f} m. "
          f"Drops/slopes: STOP < {TERRAIN_STOP_DISTANCE_M:.2f} m, CAUTION < {TERRAIN_CAUTION_DISTANCE_M:.2f} m. "
          f"Corridor +/-{mt.CORRIDOR_HALF_WIDTH_M:.2f} m")

    detector = ObjectDetector(ROCK_MODEL_PATHS, conf=ROCK_CONF)
    if detector.is_rock_model:
        detector_note = "Detected rocks"
        print(f"Rock detector: {detector.model_path}")
    else:
        detector_note = "Detected objects"
        print("WARNING: no rock-trained weights found -- YOLO is running generic COCO classes, "
              "which do not include rocks. Rocks are still caught by the geometric obstacle "
              "check. See scripts/build_rock_model.py.")

    if not args.no_display:
        cv2.namedWindow("Rover autonomy (OAK-D Lite)")

    panel_h = TARGET_HEIGHT // 2
    perception = Worker(Perception(detector, K, (PANEL_W, panel_h)))
    perception.start()

    last_frame = None
    t0 = time.time()
    last_snapshot = 0.0
    frame_s = 1 / 30

    while not perception.failed:
        frame, depth_mm = oak_cam.read()
        if frame is None or depth_mm is None or frame is last_frame:
            # No new camera frame yet (read() hands back the cached one).
            # Sleeping a few ms rather than spinning leaves the interpreter
            # to the perception thread between frames.
            time.sleep(0.004)
            continue
        last_frame = frame

        perception.submit((frame, depth_mm))
        result = perception.output
        if result is None:
            continue   # first perception pass still running

        terrain, rocks, decision = result["terrain"], result["rocks"], result["decision"]

        # -------- Drawing (on a copy: the workers are still reading `frame`) --------
        view = frame.copy()
        for ob in terrain.obstacles:
            if not ob.in_corridor:
                continue
            x1, y1, x2, y2 = ob.xyxy
            c = distance_color(ob.distance_m)
            cv2.rectangle(view, (x1, y1), (x2, y2), c, 1, cv2.LINE_AA)
            put(view, f"obstacle {ob.distance_m:.2f} m", (x1, min(y2 + 14, TARGET_HEIGHT - 4)), 0.4, c)

        if terrain.hazard_kind is not None:
            c = terrain_color(terrain)
            line = mt.edge_line_px(terrain, K, terrain.hazard_m)
            if line is not None:
                cv2.line(view, line[0], line[1], c, 3, cv2.LINE_AA)
            text = f"{TERRAIN_NAMES[terrain.hazard_kind]}  {terrain.hazard_m:.2f} m"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            x = (TARGET_WIDTH - tw) // 2
            cv2.rectangle(view, (x - 10, 12), (x + tw + 10, 24 + th + 8), (0, 0, 0), -1)
            put(view, text, (x, 22 + th), 0.8, c, 2)

        for (x1, y1, x2, y2), label, dist, in_corridor in rocks:
            c = distance_color(dist) if in_corridor else (200, 200, 200)
            cv2.rectangle(view, (x1, y1), (x2, y2), c, 2, cv2.LINE_AA)
            text = f"{label} {dist:.2f} m" if dist is not None else f"{label} (no depth)"
            put(view, text, (x1, max(y1 - 8, 16)), 0.5, c)

        t1 = time.time()
        frame_s = 0.9 * frame_s + 0.1 * (t1 - t0)   # average the interval, not its inverse
        t0 = t1
        fps = 1.0 / frame_s

        items = sorted(((label, d, in_corr) for _, label, d, in_corr in rocks),
                       key=lambda it: (it[1] is None, it[1]))
        panel = status_panel((PANEL_W, TARGET_HEIGHT - panel_h), decision, result["reason"],
                             result["hazard_m"], result["hazard_source"], terrain, items,
                             (fps, perception.step_s), detector_note)
        cv2.rectangle(view, (0, 0), (TARGET_WIDTH - 1, TARGET_HEIGHT - 1), DECISION_COLORS[decision], 3)
        display = np.hstack([view, np.vstack([result["heatmap"], panel])])

        if args.snapshot and t1 - last_snapshot > 1.0:
            cv2.imwrite(args.snapshot, display)
            last_snapshot = t1

        if not args.no_display:
            cv2.imshow("Rover autonomy (OAK-D Lite)", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    perception.stop()
    perception.join(timeout=2)
    oak_cam.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
