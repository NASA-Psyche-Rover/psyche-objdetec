# pip install opencv-python numpy torch torchvision timm

import cv2
import numpy as np
import time 
import threading 
import queue
import onnxruntime as ort 
from pathlib import Path

# configs
DEPTH_DROP_RATIO  = 1.4
WARNING_ZONE      = 0.25  # middle X% of frame = watch zone (drop appears here)
SURFACE_ZONE      = 0.25  # bottom X% of frame = surface reference (rover's ground)
FRAME_SCALE       = 0.5

print("loading midas")
ONNX_PATH = Path(__file__).parent.parent / "models" / "midas_small.onnx"
print(f"loading MiDaS ONNX from {ONNX_PATH}")
sess       = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
INPUT_NAME = sess.get_inputs()[0].name   # "input"
print("Model ready. Press Q to quit.")

MIDAS_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
MIDAS_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# estimate depth
def estimate_depth(frame):
    """
    Returns normalized depth map (0 to 1) matching frame's original size.
    Higher value = closer to camera (same convention as before).
    
    Changes from PyTorch version:
      - Input resized to 256×256 manually (MiDaS_small's expected input size)
      - Normalized to [0,1] float32 instead of using MiDaS transform
      - ONNX session replaces torch model + interpolate
      - Output resized back to original frame size using OpenCV
    """
    # resize to MiDaS input size and normalize to [0, 1]
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (256, 256)).astype(np.float32) / 255.0
 
    # reshape to 1 × 3 × 256 × 256 (batch × channels × height × width)
    inp   = resized.transpose(2, 0, 1)[np.newaxis, ...]
 
    # run inference — replaces: with torch.no_grad(): pred = model(batch)
    raw   = sess.run(None, {INPUT_NAME: inp})[0].squeeze()
 
    # resize depth map back to match the frame size — replaces torch interpolate
    depth = cv2.resize(raw, (frame.shape[1], frame.shape[0]))
 
    # normalize to [0, 1] — same as before
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
    return depth

# adding threading 
frame_queue = queue.Queue(maxsize=1) # frames being processed 
depth_queue = queue.Queue(maxsize=1) # results being displayed 

# benchmark counters (used in benchmark_pipeline)
total_frames_read = 0   # frames captured by cap.read()
total_frames_dropped = 0    # times queue.Full fired (frames skipped)

# per-inference latency log (ms), kept in memory
inference_latencies: list[float] = []
lat_lock = threading.Lock()

def inference_worker():
    '''Runs own thread; pulls frame pushes depth maps'''
    while True:
        frame = frame_queue.get() # gets frame
        if frame is None: 
            break 
        t0 = time.perf_counter()
        depth = estimate_depth(frame)
        t1 = time.perf_counter()
        inference_latencies.append((t1-t0)*1000)
        if depth_queue.full(): 
            try:
                depth_queue.get_nowait()
            except queue.Empty:
                pass 
        depth_queue.put(depth)

worker=threading.Thread(target=inference_worker, daemon=True) # thread auto die if main prog crashes 
worker.start()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("cannot open webcam.")

#fps count 
prev_time = time.time()
fps = 0.0
alpha = 0.05 # how much weight on prev sample 5% new, 95% old 
last_depth=None 

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        total_frames_read += 1
        small = cv2.resize(frame, (0, 0), fx=FRAME_SCALE, fy=FRAME_SCALE)

        # bfr: depth = estimate_depth(small) -> issue froze loop 200ms for each frame 
        # avoids waiting for inference to finish in main loop 
        try: 
            frame_queue.put_nowait(small)
        except queue.Full:
            total_frames_dropped += 1
        
        try: 
            last_depth=depth_queue.get_nowait()
        except queue.Empty:
            pass
        if last_depth is None:
            continue 
        depth=last_depth

        # terrain risk 
        #depth = estimate_depth(small)
        h, w  = depth.shape

        surf_top = int(h * (1 - SURFACE_ZONE))   # bottom strip = ground under rover
        surf_bot = h
        zone_top = int(h * (0.5 - WARNING_ZONE / 2))  # middle strip = where drop shows
        zone_bot = int(h * (0.5 + WARNING_ZONE / 2))

        surface_depth = np.mean(depth[surf_top:surf_bot, :])
        zone_depth    = np.mean(depth[zone_top:zone_bot, :])

        # ratio inverted — drop means watch zone is much LESS deep, (MiDaS: close surface = high value, open crater = low value)
        ratio = surface_depth / (zone_depth + 1e-6)
        stop  = ratio > DEPTH_DROP_RATIO

        # visual
        depth_vis   = (depth * 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)

        display      = small.copy()
        status_color = (0, 0, 255) if stop else (0, 255, 0)

        # zones in new positions 
        cv2.rectangle(display, (0, surf_top), (w, surf_bot), (255, 200, 0), 1)
        cv2.rectangle(display, (0, zone_top), (w, zone_bot), status_color,  2)
        cv2.putText(display, "surface ref", (4, surf_top - 4),
                    cv2.FONT_HERSHEY_PLAIN, 0.9, (255, 200, 0), 1)
        cv2.putText(display, "watch zone",  (4, zone_top - 4),
                    cv2.FONT_HERSHEY_PLAIN, 0.9, status_color, 1)
        cv2.rectangle(depth_color, (0, surf_top), (w, surf_bot), (255, 200, 0), 1)
        cv2.rectangle(depth_color, (0, zone_top), (w, zone_bot), status_color,  2)

        label = "STOP - DROP AHEAD" if stop else "CLEAR"
        cv2.putText(display, label, (10, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        # FPS 

        curr_time = time.time()
        instant_fps = 1.0 / max(curr_time - prev_time, 1e-6)
        fps = alpha * instant_fps + (1 - alpha) * fps if fps > 0 else instant_fps
        prev_time = curr_time

        cv2.putText(display, f"FPS: {fps:.1f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # live drop-rate HUD 
        if total_frames_read > 0:
            live_drop_pct = total_frames_dropped / total_frames_read * 100
            cv2.putText(display, f"drops: {total_frames_dropped}/{total_frames_read} ({live_drop_pct:.1f}%)",
                        (10, h - 44), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 255), 1)

        cv2.putText(display, f"surface depth: {surface_depth:.3f}", (10, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(display, f"watch depth:   {zone_depth:.3f}",   (10, h - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(display, f"ratio: {ratio:.2f}  (threshold: {DEPTH_DROP_RATIO})",
                    (10, h - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        cv2.imshow("terrain detection", np.hstack([display, depth_color]))

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
except KeyboardInterrupt:
    print("'ctrl c caught - printing summary before exit")

finally:
    # shutdown 
    try:
        frame_queue.put(None)
    except queue.Full:
        worker.join(timeout=10)
        cap.release()
        cv2.destroyAllWindows()

    # benchmark summary 
    print("\n" + "=" * 55)
    print("  SESSION BENCHMARK SUMMARY")
    print("=" * 55)
    print(f"  Frames read          : {total_frames_read}")
    print(f"  Frames dropped       : {total_frames_dropped}")
    if total_frames_read > 0:
        drop_pct = total_frames_dropped / total_frames_read * 100
        print(f"  Drop rate            : {drop_pct:.2f}%")
    
    if inference_latencies:
        lats = np.array(inference_latencies)
        print(f"  Inference samples    : {len(lats)}")
        print(f"  Mean latency         : {np.mean(lats):.1f} ms")
        print(f"  P95 latency          : {np.percentile(lats, 95):.1f} ms")
        print(f"  P99 latency          : {np.percentile(lats, 99):.1f} ms")
    print("=" * 55)