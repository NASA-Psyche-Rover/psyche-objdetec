# pip install opencv-python numpy torch torchvision timm

import cv2
import torch
import numpy as np
import time 
import threading 
import queue

# configs
DEPTH_DROP_RATIO  = 1.4
WARNING_ZONE      = 0.25  # middle X% of frame = watch zone (drop appears here)
SURFACE_ZONE      = 0.25  # bottom X% of frame = surface reference (rover's ground)
FRAME_SCALE       = 0.5

print("loading midas")
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model     = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
model.to(device).eval()
transform = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True).small_transform
print(f"Model ready on {device}. Press Q to quit.")

# estimate depth
def estimate_depth(frame):
    """returns normalized depth (0 to 1). Higher = farther away."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    batch = transform(rgb).to(device)
    with torch.no_grad():
        pred = model(batch)
        pred = torch.nn.functional.interpolate(pred.unsqueeze(1), size=frame.shape[:2], mode="bicubic", align_corners=False,).squeeze()
    depth = pred.cpu().numpy().astype(np.float32)
    # MiDaS closer=higher, further=lower (watch zone < current surface)
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
    return depth

# adding threading 
frame_queue = queue.Queue(maxsize=1) # frames being processed 
depth_queue = queue.Queue(maxsize=1) # results being displayed 

def inference_worker():
    '''Runs own thread; pulls frame pushes depth maps'''
    while True:
        frame = frame_queue.get() # gets frame
        if frame is None: 
            break 
        depth = estimate_depth(frame)
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
alpha = 0.05 # how much weight on prev sample
last_depth=None 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    small = cv2.resize(frame, (0, 0), fx=FRAME_SCALE, fy=FRAME_SCALE)

    # bfr: depth = estimate_depth(small) -> issue froze loop 200ms for each frame 
    # avoids waiting for inference to finish in main loop 
    try: 
        frame_queue.put_nowait(small)
    except queue.Full:
        pass
    
    try: 
        last_depth=depth_queue.get_nowait()
    except queue.Empty:
        pass
    if last_depth is None:
        continue 
    depth=last_depth
        
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
    
    curr_time = time.time()
    instant_fps = 1.0 / max(curr_time - prev_time, 1e-6)
    fps = alpha * fps + (1 - alpha) * instant_fps if fps > 0 else instant_fps
    prev_time = curr_time

    cv2.putText(display, f"FPS: {fps:.1f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

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
frame_queue.put(None)
worker.join(timeout=2)
cap.release()
cv2.destroyAllWindows()