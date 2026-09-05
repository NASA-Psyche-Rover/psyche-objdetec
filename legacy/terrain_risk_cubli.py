# LEGACY / REFERENCE ONLY -- not imported by main.py or any src/ module.
#
# Original terrain-risk prototype built for the "cubli" test platform, using
# Depth-Anything-V2 via transformers (GPU-oriented, calibrated for a jumping
# rather than a wheeled rover). Superseded on the rover path by
# src/terrain_risk.py (ONNX Runtime MiDaS-small, CPU-friendly, slope +
# roughness + drop-ratio combined into one risk score). Kept here for
# reference in case the cubli platform work continues separately.

import cv2
import torch
import numpy as np
from PIL import Image
# PIL is python imaging library process
from transformers import pipeline

# terrain analysis class 
class TerrainAnalyzer:
    def __init__(self, model_name="depth-anything/Depth-Anything-V2-Small-hf"):
        print("intializing model")
        # device=0 uses GPU, device=-1 uses CPU
        self.device = 0 if torch.cuda.is_available() else -1
        self.pipe = pipeline(task="depth-estimation", model=model_name, device=self.device)
        # model calibration for cubli 
        self.SLOPE_THRESHOLD=0.15 # max tilt cubli handle
        self.ROUGHNESS_THRESHOLD=20.0 # max terain roughness for it to be able to jump
        print("model loaded")

    def get_risk_assessment(self, frame):
        # covert OpenCV BGR to PIL RGB for the transformer
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # get depth map; 'predicted_depth' = tensor of relative distances
        output = self.pipe(pil_img)
        depth_map = output["predicted_depth"].squeeze().cpu().numpy()
        #normalize 0-1 
        depth_min, depth_max = depth_map.min(), depth_map.max()
        depth_norm = (depth_map - depth_min) / (depth_max - depth_min + 1e-6)
        # extract height, width 
        h, w = depth_norm.shape
        # reigon of interest = bottom 40% of screen 
        roi=depth_norm[int(h*0.6):, int(w*0.2):int(w*0.8)]
        # calculate gradient (slope: tells if cliff or flat)
        grad_x = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
        slope_mag = np.sqrt(grad_x**2 + grad_y**2)
        avg_slope = np.mean(slope_mag)
        roughness = np.std(roi) * 100 # sd = terrain var 
        # cubli logic
        if avg_slope > self.SLOPE_THRESHOLD:
            status="UNSTABLE (SLOPE)"
            color=(0,0,255)
        elif roughness>self.ROUGHNESS_THRESHOLD:
            status="UNSAFE (ROCKY)"
            color=(0,165,255)
        else:
            status="SAFE"
            color=(0,255,0)
        return depth_norm, status, avg_slope, color

# main to run webcam 
if __name__ == "__main__":
    analyzer = TerrainAnalyzer()
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        # run function 
        depth_map, status, slope, color= analyzer.get_risk_assessment(frame)
        # visualizations
        depth_viz = (depth_map * 255).astype(np.uint8)
        depth_viz = cv2.applyColorMap(depth_viz, cv2.COLORMAP_VIRIDIS)
        # roi box on original frame
        h, w, _ = frame.shape
        cv2.rectangle(frame, (int(w*0.2), int(h*0.6)), (int(w*0.8), h), color, 2)
        # slope + status text
        cv2.putText(frame, f"STATUS: {status}", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"Slope: {slope:.3f}", (30, 85), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        display = np.hstack((frame, depth_viz))
        cv2.imshow('terrain risk analysis', display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()