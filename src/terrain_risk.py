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
        print("model loaded")

    def get_risk_assessment(self, frame):
        # covert OpenCV BGR to PIL RGB for the transformer
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # get depth map; 'predicted_depth' = tensor of relative distances
        output = self.pipe(pil_img)
        depth_map = output["predicted_depth"].squeeze().cpu().numpy()

        # extract height, width 
        h, w = depth_map.shape
        
        # find closeness + max, input for mean is region of interest
        avg_closeness = np.mean(depth_map[int(h * 0.7):, :])
        max_possible = np.max(depth_map)
        
        # risk threshold: avg > 80% of max distance
        risk_score = (avg_closeness/max_possible) if max_possible > 0 else 0
        status = "CRITICAL" if risk_score > 0.8 else "SAFE"
        return depth_map, status, risk_score

# main to run webcam 
if __name__ == "__main__":
    analyzer = TerrainAnalyzer()
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        depth_map, status, score = analyzer.get_risk_assessment(frame)

        # visualization; normalize depth for heatmap
        depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        depth_viz = cv2.applyColorMap(depth_norm, cv2.COLORMAP_MAGMA)

        color = (0, 0, 255) if status == "CRITICAL" else (0, 255, 0)
        cv2.putText(frame, f"STATUS: {status} ({score:.2f})", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

        # split screen view 
        display = np.hstack((frame, depth_viz))
        cv2.imshow('Psyche Rover: Terrain Risk Detection', display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()