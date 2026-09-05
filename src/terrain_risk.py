"""
Terrain-risk analysis: monocular depth (MiDaS-small, ONNX Runtime) -> rover
traversability risk.

This consolidates two earlier prototypes (see legacy/terrain_risk_cubli.py and
scripts/terrain_demo.py) into one module:
  - depth backend: ONNX Runtime instead of the transformers/PyTorch pipeline
    (~10x lower latency on CPU, see quantize_results.json / scripts/quantize_profile.py)
  - risk signal: slope + roughness (near-field ROI) combined with a
    surface-vs-mid-frame drop ratio (ledge/crater ahead), unified into one
    comparable risk score instead of two separate, incompatible metrics.

Once the OAK-D Lite is integrated, its native stereo depth (metric, hardware-
accelerated) should replace _estimate_depth() here entirely -- see README
"Future Implementation" for the migration plan and why the thresholds below
will need re-tuning against absolute (mm) depth.
"""

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "midas_small.onnx"
MIDAS_INPUT_SIZE = (256, 256)  # MiDaS-small's expected input resolution


class TerrainAnalyzer:
    def __init__(
        self,
        model_path=DEFAULT_MODEL_PATH,
        slope_threshold=0.15,
        roughness_threshold=20.0,
        drop_ratio_threshold=1.4,
        roi_top=0.6, roi_left=0.2, roi_right=0.8,  # near-field ROI for slope/roughness
        surface_zone=0.25,                          # bottom band = ground under the rover
        warning_zone=0.25,                          # mid band = where a drop-off first appears
        providers=None,
    ):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"MiDaS ONNX model not found at {model_path}. "
                "Run scripts/quantize_profile.py to export it from torch.hub, "
                "or point model_path at an existing models/midas_small.onnx."
            )
        self.session = ort.InferenceSession(str(model_path), providers=providers or ["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

        self.slope_threshold = slope_threshold
        self.roughness_threshold = roughness_threshold
        self.drop_ratio_threshold = drop_ratio_threshold
        self.roi_top, self.roi_left, self.roi_right = roi_top, roi_left, roi_right
        self.surface_zone = surface_zone
        self.warning_zone = warning_zone

    def _estimate_depth(self, frame):
        """Runs MiDaS-small and returns a depth map normalized to [0, 1], resized
        back to the input frame's size. Higher value = closer to the camera."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, MIDAS_INPUT_SIZE).astype(np.float32) / 255.0
        inp = resized.transpose(2, 0, 1)[np.newaxis, ...]

        raw = self.session.run(None, {self.input_name: inp})[0].squeeze()
        depth = cv2.resize(raw, (frame.shape[1], frame.shape[0]))

        depth_min, depth_max = depth.min(), depth.max()
        return (depth - depth_min) / (depth_max - depth_min + 1e-6)

    def get_risk_assessment(self, frame):
        """
        Returns (depth_map, status, risk_score, color):
          depth_map  : normalized [0, 1] depth map, same size as `frame`
          status     : "SAFE" | "UNSAFE (ROCKY)" | "UNSTABLE (SLOPE)" | "STOP (DROP AHEAD)"
          risk_score : max of three ratios, each = signal / its own threshold, so
                       1.0 means that signal alone has crossed its threshold.
                       Lets should_proceed() use one cutoff regardless of which
                       hazard (slope, roughness, or drop) actually tripped it.
          color      : BGR tuple for HUD drawing
        """
        depth = self._estimate_depth(frame)
        h, w = depth.shape

        # Near-field ROI: bottom 40% of the frame, center 60% width -- the ground
        # just ahead of the rover, where slope/roughness matter most.
        roi = depth[int(h * self.roi_top):, int(w * self.roi_left):int(w * self.roi_right)]
        grad_x = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
        avg_slope = float(np.mean(np.sqrt(grad_x ** 2 + grad_y ** 2)))
        roughness = float(np.std(roi) * 100)

        # Surface band (under the rover) vs. mid-frame band (a bit further out):
        # a sharp drop in "closeness" between the two means a ledge or crater ahead.
        surf_top = int(h * (1 - self.surface_zone))
        zone_top = int(h * (0.5 - self.warning_zone / 2))
        zone_bot = int(h * (0.5 + self.warning_zone / 2))
        surface_depth = float(np.mean(depth[surf_top:, :]))
        watch_depth = float(np.mean(depth[zone_top:zone_bot, :]))
        drop_ratio = surface_depth / (watch_depth + 1e-6)

        r_slope = avg_slope / self.slope_threshold
        r_rough = roughness / self.roughness_threshold
        r_drop = max(0.0, (drop_ratio - 1.0) / (self.drop_ratio_threshold - 1.0))
        risk_score = max(r_slope, r_rough, r_drop)

        if r_drop >= 1.0:
            status, color = "STOP (DROP AHEAD)", (0, 0, 255)
        elif r_slope >= 1.0:
            status, color = "UNSTABLE (SLOPE)", (0, 0, 255)
        elif r_rough >= 1.0:
            status, color = "UNSAFE (ROCKY)", (0, 165, 255)
        else:
            status, color = "SAFE", (0, 255, 0)

        return depth, status, risk_score, color
