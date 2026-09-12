"""
Calibration I/O for the LiDAR-camera fusion pipeline.

Two things live on disk here:
  - calibration/oak_intrinsics.yaml  -> the OAK-D-Lite's own factory-calibrated
    K matrix + distortion coefficients. Generate this with
    scripts/get_oak_calibration.py while the camera is plugged in.
  - calibration/extrinsics.yaml      -> the LiDAR->camera R and t you solve
    for once with a calibration tool (see README). This is the one that
    silently goes wrong if the mount ever flexes or gets re-printed.
"""
from dataclasses import dataclass
import numpy as np
import yaml


@dataclass
class CameraIntrinsics:
    K: np.ndarray       # (3,3)
    dist: np.ndarray    # (n,) distortion coefficients
    width: int
    height: int


@dataclass
class Extrinsics:
    R: np.ndarray  # (3,3) rotation, LiDAR frame -> camera frame
    t: np.ndarray  # (3,)  translation, LiDAR frame -> camera frame, in meters


def load_camera_intrinsics(path: str) -> CameraIntrinsics:
    with open(path) as f:
        data = yaml.safe_load(f)
    return CameraIntrinsics(
        K=np.array(data["K"], dtype=np.float64).reshape(3, 3),
        dist=np.array(data["dist"], dtype=np.float64),
        width=int(data["width"]),
        height=int(data["height"]),
    )


def save_camera_intrinsics(path: str, K, dist, width: int, height: int) -> None:
    data = {
        "K": np.asarray(K, dtype=np.float64).reshape(-1).tolist(),
        "dist": np.asarray(dist, dtype=np.float64).reshape(-1).tolist(),
        "width": width,
        "height": height,
    }
    with open(path, "w") as f:
        yaml.safe_dump(data, f)


def load_extrinsics(path: str) -> Extrinsics:
    with open(path) as f:
        data = yaml.safe_load(f)
    return Extrinsics(
        R=np.array(data["R"], dtype=np.float64).reshape(3, 3),
        t=np.array(data["t"], dtype=np.float64).reshape(3),
    )


def save_extrinsics(path: str, R, t) -> None:
    data = {
        "R": np.asarray(R, dtype=np.float64).reshape(-1).tolist(),
        "t": np.asarray(t, dtype=np.float64).reshape(-1).tolist(),
    }
    with open(path, "w") as f:
        yaml.safe_dump(data, f)
