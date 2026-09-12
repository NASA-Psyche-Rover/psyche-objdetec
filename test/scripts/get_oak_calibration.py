"""
Run this ON THE JETSON with the OAK-D-Lite plugged in over USB-C.

Pulls the camera's own factory-calibrated K matrix + distortion coefficients
straight off the device (no manual chessboard calibration needed for this
part -- that's only needed for the LiDAR<->camera EXTRINSICS, see the README)
and saves them so the fusion pipeline can load them.

Usage:
    pip install depthai
    python scripts/get_oak_calibration.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import depthai as dai

from fusion_lib.calibration import save_camera_intrinsics

# Match whatever resolution you actually run YOLOv6 detection at -- the
# intrinsics scale with resolution, so this has to match your live pipeline.
WIDTH, HEIGHT = 640, 480

# On older depthai versions the RGB socket enum is dai.CameraBoardSocket.RGB
# instead of CAM_A. If this throws an AttributeError, swap it.
RGB_SOCKET = dai.CameraBoardSocket.CAM_A


def main():
    with dai.Device() as device:
        calib = device.readCalibration()
        K = calib.getCameraIntrinsics(RGB_SOCKET, WIDTH, HEIGHT)
        dist = calib.getDistortionCoefficients(RGB_SOCKET)

        save_camera_intrinsics("calibration/oak_intrinsics.yaml", K, dist, WIDTH, HEIGHT)
        print("Saved calibration/oak_intrinsics.yaml")
        print("K =", K)
        print("dist =", dist)


if __name__ == "__main__":
    main()
