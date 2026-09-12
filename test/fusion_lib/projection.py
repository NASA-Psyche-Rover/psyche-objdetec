"""
The actual overlay step: take raw 3D LiDAR points and figure out which pixel
each one lands on in the OAK-D-Lite's image.

cv2.projectPoints does the rotate -> translate -> apply-K -> divide-by-depth
chain in one call, so this file is mostly bookkeeping (filtering points that
are behind the camera or outside the image) around that one call.
"""
import cv2
import numpy as np


def project_lidar_to_image(points_lidar, R, t, K, dist_coeffs, img_w, img_h):
    """
    points_lidar: (N,3) array, in the LiDAR's own frame (meters)
    R, t:         LiDAR -> camera extrinsics from calibration/extrinsics.yaml
    K, dist_coeffs: camera intrinsics from calibration/oak_intrinsics.yaml

    Returns:
      pixels:     (N,2) float, pixel (u,v) each point projects to
                  (garbage/-1 for points filtered out -- check valid_mask)
      valid_mask: (N,) bool, True if the point is in front of the camera
                  AND lands inside the image bounds
      cam_pts:    (N,3) the same points expressed in the camera's frame
                  (useful if you want camera-frame depth instead of range)
    """
    points_lidar = np.asarray(points_lidar, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3)

    rvec, _ = cv2.Rodrigues(R)
    tvec = t.reshape(3, 1)

    # Points behind the camera would otherwise wrap around and "project" onto
    # a nonsense pixel, so check this in the camera frame ourselves first.
    cam_pts = (R @ points_lidar.T).T + t.reshape(1, 3)
    in_front = cam_pts[:, 2] > 0.05  # 5cm buffer past the lens

    pixels = np.full((points_lidar.shape[0], 2), -1.0)
    if in_front.any():
        projected, _ = cv2.projectPoints(
            points_lidar[in_front], rvec, tvec,
            np.asarray(K, dtype=np.float64),
            np.asarray(dist_coeffs, dtype=np.float64),
        )
        pixels[in_front] = projected.reshape(-1, 2)

    valid_mask = (
        in_front
        & (pixels[:, 0] >= 0) & (pixels[:, 0] < img_w)
        & (pixels[:, 1] >= 0) & (pixels[:, 1] < img_h)
    )
    return pixels, valid_mask, cam_pts


def overlay_points_on_image(image, pixels, valid_mask, ranges, max_range=8.0):
    """
    Draws every valid projected point onto the camera image, colored by
    distance. This is the fastest way to sanity-check a calibration -- point
    the rover at something with a known distance and see if the dots line up
    with its actual edges in the picture. If they're offset, recalibrate
    before trusting anything downstream.
    """
    out = image.copy()
    for i in np.where(valid_mask)[0]:
        u, v = int(pixels[i, 0]), int(pixels[i, 1])
        r = float(np.clip(ranges[i] / max_range, 0.0, 1.0))
        # near = red, far = blue (BGR order for OpenCV)
        color = (int(255 * r), 0, int(255 * (1 - r)))
        cv2.circle(out, (u, v), 2, color, -1)
    return out
