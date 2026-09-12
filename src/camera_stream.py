import cv2
import numpy as np


def get_video_source(source=0):
    """
    Returns an OpenCV video capture object.
    Use source=0 for webcam or a file path for a video.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise IOError("Cannot open video source")
    return cap


class OakDLiteCamera:
    """DepthAI v3 pipeline for the OAK-D Lite: RGB frame + stereo depth (mm).

    Uses the unified v3 Camera/StereoDepth API (not the old ColorCamera/
    MonoCamera/XLinkOut v2 nodes) -- validated against real OAK-D Lite hardware.
    `depthai` is imported lazily so the rest of the codebase keeps working
    without it installed.

    Note: this currently only returns RGB + raw stereo depth (mm) for camera
    acquisition. It does NOT yet feed real metric depth into TerrainAnalyzer --
    that swap is a separate, bigger change (see README > Future Implementation)
    since the slope/roughness/drop-ratio thresholds were tuned against MiDaS's
    *relative* depth and need to be re-derived against real millimeter values.
    """

    def __init__(self, rgb_size=(640, 480), fps=30):
        import depthai as dai

        self.pipeline = dai.Pipeline()

        cam_rgb = self.pipeline.create(dai.node.Camera).build(
            dai.CameraBoardSocket.CAM_A, sensorFps=fps
        )

        mono_left = self.pipeline.create(dai.node.Camera).build(
            dai.CameraBoardSocket.CAM_B, sensorFps=fps
        )
        mono_right = self.pipeline.create(dai.node.Camera).build(
            dai.CameraBoardSocket.CAM_C, sensorFps=fps
        )
        stereo = self.pipeline.create(dai.node.StereoDepth).build(
            mono_left.requestOutput((640, 400)), mono_right.requestOutput((640, 400))
        )
        stereo.setLeftRightCheck(True)
        # Short baseline (~7.5cm) on the Lite means poor close-range depth by
        # default -- this roughly halves the minimum usable distance.
        stereo.setExtendedDisparity(True)

        self.rgb_queue = cam_rgb.requestOutput(rgb_size).createOutputQueue(
            maxSize=1, blocking=False
        )
        self.depth_queue = stereo.depth.createOutputQueue(maxSize=1, blocking=False)

        self.pipeline.start()

        # Cache the last valid frames so a single missed tick on either
        # stream doesn't flicker the display between "have data" and "no
        # data" -- RGB and depth aren't guaranteed to land on the exact same
        # tick, so read() falls back to the last known-good frame briefly.
        self._last_rgb = None
        self._last_depth = None

    def read(self):
        """Returns (rgb_frame, depth_frame_mm). Falls back to the last known
        frame for whichever stream isn't ready yet this tick, to avoid
        flicker; both are None only before the very first frames arrive."""
        rgb_pkt = self.rgb_queue.tryGet()
        depth_pkt = self.depth_queue.tryGet()

        if rgb_pkt is not None:
            self._last_rgb = rgb_pkt.getCvFrame()
        if depth_pkt is not None:
            self._last_depth = depth_pkt.getFrame()  # uint16, millimeters

        return self._last_rgb, self._last_depth

    def close(self):
        self.pipeline.stop()


# ---------------------------------------------------------------------------
# OAK-D Lite distance sensing (real metric depth, millimeters)
#
# These are intentionally SEPARATE from src/utils.py's estimate_object_proximity(),
# which expects MiDaS's relative, inverted-and-normalized [0,1] "closeness"
# values (higher = closer). The OAK-D's raw stereo depth is the opposite
# convention: real millimeters, where a LARGER number means farther away.
# Feeding one into a function built for the other will silently produce
# backwards, wrongly-scaled results -- so use these instead once main.py is
# ready to consume real metric distance (see README > Future Implementation
# for the bigger swap of replacing MiDaS terrain risk with this entirely).
# ---------------------------------------------------------------------------

def estimate_object_distances_mm(boxes, depth_frame, frame_size, sample_frac=0.3):
    """
    Real metric distance (millimeters) from the OAK-D Lite to each detected
    2D box, using host-side YOLO boxes (from ultralytics/Detector) against the
    OAK-D's own depth_frame_mm (from OakDLiteCamera.read()).

    Samples a small region around each box's center rather than a single
    pixel -- raw stereo depth has dropout/noise, so one pixel is unreliable --
    and takes the median of valid (nonzero) readings in that region. Returns a
    list of (distance_mm or None) per box, in the same order as `boxes`. None
    means no valid depth was found there (e.g. object outside the stereo
    pair's overlapping field of view, or closer than the Lite's minimum range).

    Box coordinates are in frame_size (w, h) pixels; depth_frame may be a
    different resolution (OAK-D's stereo output, typically 640x400) so
    coordinates are rescaled before sampling.
    """
    if not boxes:
        return []

    frame_w, frame_h = frame_size
    depth_h, depth_w = depth_frame.shape
    distances = []

    for x1, y1, x2, y2 in boxes:
        cx = int((x1 + x2) / 2 * depth_w / frame_w)
        cy = int((y1 + y2) / 2 * depth_h / frame_h)

        box_w = max(int((x2 - x1) * sample_frac * depth_w / frame_w), 2)
        box_h = max(int((y2 - y1) * sample_frac * depth_h / frame_h), 2)

        sx1 = max(cx - box_w // 2, 0)
        sx2 = min(cx + box_w // 2, depth_w - 1)
        sy1 = max(cy - box_h // 2, 0)
        sy2 = min(cy + box_h // 2, depth_h - 1)

        region = depth_frame[sy1:sy2 + 1, sx1:sx2 + 1]
        valid = region[region > 0]
        distances.append(float(np.median(valid)) if valid.size > 0 else None)

    return distances


def nearest_object_distance_m(boxes, depth_frame, frame_size):
    """Convenience wrapper: nearest valid distance across all boxes, in meters.
    Returns None if no box had a valid depth reading anywhere."""
    distances_mm = estimate_object_distances_mm(boxes, depth_frame, frame_size)
    valid = [d for d in distances_mm if d is not None]