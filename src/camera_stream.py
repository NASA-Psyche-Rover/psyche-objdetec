import cv2


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
    """Minimal DepthAI pipeline for the OAK-D Lite: RGB preview + stereo depth.

    Scaffold, not yet validated against hardware. `depthai` is imported lazily
    so the rest of the codebase keeps working without it installed. Once the
    camera is in hand, verify mono camera board sockets, resolution/FPS, and
    depth-to-RGB alignment -- see README > Future Implementation for the
    on-device YOLOv6 migration this also unlocks.
    """

    def __init__(self, rgb_size=(640, 480), fps=30):
        import depthai as dai

        self.pipeline = dai.Pipeline()

        cam_rgb = self.pipeline.create(dai.node.ColorCamera)
        cam_rgb.setPreviewSize(*rgb_size)
        cam_rgb.setInterleaved(False)
        cam_rgb.setFps(fps)

        mono_left = self.pipeline.create(dai.node.MonoCamera)
        mono_right = self.pipeline.create(dai.node.MonoCamera)
        mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
        mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)

        stereo = self.pipeline.create(dai.node.StereoDepth)
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        stereo.setLeftRightCheck(True)
        mono_left.out.link(stereo.left)
        mono_right.out.link(stereo.right)

        xout_rgb = self.pipeline.create(dai.node.XLinkOut)
        xout_rgb.setStreamName("rgb")
        cam_rgb.preview.link(xout_rgb.input)

        xout_depth = self.pipeline.create(dai.node.XLinkOut)
        xout_depth.setStreamName("depth")
        stereo.depth.link(xout_depth.input)

        self.device = dai.Device(self.pipeline)
        self.rgb_queue = self.device.getOutputQueue("rgb", maxSize=1, blocking=False)
        self.depth_queue = self.device.getOutputQueue("depth", maxSize=1, blocking=False)

    def read(self):
        """Returns (rgb_frame, depth_frame_mm). Either may be None if a frame
        isn't ready yet -- this is non-blocking, unlike cv2.VideoCapture.read()."""
        rgb_pkt = self.rgb_queue.tryGet()
        depth_pkt = self.depth_queue.tryGet()
        rgb = rgb_pkt.getCvFrame() if rgb_pkt else None
        depth = depth_pkt.getFrame() if depth_pkt else None  # uint16, millimeters
        return rgb, depth

    def close(self):
        self.device.close()
