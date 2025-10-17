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