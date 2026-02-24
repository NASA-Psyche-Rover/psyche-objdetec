from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional


Box = Tuple[int, int, int, int]


@dataclass
class ProximityStopConfig:
    # Stop if a single object's box covers this fraction of the frame
    stop_threshold: float = 0.80  # try 0.75–0.80
    # Resume when below this fraction (hysteresis to prevent flicker)
    resume_threshold: float = 0.65
    # Require N consecutive "close" frames before stopping
    stop_confirm_frames: int = 3
    # Require N consecutive "not close" frames before resuming
    resume_confirm_frames: int = 3


class ProximityStopper:
    """
    Stops ONLY when a *single* object is "close" (box area ratio high).
    Uses hysteresis + consecutive-frame confirmation to avoid jitter.
    Optionally uses labels to ensure the SAME class is the close object.
    """

    def __init__(self, cfg: ProximityStopConfig = ProximityStopConfig()):
        self.cfg = cfg
        self._close_count = 0
        self._far_count = 0
        self._stopped = False
        self._locked_label: Optional[str] = None

    @staticmethod
    def _box_area(b: Box) -> int:
        x1, y1, x2, y2 = b
        w = max(0, x2 - x1)
        h = max(0, y2 - y1)
        return w * h

    def update(
        self,
        boxes: List[Box],
        labels: Optional[List[str]],
        frame_area: int,
    ) -> Tuple[str, float, Optional[str]]:
        """
        Returns: (decision, max_coverage, max_label)
        decision in {"PROCEED", "STOP"}.
        """

        if frame_area <= 0 or not boxes:
            # No detections -> proceed (and decay state)
            self._close_count = 0
            self._far_count += 1
            if self._stopped and self._far_count >= self.cfg.resume_confirm_frames:
                self._stopped = False
                self._locked_label = None
            return ("STOP" if self._stopped else "PROCEED", 0.0, None)

        # Find the single biggest box (closest proxy)
        areas = [self._box_area(b) for b in boxes]
        i_max = max(range(len(areas)), key=lambda i: areas[i])
        max_cov = areas[i_max] / float(frame_area)
        max_label = labels[i_max] if labels and i_max < len(labels) else None

        # If we already stopped, prefer to stay stopped until clearly far
        if self._stopped:
            if max_cov <= self.cfg.resume_threshold:
                self._far_count += 1
                if self._far_count >= self.cfg.resume_confirm_frames:
                    self._stopped = False
                    self._locked_label = None
                    self._close_count = 0
            else:
                self._far_count = 0
            return ("STOP" if self._stopped else "PROCEED", max_cov, max_label)

        # Not stopped: decide whether to stop
        # "SAME object" proxy: lock onto label once it starts getting close
        if max_cov >= self.cfg.stop_threshold:
            if self._locked_label is None and max_label is not None:
                self._locked_label = max_label

            # If we have a locked label, only stop when the biggest thing matches it
            if self._locked_label is None or max_label == self._locked_label:
                self._close_count += 1
                self._far_count = 0
                if self._close_count >= self.cfg.stop_confirm_frames:
                    self._stopped = True
                    return ("STOP", max_cov, max_label)
            else:
                # different label than locked target -> treat as not confirmed
                self._close_count = 0
                self._far_count += 1
        else:
            self._close_count = 0
            self._far_count += 1
            if self._far_count >= self.cfg.resume_confirm_frames:
                self._locked_label = None

        return ("PROCEED", max_cov, max_label)
