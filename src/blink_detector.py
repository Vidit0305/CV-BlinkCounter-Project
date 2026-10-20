"""
blink_detector.py - Eye Aspect Ratio (EAR) Calculation and Temporal Blink Logic

Implements the 6-point Eye Aspect Ratio calculation, dual-threshold hysteresis,
temporal confirmation, smoothing, and state tracking for accurate blink counting.
"""

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from config.settings import (
    EAR_THRESH_CLOSE,
    EAR_THRESH_OPEN,
    CONSECUTIVE_FRAMES_MIN,
    CONSECUTIVE_FRAMES_MAX,
    BLINK_STATUS_DURATION_SEC,
    EAR_SMOOTHING_FACTOR,
    STATUS_TEXT_NO_FACE,
    STATUS_TEXT_OPEN,
    STATUS_TEXT_CLOSED,
    STATUS_TEXT_BLINK,
)
from src.face_detector import FaceLandmarksData


@dataclass
class BlinkState:
    """Current state and metrics of the eye blink detector."""
    blink_count: int
    status_text: str
    is_eyes_closed: bool
    ear_left: Optional[float]
    ear_right: Optional[float]
    ear_avg: Optional[float]
    ear_smoothed: Optional[float]
    blink_detected_this_frame: bool


class BlinkDetector:
    """
    Computes Eye Aspect Ratio (EAR) and manages the blink detection state machine.

    Temporal logic guarantees:
    1. A blink requires a verified transition from OPEN -> CLOSED (for >= CONSECUTIVE_FRAMES_MIN) -> OPEN.
    2. Prolonged closure (e.g. resting eyes or sleeping) increments only 1 blink when reopened.
    3. Single closed frames (sensor noise or occlusion) are filtered out.
    4. Lost face tracking safely resets incomplete blink states without duplicate counting.
    """

    def __init__(
        self,
        thresh_close: float = EAR_THRESH_CLOSE,
        thresh_open: float = EAR_THRESH_OPEN,
        consec_frames_min: int = CONSECUTIVE_FRAMES_MIN,
        consec_frames_max: int = CONSECUTIVE_FRAMES_MAX,
        status_duration: float = BLINK_STATUS_DURATION_SEC,
        smoothing_alpha: float = EAR_SMOOTHING_FACTOR,
    ):
        self.thresh_close = thresh_close
        self.thresh_open = thresh_open
        self.consec_frames_min = consec_frames_min
        self.consec_frames_max = consec_frames_max
        self.status_duration = status_duration
        self.smoothing_alpha = smoothing_alpha

        # Internal state counters
        self.blink_count: int = 0
        self.consecutive_closed_frames: int = 0
        self.blink_confirmed_pending: bool = False
        self.is_currently_closed: bool = False
        self.smoothed_ear: Optional[float] = None
        self.last_blink_timestamp: float = 0.0

    @staticmethod
    def calculate_ear(eye_pts: Optional[np.ndarray]) -> Optional[float]:
        """
        Calculates Eye Aspect Ratio (EAR) from 6 landmark points.

        Formula:
            EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)

        Args:
            eye_pts: (6, 2) array of landmark coordinates [p1, p2, p3, p4, p5, p6].

        Returns:
            EAR float value or None if points are invalid/degenerate.
        """
        if eye_pts is None or len(eye_pts) != 6:
            return None

        # p1: outer corner, p4: inner corner
        p1, p2, p3, p4, p5, p6 = eye_pts

        # Vertical distances between upper and lower eyelids
        dist_vert_1 = np.linalg.norm(p2 - p6)
        dist_vert_2 = np.linalg.norm(p3 - p5)

        # Horizontal distance between eye corners
        dist_horiz = np.linalg.norm(p1 - p4)

        if dist_horiz < 1e-6:
            # Degenerate eye points (e.g. extreme angle or bad fit)
            return None

        ear = float((dist_vert_1 + dist_vert_2) / (2.0 * dist_horiz))
        return ear

    def _compute_combined_ear(
        self, ear_left: Optional[float], ear_right: Optional[float]
    ) -> Optional[float]:
        """
        Combines left and right eye EARs.
        If both eyes are visible, takes the arithmetic mean.
        If one eye is occluded, gracefully falls back to the single available eye.
        """
        if ear_left is not None and ear_right is not None:
            return (ear_left + ear_right) / 2.0
        elif ear_left is not None:
            return ear_left
        elif ear_right is not None:
            return ear_right
        return None

    def update(self, face_data: FaceLandmarksData) -> BlinkState:
        """
        Updates the blink detection state machine with the latest frame's landmarks.

        Args:
            face_data: Detected landmark coordinates from FaceDetector.

        Returns:
            BlinkState with updated count, status message, and metrics.
        """
        now = time.time()
        blink_this_frame = False

        # Case 1: Face not detected in frame
        if not face_data.detected:
            # Safely reset incomplete closure to avoid false blinks on face re-entry
            self.consecutive_closed_frames = 0
            self.blink_confirmed_pending = False
            self.is_currently_closed = False
            self.smoothed_ear = None

            return BlinkState(
                blink_count=self.blink_count,
                status_text=STATUS_TEXT_NO_FACE,
                is_eyes_closed=False,
                ear_left=None,
                ear_right=None,
                ear_avg=None,
                ear_smoothed=None,
                blink_detected_this_frame=False,
            )

        # Calculate EAR for both eyes
        ear_right = self.calculate_ear(face_data.right_eye_ear_pts)
        ear_left = self.calculate_ear(face_data.left_eye_ear_pts)
        ear_avg = self._compute_combined_ear(ear_left, ear_right)

        if ear_avg is None:
            # Face was detected but eye landmarks were invalid or degenerate
            return BlinkState(
                blink_count=self.blink_count,
                status_text="Status: Eye Landmarks Unreliable",
                is_eyes_closed=self.is_currently_closed,
                ear_left=ear_left,
                ear_right=ear_right,
                ear_avg=None,
                ear_smoothed=self.smoothed_ear,
                blink_detected_this_frame=False,
            )

        # Apply Exponential Moving Average (EMA) smoothing
        if self.smoothed_ear is None:
            self.smoothed_ear = ear_avg
        else:
            self.smoothed_ear = (
                self.smoothing_alpha * ear_avg
                + (1.0 - self.smoothing_alpha) * self.smoothed_ear
            )

        metric_ear = self.smoothed_ear

        # ======================================================================
        # State Machine: Open / Closed Transitions with Hysteresis
        # ======================================================================

        if metric_ear < self.thresh_close:
            # Eyes are currently below the closure threshold
            self.consecutive_closed_frames += 1
            self.is_currently_closed = True

            # Confirm valid closure once minimum consecutive frame requirement is met
            if self.consecutive_closed_frames >= self.consec_frames_min:
                self.blink_confirmed_pending = True

        elif metric_ear >= self.thresh_open:
            # Eyes have reopened above the opening hysteresis threshold
            if self.blink_confirmed_pending:
                # Valid completed blink cycle: OPEN -> CLOSED -> OPEN!
                self.blink_count += 1
                self.last_blink_timestamp = now
                blink_this_frame = True
                self.blink_confirmed_pending = False

            # Reset closure tracking
            self.consecutive_closed_frames = 0
            self.is_currently_closed = False

        else:
            # Metric is in the hysteresis deadband [thresh_close, thresh_open]
            # Preserve current state (prevents jitter near threshold)
            if self.is_currently_closed:
                self.consecutive_closed_frames += 1

        # ======================================================================
        # Status Message Resolution
        # ======================================================================
        if self.is_currently_closed and self.consecutive_closed_frames >= self.consec_frames_min:
            status_text = STATUS_TEXT_CLOSED
        elif (now - self.last_blink_timestamp) < self.status_duration:
            status_text = STATUS_TEXT_BLINK
        else:
            status_text = STATUS_TEXT_OPEN

        return BlinkState(
            blink_count=self.blink_count,
            status_text=status_text,
            is_eyes_closed=self.is_currently_closed,
            ear_left=ear_left,
            ear_right=ear_right,
            ear_avg=ear_avg,
            ear_smoothed=self.smoothed_ear,
            blink_detected_this_frame=blink_this_frame,
        )

    def reset(self) -> None:
        """
        Resets the session blink counter to zero and clears all state machines.
        Guarantees that resetting never accidentally counts a blink.
        """
        self.blink_count = 0
        self.consecutive_closed_frames = 0
        self.blink_confirmed_pending = False
        self.is_currently_closed = False
        self.smoothed_ear = None
        self.last_blink_timestamp = 0.0
