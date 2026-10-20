"""
face_detector.py - MediaPipe Facial Landmark Detection Module

Responsible for initializing MediaPipe FaceLandmarker, verifying/downloading the
model asset, and extracting pixel-space landmark coordinates for eyes and face contours.
"""

import os
import sys
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


@contextmanager
def suppress_c_stderr():
    """Suppresses low-level C++ stderr messages from TensorFlow Lite / glog."""
    try:
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        saved_stderr_fd = os.dup(2)
        os.dup2(devnull_fd, 2)
        os.close(devnull_fd)
        yield
    except Exception:
        yield
    finally:
        try:
            os.dup2(saved_stderr_fd, 2)
            os.close(saved_stderr_fd)
        except Exception:
            pass

from config.settings import (
    MODEL_PATH,
    MODEL_DOWNLOAD_URL,
    RIGHT_EYE_EAR_INDICES,
    LEFT_EYE_EAR_INDICES,
    RIGHT_EYE_CONTOUR,
    LEFT_EYE_CONTOUR,
)


@dataclass
class FaceLandmarksData:
    """Encapsulates extracted facial landmarks for the detected face."""
    detected: bool = False
    pixel_landmarks: List[Tuple[float, float]] = field(default_factory=list)
    right_eye_ear_pts: Optional[np.ndarray] = None      # Shape: (6, 2)
    left_eye_ear_pts: Optional[np.ndarray] = None       # Shape: (6, 2)
    right_eye_contour_pts: Optional[np.ndarray] = None  # Shape: (16, 2)
    left_eye_contour_pts: Optional[np.ndarray] = None   # Shape: (16, 2)


class FaceDetector:
    """Initializes and runs MediaPipe FaceLandmarker on CPU with error handling."""

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = Path(model_path or MODEL_PATH)
        self._ensure_model_exists()
        self.detector = self._initialize_detector()

    def _ensure_model_exists(self) -> None:
        """Verifies the model asset exists, or attempts downloading it."""
        if self.model_path.is_file() and self.model_path.stat().st_size > 100_000:
            return

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Model asset not found at '{self.model_path}'.")
        print(f"Attempting to download from {MODEL_DOWNLOAD_URL}...")

        try:
            urllib.request.urlretrieve(MODEL_DOWNLOAD_URL, str(self.model_path))
            print(f"Model successfully downloaded to '{self.model_path}'.")
        except Exception as e:
            print(
                f"\n[ERROR] Failed to download MediaPipe model asset: {e}\n"
                f"Please manually download the file from:\n"
                f"  {MODEL_DOWNLOAD_URL}\n"
                f"and save it to:\n"
                f"  {self.model_path.resolve()}\n",
                file=sys.stderr,
            )
            raise RuntimeError(f"Missing model asset: {self.model_path}") from e

    def _initialize_detector(self) -> vision.FaceLandmarker:
        """Initializes the MediaPipe FaceLandmarker with CPU delegate."""
        base_options = python.BaseOptions(
            model_asset_path=str(self.model_path)
        )
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
        )
        with suppress_c_stderr():
            return vision.FaceLandmarker.create_from_options(options)

    def detect(self, frame_bgr: np.ndarray) -> FaceLandmarksData:
        """
        Runs face landmark detection on a BGR video frame.

        Args:
            frame_bgr: Current video frame from OpenCV (H, W, 3).

        Returns:
            FaceLandmarksData with extracted landmarks in pixel coordinates.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return FaceLandmarksData(detected=False)

        h, w = frame_bgr.shape[:2]

        try:
            # Convert BGR to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = self.detector.detect(mp_image)
        except Exception as err:
            # Prevent camera loop crash on transient inference failure
            print(f"[WARN] FaceLandmarker inference error: {err}", file=sys.stderr)
            return FaceLandmarksData(detected=False)

        if not result.face_landmarks:
            return FaceLandmarksData(detected=False)

        # Single face detection
        raw_landmarks = result.face_landmarks[0]
        num_pts = len(raw_landmarks)

        # Convert normalized coordinates (0..1) to pixel coordinates (x, y)
        pixel_landmarks = [
            (lm.x * w, lm.y * h) for lm in raw_landmarks
        ]

        def extract_points(indices: List[int]) -> Optional[np.ndarray]:
            if any(idx >= num_pts for idx in indices):
                return None
            return np.array([pixel_landmarks[idx] for idx in indices], dtype=np.float32)

        right_ear_pts = extract_points(RIGHT_EYE_EAR_INDICES)
        left_ear_pts = extract_points(LEFT_EYE_EAR_INDICES)
        right_contour_pts = extract_points(RIGHT_EYE_CONTOUR)
        left_contour_pts = extract_points(LEFT_EYE_CONTOUR)

        return FaceLandmarksData(
            detected=True,
            pixel_landmarks=pixel_landmarks,
            right_eye_ear_pts=right_ear_pts,
            left_eye_ear_pts=left_ear_pts,
            right_eye_contour_pts=right_contour_pts,
            left_eye_contour_pts=left_contour_pts,
        )

    def close(self) -> None:
        """Closes the underlying detector and releases resources."""
        if hasattr(self, "detector") and self.detector:
            try:
                self.detector.close()
            except Exception:
                pass
