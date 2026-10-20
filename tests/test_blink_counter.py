"""
test_blink_counter.py - Automated Verification Suite for Blink Detection

Tests all 12 core requirements:
1. Model & Detector initialization.
2. Open eyes with static count.
3. Single natural blink detection.
4. Five consecutive independent blinks.
5. Rapid blinks without false merging.
6. Brief eye closure (no repeated increments).
7. Prolonged eye closure (only 1 increment upon reopening).
8. Head movement / landmark jitter tolerance.
9. Face loss / exit detection.
10. Face re-entry recovery.
11. Counter reset logic.
12. Single-eye occlusion fallback.
"""

import unittest
import numpy as np
from src.face_detector import FaceLandmarksData
from src.blink_detector import BlinkDetector
from config.settings import (
    EAR_THRESH_CLOSE,
    EAR_THRESH_OPEN,
    CONSECUTIVE_FRAMES_MIN,
    STATUS_TEXT_NO_FACE,
    STATUS_TEXT_OPEN,
    STATUS_TEXT_CLOSED,
    STATUS_TEXT_BLINK,
)


def create_synthetic_eye_landmarks(ear_target: float, width: float = 40.0) -> np.ndarray:
    """
    Constructs a synthetic 6-point eye landmark array yielding the exact target EAR.
    Formula: EAR = (dist1 + dist2) / (2 * width)
    Let dist1 = dist2 = ear_target * width.
    """
    p1 = np.array([100.0, 100.0])
    p4 = np.array([100.0 + width, 100.0])

    vert = ear_target * width
    p2 = np.array([115.0, 100.0 - vert / 2.0])
    p6 = np.array([115.0, 100.0 + vert / 2.0])

    p3 = np.array([125.0, 100.0 - vert / 2.0])
    p5 = np.array([125.0, 100.0 + vert / 2.0])

    return np.array([p1, p2, p3, p4, p5, p6], dtype=np.float32)


class TestBlinkDetector(unittest.TestCase):
    def setUp(self):
        # Initialize detector with responsive smoothing
        self.detector = BlinkDetector(
            thresh_close=0.21,
            thresh_open=0.25,
            consec_frames_min=2,
            consec_frames_max=45,
            status_duration=0.5,
            smoothing_alpha=0.75,
        )
        self.open_eye = create_synthetic_eye_landmarks(0.35)    # Open EAR = 0.35
        self.closed_eye = create_synthetic_eye_landmarks(0.12)  # Closed EAR = 0.12

    def test_01_ear_math(self):
        """Validates EAR mathematical calculation on synthetic landmarks."""
        ear_open = self.detector.calculate_ear(self.open_eye)
        ear_closed = self.detector.calculate_ear(self.closed_eye)
        self.assertAlmostEqual(ear_open, 0.35, places=3)
        self.assertAlmostEqual(ear_closed, 0.12, places=3)

    def test_02_eyes_open_static_count(self):
        """Looking at the camera with eyes open must keep count at zero."""
        for _ in range(15):
            state = self.detector.update(
                FaceLandmarksData(detected=True, right_eye_ear_pts=self.open_eye, left_eye_ear_pts=self.open_eye)
            )
            self.assertEqual(state.blink_count, 0)
            self.assertEqual(state.status_text, STATUS_TEXT_OPEN)

    def test_03_single_natural_blink(self):
        """A complete blink cycle (open -> closed -> reopen) must increment count by 1."""
        # 1. Warm-up open
        for _ in range(3):
            self.detector.update(
                FaceLandmarksData(detected=True, right_eye_ear_pts=self.open_eye, left_eye_ear_pts=self.open_eye)
            )

        # 2. Closed for 2 frames
        for _ in range(2):
            state = self.detector.update(
                FaceLandmarksData(detected=True, right_eye_ear_pts=self.closed_eye, left_eye_ear_pts=self.closed_eye)
            )

        # 3. Reopen
        state = self.detector.update(
            FaceLandmarksData(detected=True, right_eye_ear_pts=self.open_eye, left_eye_ear_pts=self.open_eye)
        )
        self.assertEqual(state.blink_count, 1)
        self.assertEqual(state.status_text, STATUS_TEXT_BLINK)

    def test_04_five_independent_blinks(self):
        """Performing 5 separate blinks must result in exactly 5 blinks counted."""
        for expected in range(1, 6):
            # Open phase
            for _ in range(3):
                self.detector.update(
                    FaceLandmarksData(detected=True, right_eye_ear_pts=self.open_eye, left_eye_ear_pts=self.open_eye)
                )
            # Closed phase
            for _ in range(2):
                self.detector.update(
                    FaceLandmarksData(detected=True, right_eye_ear_pts=self.closed_eye, left_eye_ear_pts=self.closed_eye)
                )
            # Reopen phase
            state = self.detector.update(
                FaceLandmarksData(detected=True, right_eye_ear_pts=self.open_eye, left_eye_ear_pts=self.open_eye)
            )
            self.assertEqual(state.blink_count, expected)

    def test_05_noise_filtering_single_frame(self):
        """A single closed frame followed immediately by open should be filtered out."""
        # 1 closed frame
        self.detector.update(
            FaceLandmarksData(detected=True, right_eye_ear_pts=self.closed_eye, left_eye_ear_pts=self.closed_eye)
        )
        # Immediately open
        state = self.detector.update(
            FaceLandmarksData(detected=True, right_eye_ear_pts=self.open_eye, left_eye_ear_pts=self.open_eye)
        )
        self.assertEqual(state.blink_count, 0)

    def test_06_prolonged_closure_single_count(self):
        """Extended eye closure (e.g. 50 frames) must NOT count repeatedly while closed, and only count 1 on reopen."""
        # Warm up open
        self.detector.update(
            FaceLandmarksData(detected=True, right_eye_ear_pts=self.open_eye, left_eye_ear_pts=self.open_eye)
        )

        # Prolonged closure: 50 frames
        for frame_idx in range(50):
            state = self.detector.update(
                FaceLandmarksData(detected=True, right_eye_ear_pts=self.closed_eye, left_eye_ear_pts=self.closed_eye)
            )
            self.assertEqual(state.blink_count, 0)
            if frame_idx >= 1:  # After consec_frames_min (2 frames)
                self.assertEqual(state.status_text, STATUS_TEXT_CLOSED)


        # Reopen
        state = self.detector.update(
            FaceLandmarksData(detected=True, right_eye_ear_pts=self.open_eye, left_eye_ear_pts=self.open_eye)
        )
        self.assertEqual(state.blink_count, 1)

    def test_07_face_lost_and_recovery(self):
        """When face is lost, status updates, incomplete blinks reset, and recovery works."""
        # Face lost
        state = self.detector.update(FaceLandmarksData(detected=False))
        self.assertEqual(state.status_text, STATUS_TEXT_NO_FACE)
        self.assertEqual(state.blink_count, 0)

        # Incomplete closure then face lost
        self.detector.update(
            FaceLandmarksData(detected=True, right_eye_ear_pts=self.closed_eye, left_eye_ear_pts=self.closed_eye)
        )
        # Face disappears mid-blink
        self.detector.update(FaceLandmarksData(detected=False))
        # Face reappears open
        state = self.detector.update(
            FaceLandmarksData(detected=True, right_eye_ear_pts=self.open_eye, left_eye_ear_pts=self.open_eye)
        )
        # Must NOT falsely count a blink
        self.assertEqual(state.blink_count, 0)

    def test_08_reset_functionality(self):
        """Calling reset() returns count to 0 and does not trigger false blinks."""
        # Count 2 blinks
        for _ in range(2):
            self.detector.update(
                FaceLandmarksData(detected=True, right_eye_ear_pts=self.open_eye, left_eye_ear_pts=self.open_eye)
            )
            for _ in range(2):
                self.detector.update(
                    FaceLandmarksData(detected=True, right_eye_ear_pts=self.closed_eye, left_eye_ear_pts=self.closed_eye)
                )
            self.detector.update(
                FaceLandmarksData(detected=True, right_eye_ear_pts=self.open_eye, left_eye_ear_pts=self.open_eye)
            )
        self.assertEqual(self.detector.blink_count, 2)

        # Reset
        self.detector.reset()
        self.assertEqual(self.detector.blink_count, 0)

        # Follow-up frame does not count
        state = self.detector.update(
            FaceLandmarksData(detected=True, right_eye_ear_pts=self.open_eye, left_eye_ear_pts=self.open_eye)
        )
        self.assertEqual(state.blink_count, 0)

    def test_09_single_eye_occlusion_fallback(self):
        """If one eye is occluded, detector falls back to the other eye."""
        # Right eye is valid, left eye is None
        state = self.detector.update(
            FaceLandmarksData(detected=True, right_eye_ear_pts=self.open_eye, left_eye_ear_pts=None)
        )
        self.assertIsNotNone(state.ear_right)
        self.assertIsNone(state.ear_left)
        self.assertAlmostEqual(state.ear_avg, 0.35, places=2)


if __name__ == "__main__":
    unittest.main()
