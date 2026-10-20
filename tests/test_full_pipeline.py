"""
test_full_pipeline.py - Integration Testing for Blink Counter Components
"""

import unittest
import numpy as np
import cv2
from src.face_detector import FaceDetector, FaceLandmarksData
from src.blink_detector import BlinkDetector
from src.visualizer import Visualizer


class TestFullPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.face_detector = FaceDetector()
        cls.blink_detector = BlinkDetector()
        cls.visualizer = Visualizer()

    @classmethod
    def tearDownClass(cls):
        cls.face_detector.close()

    def test_detector_on_blank_frame(self):
        """Blank image should gracefully yield detected=False without crashing."""
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        data = self.face_detector.detect(blank)
        self.assertFalse(data.detected)
        self.assertIsNone(data.right_eye_ear_pts)

        # Update blink detector on no-face data
        state = self.blink_detector.update(data)
        self.assertEqual(state.status_text, "Face not detected — Look toward the camera")

        # Visualizer rendering
        output = self.visualizer.render(blank, data, state, fps=30.0)
        self.assertEqual(output.shape, blank.shape)

    def test_visualizer_rendering(self):
        """Visualizer must render HUD elements without altering frame dimensions."""
        frame = np.full((480, 640, 3), 50, dtype=np.uint8)
        dummy_state = self.blink_detector.update(FaceLandmarksData(detected=False))
        output = self.visualizer.render(frame, FaceLandmarksData(detected=False), dummy_state, fps=28.5)
        self.assertEqual(output.shape, (480, 640, 3))
        # Ensure HUD modified pixel values
        self.assertFalse(np.array_equal(frame, output))

    def test_reset_behavior(self):
        """Blink detector reset must restore counters without counting blinks."""
        self.blink_detector.reset()
        self.assertEqual(self.blink_detector.blink_count, 0)
        self.assertFalse(self.blink_detector.blink_confirmed_pending)
        self.assertEqual(self.blink_detector.consecutive_closed_frames, 0)


if __name__ == "__main__":
    unittest.main()
