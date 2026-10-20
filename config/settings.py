"""
settings.py - Global Configuration Settings for Real-Time Eye Blink Counter

All detection thresholds, camera settings, landmark indices, and visualization
parameters are centralized here for easy tuning and calibration.
"""

import os
from pathlib import Path

# ==============================================================================
# Path Configuration
# ==============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "face_landmarker.task"
MODEL_DOWNLOAD_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

# ==============================================================================
# Webcam & Capture Settings
# ==============================================================================
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 30
FULLSCREEN_DEFAULT = True
MIRROR_VIEW = True  # Natural selfie/mirror orientation

# ==============================================================================
# Eye Aspect Ratio (EAR) & Blink Detection Parameters
# ==============================================================================
# Threshold below which eyes are classified as closed.
# Calibrated for MediaPipe Face Mesh standard normalized coordinate projection.
EAR_THRESH_CLOSE = 0.21

# Hysteresis threshold: eyes must reopen above this value to register completion.
# Using EAR_THRESH_OPEN > EAR_THRESH_CLOSE eliminates boundary jitter.
EAR_THRESH_OPEN = 0.25

# Minimum consecutive frames where EAR < EAR_THRESH_CLOSE to confirm eye closure.
# Filters out single-frame tracking noise or sensor glitches.
CONSECUTIVE_FRAMES_MIN = 2

# Maximum consecutive frames before classifying as prolonged closure (e.g. resting eyes).
# When the eyes eventually reopen, it counts as exactly 1 blink, not multiple blinks.
CONSECUTIVE_FRAMES_MAX = 45

# Duration in seconds to keep "Status: Blink Detected!" visible on the HUD
BLINK_STATUS_DURATION_SEC = 0.75

# Exponential Moving Average (EMA) smoothing for EAR:
# smoothed_ear = alpha * current_ear + (1 - alpha) * prev_ear
# 1.0 = raw signal (no smoothing), lower values = smoother signal
EAR_SMOOTHING_FACTOR = 0.70

# ==============================================================================
# MediaPipe Face Landmark Indices
# ==============================================================================
# 6 landmarks per eye for EAR calculation:
# p1: lateral/medial corner 1
# p2: upper eyelid 1
# p3: upper eyelid 2
# p4: opposite corner
# p5: lower eyelid 2 (below p3)
# p6: lower eyelid 1 (below p2)
#
# Subject's Right Eye (Viewer's Left on unmirrored camera):
# p1=33 (outer corner), p2=160, p3=158, p4=133 (inner corner), p5=153, p6=144
RIGHT_EYE_EAR_INDICES = [33, 160, 158, 133, 153, 144]

# Subject's Left Eye (Viewer's Right on unmirrored camera):
# p1=362 (inner corner), p2=385, p3=387, p4=263 (outer corner), p5=373, p6=380
LEFT_EYE_EAR_INDICES = [362, 385, 387, 263, 373, 380]

# Complete 16-point eyelid contours for subtle, accurate visual boundary rendering
RIGHT_EYE_CONTOUR = [
    33, 7, 163, 144, 145, 153, 154, 155,
    133, 173, 157, 158, 159, 160, 161, 246
]

LEFT_EYE_CONTOUR = [
    362, 382, 381, 380, 374, 373, 390, 249,
    263, 466, 388, 387, 386, 385, 384, 398
]

# ==============================================================================
# Visualizer & UI Configuration
# ==============================================================================
WINDOW_NAME = "Real-Time Eye Blink Counter"

# Debug EAR display (can be toggled in real time with 'D' key)
SHOW_DEBUG_EAR = False

# Subtle Eye Landmark Styling
DRAW_EYE_CONTOURS = True
DRAW_EYE_POINTS = True
EYE_POINT_RADIUS = 2
EYE_CONTOUR_THICKNESS = 1

# Color Palette (BGR format for OpenCV)
# Modern restrained, high-contrast palette
COLOR_OPEN = (100, 215, 115)       # Emerald Green
COLOR_CLOSED = (60, 150, 255)      # Warm Amber / Orange
COLOR_BLINK = (245, 195, 30)       # Electric Sky Blue / Cyan
COLOR_NO_FACE = (85, 85, 235)      # Soft Coral Red

COLOR_TEXT_PRIMARY = (255, 255, 255)    # Pure White
COLOR_TEXT_MUTED = (180, 185, 190)      # Light Slate Gray
COLOR_PANEL_BG = (18, 20, 24)           # Deep Charcoal
COLOR_PANEL_BORDER = (55, 60, 70)       # Subtle Border

# Status text mappings
STATUS_TEXT_NO_FACE = "Face not detected — Look toward the camera"
STATUS_TEXT_OPEN = "Status: Eyes Open"
STATUS_TEXT_CLOSED = "Status: Eyes Closed"
STATUS_TEXT_BLINK = "Status: Blink Detected!"
