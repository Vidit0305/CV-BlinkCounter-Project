"""
visualizer.py - Clean OpenCV HUD and Eye Landmark Visualization

Renders a minimal, high-contrast, professional camera overlay displaying:
- Eye contour outlines and key landmark points.
- Prominent blink counter: 'BLINKS: X'.
- Responsive status messages.
- Keyboard navigation instructions.
- Optional EAR debug metrics and FPS display.
"""

from typing import Optional, Tuple

import cv2
import numpy as np

from config.settings import (
    COLOR_OPEN,
    COLOR_CLOSED,
    COLOR_BLINK,
    COLOR_NO_FACE,
    COLOR_TEXT_PRIMARY,
    COLOR_TEXT_MUTED,
    COLOR_PANEL_BG,
    COLOR_PANEL_BORDER,
    DRAW_EYE_CONTOURS,
    DRAW_EYE_POINTS,
    EYE_POINT_RADIUS,
    EYE_CONTOUR_THICKNESS,
    STATUS_TEXT_NO_FACE,
    STATUS_TEXT_CLOSED,
    STATUS_TEXT_BLINK,
)
from src.face_detector import FaceLandmarksData
from src.blink_detector import BlinkState


class Visualizer:
    """Manages clean desktop overlay rendering on the live video stream."""

    def __init__(self, show_debug_ear: bool = False):
        self.show_debug_ear = show_debug_ear

    def toggle_debug_ear(self) -> bool:
        """Toggles EAR debug information overlay."""
        self.show_debug_ear = not self.show_debug_ear
        return self.show_debug_ear

    def _get_status_color(self, state: BlinkState) -> Tuple[int, int, int]:
        """Maps current state to appropriate restrained BGR accent color."""
        if state.status_text == STATUS_TEXT_NO_FACE:
            return COLOR_NO_FACE
        elif state.status_text == STATUS_TEXT_BLINK:
            return COLOR_BLINK
        elif state.is_eyes_closed or state.status_text == STATUS_TEXT_CLOSED:
            return COLOR_CLOSED
        return COLOR_OPEN

    def draw_eye_indicators(
        self,
        frame: np.ndarray,
        face_data: FaceLandmarksData,
        state: BlinkState,
    ) -> None:
        """
        Draws subtle outlines and landmark points around both eyes.
        Does not clutter the user's face with unnecessary mesh lines.
        """
        if not face_data.detected:
            return

        accent_color = self._get_status_color(state)
        h, w = frame.shape[:2]
        scale = max(0.85, min(w / 1280.0, 1.65))
        contour_thick = max(1, int(EYE_CONTOUR_THICKNESS * scale))
        point_rad = max(2, int(EYE_POINT_RADIUS * scale))

        # Draw left and right eye contour outlines
        if DRAW_EYE_CONTOURS:
            for contour_pts in [face_data.left_eye_contour_pts, face_data.right_eye_contour_pts]:
                if contour_pts is not None and len(contour_pts) > 0:
                    pts_int = np.round(contour_pts).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(
                        frame,
                        [pts_int],
                        isClosed=True,
                        color=accent_color,
                        thickness=contour_thick,
                        lineType=cv2.LINE_AA,
                    )

        # Draw subtle landmark points on the 6 EAR keypoints
        if DRAW_EYE_POINTS:
            for ear_pts in [face_data.left_eye_ear_pts, face_data.right_eye_ear_pts]:
                if ear_pts is not None:
                    for pt in ear_pts:
                        center = (int(round(pt[0])), int(round(pt[1])))
                        cv2.circle(
                            frame,
                            center,
                            point_rad,
                            (255, 255, 255),
                            -1,
                            lineType=cv2.LINE_AA,
                        )


    def draw_hud(
        self,
        frame: np.ndarray,
        state: BlinkState,
        fps: Optional[float] = None,
    ) -> np.ndarray:
        """
        Renders a clean, high-contrast semi-transparent HUD panel onto the frame.
        Contains:
        - Total blink count: 'BLINKS: X'
        - Current detection status with status dot
        - Keyboard instructions
        - Optional EAR metrics and FPS
        """
        h, w = frame.shape[:2]
        accent_color = self._get_status_color(state)

        # Scale typography and panel geometry proportionally based on frame resolution
        scale = max(0.85, min(w / 1280.0, 1.65))

        panel_w = int(420 * scale)
        panel_h = int((130 if self.show_debug_ear else 92) * scale)
        panel_x = int(24 * scale)
        panel_y = int(24 * scale)

        # Create semi-transparent background card for the top HUD
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + panel_h),
            COLOR_PANEL_BG,
            -1,
        )
        # Subtle border
        cv2.rectangle(
            overlay,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + panel_h),
            COLOR_PANEL_BORDER,
            max(1, int(scale)),
        )

        # Alpha blend HUD overlay
        alpha = 0.85
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # ----------------------------------------------------------------------
        # 1. Total Blink Count Display
        # ----------------------------------------------------------------------
        counter_text = f"BLINKS: {state.blink_count}"
        font_scale_blinks = 0.95 * scale
        cv2.putText(
            frame,
            counter_text,
            (panel_x + int(18 * scale), panel_y + int(38 * scale)),
            cv2.FONT_HERSHEY_DUPLEX,
            font_scale_blinks,
            COLOR_TEXT_PRIMARY,
            max(2, int(2 * scale)),
            lineType=cv2.LINE_AA,
        )

        # ----------------------------------------------------------------------
        # 2. Detection Status Indicator & Message
        # ----------------------------------------------------------------------
        dot_radius = max(4, int(6 * scale))
        status_dot_center = (panel_x + int(24 * scale), panel_y + int(68 * scale))
        cv2.circle(
            frame,
            status_dot_center,
            dot_radius,
            accent_color,
            -1,
            lineType=cv2.LINE_AA,
        )

        font_scale_status = 0.54 * scale
        cv2.putText(
            frame,
            state.status_text,
            (panel_x + int(42 * scale), panel_y + int(73 * scale)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale_status,
            COLOR_TEXT_PRIMARY,
            1,
            lineType=cv2.LINE_AA,
        )

        # ----------------------------------------------------------------------
        # 3. Optional Debug EAR Metrics
        # ----------------------------------------------------------------------
        if self.show_debug_ear:
            ear_l_str = f"{state.ear_left:.3f}" if state.ear_left is not None else "N/A"
            ear_r_str = f"{state.ear_right:.3f}" if state.ear_right is not None else "N/A"
            ear_avg_str = f"{state.ear_smoothed:.3f}" if state.ear_smoothed is not None else "N/A"
            debug_str = f"L: {ear_l_str} | R: {ear_r_str} | Avg: {ear_avg_str}"

            cv2.putText(
                frame,
                debug_str,
                (panel_x + int(18 * scale), panel_y + int(110 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46 * scale,
                COLOR_TEXT_MUTED,
                1,
                lineType=cv2.LINE_AA,
            )

        # ----------------------------------------------------------------------
        # 4. Bottom Footer: Instructions & FPS
        # ----------------------------------------------------------------------
        footer_y = h - int(18 * scale)
        instruction_text = "Press Q / ESC: quit | R: reset | F: toggle fullscreen | D: debug EAR"
        cv2.putText(
            frame,
            instruction_text,
            (int(24 * scale), footer_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44 * scale,
            COLOR_TEXT_MUTED,
            1,
            lineType=cv2.LINE_AA,
        )

        if fps is not None and fps > 0:
            fps_text = f"{fps:.1f} FPS"
            (fps_w, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.44 * scale, 1)
            cv2.putText(
                frame,
                fps_text,
                (w - fps_w - int(24 * scale), footer_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.44 * scale,
                COLOR_TEXT_MUTED,
                1,
                lineType=cv2.LINE_AA,
            )


        return frame

    def render(
        self,
        frame: np.ndarray,
        face_data: FaceLandmarksData,
        state: BlinkState,
        fps: Optional[float] = None,
    ) -> np.ndarray:
        """
        Coordinates full frame rendering: eye indicators followed by HUD.
        """
        output_frame = frame.copy()
        self.draw_eye_indicators(output_frame, face_data, state)
        self.draw_hud(output_frame, state, fps)
        return output_frame
