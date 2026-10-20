#!/usr/bin/env python3
"""
main.py - Entry Point for Real-Time Eye Blink Detection and Counter

Standalone Computer Vision application using OpenCV and MediaPipe.
Runs directly from the terminal, opens the webcam, detects eye landmarks,
computes Eye Aspect Ratio (EAR), and tracks blinks in real time.

Controls:
    Q / ESC: Quit program
    R:       Reset blink counter
    D:       Toggle EAR debug metrics overlay
"""

import argparse
import os
import sys
import time
from typing import Optional

# Suppress third-party C++ logs (TensorFlow Lite, glog, Qt, OpenCV)
os.environ["QT_LOGGING_RULES"] = "*=false"
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOGLEVEL"] = "3"
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

import cv2
import numpy as np

from config.settings import (
    CAMERA_INDEX,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    WINDOW_NAME,
    SHOW_DEBUG_EAR,
    MIRROR_VIEW,
)
from src.face_detector import FaceDetector
from src.blink_detector import BlinkDetector
from src.visualizer import Visualizer


def parse_arguments() -> argparse.Namespace:
    """Parses optional command-line arguments for flexibility."""
    parser = argparse.ArgumentParser(
        description="Real-Time Eye Blink Detection & Counter Using Python and MediaPipe."
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=CAMERA_INDEX,
        help=f"Index of the camera device to open (default: {CAMERA_INDEX}).",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Optional path to a video file for offline testing instead of webcam.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=SHOW_DEBUG_EAR,
        help="Display EAR values on the HUD overlay.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="Optional maximum number of frames to process before exiting (useful for testing).",
    )
    parser.add_argument(
        "--windowed",
        action="store_true",
        default=False,
        help="Launch in a standard floating window instead of FullScreen.",
    )
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        default=not MIRROR_VIEW,
        help="Disable horizontal flip (mirror/selfie view).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run without opening GUI windows (for headless automated testing).",
    )
    return parser.parse_args()


def get_screen_resolution(default_w: int = 1920, default_h: int = 1080) -> tuple[int, int]:
    """Attempts to detect primary display resolution on Linux/X11 with safe fallback."""
    try:
        import subprocess
        out = subprocess.check_output(["xrandr", "--current"], text=True, stderr=subprocess.DEVNULL)
        for line in out.splitlines():
            if "*" in line:
                parts = line.split()[0].split("x")
                return int(parts[0]), int(parts[1])
    except Exception:
        pass
    return default_w, default_h


def crop_to_aspect(frame: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Crops frame to match the target aspect ratio before scaling, preventing distortion."""
    h, w = frame.shape[:2]
    target_aspect = target_w / target_h
    current_aspect = w / h

    if current_aspect > target_aspect:
        # Frame is wider than target aspect ratio -> crop width
        new_w = int(h * target_aspect)
        offset_x = (w - new_w) // 2
        return frame[:, offset_x : offset_x + new_w]
    elif current_aspect < target_aspect:
        # Frame is taller than target aspect ratio -> crop height
        new_h = int(w / target_aspect)
        offset_y = (h - new_h) // 2
        return frame[offset_y : offset_y + new_h, :]
    return frame


def initialize_camera(
    source: int | str,
    width: int = FRAME_WIDTH,
    height: int = FRAME_HEIGHT,
) -> Optional[cv2.VideoCapture]:
    """
    Initializes and configures the OpenCV video capture device.

    Args:
        source: Camera device index (int) or video file path (str).
        width: Desired capture frame width.
        height: Desired capture frame height.

    Returns:
        Opened cv2.VideoCapture instance or None on failure.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        return None

    # Apply preferred resolution if capturing from hardware webcam
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    return cap


def main() -> int:
    """Main execution flow for eye blink counter."""
    args = parse_arguments()

    print("Initializing eye blink detector...")
    print("Loading facial landmark model...")

    try:
        face_detector = FaceDetector()
    except Exception as exc:
        print(f"\n[ERROR] Failed to initialize facial landmark detector: {exc}", file=sys.stderr)
        return 1

    blink_detector = BlinkDetector()
    visualizer = Visualizer(show_debug_ear=args.debug)

    # Initialize video capture source
    source = args.video if args.video else args.camera
    source_desc = f"video file '{args.video}'" if args.video else f"webcam index {args.camera}"

    cap = initialize_camera(source, FRAME_WIDTH, FRAME_HEIGHT)
    if cap is None:
        print(
            f"\n[ERROR] Could not open {source_desc}.\n"
            f"Please ensure the camera is connected, permissions are granted,\n"
            f"and no other program is currently accessing it.",
            file=sys.stderr,
        )
        face_detector.close()
        return 1

    screen_w, screen_h = get_screen_resolution()
    is_fullscreen = not args.windowed
    is_mirrored = not args.no_mirror

    if not args.headless:
        # WINDOW_GUI_NORMAL completely removes Qt navigation, zoom, and save buttons!
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
        if is_fullscreen:
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.resizeWindow(WINDOW_NAME, 960, 540)

    print("Webcam initialized successfully.")
    print(f"Eye blink detection started ({'FullScreen' if is_fullscreen else 'Windowed'}, Mirror View: {'ON' if is_mirrored else 'OFF'}). Press Q to quit.")

    # FPS tracking variables
    fps: float = 0.0
    prev_time = time.time()
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                if args.video:
                    print("Reached end of video stream.")
                    break
                print("[WARN] Failed to read frame from webcam. Retrying...", file=sys.stderr)
                time.sleep(0.05)
                continue

            # Invert left-right for a natural selfie/mirror experience
            if is_mirrored:
                frame = cv2.flip(frame, 1)

            frame_count += 1
            current_time = time.time()
            dt = current_time - prev_time
            prev_time = current_time
            if dt > 0:
                instant_fps = 1.0 / dt
                fps = instant_fps if fps == 0.0 else 0.9 * fps + 0.1 * instant_fps

            # Scale to fullscreen aspect ratio without distorting face proportions
            if is_fullscreen:
                cropped = crop_to_aspect(frame, screen_w, screen_h)
                proc_frame = cv2.resize(cropped, (screen_w, screen_h), interpolation=cv2.INTER_LINEAR)
            else:
                proc_frame = frame

            # 1. Detect facial landmarks
            face_data = face_detector.detect(proc_frame)

            # 2. Update blink detection state machine
            blink_state = blink_detector.update(face_data)

            # 3. Render visual overlay
            output_frame = visualizer.render(proc_frame, face_data, blink_state, fps)

            # 4. Display or headless handle
            if not args.headless:
                cv2.imshow(WINDOW_NAME, output_frame)

                # Check if window was closed via OS 'X' button
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    print("Camera window closed by user.")
                    break

                key = cv2.waitKey(1) & 0xFF

                # Q or ESC to quit
                if key in (ord("q"), ord("Q"), 27):
                    print("Quit requested by user.")
                    break
                # R to reset counter
                elif key in (ord("r"), ord("R")):
                    blink_detector.reset()
                    print("Blink counter reset to 0.")
                # F to toggle fullscreen
                elif key in (ord("f"), ord("F")):
                    is_fullscreen = not is_fullscreen
                    if is_fullscreen:
                        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                        print("Switched to FullScreen mode.")
                    else:
                        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                        cv2.resizeWindow(WINDOW_NAME, 960, 540)
                        print("Switched to Windowed mode.")
                # M to toggle mirror mode
                elif key in (ord("m"), ord("M")):
                    is_mirrored = not is_mirrored
                    print(f"Mirror mode {'enabled' if is_mirrored else 'disabled'}.")
                # D to toggle debug EAR metrics
                elif key in (ord("d"), ord("D")):
                    is_debug = visualizer.toggle_debug_ear()
                    print(f"Debug EAR metrics {'enabled' if is_debug else 'disabled'}.")


            # If frame limit specified (for automated testing)
            if args.frames and frame_count >= args.frames:
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C).")
    finally:
        print("\nCleaning up resources...")
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()
        face_detector.close()

    print(f"Session completed successfully.")
    print(f"Total blinks detected: {blink_detector.blink_count}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

