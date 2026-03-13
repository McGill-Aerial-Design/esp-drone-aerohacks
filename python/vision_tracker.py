"""
Dual-camera drone position tracker  –  hackathon starter code.

Camera layout
─────────────
  cam_front (index 0)  –  looks at the FRONT face of the cage
      detected (px, py)  →  cage X (left/right)  +  cage Z (height)

  cam_side  (index 1)  –  looks at the LEFT SIDE face of the cage
      detected (px, py)  →  cage Y (depth in/out)  +  cage Z (height)

Combining both cameras gives a full 3-D position estimate.
Z is averaged from both cameras; each camera owns one horizontal axis.

Detection approach
──────────────────
MOG2 background subtraction: the drone is the only moving object, so
subtracting a learned static background leaves just the drone as a bright
blob.  The centroid of the largest contour is taken as the drone's pixel
position.

  TODO: if the cage/lighting changes a lot, try:
    - HSV colour threshold on the drone's LED colour
    - cv2.inRange on a known colour
    - Frame-differencing instead of MOG2

Coordinate frame  (cage-centred, in metres)
───────────────────────────────────────────
  Origin  = bottom-left-front corner of cage
  +X      = right   (front camera's horizontal axis)
  +Y      = into cage  (side camera's horizontal axis)
  +Z      = up    (inverted image-y)

  Cage is 1 m × 1 m × 1 m.  Target hover ≈ (0.5, 0.5, 0.5).

ROI calibration
───────────────
Set FRONT_ROI and SIDE_ROI at the top of this file to the pixel
bounding-box of the cage in each camera image:  (x1, y1, x2, y2).
Run calibrate_roi() once (with display=True) to visually determine
the right values.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────── #
#  PER-SETUP CONSTANTS  –  tune these for your camera positions        #
# ──────────────────────────────────────────────────────────────────── #

CAGE_SIZE_M = 1.0          # physical cage side length in metres

# Pixel bounding-boxes of the cage in each camera frame (x1, y1, x2, y2).
# Run  calibrate_roi(cam_index=0)  and  calibrate_roi(cam_index=1)
# to find these values interactively.
FRONT_ROI = (0, 0, 1280, 720)  # TODO: set to actual cage region in cam 2 (Logi C270)
SIDE_ROI  = (0, 0, 1280, 720)  # TODO: set to actual cage region in cam 3 (Logi C270)

# MOG2 background subtractor settings
MOG2_HISTORY        = 200
MOG2_VAR_THRESHOLD  = 50
MOG2_DETECT_SHADOWS = False

# Contour area filter – ignore blobs smaller than this (noise)
MIN_CONTOUR_AREA = 150   # pixels²  – lower if drone looks small


# ──────────────────────────────────────────────────────────────────── #
#  Helper: interactive ROI selector                                    #
# ──────────────────────────────────────────────────────────────────── #

def calibrate_roi(cam_index: int = 0) -> tuple[int, int, int, int]:
    """
    Open camera cam_index, show the first frame, and let the user draw
    a rectangle around the cage.  Returns (x1, y1, x2, y2).

    Use this once to find the right ROI constants for your setup:
        front_roi = calibrate_roi(0)
        side_roi  = calibrate_roi(1)
        print(front_roi, side_roi)
    """
    cap = cv2.VideoCapture(cam_index)
    for _ in range(5):              # flush a few frames
        cap.read()
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read from camera {cam_index}")

    roi = cv2.selectROI(
        f"Select cage region – cam {cam_index}  (ENTER to confirm, ESC to cancel)",
        frame, fromCenter=False, showCrosshair=True,
    )
    cv2.destroyAllWindows()
    x, y, w, h = roi
    return (x, y, x + w, y + h)


# ──────────────────────────────────────────────────────────────────── #
#  VisionTracker                                                       #
# ──────────────────────────────────────────────────────────────────── #

class VisionTracker:
    """
    Tracks the drone using background subtraction on two orthogonal cameras.

    Parameters
    ----------
    cam_front   : OpenCV VideoCapture index for the front camera
    cam_side    : OpenCV VideoCapture index for the side camera
    front_roi   : (x1,y1,x2,y2) pixel bounding-box of the cage in cam_front
    side_roi    : (x1,y1,x2,y2) pixel bounding-box of the cage in cam_side
    display     : show annotated frames in an OpenCV window
    """

    def __init__(
        self,
        cam_front: int = 0,
        cam_side:  int = 1,
        front_roi: tuple = FRONT_ROI,
        side_roi:  tuple = SIDE_ROI,
        display:   bool  = True,
        allow_no_camera: bool = True,
    ) -> None:
        self._display   = display
        self._front_roi = front_roi
        self._side_roi  = side_roi
        self._enabled   = True

        # Background subtractors (one per camera)
        self._bg_front = cv2.createBackgroundSubtractorMOG2(
            history=MOG2_HISTORY,
            varThreshold=MOG2_VAR_THRESHOLD,
            detectShadows=MOG2_DETECT_SHADOWS,
        )
        self._bg_side = cv2.createBackgroundSubtractorMOG2(
            history=MOG2_HISTORY,
            varThreshold=MOG2_VAR_THRESHOLD,
            detectShadows=MOG2_DETECT_SHADOWS,
        )

        # Open cameras and request 720p from the Logitech C270
        self._cap_front = cv2.VideoCapture(cam_front, cv2.CAP_DSHOW)
        self._cap_front.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        self._cap_front.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
        self._cap_side  = cv2.VideoCapture(cam_side,  cv2.CAP_DSHOW)
        self._cap_side.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        self._cap_side.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
        front_ok = self._cap_front.isOpened()
        side_ok = self._cap_side.isOpened()
        if not (front_ok and side_ok):
            self._cap_front.release()
            self._cap_side.release()
            if not allow_no_camera:
                missing = []
                if not front_ok:
                    missing.append(f"front camera (index {cam_front})")
                if not side_ok:
                    missing.append(f"side camera (index {cam_side})")
                raise RuntimeError("Cannot open " + " and ".join(missing))
            self._enabled = False
            log.warning(
                "VisionTracker disabled: camera(s) unavailable (front=%s side=%s). "
                "Running without camera-based position.",
                front_ok,
                side_ok,
            )

        # Shared state (written by background thread, read by main thread)
        self._position: Optional[np.ndarray] = None   # (x, y, z) metres
        self._frames:   Optional[tuple]       = None   # (front, side) for display
        self._lock    = threading.Lock()
        self._running = False

        if self._enabled:
            log.info("VisionTracker ready  front=%d  side=%d", cam_front, cam_side)

    # ────────────────────────────────────────────────────────────────
    # Public API
    # ────────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start background capture / detection thread."""
        if not self._enabled:
            return
        self._running = True
        self._thread  = threading.Thread(
            target=self._loop, name="vision", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the thread and release cameras."""
        if not self._enabled:
            cv2.destroyAllWindows()
            return
        self._running = False
        if hasattr(self, "_thread"):
            self._thread.join(timeout=2.0)
        self._cap_front.release()
        self._cap_side.release()
        cv2.destroyAllWindows()

    def get_position(self) -> Optional[np.ndarray]:
        """
        Latest estimated position (x, y, z) in metres inside the cage,
        or None if the drone was not detected.  Thread-safe.
        """
        with self._lock:
            return self._position.copy() if self._position is not None else None

    def show(self) -> None:
        """
        Call this from the MAIN thread to render the debug display.
        Must be called from the main thread on Windows.
        """
        if not self._enabled:
            return

        with self._lock:
            frames = self._frames
            pos    = self._position.copy() if self._position is not None else None

        if frames is None:
            return

        DISP_W, DISP_H = 640, 360   # display size per camera (half of 1280×720)

        front_ann, side_ann = frames
        front_disp = cv2.resize(front_ann, (DISP_W, DISP_H))
        side_disp  = cv2.resize(side_ann,  (DISP_W, DISP_H))

        # Camera labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.rectangle(front_disp, (0, 0), (110, 28), (0, 0, 0), -1)
        cv2.putText(front_disp, "FRONT", (6, 20), font, 0.65, (0, 220, 255), 2)
        cv2.rectangle(side_disp, (0, 0), (100, 28), (0, 0, 0), -1)
        cv2.putText(side_disp, "SIDE", (6, 20), font, 0.65, (0, 220, 255), 2)

        # Divider line between the two feeds
        combined = np.hstack([front_disp, side_disp])
        cv2.line(combined, (DISP_W, 0), (DISP_W, DISP_H), (80, 80, 80), 2)

        # Position overlay at the bottom
        if pos is not None:
            label = f"x={pos[0]:.2f}  y={pos[1]:.2f}  z={pos[2]:.2f} m"
            color = (0, 255, 80)
        else:
            label = "drone not detected"
            color = (0, 80, 255)
        cv2.rectangle(combined, (0, DISP_H - 30), (combined.shape[1], DISP_H), (0, 0, 0), -1)
        cv2.putText(combined, label, (10, DISP_H - 8), font, 0.65, color, 2)

        cv2.imshow("Tracker  [front | side]  (q=quit)", combined)

    # ────────────────────────────────────────────────────────────────
    # Background thread
    # ────────────────────────────────────────────────────────────────

    def _loop(self) -> None:
        while self._running:
            ok0, frame_front = self._cap_front.read()
            ok1, frame_side  = self._cap_side.read()
            if not ok0 or not ok1:
                log.warning("Camera read failure")
                time.sleep(0.02)
                continue

            # Detect drone centroid in each camera  →  normalised [0,1] coords
            cx_front, cy_front = self._detect(frame_front, self._bg_front, self._front_roi)
            cx_side,  cy_side  = self._detect(frame_side,  self._bg_side,  self._side_roi)

            pos = None
            if (cx_front is not None and cy_front is not None
                    and cx_side is not None and cy_side is not None):
                # Front camera  →  X (left/right)  and  Z_front (height)
                # Side  camera  →  Y (depth)         and  Z_side  (height)
                x       = cx_front * CAGE_SIZE_M
                y       = cx_side  * CAGE_SIZE_M
                z_front = (1.0 - cy_front) * CAGE_SIZE_M   # invert y → up
                z_side  = (1.0 - cy_side)  * CAGE_SIZE_M
                z       = (z_front + z_side) / 2.0          # average Z
                pos = np.array([x, y, z], dtype=np.float32)

            with self._lock:
                self._position = pos
                self._frames   = (frame_front, frame_side)

    def _detect(
        self,
        frame:         np.ndarray,
        bg_subtractor: cv2.BackgroundSubtractor,
        roi:           tuple,
    ) -> tuple[Optional[float], Optional[float]]:
        """
        Detect the drone in one frame.

        Returns (cx_norm, cy_norm) in [0.0, 1.0] relative to the ROI,
        or (None, None) if not detected.

        TODO: swap in a better detection method here.
        """
        x1, y1, x2, y2 = roi
        crop = frame[y1:y2, x1:x2]

        # ── background subtraction ────────────────────────────────── #
        fg_mask = bg_subtractor.apply(crop)

        # Morphological clean-up to remove small noise
        kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        # ── find largest contour (= drone) ────────────────────────── #
        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None, None

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < MIN_CONTOUR_AREA:
            return None, None

        # Centroid via moments
        M  = cv2.moments(largest)
        if M["m00"] == 0:
            return None, None
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        # Draw contour + centroid for debug display
        h_crop, w_crop = crop.shape[:2]
        cv2.drawContours(crop, [largest], -1, (0, 255, 0), 2)
        cv2.circle(crop, (int(cx), int(cy)), 6, (0, 0, 255), -1)

        # Normalise to [0, 1] within the ROI
        return cx / w_crop, cy / h_crop
