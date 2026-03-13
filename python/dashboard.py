"""
OpenCV dashboard for ESP-Drone  –  hackathon starter code.

Displays a live panel with:
  • Vision-derived position (x, y, z)
  • Attitude from telemetry  (roll, pitch, yaw)
  • Raw IMU bars             (accelerometer, gyroscope)
  • Battery voltage
  • Flight status + target position + position error

Usage
─────
    dash = Dashboard()
    # inside your loop:
    dash.update(position=pos, target=target, tel=tel.get_data(),
                flying=True, connected=True)
    dash.show()          # call from the main thread
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

# ──────────────────────────────────────────────────────────────────── #
# Layout constants                                                      #
# ──────────────────────────────────────────────────────────────────── #
W, H = 960, 520
MARGIN_X = 12
COL_GAP = 14
COL_W = (W - (2 * MARGIN_X) - COL_GAP) // 2

# Colour palette  (BGR)
BG      = ( 22,  22,  22)
HDR_BG  = ( 38,  38,  55)
SEC_BG  = ( 30,  30,  30)
DIV     = ( 55,  55,  55)
WHITE   = (230, 230, 230)
GRAY    = (145, 145, 145)
GREEN   = ( 60, 210,  80)
AMBER   = ( 40, 175, 245)
RED_CV  = ( 55,  55, 215)
CYAN    = (210, 210,  50)
BAR_BG  = ( 50,  50,  50)
TEAL    = (170, 210,  60)

FONT  = cv2.FONT_HERSHEY_SIMPLEX
MONO  = cv2.FONT_HERSHEY_PLAIN


def _rect(img, x1, y1, x2, y2, color, filled=True):
    if filled:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
    else:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)


def _text(img, s, x, y, scale=0.55, color=WHITE, thickness=1):
    cv2.putText(img, s, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)


def _bar(img, x, y, w, h, value, vmin, vmax, bar_color=GREEN):
    """Draw a filled horizontal progress bar."""
    _rect(img, x, y, x + w, y + h, BAR_BG)
    ratio = max(0.0, min(1.0, (value - vmin) / (vmax - vmin) if vmax != vmin else 0.0))
    fill  = int(ratio * w)
    if fill > 0:
        _rect(img, x, y, x + fill, y + h, bar_color)
    # border
    _rect(img, x, y, x + w, y + h, DIV, filled=False)


def _signed_bar(img, x, y, w, h, value, vmax, pos_color=GREEN, neg_color=RED_CV):
    """Draw a centre-origin bar (positive → right, negative → left)."""
    _rect(img, x, y, x + w, y + h, BAR_BG)
    mid   = x + w // 2
    ratio = max(-1.0, min(1.0, value / vmax if vmax != 0 else 0.0))
    px    = int(abs(ratio) * (w // 2))
    color = pos_color if ratio >= 0 else neg_color
    if px > 0:
        if ratio >= 0:
            _rect(img, mid, y + 2, mid + px, y + h - 2, color)
        else:
            _rect(img, mid - px, y + 2, mid, y + h - 2, color)
    # centre line
    cv2.line(img, (mid, y), (mid, y + h), DIV, 1)
    _rect(img, x, y, x + w, y + h, DIV, filled=False)


def _section(img, x1, y1, x2, y2, title):
    """Draw a labelled section box."""
    _rect(img, x1, y1, x2, y2, SEC_BG)
    _rect(img, x1, y1, x2, y1 + 22, HDR_BG)
    _text(img, title, x1 + 8, y1 + 15, scale=0.48, color=CYAN, thickness=1)
    _rect(img, x1, y1, x2, y2, DIV, filled=False)


# ──────────────────────────────────────────────────────────────────── #
# Dashboard class                                                       #
# ──────────────────────────────────────────────────────────────────── #

class Dashboard:
    """
    Renders a telemetry dashboard into an OpenCV window.

    Call update() with the latest data, then show() each frame.
    """

    def __init__(self, window_name: str = "ESP-Drone Dashboard") -> None:
        self._win   = window_name
        self._canvas = np.zeros((H, W, 3), dtype=np.uint8)
        self._data: dict = {}

    # ────────────────────────────────────────────────────────────────
    # Public API
    # ────────────────────────────────────────────────────────────────

    def update(
        self,
        position:  Optional[np.ndarray],         # (x,y,z) metres or None
        target:    Optional[np.ndarray],         # (x,y,z) metres or None
        tel:       dict[str, float],             # telemetry dict from Telemetry.get_data()
        flying:    bool,
        connected: bool,
    ) -> None:
        """Recompute the canvas with the latest data.  Call before show()."""
        self._data = dict(
            position  = position,
            target    = target,
            tel       = tel,
            flying    = flying,
            connected = connected,
        )
        self._draw()

    def show(self) -> None:
        """Display the dashboard window.  Must be called from the main thread."""
        cv2.imshow(self._win, self._canvas)

    # ────────────────────────────────────────────────────────────────
    # Drawing
    # ────────────────────────────────────────────────────────────────

    def _draw(self) -> None:
        img = self._canvas
        img[:] = BG

        pos   = self._data.get("position")
        tgt   = self._data.get("target")
        tel   = self._data.get("tel", {})
        fly   = self._data.get("flying", False)
        conn  = self._data.get("connected", False)

        left_col_x = MARGIN_X
        right_col_x = MARGIN_X + COL_W + COL_GAP

        self._draw_header(img, fly, conn)
        self._draw_position(img, pos, x0=left_col_x,  y0=50)
        self._draw_attitude(img, tel,  x0=right_col_x, y0=50)
        self._draw_accel   (img, tel,  x0=left_col_x,  y0=220)
        self._draw_gyro    (img, tel,  x0=right_col_x, y0=220)
        self._draw_footer  (img, tel, pos, tgt)

    # ── Header ───────────────────────────────────────────────────── #

    def _draw_header(self, img, flying: bool, connected: bool) -> None:
        _rect(img, 0, 0, W, 44, HDR_BG)
        _text(img, "ESP-DRONE  DASHBOARD", 14, 28, scale=0.75,
              color=WHITE, thickness=2)

        # Connection pill
        dot_color = GREEN if connected else RED_CV
        status_txt = "CONNECTED" if connected else "NO LINK"
        cv2.circle(img, (580, 22), 7, dot_color, -1)
        _text(img, status_txt, 594, 27, scale=0.5, color=dot_color)

        # Flight mode pill
        mode_color = AMBER if flying else GRAY
        mode_txt   = "FLYING" if flying else "IDLE"
        _rect(img, 710, 10, 806, 34, mode_color, filled=False)
        _text(img, mode_txt, 720, 27, scale=0.55, color=mode_color)

        cv2.line(img, (0, 44), (W, 44), DIV, 1)

    # ── Position section ─────────────────────────────────────────── #

    def _draw_position(self, img, pos, x0, y0) -> None:
        sec_x2 = x0 + COL_W
        _section(img, x0, y0, sec_x2, y0 + 165, "VISION POSITION (m)")
        row_h = 42
        axes  = [("X", 0), ("Y", 1), ("Z", 2)]
        bar_x = x0 + 40
        bar_w = COL_W - 135
        val_x = sec_x2 - 85
        for i, (label, idx) in enumerate(axes):
            ry = y0 + 28 + i * row_h
            val = float(pos[idx]) if pos is not None else 0.0
            det = pos is not None

            _text(img, label, x0 + 12, ry + 14, scale=0.55, color=GRAY)
            _bar(img, bar_x, ry, bar_w, 20, val if det else 0.0, 0.0, 1.0,
                 bar_color=TEAL if det else BAR_BG)
            val_str = f"{val:+.3f}" if det else "  ---"
            _text(img, val_str, val_x, ry + 14, scale=0.5,
                  color=WHITE if det else GRAY)

    # ── Attitude section ─────────────────────────────────────────── #

    def _draw_attitude(self, img, tel: dict, x0, y0) -> None:
        sec_x2 = x0 + COL_W
        _section(img, x0, y0, sec_x2, y0 + 165, "ATTITUDE (°)  from telemetry")
        keys   = [("Roll",  "stateEstimate.roll",  -45, 45),
                  ("Pitch", "stateEstimate.pitch", -45, 45),
                  ("Yaw",   "stateEstimate.yaw",  -180, 180)]
        row_h  = 42
        bar_x = x0 + 60
        bar_w = COL_W - 165
        val_x = sec_x2 - 95
        for i, (label, key, vmin, vmax) in enumerate(keys):
            ry  = y0 + 28 + i * row_h
            val = tel.get(key, 0.0)
            _text(img, label, x0 + 8, ry + 14, scale=0.48, color=GRAY)
            # signed bar centred at 0
            _signed_bar(img, bar_x, ry, bar_w, 20, val, max(abs(vmin), abs(vmax)))
            _text(img, f"{val:+7.1f}", val_x, ry + 14, scale=0.5, color=AMBER)

    # ── Accelerometer section ─────────────────────────────────────── #

    def _draw_accel(self, img, tel: dict, x0, y0) -> None:
        sec_x2 = x0 + COL_W
        _section(img, x0, y0, sec_x2, y0 + 165, "ACCELEROMETER (m/s²)")
        self._draw_imu_bars(img, tel, x0, y0,
                            keys=[("ax", "acc.x"), ("ay", "acc.y"), ("az", "acc.z")],
                            vmax=20.0, bar_color=GREEN)

    # ── Gyroscope section ─────────────────────────────────────────── #

    def _draw_gyro(self, img, tel: dict, x0, y0) -> None:
        sec_x2 = x0 + COL_W
        _section(img, x0, y0, sec_x2, y0 + 165, "GYROSCOPE (°/s)")
        self._draw_imu_bars(img, tel, x0, y0,
                            keys=[("rx", "gyro.x"), ("ry", "gyro.y"), ("rz", "gyro.z")],
                            vmax=500.0, bar_color=AMBER)

    def _draw_imu_bars(self, img, tel, x0, y0, keys, vmax, bar_color) -> None:
        row_h = 42
        sec_x2 = x0 + COL_W
        bar_x = x0 + 45
        bar_w = COL_W - 160
        val_x = sec_x2 - 105
        for i, (label, key) in enumerate(keys):
            ry  = y0 + 28 + i * row_h
            val = tel.get(key, 0.0)
            _text(img, label, x0 + 8, ry + 14, scale=0.5, color=GRAY)
            _signed_bar(img, bar_x, ry, bar_w, 20, val, vmax, bar_color, RED_CV)
            _text(img, f"{val:+8.2f}", val_x, ry + 14, scale=0.5, color=WHITE)

    # ── Footer ───────────────────────────────────────────────────── #

    def _draw_footer(self, img, tel: dict, pos, tgt) -> None:
        fy = H - 110
        _rect(img, 0, fy, W, H, SEC_BG)
        cv2.line(img, (0, fy), (W, fy), DIV, 1)

        # Battery
        vbat  = tel.get("pm.vbat", 0.0)
        bat_w = 200
        _text(img, "BATTERY", 14, fy + 20, scale=0.48, color=CYAN)
        bat_pct = max(0.0, min(1.0, (vbat - 3.2) / (4.2 - 3.2))) if vbat > 0.1 else 0.0
        bat_color = GREEN if bat_pct > 0.4 else (AMBER if bat_pct > 0.2 else RED_CV)
        _bar(img, 90, fy + 6, bat_w, 18, bat_pct, 0.0, 1.0, bat_color)
        bat_str = f"{vbat:.2f} V" if vbat > 0.1 else "N/A"
        _text(img, bat_str, 300, fy + 20, scale=0.5, color=bat_color)

        # Target
        if tgt is not None:
            tgt_str = f"TARGET  x={tgt[0]:.2f}  y={tgt[1]:.2f}  z={tgt[2]:.2f} m"
        else:
            tgt_str = "TARGET  not set"
        _text(img, tgt_str, 14, fy + 50, scale=0.5, color=GRAY)

        # Position error
        if pos is not None and tgt is not None:
            err = tgt - pos
            err_str = (f"ERROR   dx={err[0]:+.3f}  dy={err[1]:+.3f}  dz={err[2]:+.3f} m"
                       f"   |e|={float(np.linalg.norm(err)):.3f} m")
            err_color = GREEN if float(np.linalg.norm(err)) < 0.1 else AMBER
        else:
            err_str   = "ERROR   ---"
            err_color = GRAY
        _text(img, err_str, 14, fy + 76, scale=0.5, color=err_color)

        # Yaw heading compass (simple text arc indicator)
        yaw = tel.get("stateEstimate.yaw", 0.0)
        cx, cy, r = W - 60, fy + 55, 40
        cv2.circle(img, (cx, cy), r, DIV, 1)
        import math
        rad = math.radians(-yaw)
        nx  = int(cx + r * 0.75 * math.sin(rad))
        ny  = int(cy - r * 0.75 * math.cos(rad))
        cv2.arrowedLine(img, (cx, cy), (nx, ny), AMBER, 2, tipLength=0.3)
        _text(img, "N", cx - 4, fy + 20, scale=0.4, color=GRAY)
        _text(img, f"{yaw:.0f}°", cx - 14, fy + 108, scale=0.42, color=AMBER)
