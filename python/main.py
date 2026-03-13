"""
ESP-Drone hover controller  –  hackathon starter code.

Quick-start
───────────
1.  pip install -r requirements.txt
2.  Connect laptop to the ESP-Drone Wi-Fi (SSID: ESP-DRONE)
3.  Run:  python main.py
4.  Press T to take off, then H to hold position.

Windows opened
──────────────
  "Tracker"    – live annotated frames from both cameras
  "Dashboard"  – telemetry (IMU, attitude, position, battery …)

Controls  (any OpenCV window must have focus)
─────────────────────────────────────────────
  T       – takeoff (brief liftoff, then pilot controls altitude)
  L       – land
  H       – hold (hover in place using on-board estimator)
  W/S     – fly ±Y  (depth axis)
  A/D     – fly ±X  (left/right)
  E/C     – fly ±Z  (height)
  Q / ESC – emergency stop and quit
"""

from __future__ import annotations

import logging
import signal
import time

import cv2
import numpy as np

from crtp_driver     import CRTPDriver
from drone_controller import DroneController
from pid_controller  import PositionPID
from vision_tracker  import VisionTracker
from telemetry       import Telemetry
from dashboard       import Dashboard

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger("main")

# ──────────────────────────────────────────────────────────────────── #
#  CONFIGURATION  –  tune these for your setup                        #
# ──────────────────────────────────────────────────────────────────── #

DRONE_IP     = "192.168.43.42"   # ESP-Drone AP default
CAM_FRONT    = 2                  # Logi C270 HD WebCam (front view)
CAM_SIDE     = 3                  # Logi C270 HD WebCam (side view)

# Target hover position inside the 1 m cage (metres)
HOVER_TARGET = (0.5, 0.5, 0.5)

# PID gains  –  start with kp only; add kd to damp oscillation
PID_KP  = 0.7
PID_KI  = 0.04
PID_KD  = 0.12
MAX_VEL = 0.35    # m/s  maximum velocity command

# Takeoff / landing
TAKEOFF_VZ       = 0.20    # m/s upward burst velocity
TAKEOFF_TIME     = 1.0     # seconds of burst – just enough to clear the ground
LAND_VZ          = -0.15   # m/s downward
LAND_Z_THRESHOLD =  0.08   # m – cut motors below this height

MANUAL_VEL       = 0.05    # m/s per key-hold (WASD / E / C)
CONTROL_HZ       = 20      # command rate (Hz)


# ──────────────────────────────────────────────────────────────────── #
#  Application                                                         #
# ──────────────────────────────────────────────────────────────────── #

class App:
    def __init__(self) -> None:
        self._driver    = CRTPDriver(DRONE_IP)
        self._ctrl      = DroneController(self._driver)
        self._pid       = PositionPID(PID_KP, PID_KI, PID_KD, MAX_VEL)
        self._tracker   = VisionTracker(cam_front=CAM_FRONT, cam_side=CAM_SIDE, display=True)
        self._telemetry = Telemetry(self._driver)
        self._dashboard = Dashboard()

        self._target  = np.array(list(HOVER_TARGET), dtype=float)
        self._vel_cmd = (0.0, 0.0, 0.0)   # manual velocity command (m/s)
        self._flying  = False
        self._running = True
        self._dt      = 1.0 / CONTROL_HZ

        signal.signal(signal.SIGINT, lambda *_: self._quit())

    def _is_connected(self) -> bool:
        """True when telemetry is actively receiving packets from the drone."""
        return self._telemetry.is_connected()

    # ────────────────────────────────────────────────────────────────

    def run(self) -> None:
        # ── connect ──────────────────────────────────────────────── #
        log.info("Connecting to drone at %s …", DRONE_IP)
        self._driver.connect()

        # ── telemetry (background – don't block cameras) ──────────── #
        import threading
        def _start_tel():
            ok = self._telemetry.start(timeout=30.0)
            if ok:
                log.info("Telemetry streaming.")
            else:
                log.warning("Telemetry unavailable – dashboard shows zeros. "
                            "(Is the drone on the network?)")
        threading.Thread(target=_start_tel, daemon=True).start()

        # ── vision tracker ────────────────────────────────────────── #
        log.info("Starting vision tracker …")
        self._tracker.start()

        log.info("Ready.  T=takeoff  L=land  H=hold  WASD/EC=move  Q=quit")

        last_cmd = time.monotonic()
        try:
            while self._running:
                # -- render both windows + read keyboard --------------
                self._tracker.show()
                self._update_dashboard()
                key = cv2.waitKey(1) & 0xFF
                self._handle_key(key)

                # -- fixed-rate control loop --------------------------
                now = time.monotonic()
                if now - last_cmd < self._dt:
                    continue
                last_cmd = now
                self._control_step()

        finally:
            self._shutdown()

    # ────────────────────────────────────────────────────────────────
    # Dashboard
    # ────────────────────────────────────────────────────────────────

    def _update_dashboard(self) -> None:
        self._dashboard.update(
            position  = self._tracker.get_position(),
            target    = self._target if self._flying else None,
            tel       = self._telemetry.get_data(),
            flying    = self._flying,
            connected = self._is_connected(),
        )
        self._dashboard.show()

    # ────────────────────────────────────────────────────────────────
    # Control step
    # ────────────────────────────────────────────────────────────────

    def _control_step(self) -> None:
        if not self._is_connected():
            return

        if not self._flying:
            self._ctrl.stop()  # keepalive: prevent watchdog timeout without arming
            return

        # ── TODO (participants): replace with vision-based position control ──
        # pos = self._tracker.get_position()
        # if pos is not None:
        #     self._ctrl.send_ext_position(*pos)
        #     vx, vy, vz = self._pid.update(tuple(self._target), tuple(pos))
        #     self._ctrl.send_velocity_world(vx, vy, vz)
        #     return

        self._ctrl.send_velocity_world(*self._vel_cmd)

    # ────────────────────────────────────────────────────────────────
    # Keyboard
    # ────────────────────────────────────────────────────────────────

    def _handle_key(self, key: int) -> None:
        if key in (ord("q"), 27):
            self._quit()
            return

        if not self._is_connected():
            if key in (ord("t"), ord("l"), ord("h"), ord("w"), ord("s"), ord("a"), ord("d"), ord("e"), ord("c")):
                log.warning("Ignoring flight input while disconnected.")
            return

        elif key == ord("t") and not self._flying:
            self._takeoff()
        elif key == ord("l") and self._flying:
            self._land()
        elif self._flying:
            vx = vy = vz = 0.0
            if   key == ord("h"):  log.info("Hold")
            elif key == ord("w"):  vy = +MANUAL_VEL
            elif key == ord("s"):  vy = -MANUAL_VEL
            elif key == ord("d"):  vx = +MANUAL_VEL
            elif key == ord("a"):  vx = -MANUAL_VEL
            elif key == ord("e"):  vz = +MANUAL_VEL
            elif key == ord("c"):  vz = -MANUAL_VEL
            new_cmd = (vx, vy, vz)
            if new_cmd != self._vel_cmd:
                log.info("vel → (%.2f, %.2f, %.2f) m/s", *new_cmd)
            self._vel_cmd = new_cmd

    # ────────────────────────────────────────────────────────────────
    # Flight phases
    # ────────────────────────────────────────────────────────────────

    def _takeoff(self) -> None:
        if not self._is_connected():
            log.warning("Takeoff blocked: drone not connected.")
            return
        log.info("Takeoff – brief liftoff burst …")
        self._flying  = True
        self._vel_cmd = (0.0, 0.0, 0.0)
        t0 = time.monotonic()
        while time.monotonic() - t0 < TAKEOFF_TIME:
            self._ctrl.send_velocity_world(0.0, 0.0, TAKEOFF_VZ)
            time.sleep(self._dt)
        log.info("Airborne – E/C = altitude  WASD = lateral  H = hold  L = land")

    def _land(self) -> None:
        if not self._is_connected():
            log.warning("Land blocked: drone not connected.")
            return
        log.info("Landing …")
        self._vel_cmd = (0.0, 0.0, 0.0)
        t0       = time.monotonic()
        deadline = t0 + abs(1.5 / LAND_VZ)   # ~10 s: covers up to 1.5 m altitude
        while time.monotonic() < deadline:
            # After a brief descent, trust stateEstimate.z to detect ground contact
            if time.monotonic() - t0 > 1.5 and self._telemetry.active:
                if self._telemetry.get_data().get("stateEstimate.z", 1.0) < LAND_Z_THRESHOLD:
                    break
            self._ctrl.send_velocity_world(0.0, 0.0, LAND_VZ)
            time.sleep(self._dt)
        self._ctrl.stop()
        self._flying = False
        self._pid.reset()
        log.info("Landed.")

    def _quit(self) -> None:
        log.info("Quit")
        self._running = False

    def _shutdown(self) -> None:
        self._ctrl.emergency_stop()
        log.warning("Emergency stop sent.")
        self._telemetry.stop()
        self._tracker.stop()
        self._driver.disconnect()
        log.info("Shutdown complete.")


# ──────────────────────────────────────────────────────────────────── #

if __name__ == "__main__":
    App().run()
