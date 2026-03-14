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

# Flight phase constants (state machine)
_IDLE       = "idle"
_TAKING_OFF = "taking_off"
_FLYING     = "flying"
_LANDING    = "landing"


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

        self._target      = np.array(list(HOVER_TARGET), dtype=float)
        self._vel_cmd     = (0.0, 0.0, 0.0)   # manual velocity command (m/s)
        self._vel_key_t   = 0.0               # time of last WASD/EC keypress
        self._phase       = _IDLE              # flight state machine
        self._phase_t0    = 0.0               # time.monotonic() when phase started
        self._running     = True
        self._dt          = 1.0 / CONTROL_HZ

        signal.signal(signal.SIGINT, lambda *_: self._quit())

    @property
    def _flying(self) -> bool:
        """True while in any active flight phase (used by dashboard)."""
        return self._phase != _IDLE

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

        last_cmd  = time.monotonic()
        last_draw = time.monotonic()
        _draw_dt  = 1.0 / 20   # render at 20 fps – keeps CPU free for key polling
        try:
            while self._running:
                now = time.monotonic()

                # -- key is always polled, never blocked by rendering --
                key = cv2.waitKey(1) & 0xFF
                self._handle_key(key)

                # -- rendering at 20 fps ------------------------------
                if now - last_draw >= _draw_dt:
                    last_draw = now
                    self._tracker.show()
                    self._update_dashboard()

                # -- fixed-rate control loop --------------------------
                if now - last_cmd >= self._dt:
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
        now = time.monotonic()

        if self._phase == _IDLE:
            # Keepalive only makes sense when actually connected
            if self._is_connected():
                self._ctrl.stop()
            return

        # During active flight keep sending commands even if telemetry goes
        # quiet (motor RF noise can drop telemetry without breaking the UDP
        # command link).  Stopping here would trigger the on-board watchdog.

        if self._phase == _TAKING_OFF:
            if now - self._phase_t0 < TAKEOFF_TIME:
                self._ctrl.send_velocity_world(0.0, 0.0, TAKEOFF_VZ)
            else:
                self._phase = _FLYING
                log.info("Airborne – E/C = altitude  WASD = lateral  H = hold  L = land")
            return

        if self._phase == _LANDING:
            elapsed  = now - self._phase_t0
            deadline = self._phase_t0 + abs(1.5 / LAND_VZ)
            if elapsed > 1.5 and self._telemetry.active:
                if self._telemetry.get_data().get("stateEstimate.z", 1.0) < LAND_Z_THRESHOLD:
                    self._on_landed()
                    return
            if now >= deadline:
                self._on_landed()
                return
            self._ctrl.send_velocity_world(0.0, 0.0, LAND_VZ)
            return

        # _FLYING

        # Auto-stop when a WASD/EC key is released: OS key-repeat fires every
        # ~30 ms while held; if we haven't seen one for 200 ms, the key is up.
        if self._vel_cmd != (0.0, 0.0, 0.0) and self._vel_key_t > 0.0:
            if now - self._vel_key_t > 0.20:
                self._vel_cmd = (0.0, 0.0, 0.0)
                log.info("Key released – hovering")

        # ── TODO (participants): replace with vision-based position control ──
        # pos = self._tracker.get_position()
        # if pos is not None:
        #     self._ctrl.send_ext_position(*pos)
        #     vx, vy, vz = self._pid.update(tuple(self._target), tuple(pos))
        #     self._ctrl.send_velocity_world(vx, vy, vz)
        #     return

        self._ctrl.send_velocity_world(*self._vel_cmd)

    def _on_landed(self) -> None:
        self._ctrl.stop()
        self._phase   = _IDLE
        self._vel_cmd = (0.0, 0.0, 0.0)
        self._pid.reset()
        log.info("Landed.")

    # ────────────────────────────────────────────────────────────────
    # Keyboard
    # ────────────────────────────────────────────────────────────────

    def _handle_key(self, key: int) -> None:
        if key in (ord("q"), 27):           # Q / ESC – always works immediately
            self._emergency_stop_now()
            self._quit()
            return

        if key == ord("t") and self._phase == _IDLE:
            # Require a live telemetry link before arming – proves drone is reachable
            if not self._is_connected():
                log.warning("Takeoff blocked: no telemetry from drone.")
                return
            self._takeoff()
        elif key == ord("l") and self._phase in (_FLYING, _TAKING_OFF):
            self._land()
        elif self._phase == _FLYING:
            _VEL_MAP = {
                ord("w"): (0.0, +MANUAL_VEL, 0.0),
                ord("s"): (0.0, -MANUAL_VEL, 0.0),
                ord("d"): (+MANUAL_VEL, 0.0, 0.0),
                ord("a"): (-MANUAL_VEL, 0.0, 0.0),
                ord("e"): (0.0, 0.0, +MANUAL_VEL),
                ord("c"): (0.0, 0.0, -MANUAL_VEL),
            }
            if key == ord("h"):
                self._vel_cmd   = (0.0, 0.0, 0.0)
                self._vel_key_t = 0.0   # disable auto-release
                log.info("Hold")
            elif key in _VEL_MAP:
                new_cmd = _VEL_MAP[key]
                if new_cmd != self._vel_cmd:
                    log.info("vel → (%.2f, %.2f, %.2f) m/s", *new_cmd)
                self._vel_cmd   = new_cmd
                self._vel_key_t = time.monotonic()
            # key == 0xFF (no key): do NOT touch _vel_cmd

    # ────────────────────────────────────────────────────────────────
    # Flight phases
    # ────────────────────────────────────────────────────────────────

    def _takeoff(self) -> None:
        log.info("Takeoff – brief liftoff burst …")
        self._phase    = _TAKING_OFF
        self._phase_t0 = time.monotonic()
        self._vel_cmd  = (0.0, 0.0, 0.0)

    def _land(self) -> None:
        log.info("Landing …")
        self._vel_cmd  = (0.0, 0.0, 0.0)
        self._phase    = _LANDING
        self._phase_t0 = time.monotonic()

    def _emergency_stop_now(self) -> None:
        """Send stop commands immediately and repeatedly to cut motors fast.

        The firmware watchdog takes up to 2 s to cut motors if these packets
        are lost; sending several bursts ensures at least one gets through
        despite RF noise from spinning motors.
        """
        for _ in range(5):
            self._ctrl.stop()            # TYPE_STOP  – nulls the setpoint queue
            self._ctrl.emergency_stop()  # LOC_EMERGENCY_STOP – sets firmware flag
        log.warning("Emergency stop burst sent.")

    def _quit(self) -> None:
        log.info("Quit")
        self._running = False

    def _shutdown(self) -> None:
        self._emergency_stop_now()       # one more burst during cleanup
        self._telemetry.stop()
        self._tracker.stop()
        self._driver.disconnect()
        log.info("Shutdown complete.")


# ──────────────────────────────────────────────────────────────────── #

if __name__ == "__main__":
    App().run()
