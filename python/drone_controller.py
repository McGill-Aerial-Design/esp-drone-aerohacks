"""
High-level drone controller for ESP-Drone.

Wraps the CRTP packet types defined in:
  components/core/crazyflie/modules/src/crtp_commander_generic.c
  components/core/crazyflie/modules/src/crtp_localization_service.c

All setpoint methods should be called at a fixed rate (20–50 Hz).
The on-board watchdog cuts thrust if no setpoint arrives within ~500 ms.
"""

import struct
import logging
from crtp_driver import (
    CRTPDriver,
    PORT_COMMANDER,
    PORT_SETPOINT_GENERIC,
    PORT_LOCALIZATION,
)

log = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Generic setpoint packet types  (crtp_commander_generic.c)           #
# ------------------------------------------------------------------ #
TYPE_STOP          = 0   # kill motors
TYPE_VEL_WORLD     = 1   # velocity in world frame
TYPE_Z_DISTANCE    = 2   # abs height + roll/pitch angles
TYPE_ALT_HOLD      = 4   # z velocity + roll/pitch
TYPE_HOVER         = 5   # abs height + body-frame xy velocity
TYPE_FULL_STATE    = 6   # full state setpoint
TYPE_POSITION      = 7   # absolute x, y, z + yaw

# ------------------------------------------------------------------ #
# Localization GENERIC_TYPE sub-types  (crtp_localization_service.h)  #
# ------------------------------------------------------------------ #
LOC_EMERGENCY_STOP = 3   # triggers stabilizerSetEmergencyStop()
LOC_EXT_POSE       = 8   # inject position + quaternion into Kalman


class DroneController:
    """
    High-level API for controlling an ESP-Drone over CRTP/UDP.

    Parameters
    ----------
    driver : CRTPDriver
        A connected CRTPDriver instance.
    """

    def __init__(self, driver: CRTPDriver) -> None:
        self._drv = driver

    # ---------------------------------------------------------------- #
    # Safety commands                                                   #
    # ---------------------------------------------------------------- #

    def stop(self) -> None:
        """Immediately cut motors (freefall). Use for emergency."""
        self._drv.send_packet(PORT_SETPOINT_GENERIC, 0, bytes([TYPE_STOP]))

    def emergency_stop(self) -> None:
        """Trigger the on-board emergency-stop flag via localization port."""
        self._drv.send_packet(PORT_LOCALIZATION, 1, bytes([LOC_EMERGENCY_STOP]))

    # ---------------------------------------------------------------- #
    # Setpoint commands – call continuously at ~20 Hz                   #
    # ---------------------------------------------------------------- #

    def send_velocity_world(
        self,
        vx: float = 0.0,
        vy: float = 0.0,
        vz: float = 0.0,
        yawrate: float = 0.0,
    ) -> None:
        """
        Velocity setpoint in the world (inertial) frame.

        Parameters
        ----------
        vx, vy, vz : float  Velocity in m/s.
        yawrate    : float  Yaw rate in deg/s.

        Note: the drone must have a working altitude estimator for vz
        to have the expected effect (barometer or ToF sensor required).
        """
        data = bytes([TYPE_VEL_WORLD]) + struct.pack("<ffff", vx, vy, vz, yawrate)
        self._drv.send_packet(PORT_SETPOINT_GENERIC, 0, data)

    def send_hover(
        self,
        vx: float = 0.0,
        vy: float = 0.0,
        yawrate: float = 0.0,
        z_distance: float = 0.5,
    ) -> None:
        """
        Hover mode: body-frame xy velocity + absolute altitude hold.

        Parameters
        ----------
        vx, vy     : float  Velocity in the drone's body frame (m/s).
        yawrate    : float  Yaw rate in deg/s.
        z_distance : float  Target altitude in metres (requires barometer/ToF).
        """
        data = bytes([TYPE_HOVER]) + struct.pack("<ffff", vx, vy, yawrate, z_distance)
        self._drv.send_packet(PORT_SETPOINT_GENERIC, 0, data)

    def send_position(
        self,
        x: float,
        y: float,
        z: float,
        yaw: float = 0.0,
    ) -> None:
        """
        Absolute position setpoint.

        The drone must be configured with the Kalman estimator for this
        to work properly.  Use send_ext_position() to feed camera-derived
        position measurements into the Kalman filter.

        Parameters
        ----------
        x, y, z : float  Target position in metres.
        yaw     : float  Target yaw in degrees.
        """
        data = bytes([TYPE_POSITION]) + struct.pack("<ffff", x, y, z, yaw)
        self._drv.send_packet(PORT_SETPOINT_GENERIC, 0, data)

    def send_alt_hold(
        self,
        roll: float,
        pitch: float,
        yawrate: float,
        z_velocity: float,
    ) -> None:
        """
        Altitude-hold mode: roll/pitch angles + vertical velocity.

        Parameters
        ----------
        roll, pitch : float  Angles in radians.
        yawrate     : float  Yaw rate in deg/s.
        z_velocity  : float  Vertical velocity in m/s.
        """
        data = bytes([TYPE_ALT_HOLD]) + struct.pack(
            "<ffff", roll, pitch, yawrate, z_velocity
        )
        self._drv.send_packet(PORT_SETPOINT_GENERIC, 0, data)

    def send_rpyt(
        self,
        roll: float,
        pitch: float,
        yaw: float,
        thrust: int,
    ) -> None:
        """
        Legacy RPYT setpoint (bare attitude + raw thrust).

        Parameters
        ----------
        roll, pitch, yaw : float  Angles in degrees.
        thrust           : int    Raw thrust value 0–65535.
        """
        data = struct.pack("<fffH", roll, pitch, yaw, int(thrust))
        self._drv.send_packet(PORT_COMMANDER, 0, data)

    # ---------------------------------------------------------------- #
    # External measurement injection (feeds the on-board Kalman filter) #
    # ---------------------------------------------------------------- #

    def send_ext_position(self, x: float, y: float, z: float) -> None:
        """
        Inject a camera-derived position measurement into the drone's
        Kalman filter (localization port, channel EXT_POSITION).

        Call this every time a new vision estimate is available.
        The drone firmware then fuses this with IMU data internally.

        Parameters
        ----------
        x, y, z : float  Measured position in metres.
        """
        data = struct.pack("<fff", x, y, z)
        self._drv.send_packet(PORT_LOCALIZATION, 0, data)

    def send_ext_pose(
        self,
        x: float,
        y: float,
        z: float,
        qx: float,
        qy: float,
        qz: float,
        qw: float,
    ) -> None:
        """
        Inject position + orientation into the drone's Kalman filter.

        Parameters
        ----------
        x, y, z          : float  Position in metres.
        qx, qy, qz, qw   : float  Orientation quaternion (Hamilton convention).
        """
        data = bytes([LOC_EXT_POSE]) + struct.pack(
            "<fffffff", x, y, z, qx, qy, qz, qw
        )
        self._drv.send_packet(PORT_LOCALIZATION, 1, data)
