"""
Simple PID controller  –  hackathon starter code.

Maps position error  →  velocity command (m/s).

Tune kp / ki / kd in main.py.  Start with only kp non-zero, add kd to
damp oscillations, add a small ki only if you see steady-state drift.
"""

import time


class PID:
    """Single-axis PID with integral anti-windup."""

    def __init__(
        self,
        kp: float = 1.0,
        ki: float = 0.0,
        kd: float = 0.0,
        output_limit: float = 1.0,
    ) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.limit = output_limit

        self._integral   = 0.0
        self._prev_error = 0.0
        self._prev_t: float | None = None

    def reset(self) -> None:
        self._integral   = 0.0
        self._prev_error = 0.0
        self._prev_t     = None

    def update(self, setpoint: float, measurement: float) -> float:
        now = time.monotonic()
        dt  = (now - self._prev_t) if self._prev_t is not None else 0.02
        self._prev_t = now

        error = setpoint - measurement

        self._integral = max(
            -self.limit, min(self.limit, self._integral + error * dt)
        )
        derivative = (error - self._prev_error) / max(dt, 1e-6)
        self._prev_error = error

        out = self.kp * error + self.ki * self._integral + self.kd * derivative
        return max(-self.limit, min(self.limit, out))


class PositionPID:
    """Three independent PIDs for x, y, z.  Returns (vx, vy, vz) in m/s."""

    def __init__(
        self,
        kp: float = 0.8,
        ki: float = 0.05,
        kd: float = 0.10,
        max_vel: float = 0.4,
    ) -> None:
        args = dict(kp=kp, ki=ki, kd=kd, output_limit=max_vel)
        self.x = PID(**args)
        self.y = PID(**args)
        self.z = PID(**args)

    def reset(self) -> None:
        self.x.reset(); self.y.reset(); self.z.reset()

    def update(
        self,
        target:  tuple[float, float, float],
        current: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        return (
            self.x.update(target[0], current[0]),
            self.y.update(target[1], current[1]),
            self.z.update(target[2], current[2]),
        )
