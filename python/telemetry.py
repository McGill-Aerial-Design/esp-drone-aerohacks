"""
Drone telemetry via CRTP LOG port (0x05)  –  hackathon starter code.

Protocol flow
─────────────
1. fetch_toc()   – ask drone for its variable list  (name → id + type)
2. start()       – create log blocks, start streaming at 20 Hz
3. get_data()    – returns latest values as a plain dict

Variables streamed (two blocks to fit within the 30-byte CRTP limit):

  Block 1 – IMU raw
    acc.x   acc.y   acc.z      accelerometer  m/s²  (or g, firmware-dependent)
    gyro.x  gyro.y  gyro.z     gyroscope      °/s

  Block 2 – state estimate + power
    stateEstimate.roll   °
    stateEstimate.pitch  °
    stateEstimate.yaw    °
    pm.vbat              V

If a variable is not found in the TOC it is silently skipped.
"""

from __future__ import annotations

import struct
import threading
import time
import logging

from crtp_driver import CRTPDriver, PORT_LOG

log = logging.getLogger(__name__)

# ── CRTP LOG channel numbers ──────────────────────────────────────── #
CH_TOC  = 0
CH_CTRL = 1
CH_DATA = 2

# ── TOC command bytes ─────────────────────────────────────────────── #
TOC_GET_ITEM = 0
TOC_GET_INFO = 1

# ── Control command bytes ─────────────────────────────────────────── #
CTRL_CREATE = 0
CTRL_START  = 3
CTRL_STOP   = 4
CTRL_RESET  = 5

# ── Log variable type IDs → (struct format char, byte size) ──────── #
# Matches log.h in the firmware
LOG_TYPES: dict[int, tuple[str, int]] = {
    1: ("B", 1),   # UINT8
    2: ("H", 2),   # UINT16
    3: ("I", 4),   # UINT32
    4: ("b", 1),   # INT8
    5: ("h", 2),   # INT16
    6: ("i", 4),   # INT32
    7: ("f", 4),   # FLOAT
}

# ── Log blocks to create ──────────────────────────────────────────── #
# Max 6 floats (24 bytes) per block; CRTP data limit is 30 bytes and
# 4 bytes are consumed by block_id + 3-byte timestamp.
LOG_BLOCKS = [
    {
        "id":     1,
        "period": 5,   # × 10 ms = 50 ms → 20 Hz
        "vars":   [
            ("acc",  "x"), ("acc",  "y"), ("acc",  "z"),
            ("gyro", "x"), ("gyro", "y"), ("gyro", "z"),
        ],
    },
    {
        "id":     2,
        "period": 5,
        "vars":   [
            ("stateEstimate", "roll"),
            ("stateEstimate", "pitch"),
            ("stateEstimate", "yaw"),
            ("stateEstimate", "z"),
            ("pm", "vbat"),
        ],
    },
]

# Flat default dict returned before telemetry is ready
_DEFAULT: dict[str, float] = {
    "acc.x": 0.0, "acc.y": 0.0, "acc.z": 0.0,
    "gyro.x": 0.0, "gyro.y": 0.0, "gyro.z": 0.0,
    "stateEstimate.roll":  0.0,
    "stateEstimate.pitch": 0.0,
    "stateEstimate.yaw":   0.0,
    "stateEstimate.z":     0.0,
    "pm.vbat": 0.0,
}


class Telemetry:
    """
    Streams sensor data from the drone via the CRTP LOG protocol.

    Usage
    -----
    tel = Telemetry(driver)
    ok  = tel.start()          # blocks ~2-5 s during TOC fetch
    data = tel.get_data()      # non-blocking, call any time
    """

    def __init__(self, driver: CRTPDriver) -> None:
        self._drv  = driver
        self._toc: dict[str, dict] = {}          # "group.name" → {id, type_id}
        self._data = dict(_DEFAULT)              # latest values
        self._lock = threading.Lock()
        # block_id → ordered list of "group.name" keys
        self._block_layout: dict[int, list[str]] = {}
        self.active = False
        self._last_rx_ts = 0.0

    # ── Public API ────────────────────────────────────────────────── #

    def start(self, timeout: float = 8.0) -> bool:
        """
        Fetch the TOC, create log blocks and start streaming.
        Returns True if at least one variable is being streamed.

        This call blocks for up to `timeout` seconds (mostly TOC fetching).
        """
        try:
            # Reset any existing blocks from a previous run
            self._send_ctrl(bytes([CTRL_RESET]))
            time.sleep(0.15)

            # --- step 1: get TOC length ---
            n_vars = self._get_toc_length(timeout=3.0)
            if n_vars is None:
                log.warning("Telemetry: TOC info request timed out – skipping")
                return False
            log.info("Telemetry: TOC contains %d variables", n_vars)

            # --- step 2: fetch all TOC items ---
            self._fetch_toc(n_vars, deadline=time.monotonic() + timeout * 0.7)

            # --- step 3: create + start each log block ---
            any_started = False
            for block_cfg in LOG_BLOCKS:
                block_id = block_cfg["id"]
                layout   = self._build_block(block_id, block_cfg["vars"])
                if not layout:
                    continue
                self._block_layout[block_id] = layout
                self._send_ctrl(
                    bytes([CTRL_CREATE, block_id])
                    + b"".join(
                        bytes([self._toc[k]["type_id"], self._toc[k]["id"]])
                        for k in layout
                    )
                )
                self._drv.recv_packet_from(PORT_LOG, CH_CTRL, timeout=1.0)

                self._send_ctrl(bytes([CTRL_START, block_id, block_cfg["period"]]))
                self._drv.recv_packet_from(PORT_LOG, CH_CTRL, timeout=1.0)

                log.info("Telemetry: block %d started  vars=%s", block_id, layout)
                any_started = True

            # --- step 4: register data callback ---
            self._drv.register_callback(PORT_LOG, CH_DATA, self._on_data)
            self.active = any_started
            return any_started

        except Exception as exc:
            log.error("Telemetry.start failed: %s", exc, exc_info=True)
            return False

    def stop(self) -> None:
        for block in LOG_BLOCKS:
            self._send_ctrl(bytes([CTRL_STOP, block["id"]]))
        self.active = False

    def get_data(self) -> dict[str, float]:
        """Return a snapshot of the latest telemetry values.  Thread-safe."""
        with self._lock:
            return dict(self._data)

    def is_connected(self, stale_after_s: float = 1.0) -> bool:
        """
        Return True when telemetry is active and data is arriving recently.

        Parameters
        ----------
        stale_after_s : float
            Maximum age of the latest telemetry packet in seconds.
        """
        if not self.active:
            return False
        with self._lock:
            last_rx = self._last_rx_ts
        return (time.monotonic() - last_rx) <= stale_after_s

    # ── Protocol helpers ──────────────────────────────────────────── #

    def _send_ctrl(self, data: bytes) -> None:
        self._drv.send_packet(PORT_LOG, CH_CTRL, data)

    def _send_toc(self, data: bytes) -> None:
        self._drv.send_packet(PORT_LOG, CH_TOC, data)

    def _get_toc_length(self, timeout: float) -> int | None:
        """Return number of TOC entries, or None on failure."""
        for _ in range(3):
            self._send_toc(bytes([TOC_GET_INFO]))
            pkt = self._drv.recv_packet_from(PORT_LOG, CH_TOC, timeout=1.0)
            if pkt and len(pkt[2]) >= 2 and pkt[2][0] == TOC_GET_INFO:
                return pkt[2][1]   # uint8 count
        return None

    # All variable names we actually need
    _WANTED: frozenset = frozenset([
        "acc.x", "acc.y", "acc.z",
        "gyro.x", "gyro.y", "gyro.z",
        "stateEstimate.roll", "stateEstimate.pitch", "stateEstimate.yaw",
        "stateEstimate.z",
        "pm.vbat",
    ])

    def _fetch_toc(self, n_vars: int, deadline: float) -> None:
        """Populate self._toc by requesting each variable entry."""
        for var_id in range(n_vars):
            if len(self._toc) == len(self._WANTED):
                log.info("Telemetry: all wanted variables found after %d items", var_id)
                break
            if time.monotonic() > deadline:
                log.warning("Telemetry: TOC fetch deadline reached at %d/%d", var_id, n_vars)
                break
            self._send_toc(bytes([TOC_GET_ITEM, var_id]))
            pkt = self._drv.recv_packet_from(PORT_LOG, CH_TOC, timeout=0.5)
            if not pkt:
                continue
            data = pkt[2]
            # Response v1: [CMD_GET_ITEM, id, type, group\0name\0]
            if len(data) < 4 or data[0] != TOC_GET_ITEM:
                continue
            type_id = data[2]
            # Split null-terminated strings
            parts = data[3:].split(b"\x00")
            if len(parts) >= 2:
                group = parts[0].decode("utf-8", errors="replace")
                name  = parts[1].decode("utf-8", errors="replace")
                key   = f"{group}.{name}"
                if key in self._WANTED:
                    self._toc[key] = {"id": var_id, "type_id": type_id}

        log.info("Telemetry: fetched %d TOC entries", len(self._toc))

    def _build_block(self, block_id: int, var_pairs: list) -> list[str]:
        """Return list of "group.name" keys that exist in the TOC."""
        found = []
        for group, name in var_pairs:
            key = f"{group}.{name}"
            if key in self._toc:
                found.append(key)
            else:
                log.debug("Telemetry: %s not in TOC", key)
        return found

    # ── Data callback (called from CRTPDriver rx thread) ─────────── #

    def _on_data(self, payload: bytes) -> None:
        """Parse a LOG data packet and update self._data."""
        if len(payload) < 4:
            return
        block_id = payload[0]
        layout   = self._block_layout.get(block_id)
        if layout is None:
            return

        raw    = payload[4:]   # skip block_id + 3-byte timestamp
        offset = 0
        updates: dict[str, float] = {}

        for key in layout:
            type_id = self._toc[key]["type_id"]
            if type_id not in LOG_TYPES:
                continue
            fmt, size = LOG_TYPES[type_id]
            if offset + size > len(raw):
                break
            val = struct.unpack_from("<" + fmt, raw, offset)[0]
            updates[key] = float(val)
            offset += size

        with self._lock:
            self._last_rx_ts = time.monotonic()
            self._data.update(updates)
