"""
Low-level CRTP-over-UDP driver for ESP-Drone.

Packet wire format (from wifi_esp32.c):
    [CRTP header (1 byte)] + [CRTP data (0-30 bytes)] + [CKSUM (1 byte)]

CRTP header byte:
    bits 7-4  : port  (4 bits)
    bits 3-2  : reserved (2 bits, always 0)
    bits 1-0  : channel (2 bits)

CKSUM = sum of all preceding bytes, mod 256.

The drone listens on UDP port 2390 and replies to whatever source address
sent the packet (no fixed client port needed).
"""

import logging
import queue
import socket
import threading
import time

log = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Default network parameters                                           #
# ------------------------------------------------------------------ #

DRONE_DEFAULT_IP  = "192.168.43.42"   # ESP-Drone AP default gateway
DRONE_UDP_PORT    = 2390               # Drone listens here
UDP_BUFSIZE       = 64

# ------------------------------------------------------------------ #
# CRTP port constants (crtp.h)                                        #
# ------------------------------------------------------------------ #

PORT_CONSOLE          = 0x00
PORT_PARAM            = 0x02
PORT_COMMANDER        = 0x03   # Legacy RPYT
PORT_LOG              = 0x05
PORT_LOCALIZATION     = 0x06
PORT_SETPOINT_GENERIC = 0x07
PORT_HIGH_LEVEL       = 0x08


class CRTPDriver:
    """
    Thread-safe CRTP-over-UDP driver for ESP-Drone.

    Usage
    -----
    drv = CRTPDriver("192.168.43.42")
    drv.connect()
    drv.send_packet(PORT_SETPOINT_GENERIC, 0, bytes([0]))  # stop
    drv.disconnect()
    """

    def __init__(self, drone_ip: str = DRONE_DEFAULT_IP) -> None:
        self.drone_addr = (drone_ip, DRONE_UDP_PORT)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.settimeout(0.1)
        self._rx_queue: queue.Queue = queue.Queue(maxsize=64)
        self._callbacks: dict = {}   # (port, channel) → callable(data: bytes)
        self._running = False
        self._rx_thread: threading.Thread | None = None

    # ---------------------------------------------------------------- #
    # Lifecycle                                                          #
    # ---------------------------------------------------------------- #

    def connect(self) -> None:
        """Open socket and start receive thread."""
        self._running = True
        self._rx_thread = threading.Thread(
            target=self._recv_loop, name="crtp-rx", daemon=True
        )
        self._rx_thread.start()
        log.info("CRTP driver connected → %s:%d", *self.drone_addr)

    def disconnect(self) -> None:
        """Stop receive thread and close socket."""
        self._running = False
        if self._rx_thread:
            self._rx_thread.join(timeout=2.0)
        try:
            self._sock.close()
        except OSError:
            pass
        log.info("CRTP driver disconnected")

    # ---------------------------------------------------------------- #
    # Transmit                                                           #
    # ---------------------------------------------------------------- #

    def send_packet(self, port: int, channel: int, data: bytes) -> None:
        """
        Encode and transmit one CRTP packet.

        Parameters
        ----------
        port    : CRTP port (0x00–0x0F)
        channel : CRTP channel (0–3)
        data    : payload bytes (max 30 bytes)
        """
        header  = ((port & 0x0F) << 4) | (channel & 0x03)
        payload = bytes([header]) + data
        cksum   = sum(payload) & 0xFF
        self._sock.sendto(payload + bytes([cksum]), self.drone_addr)

    # ---------------------------------------------------------------- #
    # Receive                                                            #
    # ---------------------------------------------------------------- #

    def recv_packet(self, timeout: float = 0.1):
        """
        Return the next received packet as (port, channel, data) or None.

        Parameters
        ----------
        timeout : seconds to wait before returning None
        """
        try:
            return self._rx_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def recv_packet_from(self, port: int, channel: int, timeout: float = 1.0):
        """
        Block until a packet from (port, channel) arrives, then return it.
        Other packets received while waiting are discarded.
        Returns None on timeout.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            pkt = self.recv_packet(timeout=min(remaining, 0.05))
            if pkt is not None and pkt[0] == port and pkt[1] == channel:
                return pkt
        return None

    def register_callback(self, port: int, channel: int, fn) -> None:
        """
        Route all incoming packets on (port, channel) to fn(data: bytes)
        instead of the rx queue.  Call before connect() or at any time.
        """
        self._callbacks[(port, channel)] = fn

    # ---------------------------------------------------------------- #
    # Internal                                                           #
    # ---------------------------------------------------------------- #

    def _recv_loop(self) -> None:
        while self._running:
            try:
                raw, _ = self._sock.recvfrom(UDP_BUFSIZE)
            except socket.timeout:
                continue
            except OSError:
                break

            if len(raw) < 2:
                continue

            # Validate checksum
            expected = sum(raw[:-1]) & 0xFF
            if expected != raw[-1]:
                log.debug("Dropped packet with bad checksum")
                continue

            port    = (raw[0] >> 4) & 0x0F
            channel =  raw[0]       & 0x03
            payload =  raw[1:-1]

            key = (port, channel)
            if key in self._callbacks:
                try:
                    self._callbacks[key](payload)
                except Exception as exc:
                    log.warning("Callback error on port %x ch %d: %s", port, channel, exc, exc_info=True)
            else:
                try:
                    self._rx_queue.put_nowait((port, channel, payload))
                except queue.Full:
                    pass  # silently drop oldest
