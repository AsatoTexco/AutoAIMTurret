import http.client
import json
import math
import os
import queue
import socket
import threading
import time
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests
from requests.adapters import HTTPAdapter

try:
    import ncnn
except ImportError:
    ncnn = None

try:
    import onnxruntime as ort
except ImportError:
    ort = None


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_dotenv(env_path: str) -> None:
    """Load key=value pairs from a local .env file without overriding existing env vars."""
    if not os.path.exists(env_path):
        return
    try:
        with open(env_path, "r", encoding="utf-8") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[7:].strip()
                if "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip()
                if not key:
                    continue
                if len(val) >= 2 and val[0] == val[-1] and val[0] in ('"', "'"):
                    val = val[1:-1]
                os.environ.setdefault(key, val)
    except OSError:
        pass


_load_dotenv(os.path.join(BASE_DIR, ".env"))

DEFAULT_TURRET_URL = os.environ.get("TURRET_URL", "http://192.168.100.5")
DEFAULT_STREAM_URL = os.environ.get("STREAM_URL", "http://192.168.100.5:81/stream")
DEFAULT_BACKEND = os.environ.get("AUTOAIM_BACKEND", "auto").lower()
DEFAULT_CPU_THREADS = max(1, int(os.environ.get("AUTOAIM_CPU_THREADS", "4")))
DEFAULT_MAX_FPS = float(os.environ.get("AUTOAIM_MAX_FPS", "30.0"))
DEFAULT_CAMERA_LATENCY = max(0.0, min(0.5, float(os.environ.get("AUTOAIM_CAMERA_LATENCY", "0.08"))))
DEFAULT_DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "").strip()
DEFAULT_DISCORD_COOLDOWN_S = max(5.0, float(os.environ.get("DISCORD_COOLDOWN_S", "30.0")))
DEFAULT_DISCORD_ABSENCE_RESET_S = max(3.0, float(os.environ.get("DISCORD_ABSENCE_RESET_S", "15.0")))
DEFAULT_DISCORD_CONNECT_TIMEOUT_S = max(3.0, float(os.environ.get("DISCORD_CONNECT_TIMEOUT_S", "10.0")))
DEFAULT_DISCORD_READ_TIMEOUT_S = max(5.0, float(os.environ.get("DISCORD_READ_TIMEOUT_S", "25.0")))

DEFAULT_YOLO_PATH = os.path.join(BASE_DIR, "models", "yolov8n.onnx")
DEFAULT_YUNET_PATH = os.path.join(BASE_DIR, "models", "face_detection_yunet_2023mar.onnx")

ROTATION_CODES = {
    0: None,
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}

COCO_CLASSES = {
    "person": (0, "PESSOA"),
    "dog": (16, "CACHORRO"),
}

SOURCE_LABELS = {
    "person": "PESSOA",
    "dog": "CACHORRO",
    "face": "ROSTO",
}

RESOLUTION_OPTIONS = {
    0: ("QVGA", 320, 240),
    1: ("CIF", 400, 296),
    2: ("VGA", 640, 480),
    3: ("SVGA", 800, 600),
    4: ("WIDE", 320, 176),
    5: ("WIDE+", 400, 224),
}

CAM_GAIN_OPTIONS = {
    0: "2x",
    1: "4x",
    2: "8x",
    3: "16x",
    4: "32x",
    5: "64x",
}

DEFAULT_CAM_RES = 5
DEFAULT_CAM_QUAL = 16
DEFAULT_CAM_BRI = 1
DEFAULT_CAM_AE = 1
DEFAULT_CAM_GAIN = 4
DEFAULT_CAM_NIGHT = False


def infer_resolution_code(width: int, height: int) -> int:
    long_edge = max(width, height)
    short_edge = max(1, min(width, height))
    if (long_edge / float(short_edge)) >= 1.55:
        return 4 if long_edge <= 360 else 5
    best_code = 1
    best_diff = 10**9
    for code in (0, 1, 2, 3):
        _, rw, _ = RESOLUTION_OPTIONS[code]
        diff = abs(long_edge - rw)
        if diff < best_diff:
            best_diff = diff
            best_code = code
    return best_code


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    source: str = "person"

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) * 0.5, (self.y1 + self.y2) * 0.5)

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


class _TCPNoDelayHTTPAdapter(HTTPAdapter):
    """Requests HTTPAdapter with TCP_NODELAY (and optional SO_KEEPALIVE) enabled on every socket."""

    def __init__(self, *args, keepalive: bool = True, **kwargs) -> None:
        self._keepalive = keepalive
        super().__init__(*args, **kwargs)

    def init_poolmanager(self, connections, maxsize, block=False, **pool_kwargs):
        sock_opts = [(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)]
        if self._keepalive:
            sock_opts.append((socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1))
        pool_kwargs["socket_options"] = sock_opts
        return super().init_poolmanager(connections, maxsize, block=block, **pool_kwargs)


class MJPEGStreamReader:
    """Low-latency MJPEG reader tuned for ESP32-CAM port 81 HTTP/1.1 streaming.

    - Uses http.client + SO_KEEPALIVE + TCP_NODELAY + TCP_QUICKACK (Linux) + read1().
    - Parses multipart boundary + exact Content-Length headers (with SOI/EOI fallback).
    - Validates JPEG SOI (0xFF 0xD8), drains stale frames, and decodes only the newest frame.
    - Wakes consumers immediately via Condition (wait_for_frame) with zero sleep-polling jitter.
    - Explicitly closes the socket on stop() and reconnects so ESP32 port 81 frees immediately.
    """

    MAX_JPEG_BYTES = 2 * 1024 * 1024

    def __init__(self, stream_url: str = DEFAULT_STREAM_URL, rotation: int = 90) -> None:
        self.stream_url = stream_url
        self._rotation = rotation if rotation in ROTATION_CODES else 90
        self._lock = threading.Lock()
        self._frame_cond = threading.Condition(self._lock)
        self._conn_lock = threading.Lock()
        self._active_conn: Optional[http.client.HTTPConnection] = None
        self._stop_event = threading.Event()
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_id: int = 0
        self._last_frame_time: float = 0.0
        self._connected: bool = False
        self._last_error: str = ""
        self._stream_fps: float = 0.0
        self._last_frame_bytes: int = 0
        self._frames_dropped: int = 0
        self._thread = threading.Thread(target=self._reader_loop, daemon=True, name="MJPEGReader")

    def start(self) -> None:
        if not self._thread.is_alive():
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._reader_loop, daemon=True, name="MJPEGReader")
            self._thread.start()

    def _abort_active_conn(self) -> None:
        with self._conn_lock:
            conn = self._active_conn
            self._active_conn = None
        if conn is not None:
            sock = getattr(conn, "sock", None)
            if sock is not None:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except Exception:
                    pass
                try:
                    sock.close()
                except Exception:
                    pass
            try:
                conn.close()
            except Exception:
                pass

    def stop(self) -> None:
        self._stop_event.set()
        with self._frame_cond:
            self._frame_cond.notify_all()
        self._abort_active_conn()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def set_rotation(self, rotation: int) -> None:
        if rotation in ROTATION_CODES:
            with self._lock:
                self._rotation = rotation

    def get_rotation(self) -> int:
        with self._lock:
            return self._rotation

    def is_connected(self) -> bool:
        with self._lock:
            return self._connected and (time.time() - self._last_frame_time < 3.0)

    def get_stats(self) -> Dict[str, object]:
        with self._lock:
            return {
                "connected": self._connected and (time.time() - self._last_frame_time < 3.0),
                "stream_fps": self._stream_fps,
                "last_frame_bytes": self._last_frame_bytes,
                "frames_dropped": self._frames_dropped,
                "last_error": self._last_error,
            }

    def get_latest_frame(self) -> Tuple[Optional[np.ndarray], int, float]:
        with self._lock:
            if self._latest_frame is None:
                return None, self._frame_id, self._last_frame_time
            return self._latest_frame, self._frame_id, self._last_frame_time

    def wait_for_frame(
        self, last_frame_id: int, timeout: float = 0.05
    ) -> Tuple[Optional[np.ndarray], int, float]:
        """Block until a newer frame arrives (or timeout), avoiding sleep-polling jitter."""
        with self._frame_cond:
            if (self._latest_frame is None or self._frame_id == last_frame_id) and not self._stop_event.is_set():
                self._frame_cond.wait(timeout=timeout)
            return self._latest_frame, self._frame_id, self._last_frame_time

    @staticmethod
    def _extract_boundary(content_type: str) -> Optional[bytes]:
        for part in (content_type or "").split(";"):
            part = part.strip()
            if part.lower().startswith("boundary="):
                bval = part.split("=", 1)[1].strip().strip('"').strip("'")
                if bval:
                    if not bval.startswith("--"):
                        bval = "--" + bval
                    return bval.encode("ascii", errors="ignore")
        return None

    @staticmethod
    def _parse_content_length(header_bytes: bytes) -> Optional[int]:
        for raw_line in header_bytes.splitlines():
            low = raw_line.lower()
            if low.startswith(b"content-length:"):
                try:
                    val = int(raw_line.split(b":", 1)[1].strip())
                    if 0 < val <= MJPEGStreamReader.MAX_JPEG_BYTES:
                        return val
                except ValueError:
                    return None
        return None

    @classmethod
    def _drain_frames_from_buffer(
        cls, buf: bytearray, boundary: Optional[bytes]
    ) -> Tuple[Optional[bytes], int]:
        """Extract all complete JPEG frames from buf, returning (newest_jpeg, dropped_count)."""
        newest_jpg: Optional[bytes] = None
        dropped = 0

        if boundary is not None:
            blen = len(boundary)
            while True:
                b_idx = buf.find(boundary)
                if b_idx == -1:
                    # Wait for boundary unless buffer grew abnormally without one
                    if len(buf) > cls.MAX_JPEG_BYTES:
                        del buf[:-blen]
                    break
                if b_idx > 0:
                    del buf[:b_idx]

                hdr_end = buf.find(b"\r\n\r\n", blen)
                sep_len = 4
                if hdr_end == -1:
                    hdr_end = buf.find(b"\n\n", blen)
                    sep_len = 2
                if hdr_end == -1:
                    if len(buf) > 4096:
                        del buf[:blen]
                        continue
                    break

                header_block = bytes(buf[blen:hdr_end])
                body_start = hdr_end + sep_len
                clen = cls._parse_content_length(header_block)

                if clen is not None:
                    body_end = body_start + clen
                    if len(buf) < body_end:
                        break
                    candidate = bytes(buf[body_start:body_end])
                    del buf[:body_end]
                    if len(candidate) >= 4 and candidate[0] == 0xFF and candidate[1] == 0xD8:
                        if newest_jpg is not None:
                            dropped += 1
                        newest_jpg = candidate
                    continue

                # Boundary present but Content-Length omitted: slice until next boundary
                next_b = buf.find(boundary, body_start)
                if next_b == -1:
                    if len(buf) > cls.MAX_JPEG_BYTES:
                        buf.clear()
                    break
                candidate = bytes(buf[body_start:next_b]).strip(b"\r\n")
                del buf[:next_b]
                if len(candidate) >= 4 and candidate[0] == 0xFF and candidate[1] == 0xD8:
                    if newest_jpg is not None:
                        dropped += 1
                    newest_jpg = candidate
            return newest_jpg, dropped

        # Fallback SOI/EOI extraction for streams without multipart boundary
        while True:
            start = buf.find(b"\xff\xd8\xff")
            if start == -1:
                if len(buf) > 65536:
                    buf.clear()
                break
            if start > 0:
                del buf[:start]
            end = buf.find(b"\xff\xd9", 3)
            if end == -1:
                if len(buf) > cls.MAX_JPEG_BYTES:
                    buf.clear()
                break
            last_soi = buf.rfind(b"\xff\xd8\xff", 0, end)
            if last_soi > 0:
                del buf[:last_soi]
                end -= last_soi
            if newest_jpg is not None:
                dropped += 1
            newest_jpg = bytes(buf[: end + 2])
            del buf[: end + 2]
        return newest_jpg, dropped

    def _reader_loop(self) -> None:
        backoff_s = 0.25
        stream_fps_ema = 0.0
        tcp_quickack_opt = getattr(socket, "TCP_QUICKACK", None)

        while not self._stop_event.is_set():
            conn: Optional[http.client.HTTPConnection] = None
            prev_frame_mono = 0.0
            try:
                parsed = urllib.parse.urlparse(self.stream_url)
                use_tls = parsed.scheme.lower() == "https"
                host = parsed.hostname
                if not host:
                    raise ValueError(f"Invalid stream URL: {self.stream_url}")
                port = parsed.port or (443 if use_tls else 80)
                path = parsed.path or "/stream"
                if parsed.query:
                    path = f"{path}?{parsed.query}"

                if use_tls:
                    conn = http.client.HTTPSConnection(host, port, timeout=4.0)
                else:
                    conn = http.client.HTTPConnection(host, port, timeout=4.0)

                with self._conn_lock:
                    if self._stop_event.is_set():
                        break
                    self._active_conn = conn

                conn.connect()
                raw_sock = conn.sock
                if raw_sock is not None:
                    try:
                        raw_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                        raw_sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
                        raw_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 131072)
                        for opt_name, val in (("TCP_KEEPIDLE", 3), ("TCP_KEEPINTVL", 1), ("TCP_KEEPCNT", 3)):
                            opt = getattr(socket, opt_name, None)
                            if opt is not None:
                                try:
                                    raw_sock.setsockopt(socket.IPPROTO_TCP, opt, val)
                                except OSError:
                                    pass
                        if tcp_quickack_opt is not None:
                            try:
                                raw_sock.setsockopt(socket.IPPROTO_TCP, tcp_quickack_opt, 1)
                            except OSError:
                                pass
                    except OSError:
                        pass

                conn.request(
                    "GET",
                    path,
                    headers={
                        "Connection": "close",
                        "Cache-Control": "no-cache",
                        "Accept": "multipart/x-mixed-replace, image/jpeg, */*",
                    },
                )
                resp = conn.getresponse()
                if resp.status != 200:
                    raise ConnectionError(f"Stream HTTP {resp.status} {resp.reason}")

                boundary = self._extract_boundary(resp.getheader("Content-Type", ""))
                buf = bytearray()
                backoff_s = 0.25

                while not self._stop_event.is_set():
                    chunk = resp.read1(32768)
                    if not chunk:
                        break
                    if tcp_quickack_opt is not None and raw_sock is not None:
                        try:
                            raw_sock.setsockopt(socket.IPPROTO_TCP, tcp_quickack_opt, 1)
                        except OSError:
                            pass
                    buf.extend(chunk)

                    newest_jpg, dropped = self._drain_frames_from_buffer(buf, boundary)
                    if newest_jpg is not None:
                        arr = np.frombuffer(newest_jpg, dtype=np.uint8)
                        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if frame is not None:
                            with self._lock:
                                rot_code = ROTATION_CODES.get(self._rotation)
                            if rot_code is not None:
                                frame = cv2.rotate(frame, rot_code)
                            now_wall = time.time()
                            now_mono = time.monotonic()
                            if prev_frame_mono > 0.0:
                                dt = max(1e-3, now_mono - prev_frame_mono)
                                inst_fps = 1.0 / dt
                                alpha = 1.0 - math.exp(-dt / 0.8)
                                stream_fps_ema = (
                                    inst_fps
                                    if stream_fps_ema == 0.0
                                    else ((1.0 - alpha) * stream_fps_ema + alpha * inst_fps)
                                )
                            prev_frame_mono = now_mono
                            with self._frame_cond:
                                self._latest_frame = frame
                                self._frame_id += 1
                                self._last_frame_time = now_wall
                                self._connected = True
                                self._last_error = ""
                                self._stream_fps = stream_fps_ema
                                self._last_frame_bytes = len(newest_jpg)
                                self._frames_dropped += dropped
                                self._frame_cond.notify_all()
            except Exception as exc:
                with self._lock:
                    self._connected = False
                    self._last_error = str(exc)
            finally:
                with self._conn_lock:
                    if self._active_conn is conn:
                        self._active_conn = None
                if conn is not None:
                    sock = getattr(conn, "sock", None)
                    if sock is not None:
                        try:
                            sock.shutdown(socket.SHUT_RDWR)
                        except Exception:
                            pass
                        try:
                            sock.close()
                        except Exception:
                            pass
                    try:
                        conn.close()
                    except Exception:
                        pass
            if not self._stop_event.is_set():
                self._stop_event.wait(timeout=backoff_s)
                backoff_s = min(1.2, backoff_s * 1.4)


class TurretHardwareClient:
    """Single-flight HTTP client for ESP32-CAM port 80 (GET /servo + GET /cam).

    - Reuses a persistent keep-alive TCP connection with TCP_NODELAY on port 80 and <= 95-char query guard.
    - Endpoint separation matching ESP32 firmware:
        * GET /servo -> real-time actuators only (x=0..180, y=0..180, led=0|1, laser=0|1).
          Servos apply immediately (no ramp) with 500 ms PWM Auto-Detach on ESP32; sends minimal
          delta queries ("laser=1", "led=0") when servos are parked so Auto-Detach stays undisturbed.
          0 ms immediate priority for laser/led; 35 ms minimum interval (~28.6 Hz) to cut Port 80/81 Wi-Fi contention.
        * GET /cam -> camera settings (res=0..5, qual=10..40, bri=-3..3, ae=-5..5, gain=0..5, night=0|1),
          dispatched with 0 ms priority without waking servos.
    - Safety fail-safe:
        * Always transmits GET /servo?laser=0&led=0 on shutdown() and supports runtime
          force_failsafe() so laser/flash never stay stuck ON.
    """

    HTTP_MAX_QUERY_CHARS = 95
    HTTP_MOVE_MIN_INTERVAL_S = 0.035

    def __init__(
        self,
        base_url: str = DEFAULT_TURRET_URL,
        move_min_interval: Optional[float] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._move_min_interval: float = (
            max(0.0, float(move_min_interval))
            if move_min_interval is not None
            else self.HTTP_MOVE_MIN_INTERVAL_S
        )

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._wakeup_event = threading.Event()

        self._session = requests.Session()
        adapter = _TCPNoDelayHTTPAdapter(pool_connections=1, pool_maxsize=2, keepalive=True)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)
        self._session.headers.update({"Connection": "keep-alive"})

        # Real-time /servo targets & last-sent values
        self._target_x: int = 90
        self._target_y: int = 90
        self._target_led: bool = False
        self._target_laser: bool = False

        self._sent_x: Optional[int] = None
        self._sent_y: Optional[int] = None
        self._sent_led: Optional[bool] = None
        self._sent_laser: Optional[bool] = None

        # Eventual /cam targets & last-sent values
        self._target_cam_res: Optional[int] = None
        self._target_cam_qual: Optional[int] = None
        self._target_cam_bri: Optional[int] = None
        self._target_cam_ae: Optional[int] = None
        self._target_cam_gain: Optional[int] = None
        self._target_cam_night: Optional[bool] = None

        self._sent_cam_res: Optional[int] = None
        self._sent_cam_qual: Optional[int] = None
        self._sent_cam_bri: Optional[int] = None
        self._sent_cam_ae: Optional[int] = None
        self._sent_cam_gain: Optional[int] = None
        self._sent_cam_night: Optional[bool] = None

        self._last_send_monotonic: float = 0.0
        self._last_rtt_ms: float = 0.0
        self._last_error: str = ""
        self._last_url: str = ""
        self._last_payload: str = ""
        self._last_cam_payload: str = ""
        self._pose_history: List[Tuple[float, int, int]] = [(time.time(), 90, 90)]

        self._thread = threading.Thread(target=self._worker_loop, daemon=True, name="TurretControl")

    def start(self) -> None:
        if not self._thread.is_alive():
            self._stop_event.clear()
            self._wakeup_event.clear()
            self._thread = threading.Thread(target=self._worker_loop, daemon=True, name="TurretControl")
            self._thread.start()

    def get_effective_move_interval(self) -> float:
        return self._move_min_interval

    def set_state(
        self,
        x: Optional[float] = None,
        y: Optional[float] = None,
        led: Optional[bool] = None,
        laser: Optional[bool] = None,
    ) -> None:
        changed = False
        with self._lock:
            if x is not None:
                ix = max(0, min(180, int(round(x))))
                if ix != self._target_x or ix != self._sent_x:
                    self._target_x = ix
                    changed = True
            if y is not None:
                iy = max(0, min(180, int(round(y))))
                if iy != self._target_y or iy != self._sent_y:
                    self._target_y = iy
                    changed = True
            if led is not None:
                bled = bool(led)
                if bled != self._target_led or bled != self._sent_led:
                    self._target_led = bled
                    changed = True
            if laser is not None:
                blaser = bool(laser)
                if blaser != self._target_laser or blaser != self._sent_laser:
                    self._target_laser = blaser
                    changed = True
        if changed:
            self._wakeup_event.set()

    def set_cam_settings(
        self,
        res: Optional[int] = None,
        qual: Optional[int] = None,
        bri: Optional[int] = None,
        ae: Optional[int] = None,
        gain: Optional[int] = None,
        night: Optional[bool] = None,
        force: bool = False,
    ) -> None:
        changed = False
        with self._lock:
            if res is not None:
                ires = max(0, min(5, int(res)))
                if force or ires != self._target_cam_res or ires != self._sent_cam_res:
                    self._target_cam_res = ires
                    if force:
                        self._sent_cam_res = None
                    changed = True
            if qual is not None:
                iqual = max(10, min(40, int(qual)))
                if force or iqual != self._target_cam_qual or iqual != self._sent_cam_qual:
                    self._target_cam_qual = iqual
                    if force:
                        self._sent_cam_qual = None
                    changed = True
            if bri is not None:
                ibri = max(-3, min(3, int(bri)))
                if force or ibri != self._target_cam_bri or ibri != self._sent_cam_bri:
                    self._target_cam_bri = ibri
                    if force:
                        self._sent_cam_bri = None
                    changed = True
            if ae is not None:
                iae = max(-5, min(5, int(ae)))
                if force or iae != self._target_cam_ae or iae != self._sent_cam_ae:
                    self._target_cam_ae = iae
                    if force:
                        self._sent_cam_ae = None
                    changed = True
            if gain is not None:
                igain = max(0, min(5, int(gain)))
                if force or igain != self._target_cam_gain or igain != self._sent_cam_gain:
                    self._target_cam_gain = igain
                    if force:
                        self._sent_cam_gain = None
                    changed = True
            if night is not None:
                bnight = bool(night)
                if force or bnight != self._target_cam_night or bnight != self._sent_cam_night:
                    self._target_cam_night = bnight
                    if force:
                        self._sent_cam_night = None
                    changed = True
        if changed:
            self._wakeup_event.set()

    def set_angles(self, x: float, y: float) -> None:
        self.set_state(x=x, y=y)

    def set_led(self, state: bool) -> None:
        self.set_state(led=state)

    def set_laser(self, state: bool) -> None:
        self.set_state(laser=state)

    def center(self) -> None:
        with self._lock:
            self._target_x = 90
            self._target_y = 90
            self._sent_x = None
            self._sent_y = None
        self._wakeup_event.set()

    def force_failsafe(self) -> None:
        """Force immediate transmission of laser=0&led=0 regardless of cached sent state."""
        with self._lock:
            self._target_led = False
            self._target_laser = False
            self._sent_led = None
            self._sent_laser = None
        self._wakeup_event.set()

    def set_resolution(self, res_code: int, force: bool = False) -> None:
        self.set_cam_settings(res=res_code, force=force)

    def get_pose_at(self, timestamp: float) -> Tuple[float, float]:
        with self._lock:
            selected = self._pose_history[0]
            for t_ack, px, py in self._pose_history:
                if t_ack <= timestamp:
                    selected = (t_ack, px, py)
                else:
                    break
            return float(selected[1]), float(selected[2])

    def get_state(self) -> Dict[str, object]:
        with self._lock:
            move_ms = int(round(self._move_min_interval * 1000.0))
            return {
                "target_x": self._target_x,
                "target_y": self._target_y,
                "target_led": self._target_led,
                "target_laser": self._target_laser,
                "target_res": self._target_cam_res,
                "sent_x": self._sent_x,
                "sent_y": self._sent_y,
                "sent_led": self._sent_led,
                "sent_laser": self._sent_laser,
                "sent_res": self._sent_cam_res,
                "move_interval_ms": move_ms,
                "last_rtt_ms": round(self._last_rtt_ms, 1),
                "last_error": self._last_error,
                "last_url": self._last_url,
                "last_payload": self._last_payload,
                "last_cam_payload": self._last_cam_payload,
            }

    def shutdown(self) -> None:
        self._stop_event.set()
        self._wakeup_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.5)
        # Guaranteed HTTP confirmation so laser/flash NEVER remain stuck ON
        try:
            self._session.get(
                f"{self.base_url}/servo?laser=0&led=0",
                timeout=(0.8, 1.0),
            )
        except Exception:
            pass
        try:
            self._session.close()
        except Exception:
            pass

    @classmethod
    def _build_servo_query(
        cls,
        tx: int,
        ty: int,
        tled: bool,
        tlaser: bool,
        sx: Optional[int],
        sy: Optional[int],
        sled: Optional[bool],
        slaser: Optional[bool],
    ) -> Tuple[str, bool, bool, bool]:
        """Build optimal HTTP GET /servo query string (strictly <= 95 chars; x, y, led, laser only)."""
        led_i = 1 if tled else 0
        laser_i = 1 if tlaser else 0
        angles_dirty = (sx is None) or (sy is None) or (tx != sx) or (ty != sy)
        led_dirty = (sled is None) or (tled != sled)
        laser_dirty = (slaser is None) or (tlaser != slaser)

        parts: List[str] = []
        sent_angles = False
        sent_led = False
        sent_laser = False

        if angles_dirty:
            parts.extend((f"x={tx}", f"y={ty}", f"laser={laser_i}", f"led={led_i}"))
            sent_angles = True
            sent_led = True
            sent_laser = True
        else:
            if laser_dirty:
                parts.append(f"laser={laser_i}")
                sent_laser = True
            if led_dirty:
                parts.append(f"led={led_i}")
                sent_led = True

        if not parts:
            parts.extend((f"x={tx}", f"y={ty}", f"laser={laser_i}", f"led={led_i}"))
            sent_angles = True
            sent_led = True
            sent_laser = True

        query = "&".join(parts)
        if len(query) > cls.HTTP_MAX_QUERY_CHARS:
            raise ValueError(f"HTTP /servo query exceeds {cls.HTTP_MAX_QUERY_CHARS} chars: {query}")
        return query, sent_angles, sent_led, sent_laser

    @classmethod
    def _build_cam_query(
        cls,
        tres: Optional[int],
        tqual: Optional[int],
        tbri: Optional[int],
        tae: Optional[int],
        tgain: Optional[int],
        tnight: Optional[bool],
        sres: Optional[int],
        squal: Optional[int],
        sbri: Optional[int],
        sae: Optional[int],
        sgain: Optional[int],
        snight: Optional[bool],
    ) -> Tuple[str, Optional[int], Optional[int], Optional[int], Optional[int], Optional[int], Optional[bool]]:
        """Build optimal HTTP GET /cam query string (strictly <= 95 chars)."""
        parts: List[str] = []
        out_res: Optional[int] = None
        out_qual: Optional[int] = None
        out_bri: Optional[int] = None
        out_ae: Optional[int] = None
        out_gain: Optional[int] = None
        out_night: Optional[bool] = None

        if tres is not None and tres != sres:
            parts.append(f"res={tres}")
            out_res = tres
        if tqual is not None and tqual != squal:
            parts.append(f"qual={tqual}")
            out_qual = tqual
        if tbri is not None and tbri != sbri:
            parts.append(f"bri={tbri}")
            out_bri = tbri
        if tae is not None and tae != sae:
            parts.append(f"ae={tae}")
            out_ae = tae
        if tgain is not None and tgain != sgain:
            parts.append(f"gain={tgain}")
            out_gain = tgain
        if tnight is not None and tnight != snight:
            parts.append(f"night={1 if tnight else 0}")
            out_night = tnight

        query = "&".join(parts)
        if len(query) > cls.HTTP_MAX_QUERY_CHARS:
            raise ValueError(f"HTTP /cam query exceeds {cls.HTTP_MAX_QUERY_CHARS} chars: {query}")
        return query, out_res, out_qual, out_bri, out_ae, out_gain, out_night

    def _record_servo_sent(
        self,
        tx: int,
        ty: int,
        tled: bool,
        tlaser: bool,
        sent_angles: bool,
        sent_led: bool,
        sent_laser: bool,
        rtt_ms: float,
        last_url: str,
        last_payload: str,
    ) -> None:
        with self._lock:
            if sent_angles:
                self._sent_x = tx
                self._sent_y = ty
                self._pose_history.append((time.time(), tx, ty))
                if len(self._pose_history) > 60:
                    del self._pose_history[:-60]
            if sent_led:
                self._sent_led = tled
            if sent_laser:
                self._sent_laser = tlaser
            self._last_rtt_ms = rtt_ms
            self._last_error = ""
            self._last_url = last_url
            self._last_payload = last_payload

    def _record_cam_sent(
        self,
        sent_res: Optional[int],
        sent_qual: Optional[int],
        sent_bri: Optional[int],
        sent_ae: Optional[int],
        sent_gain: Optional[int],
        sent_night: Optional[bool],
        rtt_ms: float,
        last_url: str,
        last_payload: str,
    ) -> None:
        with self._lock:
            if sent_res is not None:
                self._sent_cam_res = sent_res
            if sent_qual is not None:
                self._sent_cam_qual = sent_qual
            if sent_bri is not None:
                self._sent_cam_bri = sent_bri
            if sent_ae is not None:
                self._sent_cam_ae = sent_ae
            if sent_gain is not None:
                self._sent_cam_gain = sent_gain
            if sent_night is not None:
                self._sent_cam_night = sent_night
            self._last_rtt_ms = rtt_ms
            self._last_error = ""
            self._last_url = last_url
            self._last_payload = last_payload
            self._last_cam_payload = last_payload

    def _http_get_with_keepalive_retry(self, full_url: str, timeout: Tuple[float, float]) -> requests.Response:
        """Execute keep-alive GET, transparently retrying once on stale pooled socket close."""
        try:
            r = self._session.get(full_url, timeout=timeout)
            _ = r.content
            return r
        except requests.exceptions.ConnectionError:
            r = self._session.get(full_url, timeout=timeout)
            _ = r.content
            return r

    def _dispatch_servo(
        self,
        tx: int,
        ty: int,
        tled: bool,
        tlaser: bool,
        sx: Optional[int],
        sy: Optional[int],
        sled: Optional[bool],
        slaser: Optional[bool],
    ) -> bool:
        query, s_ang, s_led, s_las = self._build_servo_query(
            tx, ty, tled, tlaser, sx, sy, sled, slaser
        )
        full_url = f"{self.base_url}/servo?{query}"
        t0 = time.monotonic()
        self._last_send_monotonic = t0
        try:
            r = self._http_get_with_keepalive_retry(full_url, timeout=(1.0, 2.5))
            rtt_ms = (time.monotonic() - t0) * 1000.0
            if r.status_code == 200:
                self._record_servo_sent(
                    tx=tx,
                    ty=ty,
                    tled=tled,
                    tlaser=tlaser,
                    sent_angles=s_ang,
                    sent_led=s_led,
                    sent_laser=s_las,
                    rtt_ms=rtt_ms,
                    last_url=full_url,
                    last_payload=f"GET /servo?{query}",
                )
                return True
            with self._lock:
                self._last_error = f"HTTP /servo {r.status_code}"
            self._stop_event.wait(timeout=0.12)
            return False
        except Exception as exc:
            with self._lock:
                self._last_error = str(exc)
            self._stop_event.wait(timeout=0.20)
            return False

    def _dispatch_cam(
        self,
        tres: Optional[int],
        tqual: Optional[int],
        tbri: Optional[int],
        tae: Optional[int],
        tgain: Optional[int],
        tnight: Optional[bool],
        sres: Optional[int],
        squal: Optional[int],
        sbri: Optional[int],
        sae: Optional[int],
        sgain: Optional[int],
        snight: Optional[bool],
    ) -> bool:
        query, o_res, o_qual, o_bri, o_ae, o_gain, o_night = self._build_cam_query(
            tres, tqual, tbri, tae, tgain, tnight, sres, squal, sbri, sae, sgain, snight
        )
        if not query:
            return True
        full_url = f"{self.base_url}/cam?{query}"
        t0 = time.monotonic()
        try:
            r = self._http_get_with_keepalive_retry(full_url, timeout=(1.0, 3.0))
            rtt_ms = (time.monotonic() - t0) * 1000.0
            if r.status_code == 200:
                self._record_cam_sent(
                    sent_res=o_res,
                    sent_qual=o_qual,
                    sent_bri=o_bri,
                    sent_ae=o_ae,
                    sent_gain=o_gain,
                    sent_night=o_night,
                    rtt_ms=rtt_ms,
                    last_url=full_url,
                    last_payload=f"GET /cam?{query}",
                )
                return True
            with self._lock:
                self._last_error = f"HTTP /cam {r.status_code}"
            self._stop_event.wait(timeout=0.12)
            return False
        except Exception as exc:
            with self._lock:
                self._last_error = str(exc)
            self._stop_event.wait(timeout=0.20)
            return False

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            with self._lock:
                tx, ty = self._target_x, self._target_y
                tled, tlaser = self._target_led, self._target_laser
                sx, sy = self._sent_x, self._sent_y
                sled, slaser = self._sent_led, self._sent_laser
                tres, tqual, tbri, tae, tgain, tnight = (
                    self._target_cam_res,
                    self._target_cam_qual,
                    self._target_cam_bri,
                    self._target_cam_ae,
                    self._target_cam_gain,
                    self._target_cam_night,
                )
                sres, squal, sbri, sae, sgain, snight = (
                    self._sent_cam_res,
                    self._sent_cam_qual,
                    self._sent_cam_bri,
                    self._sent_cam_ae,
                    self._sent_cam_gain,
                    self._sent_cam_night,
                )

            servo_priority_dirty = (tled != sled) or (tlaser != slaser)
            cam_dirty = (
                (tres is not None and tres != sres)
                or (tqual is not None and tqual != squal)
                or (tbri is not None and tbri != sbri)
                or (tae is not None and tae != sae)
                or (tgain is not None and tgain != sgain)
                or (tnight is not None and tnight != snight)
            )
            servo_move_dirty = (tx != sx) or (ty != sy)

            if not servo_priority_dirty and not cam_dirty and not servo_move_dirty:
                self._wakeup_event.wait(timeout=0.5)
                self._wakeup_event.clear()
                continue

            # 1) Immediate 0 ms priority for laser/led safety & responsiveness
            if servo_priority_dirty:
                with self._lock:
                    tx, ty = self._target_x, self._target_y
                    tled, tlaser = self._target_led, self._target_laser
                    sx, sy = self._sent_x, self._sent_y
                    sled, slaser = self._sent_led, self._sent_laser
                self._dispatch_servo(tx, ty, tled, tlaser, sx, sy, sled, slaser)
                continue

            # 2) Immediate 0 ms priority for /cam adjustments (independent of /servo)
            if cam_dirty:
                with self._lock:
                    tres, tqual, tbri, tae, tgain, tnight = (
                        self._target_cam_res,
                        self._target_cam_qual,
                        self._target_cam_bri,
                        self._target_cam_ae,
                        self._target_cam_gain,
                        self._target_cam_night,
                    )
                    sres, squal, sbri, sae, sgain, snight = (
                        self._sent_cam_res,
                        self._sent_cam_qual,
                        self._sent_cam_bri,
                        self._sent_cam_ae,
                        self._sent_cam_gain,
                        self._sent_cam_night,
                    )
                self._dispatch_cam(
                    tres, tqual, tbri, tae, tgain, tnight, sres, squal, sbri, sae, sgain, snight
                )
                continue

            # 3) Throttled servo Pan/Tilt motion (35 ms min interval = ~28.6 Hz to reduce Port 80/81 Wi-Fi contention)
            move_interval = self.get_effective_move_interval()
            if move_interval > 0.0:
                elapsed = time.monotonic() - self._last_send_monotonic
                remaining = move_interval - elapsed
                if remaining > 0.0:
                    self._wakeup_event.wait(timeout=remaining)
                    self._wakeup_event.clear()
                    continue

            with self._lock:
                tx, ty = self._target_x, self._target_y
                tled, tlaser = self._target_led, self._target_laser
                sx, sy = self._sent_x, self._sent_y
                sled, slaser = self._sent_led, self._sent_laser

            self._dispatch_servo(tx, ty, tled, tlaser, sx, sy, sled, slaser)


class PersonDetector:
    """YOLOv8n multi-class (person/dog) detector with GPU (ORT/ncnn Vulkan) + YuNet fallback."""

    def __init__(
        self,
        yolo_path: str = DEFAULT_YOLO_PATH,
        yunet_path: str = DEFAULT_YUNET_PATH,
        input_size: int = 640,
        iou_threshold: float = 0.45,
    ) -> None:
        self.input_size = input_size
        self.iou_threshold = iou_threshold
        self._yolo_path = yolo_path
        self._ort_session = None
        self._ort_input_name = None
        self._cv_net = None
        self._ncnn_net = None
        self._ncnn_input_name = "in0"
        self._ncnn_output_name = "out0"
        self._yunet = None
        self._yunet_input_size: Tuple[int, int] = (320, 320)
        self._lb_canvas: Optional[np.ndarray] = None
        self._lb_geom: Tuple[int, int] = (0, 0)
        self._lb_params: Tuple[int, int, int, int, float] = (0, 0, 0, 0, 1.0)
        self._empty_count: int = 0
        self.device_info: str = "CPU"

        if not os.path.exists(yolo_path):
            raise FileNotFoundError(f"YOLOv8 ONNX model not found at {yolo_path}")

        cv2.setNumThreads(DEFAULT_CPU_THREADS)
        initialized = False

        # Step A: Check ONNX Runtime GPU execution providers
        if ort is not None and DEFAULT_BACKEND in ("auto", "ort"):
            available = set(ort.get_available_providers())
            gpu_providers = [
                "ROCMExecutionProvider",
                "MIGraphXExecutionProvider",
                "DmlExecutionProvider",
                "CUDAExecutionProvider",
                "OpenVINOExecutionProvider",
            ]
            for gp in gpu_providers:
                if gp in available:
                    try:
                        opts = ort.SessionOptions()
                        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                        self._ort_session = ort.InferenceSession(
                            yolo_path,
                            sess_options=opts,
                            providers=[gp, "CPUExecutionProvider"],
                        )
                        self._ort_input_name = self._ort_session.get_inputs()[0].name
                        self.device_info = f"GPU ({gp})"
                        initialized = True
                        break
                    except Exception:
                        self._ort_session = None

        # Step B: ncnn Vulkan GPU (e.g. AMD Radeon RX 580 via Mesa RADV)
        ncnn_param = os.path.splitext(yolo_path)[0] + ".ncnn.param"
        ncnn_bin = os.path.splitext(yolo_path)[0] + ".ncnn.bin"
        if (
            not initialized
            and ncnn is not None
            and DEFAULT_BACKEND in ("auto", "vulkan")
            and os.path.exists(ncnn_param)
            and os.path.exists(ncnn_bin)
        ):
            try:
                hw_gpu_idx = next(
                    (i for i in range(ncnn.get_gpu_count()) if ncnn.get_gpu_info(i).type() in (0, 1)),
                    None,
                )
                if hw_gpu_idx is not None:
                    net = ncnn.Net()
                    net.opt.use_vulkan_compute = True
                    net.opt.use_fp16_packed = False
                    net.opt.use_fp16_storage = False
                    net.opt.use_fp16_arithmetic = False
                    net.opt.num_threads = DEFAULT_CPU_THREADS
                    net.set_vulkan_device(hw_gpu_idx)
                    net.load_param(ncnn_param)
                    net.load_model(ncnn_bin)
                    self._ncnn_net = net
                    self._ncnn_input_name = net.input_names()[0]
                    self._ncnn_output_name = net.output_names()[0]
                    _ = self._run_ncnn(np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8))
                    dev_name = ncnn.get_gpu_info(hw_gpu_idx).device_name().strip() or "Vulkan GPU"
                    self.device_info = f"GPU Vulkan ({dev_name})"
                    initialized = True
            except Exception:
                self._ncnn_net = None

        # Step C: CPU Fallback (ONNX Runtime or OpenCV DNN)
        if not initialized:
            self._init_cpu_backend()

        if os.path.exists(yunet_path) and hasattr(cv2, "FaceDetectorYN"):
            self._yunet = cv2.FaceDetectorYN.create(
                yunet_path,
                "",
                (320, 320),
                score_threshold=0.6,
                nms_threshold=0.3,
                top_k=5,
            )

    def _init_cpu_backend(self) -> None:
        if ort is not None:
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = DEFAULT_CPU_THREADS
            opts.inter_op_num_threads = 1
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._ort_session = ort.InferenceSession(
                self._yolo_path,
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            self._ort_input_name = self._ort_session.get_inputs()[0].name
            self.device_info = f"CPU ONNX ({DEFAULT_CPU_THREADS} threads)"
        else:
            self._cv_net = cv2.dnn.readNetFromONNX(self._yolo_path)
            self.device_info = "CPU OpenCV DNN"

    def _run_ncnn(self, canvas: np.ndarray) -> np.ndarray:
        mat_in = ncnn.Mat.from_pixels(
            canvas,
            ncnn.Mat.PixelType.PIXEL_BGR2RGB,
            self.input_size,
            self.input_size,
        )
        mat_in.substract_mean_normalize([], [1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0])
        ex = self._ncnn_net.create_extractor()
        ex.input(self._ncnn_input_name, mat_in)
        ret, mat_out = ex.extract(self._ncnn_output_name)
        if ret != 0:
            raise RuntimeError(f"ncnn extract failed with code {ret}")
        return np.array(mat_out)[None, ...]

    def _run_ort_or_cv(self, canvas: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        blob = np.ascontiguousarray(
            np.transpose(rgb.astype(np.float32) * (1.0 / 255.0), (2, 0, 1))[None, ...]
        )
        if self._ort_session is not None:
            return self._ort_session.run(None, {self._ort_input_name: blob})[0]
        self._cv_net.setInput(blob)
        return self._cv_net.forward()

    def detect(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.35,
        detect_person: bool = True,
        detect_dog: bool = False,
        force_fallback: bool = False,
    ) -> List[Detection]:
        if not detect_person and not detect_dog:
            return []

        enabled_classes: List[Tuple[str, int]] = []
        if detect_person:
            enabled_classes.append(("person", COCO_CLASSES["person"][0]))
        if detect_dog:
            enabled_classes.append(("dog", COCO_CLASSES["dog"][0]))

        detections = self._detect_yolo_classes(frame, conf_threshold, enabled_classes)
        if detections:
            self._empty_count = 0
            return detections

        self._empty_count += 1
        if detect_person and self._yunet is not None and (force_fallback or (self._empty_count % 20 == 0)):
            detections = self._detect_yunet_face(frame)
        return detections

    def _detect_yolo_classes(
        self,
        frame: np.ndarray,
        conf_threshold: float,
        enabled_classes: List[Tuple[str, int]],
    ) -> List[Detection]:
        h, w = frame.shape[:2]
        sz = self.input_size
        if self._lb_canvas is None or self._lb_geom != (h, w):
            scale = min(sz / w, sz / h)
            nw, nh = int(round(w * scale)), int(round(h * scale))
            dx, dy = (sz - nw) // 2, (sz - nh) // 2
            self._lb_canvas = np.full((sz, sz, 3), 114, dtype=np.uint8)
            self._lb_geom = (h, w)
            self._lb_params = (nw, nh, dx, dy, scale)
        else:
            nw, nh, dx, dy, scale = self._lb_params

        canvas = self._lb_canvas
        if nw == w and nh == h:
            # Zero-resize fast path (VGA 640x480 / rotated 480x640 matches YOLOv8n 640 1:1)
            canvas[dy : dy + nh, dx : dx + nw] = frame
        else:
            canvas[dy : dy + nh, dx : dx + nw] = cv2.resize(
                frame, (nw, nh), interpolation=cv2.INTER_LINEAR
            )

        if self._ncnn_net is not None:
            try:
                output = self._run_ncnn(canvas)
            except Exception:
                self._ncnn_net = None
                if self._ort_session is None and self._cv_net is None:
                    self._init_cpu_backend()
                output = self._run_ort_or_cv(canvas)
        else:
            output = self._run_ort_or_cv(canvas)

        # YOLOv8 output: [1, 84, 8400] — slice rows directly without transposing all 705k floats
        raw = output[0]
        inv_scale = 1.0 / scale
        results: List[Detection] = []

        for source_name, cid in enabled_classes:
            scores = raw[4 + cid]
            mask = scores >= conf_threshold
            if not np.any(mask):
                continue

            filtered_scores = scores[mask]
            cx = raw[0, mask]
            cy = raw[1, mask]
            bw = raw[2, mask]
            bh = raw[3, mask]

            half_w = 0.5 * bw
            half_h = 0.5 * bh
            x1 = np.clip((cx - half_w - dx) * inv_scale, 0.0, w - 1.0)
            y1 = np.clip((cy - half_h - dy) * inv_scale, 0.0, h - 1.0)
            x2 = np.clip((cx + half_w - dx) * inv_scale, 0.0, w - 1.0)
            y2 = np.clip((cy + half_h - dy) * inv_scale, 0.0, h - 1.0)

            bw_arr = x2 - x1
            bh_arr = y2 - y1
            valid = (bw_arr >= 8.0) & (bh_arr >= 8.0)
            if not np.any(valid):
                continue

            vx1, vy1, vx2, vy2 = x1[valid], y1[valid], x2[valid], y2[valid]
            vbw, vbh = bw_arr[valid], bh_arr[valid]
            vscores = filtered_scores[valid]

            boxes_xywh = np.column_stack([vx1, vy1, vbw, vbh]).astype(np.int32).tolist()
            confs = vscores.astype(float).tolist()

            indices = cv2.dnn.NMSBoxes(boxes_xywh, confs, conf_threshold, self.iou_threshold)
            if len(indices) == 0:
                continue

            for idx in np.array(indices).flatten():
                i = int(idx)
                results.append(
                    Detection(
                        x1=float(vx1[i]),
                        y1=float(vy1[i]),
                        x2=float(vx2[i]),
                        y2=float(vy2[i]),
                        conf=confs[i],
                        source=source_name,
                    )
                )

        return results

    def _detect_yunet_face(self, frame: np.ndarray) -> List[Detection]:
        h, w = frame.shape[:2]
        long_edge = max(w, h)
        if long_edge > 320:
            scale = 320.0 / float(long_edge)
            sw = max(16, int(round(w * scale)))
            sh = max(16, int(round(h * scale)))
            inp = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)
            inv_x = float(w) / float(sw)
            inv_y = float(h) / float(sh)
        else:
            sw, sh = w, h
            inp = frame
            inv_x = 1.0
            inv_y = 1.0

        if self._yunet_input_size != (sw, sh):
            self._yunet.setInputSize((sw, sh))
            self._yunet_input_size = (sw, sh)
        _, faces = self._yunet.detect(inp)
        if faces is None or len(faces) == 0:
            return []
        results: List[Detection] = []
        for face in faces:
            fx = float(face[0]) * inv_x
            fy = float(face[1]) * inv_y
            fw = float(face[2]) * inv_x
            fh = float(face[3]) * inv_y
            score = float(face[14]) if len(face) > 14 else 0.7
            bx1 = max(0.0, min(w - 1.0, fx))
            by1 = max(0.0, min(h - 1.0, fy))
            bx2 = max(0.0, min(w - 1.0, fx + fw))
            by2 = max(0.0, min(h - 1.0, fy + fh))
            if bx2 - bx1 >= 8.0 and by2 - by1 >= 8.0:
                results.append(Detection(x1=bx1, y1=by1, x2=bx2, y2=by2, conf=score, source="face"))
        return results

    @staticmethod
    def select_target(
        detections: List[Detection],
        frame_shape: Tuple[int, int],
        prev_target: Optional[Tuple[float, float]] = None,
    ) -> Optional[Detection]:
        if not detections:
            return None
        if len(detections) == 1:
            return detections[0]

        h, w = frame_shape
        cx, cy = w * 0.5, h * 0.5
        diag = math.hypot(w, h) + 1e-6
        frame_area = float(max(1, w * h))

        def target_score(d: Detection) -> float:
            area_norm = min(1.0, math.sqrt(d.area / frame_area))
            dist_norm = math.hypot(d.center[0] - cx, d.center[1] - cy) / diag
            continuity = 0.0
            if prev_target is not None:
                px, py = prev_target
                pdist = math.hypot(d.center[0] - px, d.center[1] - py) / (diag * 0.35)
                continuity = max(0.0, 0.35 * (1.0 - pdist))
            return (area_norm * 1.4) + (d.conf * 0.6) - (dist_norm * 0.3) + continuity

        return max(detections, key=target_score)

    @staticmethod
    def compute_aim_point(det: Detection, aim_mode: str = "center") -> Tuple[float, float]:
        tx = (det.x1 + det.x2) * 0.5
        if det.source in ("face", "dog"):
            return (tx, (det.y1 + det.y2) * 0.5)
        box_w = det.x2 - det.x1
        box_h = det.y2 - det.y1
        ref_h = min(box_h, box_w * 1.8)
        if aim_mode == "upper":
            ty = det.y1 + 0.38 * ref_h
        elif aim_mode == "head":
            ty = det.y1 + 0.20 * ref_h
        else:
            ty = (det.y1 + det.y2) * 0.5
        return (tx, ty)

    def detect_face_in_crop(self, frame: np.ndarray, det: Detection) -> Optional[Detection]:
        """Run YuNet on the upper-body crop of a detected person to locate the exact face."""
        if det.source == "face":
            return det
        if det.source != "person" or self._yunet is None:
            return None

        h, w = frame.shape[:2]
        bw = max(1.0, det.x2 - det.x1)
        bh = max(1.0, det.y2 - det.y1)
        pad_x = bw * 0.12
        pad_top = bh * 0.10
        sub_h = bh * 0.65 if bh > (bw * 1.4) else bh

        x1 = max(0, int(math.floor(det.x1 - pad_x)))
        y1 = max(0, int(math.floor(det.y1 - pad_top)))
        x2 = min(w, int(math.ceil(det.x2 + pad_x)))
        y2 = min(h, int(math.ceil(det.y1 + sub_h)))
        cw, ch = x2 - x1, y2 - y1
        if cw < 24 or ch < 24:
            return None

        crop = frame[y1:y2, x1:x2]
        long_edge = max(cw, ch)
        if long_edge > 240:
            scale = 240.0 / float(long_edge)
            sw = max(16, int(round(cw * scale)))
            sh = max(16, int(round(ch * scale)))
            inp = cv2.resize(crop, (sw, sh), interpolation=cv2.INTER_AREA)
            inv_x = float(cw) / float(sw)
            inv_y = float(ch) / float(sh)
        else:
            sw, sh = cw, ch
            inp = np.ascontiguousarray(crop)
            inv_x = 1.0
            inv_y = 1.0

        if self._yunet_input_size != (sw, sh):
            self._yunet.setInputSize((sw, sh))
            self._yunet_input_size = (sw, sh)
        _, faces = self._yunet.detect(inp)
        if faces is None or len(faces) == 0:
            return None

        best_face = max(faces, key=lambda f: float(f[14]) if len(f) > 14 else 0.7)
        fx = float(best_face[0]) * inv_x
        fy = float(best_face[1]) * inv_y
        fw = float(best_face[2]) * inv_x
        fh = float(best_face[3]) * inv_y
        score = float(best_face[14]) if len(best_face) > 14 else 0.7
        gx1 = max(0.0, min(w - 1.0, x1 + fx))
        gy1 = max(0.0, min(h - 1.0, y1 + fy))
        gx2 = max(0.0, min(w - 1.0, x1 + fx + fw))
        gy2 = max(0.0, min(h - 1.0, y1 + fy + fh))
        if (gx2 - gx1) < 8.0 or (gy2 - gy1) < 8.0:
            return None
        return Detection(x1=gx1, y1=gy1, x2=gx2, y2=gy2, conf=score, source="face")


class DiscordWebhookLogger:
    """Anti-spam Best-Shot visit logger that posts face/person captures asynchronously to Discord."""

    def __init__(
        self,
        webhook_url: str = DEFAULT_DISCORD_WEBHOOK_URL,
        cooldown_s: float = DEFAULT_DISCORD_COOLDOWN_S,
        absence_reset_s: float = DEFAULT_DISCORD_ABSENCE_RESET_S,
        connect_timeout_s: float = DEFAULT_DISCORD_CONNECT_TIMEOUT_S,
        read_timeout_s: float = DEFAULT_DISCORD_READ_TIMEOUT_S,
    ) -> None:
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._queue: "queue.Queue[Optional[Dict[str, object]]]" = queue.Queue(maxsize=4)

        self._webhook_url: str = str(webhook_url or "").strip()
        self.enabled: bool = bool(self._webhook_url)
        self.cooldown_s: float = max(5.0, min(600.0, float(cooldown_s)))
        self.absence_reset_s: float = max(3.0, min(300.0, float(absence_reset_s)))
        self.connect_timeout_s: float = max(3.0, float(connect_timeout_s))
        self.read_timeout_s: float = max(5.0, float(read_timeout_s))
        self.best_shot_window_s: float = 2.5
        self.min_confirm_frames: int = 3
        self.include_scene: bool = True
        self.log_dogs: bool = False

        # Visit session state machine: idle -> collecting -> logged
        self._state: str = "idle"
        self._confirm_streak: int = 0
        self._window_end: float = 0.0
        self._last_seen_at: float = 0.0
        self._last_eval_at: float = 0.0
        self._last_sent_at: float = -1e9
        self._best_candidate: Optional[Dict[str, object]] = None

        # Telemetry
        self._sent_count: int = 0
        self._last_sent_wall_ts: float = 0.0
        self._last_error: str = ""

        self._thread = threading.Thread(target=self._worker_loop, daemon=True, name="DiscordWebhook")

    def start(self) -> None:
        if not self._thread.is_alive():
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._worker_loop, daemon=True, name="DiscordWebhook")
            self._thread.start()

    def shutdown(self) -> None:
        self._stop_event.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    @staticmethod
    def _new_http_session() -> requests.Session:
        sess = requests.Session()
        sess.headers.update(
            {
                "User-Agent": "AutoAIM-Turret/1.0",
                "Connection": "close",
            }
        )
        return sess

    def update_config(self, cfg: Dict[str, object]) -> None:
        with self._lock:
            if "discord_enabled" in cfg:
                self.enabled = bool(cfg["discord_enabled"])
                if not self.enabled:
                    self._reset_session_locked()
            if "discord_cooldown_s" in cfg:
                self.cooldown_s = max(5.0, min(600.0, float(cfg["discord_cooldown_s"])))
            if "discord_absence_reset_s" in cfg:
                self.absence_reset_s = max(3.0, min(300.0, float(cfg["discord_absence_reset_s"])))
            if "discord_include_scene" in cfg:
                self.include_scene = bool(cfg["discord_include_scene"])
            if "discord_log_dogs" in cfg:
                self.log_dogs = bool(cfg["discord_log_dogs"])

    def _reset_session_locked(self) -> None:
        self._state = "idle"
        self._confirm_streak = 0
        self._window_end = 0.0
        self._last_eval_at = 0.0
        self._best_candidate = None

    def is_eligible_target(self, target: Optional[Detection], log_dogs: Optional[bool] = None) -> bool:
        if target is None:
            return False
        allow_dogs = self.log_dogs if log_dogs is None else log_dogs
        return target.source in ("person", "face") or (allow_dogs and target.source == "dog")

    def needs_face_check(self, target: Optional[Detection], now: float) -> bool:
        with self._lock:
            if not self.enabled or not self._webhook_url:
                return False
            if not self.is_eligible_target(target, self.log_dogs):
                return False
            if target is not None and target.source == "face":
                return False
            if self._state == "idle":
                return (
                    (self._confirm_streak + 1 >= self.min_confirm_frames)
                    and ((now - self._last_sent_at) >= self.cooldown_s)
                )
            if self._state == "collecting":
                return (self._best_candidate is None) or ((now - self._last_eval_at) >= 0.20)
            return False

    @staticmethod
    def _extract_crop(frame: np.ndarray, target: Detection, face_det: Optional[Detection]) -> Tuple[np.ndarray, bool]:
        """Crop the face region (with head/shoulder margin) or upper-body fallback cleanly."""
        h, w = frame.shape[:2]
        if not hasattr(frame, "__getitem__"):
            return frame.copy(), face_det is not None

        if face_det is not None:
            fw = max(12.0, face_det.x2 - face_det.x1)
            fh = max(12.0, face_det.y2 - face_det.y1)
            mx = fw * 0.45
            my_top = fh * 0.40
            my_bot = fh * 0.55
            x1 = max(0, int(math.floor(face_det.x1 - mx)))
            y1 = max(0, int(math.floor(face_det.y1 - my_top)))
            x2 = min(w, int(math.ceil(face_det.x2 + mx)))
            y2 = min(h, int(math.ceil(face_det.y2 + my_bot)))
            has_face = True
        else:
            bw = max(16.0, target.x2 - target.x1)
            bh = max(16.0, target.y2 - target.y1)
            if target.source == "person" and bh > (bw * 1.3):
                crop_h = bh * 0.55
                mx = bw * 0.18
                my = crop_h * 0.15
                x1 = max(0, int(math.floor(target.x1 - mx)))
                y1 = max(0, int(math.floor(target.y1 - my)))
                x2 = min(w, int(math.ceil(target.x2 + mx)))
                y2 = min(h, int(math.ceil(target.y1 + crop_h)))
            else:
                mx = bw * 0.20
                my = bh * 0.20
                x1 = max(0, int(math.floor(target.x1 - mx)))
                y1 = max(0, int(math.floor(target.y1 - my)))
                x2 = min(w, int(math.ceil(target.x2 + mx)))
                y2 = min(h, int(math.ceil(target.y2 + my)))
            has_face = False

        if x2 <= x1 + 4 or y2 <= y1 + 4:
            return frame.copy(), has_face
        return frame[y1:y2, x1:x2].copy(), has_face

    @staticmethod
    def _compute_sharpness(crop: np.ndarray) -> float:
        try:
            h, w = crop.shape[:2]
            if max(w, h) > 160:
                s = 160.0 / float(max(w, h))
                crop = cv2.resize(crop, (max(8, int(round(w * s))), max(8, int(round(h * s)))), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            lap = cv2.Laplacian(gray, cv2.CV_32F)
            return float(lap.var())
        except Exception:
            return 50.0

    def _evaluate_candidate(
        self,
        frame: np.ndarray,
        target: Detection,
        face_det: Optional[Detection],
        servo_x: int,
        servo_y: int,
        led_on: bool,
        laser_on: bool,
        include_scene: bool,
    ) -> Dict[str, object]:
        crop, has_face = self._extract_crop(frame, target, face_det)
        sharpness = self._compute_sharpness(crop)
        h, w = frame.shape[:2]
        frame_area = float(max(1, h * w))
        ref_det = face_det if face_det is not None else target
        area_ratio = min(1.0, ref_det.area / frame_area)
        conf = face_det.conf if face_det is not None else target.conf
        score = (
            (1000.0 if has_face else 0.0)
            + (conf * 120.0)
            + min(250.0, sharpness * 0.4)
            + min(150.0, math.sqrt(area_ratio) * 200.0)
        )
        source_kind = "face" if has_face else target.source
        return {
            "kind": "capture",
            "score": score,
            "source": source_kind,
            "conf": conf,
            "has_face": has_face,
            "servo_x": servo_x,
            "servo_y": servo_y,
            "led_on": led_on,
            "laser_on": laser_on,
            "crop": crop,
            "scene": frame.copy() if include_scene else None,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }

    def _enqueue_job(self, job: Dict[str, object]) -> bool:
        try:
            self._queue.put_nowait(job)
            return True
        except queue.Full:
            return False

    def observe(
        self,
        frame: np.ndarray,
        target: Optional[Detection],
        face_det: Optional[Detection],
        servo_x: int,
        servo_y: int,
        led_on: bool,
        laser_on: bool,
        now: Optional[float] = None,
    ) -> None:
        now = time.time() if now is None else now
        job_to_send: Optional[Dict[str, object]] = None

        with self._lock:
            if not self.enabled or not self._webhook_url:
                self._reset_session_locked()
                return

            eligible = self.is_eligible_target(target, self.log_dogs)

            if self._state == "idle":
                if eligible and target is not None:
                    self._confirm_streak += 1
                    self._last_seen_at = now
                    if (
                        self._confirm_streak >= self.min_confirm_frames
                        and (now - self._last_sent_at) >= self.cooldown_s
                    ):
                        self._state = "collecting"
                        self._window_end = now + self.best_shot_window_s
                        self._last_eval_at = now
                        self._best_candidate = self._evaluate_candidate(
                            frame=frame,
                            target=target,
                            face_det=face_det,
                            servo_x=servo_x,
                            servo_y=servo_y,
                            led_on=led_on,
                            laser_on=laser_on,
                            include_scene=self.include_scene,
                        )
                else:
                    self._confirm_streak = 0

            elif self._state == "collecting":
                if eligible and target is not None:
                    self._last_seen_at = now
                    if self._best_candidate is None or (now - self._last_eval_at) >= 0.20:
                        self._last_eval_at = now
                        cand = self._evaluate_candidate(
                            frame=frame,
                            target=target,
                            face_det=face_det,
                            servo_x=servo_x,
                            servo_y=servo_y,
                            led_on=led_on,
                            laser_on=laser_on,
                            include_scene=self.include_scene,
                        )
                        if self._best_candidate is None or cand["score"] > self._best_candidate["score"]:
                            self._best_candidate = cand

                lost_s = now - self._last_seen_at
                if now >= self._window_end or (not eligible and lost_s >= 1.0):
                    if self._best_candidate is not None:
                        job_to_send = self._best_candidate
                        self._best_candidate = None
                        self._last_sent_at = now
                        self._state = "logged"
                    else:
                        self._reset_session_locked()

            elif self._state == "logged":
                if eligible:
                    self._last_seen_at = now
                elif (now - self._last_seen_at) >= self.absence_reset_s:
                    self._reset_session_locked()

        if job_to_send is not None:
            self._enqueue_job(job_to_send)

    def send_test(
        self,
        frame: Optional[np.ndarray],
        servo_x: int,
        servo_y: int,
        led_on: bool,
        laser_on: bool,
    ) -> Tuple[bool, str]:
        with self._lock:
            if not self._webhook_url:
                return False, "DISCORD_WEBHOOK_URL não configurada no .env"
            job: Dict[str, object] = {
                "kind": "test",
                "source": "test",
                "conf": 1.0,
                "has_face": False,
                "servo_x": servo_x,
                "servo_y": servo_y,
                "led_on": led_on,
                "laser_on": laser_on,
                "crop": frame.copy() if frame is not None else None,
                "scene": None,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
        if not self._enqueue_job(job):
            return False, "Fila de envio do Discord cheia; tente novamente"
        return True, "Teste enfileirado para envio no Discord"

    @staticmethod
    def _encode_jpeg(
        img: Optional[np.ndarray],
        quality: int = 85,
        max_dim: Optional[int] = None,
    ) -> Optional[bytes]:
        if img is None:
            return None
        try:
            target_img = img
            if max_dim is not None and max_dim > 0 and hasattr(img, "shape") and len(img.shape) >= 2:
                h, w = int(img.shape[0]), int(img.shape[1])
                long_edge = max(h, w)
                if long_edge > max_dim:
                    scale = float(max_dim) / float(long_edge)
                    nw = max(1, int(round(w * scale)))
                    nh = max(1, int(round(h * scale)))
                    target_img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
            ok, encoded = cv2.imencode(".jpg", target_img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            if ok:
                return encoded.tobytes()
        except Exception:
            return None
        return None

    def _post_job(self, url: str, job: Dict[str, object]) -> None:
        kind = str(job.get("kind", "capture"))
        source = str(job.get("source", "person"))
        conf = float(job.get("conf", 0.0))
        servo_x = int(job.get("servo_x", 90))
        servo_y = int(job.get("servo_y", 90))
        led_on = bool(job.get("led_on", False))
        laser_on = bool(job.get("laser_on", False))
        ts_iso = str(job.get("timestamp_utc", datetime.now(timezone.utc).isoformat()))

        if kind == "test":
            title = "Teste de Conexão — AutoAIM Turret"
            color = 0x5865F2
            target_label = "Teste Manual"
        elif source == "face":
            title = "Rosto Registrado — AutoAIM"
            color = 0x22C55E
            target_label = "ROSTO"
        elif source == "dog":
            title = "Cachorro Detectado — AutoAIM"
            color = 0xF59E0B
            target_label = "CACHORRO"
        else:
            title = "Pessoa Detectada — AutoAIM"
            color = 0x38BDF8
            target_label = "PESSOA"

        light_label = "Flash ON" if led_on else "Ambiente"
        if laser_on:
            light_label += " · Laser ON"

        embed_base: Dict[str, object] = {
            "title": title,
            "color": color,
            "timestamp": ts_iso,
            "fields": [
                {"name": "Alvo", "value": target_label, "inline": True},
                {"name": "Confiança", "value": f"{conf * 100:.0f}%", "inline": True},
                {"name": "Posição", "value": f"X {servo_x}° · Y {servo_y}°", "inline": True},
                {"name": "Iluminação", "value": light_label, "inline": True},
            ],
            "footer": {"text": "AutoAIM Turret · Log Anti-Spam"},
        }

        crop_bytes = self._encode_jpeg(job.get("crop"), quality=85, max_dim=640)  # type: ignore[arg-type]
        scene_bytes = self._encode_jpeg(job.get("scene"), quality=78, max_dim=960)  # type: ignore[arg-type]

        def _build_request_parts(include_scene_attachment: bool):
            embed_main = dict(embed_base)
            embeds = [embed_main]
            files = []
            if crop_bytes is not None:
                embed_main["image"] = {"url": "attachment://capture.jpg"}
                files.append(("files[0]", ("capture.jpg", crop_bytes, "image/jpeg")))
                if include_scene_attachment and scene_bytes is not None:
                    embeds.append(
                        {
                            "title": "Cena Completa",
                            "color": color,
                            "image": {"url": "attachment://scene.jpg"},
                        }
                    )
                    files.append(("files[1]", ("scene.jpg", scene_bytes, "image/jpeg")))
            return {"embeds": embeds}, files

        max_attempts = 3
        had_timeout = False
        timeout = (self.connect_timeout_s, self.read_timeout_s)

        for attempt in range(max_attempts):
            if self._stop_event.is_set():
                return

            include_scene_att = not (had_timeout and attempt == max_attempts - 1)
            payload, files = _build_request_parts(include_scene_att)

            try:
                with self._new_http_session() as session:
                    if files:
                        resp = session.post(
                            url,
                            data={"payload_json": json.dumps(payload)},
                            files=files,
                            timeout=timeout,
                        )
                    else:
                        resp = session.post(url, json=payload, timeout=timeout)

                    try:
                        status_code = resp.status_code
                        if 200 <= status_code < 300:
                            with self._lock:
                                self._sent_count += 1
                                self._last_sent_wall_ts = time.time()
                                self._last_error = ""
                            return

                        if status_code == 429 and attempt < max_attempts - 1:
                            retry_s = 2.0
                            try:
                                retry_s = float((resp.json() or {}).get("retry_after", 2.0))
                            except Exception:
                                pass
                            if self._stop_event.wait(timeout=max(0.5, min(10.0, retry_s))):
                                return
                            continue

                        if 500 <= status_code < 600 and attempt < max_attempts - 1:
                            if self._stop_event.wait(timeout=1.5 * (attempt + 1)):
                                return
                            continue

                        with self._lock:
                            self._last_error = f"HTTP {status_code}"
                        return
                    finally:
                        try:
                            resp.close()
                        except Exception:
                            pass
            except requests.exceptions.Timeout as exc:
                had_timeout = True
                if attempt < max_attempts - 1:
                    if self._stop_event.wait(timeout=1.5 * (attempt + 1)):
                        return
                    continue
                if isinstance(exc, requests.exceptions.ConnectTimeout):
                    err_msg = f"Timeout conexão Discord ({self.connect_timeout_s:.0f}s)"
                else:
                    err_msg = f"Timeout leitura Discord ({self.read_timeout_s:.0f}s)"
                with self._lock:
                    self._last_error = err_msg
                return
            except requests.exceptions.ConnectionError as exc:
                if attempt < max_attempts - 1:
                    if self._stop_event.wait(timeout=1.5 * (attempt + 1)):
                        return
                    continue
                with self._lock:
                    self._last_error = f"Erro conexão Discord: {exc}"
                return
            except Exception as exc:
                with self._lock:
                    self._last_error = str(exc)
                return

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                job = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if job is None:
                self._queue.task_done()
                break
            try:
                with self._lock:
                    url = self._webhook_url
                if url:
                    self._post_job(url, job)
            finally:
                self._queue.task_done()

    def get_status(self) -> Dict[str, object]:
        with self._lock:
            return {
                "discord_configured": bool(self._webhook_url),
                "discord_enabled": self.enabled,
                "discord_cooldown_s": round(self.cooldown_s, 1),
                "discord_absence_reset_s": round(self.absence_reset_s, 1),
                "discord_include_scene": self.include_scene,
                "discord_log_dogs": self.log_dogs,
                "discord_sent_count": self._sent_count,
                "discord_last_sent_ts": round(self._last_sent_wall_ts, 3) if self._last_sent_wall_ts > 0 else 0,
                "discord_last_error": self._last_error,
                "discord_state": self._state,
            }


class AutoAimEngine:
    """Closed-loop visual servoing engine that tracks targets (person/dog) and controls servos, Flash LED, and Green Laser."""

    def __init__(
        self,
        stream_url: str = DEFAULT_STREAM_URL,
        turret_url: str = DEFAULT_TURRET_URL,
        auto_aim: bool = True,
    ) -> None:
        self.reader = MJPEGStreamReader(stream_url=stream_url, rotation=90)
        self.hardware = TurretHardwareClient(base_url=turret_url)
        self.detector = PersonDetector()
        self.discord = DiscordWebhookLogger()

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="AutoAimEngine")

        # Configurable parameters
        self.auto_aim: bool = auto_aim
        self.detect_person: bool = True
        self.detect_dog: bool = True
        self.detect_in_manual: bool = False
        self.eco_mode: bool = True
        self.eco_delay_s: float = 15.0
        self.eco_fps: float = 1.0
        self.eco_motion_wake: bool = True
        self.eco_motion_flash: bool = True
        self.eco_flash_dark_only: bool = False
        self.eco_motion_sensitivity: float = 50.0
        self.eco_flash_duration_s: float = 6.0
        self.eco_dark_threshold: float = 80.0
        self.auto_fire: bool = True
        self.auto_fire_laser: bool = True
        self.auto_fire_led: bool = False
        self.aim_mode: str = "center"  # center | upper | head
        self.led_mode: str = "on_lock"  # on_lock | on_detect | off
        self.kp: float = 1.0
        self.kd: float = 0.08
        self.deadzone_px: float = 12.0
        self.conf_threshold: float = 0.28
        self.invert_x: bool = False
        self.invert_y: bool = False
        self.resolution: int = DEFAULT_CAM_RES  # 0=QVGA, 1=CIF, 2=VGA, 3=SVGA, 4=WIDE (320x176), 5=WIDE+ (400x224)
        self.cam_qual: int = DEFAULT_CAM_QUAL  # 10..40 (lower = sharper; 16 = balanced)
        self.cam_bri: int = DEFAULT_CAM_BRI  # -3..3
        self.cam_ae: int = DEFAULT_CAM_AE  # -5..5
        self.cam_gain: int = DEFAULT_CAM_GAIN  # 0..5 (2x..64x; default 4 = 32x)
        self.cam_night: bool = DEFAULT_CAM_NIGHT  # 0|1 (aec2; 0 keeps max FPS indoors)
        self._res_explicit: bool = False
        self.max_step_deg: float = 45.0
        self.min_servo_interval: float = 0.035
        self.camera_latency: float = DEFAULT_CAMERA_LATENCY
        self.hold_timeout: float = 0.45

        # Internal servo, actuator & tracking state
        self._curr_x: float = 90.0
        self._curr_y: float = 90.0
        self._manual_led: bool = False
        self._manual_laser: bool = False
        self._auto_fire_active: bool = False
        self._prev_deg_err_x: float = 0.0
        self._prev_deg_err_y: float = 0.0
        self._last_servo_step_time: float = 0.0
        self._wait_frame_after: float = 0.0
        self._last_detection_time: float = 0.0
        self._last_cv_activity_time: float = time.time()
        self._eco_active: bool = False
        self._eco_wake_until: float = 0.0
        self._eco_flash_until: float = 0.0
        self._eco_flash_active: bool = False
        self._motion_prev_gray: Optional[np.ndarray] = None
        self._motion_suppress_until: float = 0.0
        self._motion_score: float = 0.0
        self._ambient_luma: float = 0.0
        self._smoothed_target: Optional[Tuple[float, float]] = None

        # Telemetry & output frames
        self._person_detected: bool = False
        self._target_type: Optional[str] = None
        self._led_on: bool = False
        self._laser_on: bool = False
        self._target_box: Optional[List[int]] = None
        self._target_center: Optional[List[int]] = None
        self._error_px: List[int] = [0, 0]
        self._frame_size: List[int] = [0, 0]
        self._fps: float = 0.0
        self._cv_fps: float = 0.0
        self._video_fps: float = 0.0
        self._inference_ms: float = 0.0
        self._detections_count: int = 0

        self._annotated_frame: Optional[np.ndarray] = None
        self._annotated_jpeg: Optional[bytes] = None
        self._annotated_id: int = 0
        self._frame_cond = threading.Condition()

        # Pipelined overlay + JPEG encoder worker (keeps CPU rendering off the 30 FPS CV critical path)
        self._video_clients: int = 0
        self._window_enabled: bool = False
        self._render_cond = threading.Condition()
        self._render_seq: int = 0
        self._render_payload: Optional[Tuple[object, ...]] = None
        self._encoder_thread = threading.Thread(
            target=self._encoder_loop, daemon=True, name="FrameEncoder"
        )

    def register_video_client(self) -> None:
        with self._render_cond:
            self._video_clients += 1
            self._render_cond.notify_all()

    def unregister_video_client(self) -> None:
        with self._render_cond:
            self._video_clients = max(0, self._video_clients - 1)

    def set_window_enabled(self, enabled: bool) -> None:
        with self._render_cond:
            self._window_enabled = bool(enabled)
            if self._window_enabled:
                self._render_cond.notify_all()

    def _has_video_consumers_locked(self) -> bool:
        return self._video_clients > 0 or self._window_enabled

    def _submit_render_job(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        target: Optional[Detection],
        aim_pt: Optional[Tuple[float, float]],
        deadzone_px: float,
        auto_aim: bool,
        led_on: bool,
        laser_on: bool,
        servo_x: int,
        servo_y: int,
        err_x: float,
        err_y: float,
        fps: float,
        eco_active: bool,
        eco_motion_woken: bool,
    ) -> None:
        with self._render_cond:
            self._render_payload = (
                frame,
                detections,
                target,
                aim_pt,
                deadzone_px,
                auto_aim,
                led_on,
                laser_on,
                servo_x,
                servo_y,
                err_x,
                err_y,
                fps,
                eco_active,
                eco_motion_woken,
            )
            self._render_seq += 1
            if self._has_video_consumers_locked():
                self._render_cond.notify_all()

    def _encoder_loop(self) -> None:
        last_seq = -1
        while not self._stop_event.is_set():
            with self._render_cond:
                while not self._stop_event.is_set() and (
                    self._render_payload is None
                    or self._render_seq == last_seq
                    or not self._has_video_consumers_locked()
                ):
                    self._render_cond.wait(timeout=0.5)
                if self._stop_event.is_set():
                    break
                payload = self._render_payload
                last_seq = self._render_seq
                need_jpeg = self._video_clients > 0
                need_frame = self._window_enabled

            if payload is None:
                continue

            (
                frame,
                detections,
                target,
                aim_pt,
                deadzone_px,
                auto_aim,
                led_on,
                laser_on,
                servo_x,
                servo_y,
                err_x,
                err_y,
                fps,
                eco_active,
                eco_motion_woken,
            ) = payload

            annotated = self._draw_overlay(
                frame=frame,  # type: ignore[arg-type]
                detections=detections,  # type: ignore[arg-type]
                target=target,  # type: ignore[arg-type]
                aim_pt=aim_pt,  # type: ignore[arg-type]
                deadzone_px=deadzone_px,  # type: ignore[arg-type]
                auto_aim=auto_aim,  # type: ignore[arg-type]
                led_on=led_on,  # type: ignore[arg-type]
                laser_on=laser_on,  # type: ignore[arg-type]
                servo_x=servo_x,  # type: ignore[arg-type]
                servo_y=servo_y,  # type: ignore[arg-type]
                err_x=err_x,  # type: ignore[arg-type]
                err_y=err_y,  # type: ignore[arg-type]
                fps=fps,  # type: ignore[arg-type]
                eco_active=eco_active,  # type: ignore[arg-type]
                eco_motion_woken=eco_motion_woken,  # type: ignore[arg-type]
            )

            jpg_bytes: Optional[bytes] = None
            if need_jpeg:
                ok, encoded = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 78])
                if ok:
                    jpg_bytes = encoded.tobytes()

            if jpg_bytes is not None or need_frame:
                with self._frame_cond:
                    if need_frame:
                        self._annotated_frame = annotated
                    if jpg_bytes is not None:
                        self._annotated_jpeg = jpg_bytes
                    self._annotated_id += 1
                    self._frame_cond.notify_all()

    def _sync_hardware_locked(self) -> None:
        """Compute effective Flash LED & Green Laser states and push unified /servo state to hardware."""
        fire_allowed = self.auto_aim and self.auto_fire and (self.led_mode != "off")
        auto_firing = fire_allowed and self._auto_fire_active
        effective_led = self._manual_led or (auto_firing and self.auto_fire_led) or self._eco_flash_active
        effective_laser = self._manual_laser or (auto_firing and self.auto_fire_laser)
        self._led_on = effective_led
        self._laser_on = effective_laser
        self.hardware.set_state(
            x=self._curr_x,
            y=self._curr_y,
            led=effective_led,
            laser=effective_laser,
        )

    def start(self) -> None:
        self.hardware.start()
        self.discord.start()
        with self._lock:
            now = time.time()
            self._curr_x = 90.0
            self._curr_y = 90.0
            self._manual_led = False
            self._manual_laser = False
            self._auto_fire_active = False
            self._last_cv_activity_time = now
            self._eco_active = False
            self._eco_wake_until = 0.0
            self._eco_flash_until = 0.0
            self._eco_flash_active = False
            self._motion_prev_gray = None
            self._motion_suppress_until = now + 1.5
            self._sync_hardware_locked()
        self.reader.start()
        self._stop_event.clear()
        if not self._encoder_thread.is_alive():
            self._encoder_thread = threading.Thread(
                target=self._encoder_loop, daemon=True, name="FrameEncoder"
            )
            self._encoder_thread.start()
        if not self._thread.is_alive():
            self._thread = threading.Thread(target=self._loop, daemon=True, name="AutoAimEngine")
            self._thread.start()

    def is_stopped(self) -> bool:
        return self._stop_event.is_set()

    def stop(self) -> None:
        self._stop_event.set()
        with self._render_cond:
            self._render_cond.notify_all()
        with self._frame_cond:
            self._frame_cond.notify_all()
        self.reader.stop()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self._encoder_thread.is_alive():
            self._encoder_thread.join(timeout=2.0)
        self.discord.shutdown()
        self.hardware.shutdown()

    def update_config(self, cfg: Dict[str, object]) -> None:
        with self._lock:
            cam_res: Optional[int] = None
            cam_qual: Optional[int] = None
            cam_bri: Optional[int] = None
            cam_ae: Optional[int] = None
            cam_gain: Optional[int] = None
            cam_night: Optional[bool] = None
            cam_dirty = False
            hw_dirty = False
            now = time.time()
            if "auto_aim" in cfg:
                new_auto = bool(cfg["auto_aim"])
                if new_auto != self.auto_aim:
                    self.auto_aim = new_auto
                    self._prev_deg_err_x = 0.0
                    self._prev_deg_err_y = 0.0
                    if new_auto:
                        self._last_cv_activity_time = now
                        self._eco_active = False
                        self._eco_wake_until = 0.0
                    else:
                        self._auto_fire_active = False
                        self._smoothed_target = None
                        self._eco_wake_until = 0.0
                        self._eco_flash_until = 0.0
                        self._eco_flash_active = False
                        if not self.detect_in_manual:
                            self._eco_active = False
                        hw_dirty = True
            if "detect_person" in cfg:
                new_dp = bool(cfg["detect_person"])
                if new_dp != self.detect_person:
                    self.detect_person = new_dp
                    self._smoothed_target = None
                    if new_dp:
                        self._last_cv_activity_time = now
            if "detect_dog" in cfg:
                new_dd = bool(cfg["detect_dog"])
                if new_dd != self.detect_dog:
                    self.detect_dog = new_dd
                    self._smoothed_target = None
                    if new_dd:
                        self._last_cv_activity_time = now
            if "detect_in_manual" in cfg:
                new_dim = bool(cfg["detect_in_manual"])
                if new_dim != self.detect_in_manual:
                    self.detect_in_manual = new_dim
                    if new_dim:
                        self._last_cv_activity_time = now
                        self._eco_active = False
                    elif not self.auto_aim:
                        self._eco_active = False
            if "eco_mode" in cfg:
                new_eco = bool(cfg["eco_mode"])
                if new_eco != self.eco_mode:
                    self.eco_mode = new_eco
                    if new_eco:
                        self._last_cv_activity_time = now
                    else:
                        self._eco_active = False
                        self._eco_wake_until = 0.0
                        if self._eco_flash_active:
                            self._eco_flash_active = False
                            self._eco_flash_until = 0.0
                            hw_dirty = True
            if "eco_delay_s" in cfg:
                self.eco_delay_s = max(3.0, min(300.0, float(cfg["eco_delay_s"])))
            if "eco_fps" in cfg:
                self.eco_fps = max(0.2, min(10.0, float(cfg["eco_fps"])))
            if "eco_motion_wake" in cfg:
                new_mw = bool(cfg["eco_motion_wake"])
                if new_mw != self.eco_motion_wake:
                    self.eco_motion_wake = new_mw
                    self._motion_prev_gray = None
                    if not new_mw:
                        self._eco_wake_until = 0.0
                        if self._eco_flash_active:
                            self._eco_flash_active = False
                            self._eco_flash_until = 0.0
                            hw_dirty = True
            if "eco_motion_flash" in cfg:
                new_mf = bool(cfg["eco_motion_flash"])
                if new_mf != self.eco_motion_flash:
                    self.eco_motion_flash = new_mf
                    if not new_mf and self._eco_flash_active:
                        self._eco_flash_active = False
                        self._eco_flash_until = 0.0
                        hw_dirty = True
            if "eco_flash_dark_only" in cfg:
                self.eco_flash_dark_only = bool(cfg["eco_flash_dark_only"])
            if "eco_motion_sensitivity" in cfg:
                self.eco_motion_sensitivity = max(10.0, min(100.0, float(cfg["eco_motion_sensitivity"])))
            if "eco_flash_duration_s" in cfg:
                self.eco_flash_duration_s = max(2.0, min(30.0, float(cfg["eco_flash_duration_s"])))
            if "eco_dark_threshold" in cfg:
                self.eco_dark_threshold = max(15.0, min(200.0, float(cfg["eco_dark_threshold"])))
            if "auto_fire" in cfg:
                self.auto_fire = bool(cfg["auto_fire"])
                if not self.auto_fire:
                    self._auto_fire_active = False
                hw_dirty = True
            if "auto_fire_laser" in cfg:
                self.auto_fire_laser = bool(cfg["auto_fire_laser"])
                hw_dirty = True
            if "auto_fire_led" in cfg:
                self.auto_fire_led = bool(cfg["auto_fire_led"])
                hw_dirty = True
            if "aim_mode" in cfg and cfg["aim_mode"] in ("center", "upper", "head"):
                self.aim_mode = str(cfg["aim_mode"])
            mode_val = cfg.get("fire_mode", cfg.get("led_mode"))
            if mode_val in ("on_lock", "on_detect", "off"):
                self.led_mode = str(mode_val)
                if self.led_mode == "off":
                    self._auto_fire_active = False
                hw_dirty = True
            if "kp" in cfg:
                self.kp = max(0.2, min(2.0, float(cfg["kp"])))
            if "kd" in cfg:
                self.kd = max(0.0, min(0.5, float(cfg["kd"])))
            if "deadzone_px" in cfg:
                self.deadzone_px = max(2.0, min(80.0, float(cfg["deadzone_px"])))
            if "conf_threshold" in cfg:
                self.conf_threshold = max(0.15, min(0.9, float(cfg["conf_threshold"])))
            if "invert_x" in cfg:
                self.invert_x = bool(cfg["invert_x"])
            if "invert_y" in cfg:
                self.invert_y = bool(cfg["invert_y"])
            if "rotation" in cfg:
                rot = int(cfg["rotation"])
                if rot in ROTATION_CODES:
                    self.reader.set_rotation(rot)
                    self._motion_prev_gray = None
                    self._motion_suppress_until = max(self._motion_suppress_until, now + 1.0)
            if "resolution" in cfg and cfg["resolution"] is not None:
                res_code = max(0, min(5, int(cfg["resolution"])))
                if res_code != self.resolution or not self._res_explicit:
                    self.resolution = res_code
                    self._res_explicit = True
                    cam_res = res_code
                    cam_dirty = True
            if "cam_qual" in cfg and cfg["cam_qual"] is not None:
                qual_val = max(10, min(40, int(cfg["cam_qual"])))
                if qual_val != self.cam_qual:
                    self.cam_qual = qual_val
                    cam_qual = qual_val
                    cam_dirty = True
            if "cam_bri" in cfg and cfg["cam_bri"] is not None:
                bri_val = max(-3, min(3, int(cfg["cam_bri"])))
                if bri_val != self.cam_bri:
                    self.cam_bri = bri_val
                    cam_bri = bri_val
                    cam_dirty = True
            if "cam_ae" in cfg and cfg["cam_ae"] is not None:
                ae_val = max(-5, min(5, int(cfg["cam_ae"])))
                if ae_val != self.cam_ae:
                    self.cam_ae = ae_val
                    cam_ae = ae_val
                    cam_dirty = True
            if "cam_gain" in cfg and cfg["cam_gain"] is not None:
                gain_val = max(0, min(5, int(cfg["cam_gain"])))
                if gain_val != self.cam_gain:
                    self.cam_gain = gain_val
                    cam_gain = gain_val
                    cam_dirty = True
            if "cam_night" in cfg and cfg["cam_night"] is not None:
                night_val = bool(cfg["cam_night"])
                if night_val != self.cam_night:
                    self.cam_night = night_val
                    cam_night = night_val
                    cam_dirty = True

            if cam_dirty:
                self._motion_prev_gray = None
                self._motion_suppress_until = max(self._motion_suppress_until, now + 1.5)
                self.hardware.set_cam_settings(
                    res=cam_res,
                    qual=cam_qual,
                    bri=cam_bri,
                    ae=cam_ae,
                    gain=cam_gain,
                    night=cam_night,
                )

            if hw_dirty:
                self._motion_prev_gray = None
                self._motion_suppress_until = max(self._motion_suppress_until, now + 1.2)
                self._sync_hardware_locked()

        self.discord.update_config(cfg)

    def apply_servo_command(
        self,
        x: Optional[float] = None,
        y: Optional[float] = None,
        led: Optional[bool] = None,
        laser: Optional[bool] = None,
        disable_auto_aim: bool = False,
    ) -> None:
        """Apply real-time /servo parameters atomically in a single hardware dispatch."""
        with self._lock:
            now = time.time()
            if disable_auto_aim and self.auto_aim:
                self.auto_aim = False
                self._auto_fire_active = False
                self._smoothed_target = None
                self._prev_deg_err_x = 0.0
                self._prev_deg_err_y = 0.0
                self._eco_wake_until = 0.0
                self._eco_flash_until = 0.0
                self._eco_flash_active = False
                if not self.detect_in_manual:
                    self._eco_active = False
            if x is not None:
                self._curr_x = max(0.0, min(180.0, float(x)))
                self._prev_deg_err_x = 0.0
                self._wait_frame_after = now + 0.12
                self._motion_suppress_until = max(self._motion_suppress_until, now + 1.2)
                self._motion_prev_gray = None
            if y is not None:
                self._curr_y = max(0.0, min(180.0, float(y)))
                self._prev_deg_err_y = 0.0
                self._wait_frame_after = now + 0.12
                self._motion_suppress_until = max(self._motion_suppress_until, now + 1.2)
                self._motion_prev_gray = None
            if led is not None:
                self._manual_led = bool(led)
                if not self._manual_led and self._eco_flash_active:
                    self._eco_flash_active = False
                    self._eco_flash_until = 0.0
                self._motion_suppress_until = max(self._motion_suppress_until, now + 2.0)
                self._motion_prev_gray = None
            if laser is not None:
                self._manual_laser = bool(laser)
            self._sync_hardware_locked()

    def apply_cam_command(
        self,
        res: Optional[int] = None,
        qual: Optional[int] = None,
        bri: Optional[int] = None,
        ae: Optional[int] = None,
        gain: Optional[int] = None,
        night: Optional[bool] = None,
        force: bool = True,
    ) -> None:
        """Apply camera settings via ESP32 GET /cam (independent from /servo)."""
        with self._lock:
            now = time.time()
            res_code: Optional[int] = None
            qual_val: Optional[int] = None
            bri_val: Optional[int] = None
            ae_val: Optional[int] = None
            gain_val: Optional[int] = None
            night_val: Optional[bool] = None
            any_set = False

            if res is not None:
                res_code = max(0, min(5, int(res)))
                self.resolution = res_code
                self._res_explicit = True
                any_set = True
            if qual is not None:
                qual_val = max(10, min(40, int(qual)))
                self.cam_qual = qual_val
                any_set = True
            if bri is not None:
                bri_val = max(-3, min(3, int(bri)))
                self.cam_bri = bri_val
                any_set = True
            if ae is not None:
                ae_val = max(-5, min(5, int(ae)))
                self.cam_ae = ae_val
                any_set = True
            if gain is not None:
                gain_val = max(0, min(5, int(gain)))
                self.cam_gain = gain_val
                any_set = True
            if night is not None:
                night_val = bool(night)
                self.cam_night = night_val
                any_set = True

            if any_set:
                self._motion_suppress_until = max(self._motion_suppress_until, now + 1.5)
                self._motion_prev_gray = None
                self.hardware.set_cam_settings(
                    res=res_code,
                    qual=qual_val,
                    bri=bri_val,
                    ae=ae_val,
                    gain=gain_val,
                    night=night_val,
                    force=force,
                )

    def reset_cam_defaults(self) -> None:
        """Restore ESP32 OV3660 balanced defaults via GET /cam."""
        self.apply_cam_command(
            qual=DEFAULT_CAM_QUAL,
            bri=DEFAULT_CAM_BRI,
            ae=DEFAULT_CAM_AE,
            gain=DEFAULT_CAM_GAIN,
            night=DEFAULT_CAM_NIGHT,
            force=True,
        )

    def center_turret(self) -> None:
        with self._lock:
            now = time.time()
            self._curr_x = 90.0
            self._curr_y = 90.0
            self._prev_deg_err_x = 0.0
            self._prev_deg_err_y = 0.0
            self._smoothed_target = None
            self._wait_frame_after = now + 0.12
            self._motion_suppress_until = max(self._motion_suppress_until, now + 1.2)
            self._motion_prev_gray = None
            self.hardware.center()
            self._sync_hardware_locked()

    def failsafe_off(self) -> None:
        """Emergency/disconnect fail-safe: immediately force Laser and Flash OFF on ESP32."""
        with self._lock:
            now = time.time()
            self._manual_led = False
            self._manual_laser = False
            self._auto_fire_active = False
            self._eco_flash_active = False
            self._eco_flash_until = 0.0
            self._led_on = False
            self._laser_on = False
            self._motion_suppress_until = max(self._motion_suppress_until, now + 2.0)
            self._motion_prev_gray = None
            self.hardware.force_failsafe()

    def _apply_stream_loss_failsafe(self) -> None:
        """If the video stream drops, immediately turn off Auto-Aim firing and Eco Flash."""
        with self._lock:
            changed = False
            if self._auto_fire_active:
                self._auto_fire_active = False
                changed = True
            if self._eco_flash_active:
                self._eco_flash_active = False
                self._eco_flash_until = 0.0
                changed = True
            if self._person_detected or self._target_type is not None:
                self._person_detected = False
                self._target_type = None
                self._detections_count = 0
                self._target_box = None
                self._target_center = None
                self._error_px = [0, 0]
                self._smoothed_target = None
            if changed:
                self._sync_hardware_locked()

    def set_manual_angles(self, x: float, y: float) -> None:
        self.apply_servo_command(x=x, y=y)

    def set_manual_led(self, state: bool) -> None:
        self.apply_servo_command(led=bool(state))

    def set_manual_laser(self, state: bool) -> None:
        self.apply_servo_command(laser=bool(state))

    def set_resolution(self, res_code: int) -> None:
        self.apply_cam_command(res=int(res_code), force=True)

    def send_discord_test(self) -> Tuple[bool, str]:
        frame, _, _ = self.reader.get_latest_frame()
        with self._lock:
            servo_x = int(round(self._curr_x))
            servo_y = int(round(self._curr_y))
            led_on = self._led_on
            laser_on = self._laser_on
        return self.discord.send_test(
            frame=frame,
            servo_x=servo_x,
            servo_y=servo_y,
            led_on=led_on,
            laser_on=laser_on,
        )

    def get_status(self) -> Dict[str, object]:
        hw = self.hardware.get_state()
        discord_status = self.discord.get_status()
        rotation = self.reader.get_rotation()
        stream_stats = self.reader.get_stats()
        stream_connected = bool(stream_stats["connected"])
        with self._lock:
            now = time.time()
            cv_enabled = (self.auto_aim or self.detect_in_manual) and (self.detect_person or self.detect_dog)
            eco_idle_s = max(0.0, now - self._last_cv_activity_time) if cv_enabled else 0.0
            eco_motion_woken = cv_enabled and self.eco_mode and (now < self._eco_wake_until)
            return {
                "auto_aim": self.auto_aim,
                "detect_person": self.detect_person,
                "detect_dog": self.detect_dog,
                "detect_in_manual": self.detect_in_manual,
                "eco_mode": self.eco_mode,
                "eco_delay_s": round(self.eco_delay_s, 1),
                "eco_fps": round(self.eco_fps, 2),
                "eco_active": self._eco_active,
                "eco_idle_s": round(eco_idle_s, 1),
                "eco_motion_wake": self.eco_motion_wake,
                "eco_motion_flash": self.eco_motion_flash,
                "eco_flash_dark_only": self.eco_flash_dark_only,
                "eco_motion_sensitivity": round(self.eco_motion_sensitivity, 0),
                "eco_flash_duration_s": round(self.eco_flash_duration_s, 1),
                "eco_dark_threshold": round(self.eco_dark_threshold, 0),
                "eco_motion_woken": eco_motion_woken,
                "eco_flash_active": self._eco_flash_active,
                "motion_score": round(self._motion_score, 2),
                "ambient_luma": round(self._ambient_luma, 1),
                "auto_fire": self.auto_fire,
                "auto_fire_laser": self.auto_fire_laser,
                "auto_fire_led": self.auto_fire_led,
                "aim_mode": self.aim_mode,
                "led_mode": self.led_mode,
                "fire_mode": self.led_mode,
                "kp": round(self.kp, 3),
                "kd": round(self.kd, 3),
                "deadzone_px": round(self.deadzone_px, 1),
                "conf_threshold": round(self.conf_threshold, 2),
                "invert_x": self.invert_x,
                "invert_y": self.invert_y,
                "rotation": rotation,
                "resolution": self.resolution,
                "cam_qual": self.cam_qual,
                "cam_bri": self.cam_bri,
                "cam_ae": self.cam_ae,
                "cam_gain": self.cam_gain,
                "cam_night": self.cam_night,
                "frame_size": list(self._frame_size),
                "person_detected": self._person_detected,
                "target_type": self._target_type,
                "detections_count": self._detections_count,
                "led_on": self._led_on,
                "laser_on": self._laser_on,
                "manual_led": self._manual_led,
                "manual_laser": self._manual_laser,
                "servo_x": int(round(self._curr_x)),
                "servo_y": int(round(self._curr_y)),
                "target_box": self._target_box,
                "target_center": self._target_center,
                "error_px": self._error_px,
                "fps": round(self._cv_fps if cv_enabled else self._video_fps, 1),
                "cv_fps": round(self._cv_fps, 1),
                "video_fps": round(self._video_fps, 1),
                "stream_fps": round(float(stream_stats["stream_fps"]), 1),
                "stream_frame_kb": round(int(stream_stats["last_frame_bytes"]) / 1024.0, 1),
                "stream_error": str(stream_stats["last_error"]),
                "inference_ms": round(self._inference_ms, 1),
                "device": self.detector.device_info,
                "stream_connected": stream_connected,
                "turret_url": self.hardware.base_url,
                "hw_move_interval_ms": hw["move_interval_ms"],
                "hw_rtt_ms": hw["last_rtt_ms"],
                "hw_error": hw["last_error"],
                "hw_last_url": hw["last_url"],
                "hw_last_payload": hw["last_payload"],
                "hw_last_cam_payload": hw["last_cam_payload"],
                **discord_status,
            }

    def get_annotated_jpeg(self) -> Tuple[Optional[bytes], int]:
        with self._frame_cond:
            return self._annotated_jpeg, self._annotated_id

    def wait_for_annotated_jpeg(self, last_id: int, timeout: float = 1.0) -> Tuple[Optional[bytes], int]:
        with self._frame_cond:
            if (self._annotated_jpeg is None or self._annotated_id == last_id) and not self._stop_event.is_set():
                self._frame_cond.wait_for(
                    lambda: self._stop_event.is_set()
                    or (self._annotated_jpeg is not None and self._annotated_id != last_id),
                    timeout=timeout,
                )
            return self._annotated_jpeg, self._annotated_id

    def get_annotated_frame(self) -> Tuple[Optional[np.ndarray], int]:
        with self._frame_cond:
            if self._annotated_frame is None:
                return None, self._annotated_id
            return self._annotated_frame.copy(), self._annotated_id

    @staticmethod
    def _compute_motion_diff(
        prev_gray: Optional[np.ndarray],
        frame: np.ndarray,
        sensitivity: float,
    ) -> Tuple[np.ndarray, bool, float, float]:
        """Lightweight frame-differencing motion detector (~0.25 ms CPU on 160x120 grayscale).

        Uses Gaussian blur + morphological opening to reject isolated dark-room sensor noise
        while adapting pixel threshold downward in dim scenes.
        Returns (curr_gray, motion_triggered, changed_pct, ambient_luma).
        """
        small = cv2.resize(frame, (160, 120), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        ambient_luma = float(np.mean(gray))

        if prev_gray is None or prev_gray.shape != gray.shape:
            return gray, False, 0.0, ambient_luma

        diff = cv2.absdiff(prev_gray, gray)
        s_norm = max(0.0, min(1.0, (sensitivity - 10.0) / 90.0))
        dark_factor = 0.72 if ambient_luma < 60.0 else 1.0
        pixel_thresh = max(8, int(round((34.0 - 22.0 * s_norm) * dark_factor)))
        area_thresh_pct = 3.6 - 3.2 * s_norm

        _, mask = cv2.threshold(diff, pixel_thresh, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        changed_pixels = int(cv2.countNonZero(mask))
        total_pixels = max(1, int(gray.size))
        changed_pct = (changed_pixels * 100.0) / float(total_pixels)
        return gray, (changed_pct >= area_thresh_pct), changed_pct, ambient_luma

    @staticmethod
    def _compute_dynamic_step(
        err_px: float,
        rem_deg: float,
        kp: float,
        deadzone_px: float,
        max_step_deg: float,
    ) -> float:
        if abs(err_px) <= deadzone_px and abs(rem_deg) <= 1.6:
            return 0.0
        abs_rem = abs(rem_deg)
        if abs_rem < 0.6:
            return 0.0
        if abs_rem <= 3.0:
            gain = 0.45 * kp
        elif abs_rem <= 10.0:
            t = (abs_rem - 3.0) / 7.0
            gain = (0.45 + 0.43 * t) * kp
        else:
            gain = 0.90 * kp
        gain = max(0.20, min(0.95, gain))
        return math.copysign(min(max_step_deg, abs_rem * gain), rem_deg)

    def _loop(self) -> None:
        last_frame_id = -1
        video_fps_ema = 0.0
        cv_fps_ema = 0.0
        prev_loop_mono = 0.0
        last_cv_exec_mono = 0.0
        last_cv_exec_time = 0.0
        last_motion_check_time = 0.0
        last_inference_ms = 0.0
        next_cv_deadline = 0.0
        was_eco_active = False
        woke_from_eco = False
        prev_led_state = False

        while not self._stop_event.is_set():
            frame, frame_id, frame_ts = self.reader.wait_for_frame(last_frame_id, timeout=0.05)
            if frame is None or frame_id == last_frame_id:
                if not self.reader.is_connected():
                    self._apply_stream_loss_failsafe()
                continue
            last_frame_id = frame_id

            now = time.time()
            now_mono = time.monotonic()
            if prev_loop_mono > 0.0:
                dt_loop = max(1e-3, now_mono - prev_loop_mono)
                inst_video_fps = 1.0 / dt_loop
                video_alpha = 1.0 - math.exp(-dt_loop / 0.8)
                video_fps_ema = (
                    inst_video_fps
                    if video_fps_ema == 0.0
                    else ((1.0 - video_alpha) * video_fps_ema + video_alpha * inst_video_fps)
                )
            prev_loop_mono = now_mono

            with self._lock:
                auto_aim = self.auto_aim
                detect_person = self.detect_person
                detect_dog = self.detect_dog
                detect_in_manual = self.detect_in_manual
                auto_fire = self.auto_fire
                aim_mode = self.aim_mode
                led_mode = self.led_mode
                kp = self.kp
                deadzone_px = self.deadzone_px
                conf_threshold = self.conf_threshold
                invert_x = self.invert_x
                invert_y = self.invert_y
                prev_smooth = self._smoothed_target
                eco_mode = self.eco_mode
                eco_delay_s = self.eco_delay_s
                eco_fps = self.eco_fps
                eco_motion_wake = self.eco_motion_wake
                eco_motion_sens = self.eco_motion_sensitivity
                eco_wake_until = self._eco_wake_until
                motion_suppress_until = self._motion_suppress_until
                prev_gray_snap = self._motion_prev_gray
                last_cv_activity = self._last_cv_activity_time

            h, w = frame.shape[:2]
            cx, cy = w * 0.5, h * 0.5

            run_cv = (auto_aim or detect_in_manual) and (detect_person or detect_dog)
            idle_elapsed = max(0.0, now - last_cv_activity) if run_cv else 0.0
            eco_idle_due = run_cv and eco_mode and (idle_elapsed >= eco_delay_s)
            eco_woken_now = run_cv and eco_mode and (now < eco_wake_until)

            # Lightweight motion + ambient brightness sampling (up to 5 Hz in Eco, 2 Hz otherwise)
            motion_interval = 0.20 if eco_idle_due else 0.50
            sampled_motion = False
            motion_triggered = False
            motion_pct = 0.0
            ambient_luma_sample: Optional[float] = None
            new_gray: Optional[np.ndarray] = None
            if (now - last_motion_check_time) >= motion_interval:
                last_motion_check_time = now
                sampled_motion = True
                if eco_idle_due and eco_motion_wake and (now >= motion_suppress_until):
                    new_gray, motion_triggered, motion_pct, ambient_luma_sample = self._compute_motion_diff(
                        prev_gray_snap, frame, eco_motion_sens
                    )
                else:
                    small = cv2.resize(frame, (160, 120), interpolation=cv2.INTER_AREA)
                    gray_only = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                    ambient_luma_sample = float(np.mean(gray_only))

            if run_cv:
                eco_active = eco_idle_due and not eco_woken_now
                if eco_active:
                    eco_period = 1.0 / max(0.2, eco_fps)
                    should_run_detector = (last_cv_exec_mono == 0.0) or ((now_mono - last_cv_exec_mono) >= eco_period)
                    if not was_eco_active:
                        cv_fps_ema = eco_fps
                        woke_from_eco = False
                else:
                    should_run_detector = True
                    if was_eco_active:
                        woke_from_eco = True
                was_eco_active = eco_active
            else:
                eco_active = False
                should_run_detector = False
                was_eco_active = False
                woke_from_eco = False
                cv_fps_ema = 0.0
                last_cv_exec_mono = 0.0
                last_cv_exec_time = 0.0

            if should_run_detector:
                t_inf_start = time.monotonic()
                detections = self.detector.detect(
                    frame,
                    conf_threshold=conf_threshold,
                    detect_person=detect_person,
                    detect_dog=detect_dog,
                    force_fallback=eco_active,
                )
                inference_ms = (time.monotonic() - t_inf_start) * 1000.0
                last_inference_ms = inference_ms
                if last_cv_exec_mono > 0.0:
                    dt_cv = max(1e-3, now_mono - last_cv_exec_mono)
                    inst_cv_fps = 1.0 / dt_cv
                    if woke_from_eco:
                        cv_fps_ema = inst_cv_fps
                        woke_from_eco = False
                    else:
                        cv_alpha = 1.0 - math.exp(-dt_cv / 0.8)
                        cv_fps_ema = (
                            inst_cv_fps
                            if cv_fps_ema == 0.0
                            else ((1.0 - cv_alpha) * cv_fps_ema + cv_alpha * inst_cv_fps)
                        )
                else:
                    cv_fps_ema = video_fps_ema if video_fps_ema > 0.0 else 0.0
                last_cv_exec_mono = now_mono
                last_cv_exec_time = now
                target = self.detector.select_target(detections, (h, w), prev_smooth)
            else:
                detections = []
                inference_ms = last_inference_ms if run_cv else 0.0
                target = None

            target_detected = target is not None
            err_x = 0.0
            err_y = 0.0
            aim_pt: Optional[Tuple[float, float]] = None

            with self._lock:
                self._frame_size = [w, h]
                if not self._res_explicit:
                    self.resolution = infer_resolution_code(w, h)

                if sampled_motion:
                    if ambient_luma_sample is not None:
                        self._ambient_luma = ambient_luma_sample
                    if new_gray is not None and now >= self._motion_suppress_until:
                        self._motion_prev_gray = new_gray
                        self._motion_score = motion_pct
                    else:
                        self._motion_prev_gray = None
                        self._motion_score = 0.0

                # Re-check self.auto_aim under lock in case manual mode was activated during inference
                active_auto_aim = self.auto_aim
                active_run_cv = (active_auto_aim or self.detect_in_manual) and (
                    self.detect_person or self.detect_dog
                )
                should_sync_hw = False

                if not active_run_cv:
                    self._last_cv_activity_time = now
                    self._eco_wake_until = 0.0
                    if self._eco_flash_active:
                        self._eco_flash_active = False
                        self._eco_flash_until = 0.0
                        should_sync_hw = True

                if target is not None:
                    tx, ty = self.detector.compute_aim_point(target, aim_mode=aim_mode)
                    self._smoothed_target = (tx, ty)
                    self._last_detection_time = now
                    self._last_cv_activity_time = now
                    self._eco_wake_until = 0.0
                    if self._eco_flash_active:
                        # Keep light on while actively tracking a target illuminated by Eco motion flash
                        self._eco_flash_until = max(self._eco_flash_until, now + 2.5)
                    aim_pt = (tx, ty)

                    err_x = tx - cx
                    err_y = ty - cy

                    if active_auto_aim:
                        effective_interval = max(
                            self.min_servo_interval, self.hardware.get_effective_move_interval()
                        )
                        if (now - self._last_servo_step_time) >= effective_interval:
                            px_per_deg_x = 10.3 * (w / 600.0)
                            px_per_deg_y = 9.2 * (h / 800.0)

                            dir_x = 1.0 if invert_x else -1.0
                            dir_y = -1.0 if invert_y else 1.0

                            delta_x = dir_x * (err_x / px_per_deg_x)
                            delta_y = dir_y * (err_y / px_per_deg_y)

                            cam_x, cam_y = self.hardware.get_pose_at(frame_ts - self.camera_latency)
                            target_world_x = max(5.0, min(175.0, cam_x + delta_x))
                            target_world_y = max(15.0, min(165.0, cam_y + delta_y))

                            rem_x = target_world_x - self._curr_x
                            rem_y = target_world_y - self._curr_y

                            step_x = self._compute_dynamic_step(
                                err_x, rem_x, kp, deadzone_px, self.max_step_deg
                            )
                            step_y = self._compute_dynamic_step(
                                err_y, rem_y, kp, deadzone_px, self.max_step_deg
                            )

                            if abs(step_x) >= 0.5 or abs(step_y) >= 0.5:
                                self._curr_x = max(5.0, min(175.0, self._curr_x + step_x))
                                self._curr_y = max(15.0, min(165.0, self._curr_y + step_y))
                                self._last_servo_step_time = now
                                should_sync_hw = True

                        prev_fire = self._auto_fire_active
                        if not auto_fire or led_mode == "off":
                            self._auto_fire_active = False
                        elif led_mode == "on_lock":
                            box_w = target.x2 - target.x1
                            box_h = target.y2 - target.y1
                            in_box = (
                                (target.x1 - deadzone_px <= cx <= target.x2 + deadzone_px)
                                and (target.y1 - deadzone_px <= cy <= target.y2 + deadzone_px)
                            )
                            lock_radius = max(deadzone_px * 2.2, min(box_w, box_h) * 0.45)
                            self._auto_fire_active = in_box or (math.hypot(err_x, err_y) <= lock_radius)
                        else:
                            self._auto_fire_active = True

                        if self._auto_fire_active != prev_fire:
                            should_sync_hw = True
                else:
                    prev_fire = self._auto_fire_active
                    if (now - self._last_detection_time) > self.hold_timeout:
                        self._smoothed_target = None
                        self._prev_deg_err_x = 0.0
                        self._prev_deg_err_y = 0.0
                        if active_auto_aim:
                            self._auto_fire_active = False
                    elif active_auto_aim and (not auto_fire or led_mode == "off"):
                        self._auto_fire_active = False
                    if self._auto_fire_active != prev_fire:
                        should_sync_hw = True

                    if self._eco_flash_active and (
                        not active_run_cv
                        or not self.eco_mode
                        or not self.eco_motion_wake
                        or not self.eco_motion_flash
                        or now >= self._eco_flash_until
                    ):
                        self._eco_flash_active = False
                        self._eco_flash_until = 0.0
                        self._motion_suppress_until = max(self._motion_suppress_until, now + 2.0)
                        self._motion_prev_gray = None
                        should_sync_hw = True

                prev_eco_locked = self._eco_active
                eco_idle_locked = (
                    active_run_cv
                    and self.eco_mode
                    and ((now - self._last_cv_activity_time) >= self.eco_delay_s)
                )

                if (
                    prev_eco_locked
                    and eco_idle_locked
                    and self.eco_motion_wake
                    and motion_triggered
                    and (now >= self._motion_suppress_until)
                    and (now >= self._eco_wake_until)
                ):
                    self._eco_wake_until = now + self.eco_flash_duration_s
                    luma_val = ambient_luma_sample if ambient_luma_sample is not None else self._ambient_luma
                    should_light = self.eco_motion_flash and (
                        (not self.eco_flash_dark_only) or (luma_val < self.eco_dark_threshold)
                    )
                    if should_light:
                        self._eco_flash_until = now + self.eco_flash_duration_s
                        if not self._eco_flash_active:
                            self._eco_flash_active = True
                            self._motion_suppress_until = max(self._motion_suppress_until, now + 2.0)
                            self._motion_prev_gray = None
                            should_sync_hw = True

                eco_woken_locked = active_run_cv and self.eco_mode and (now < self._eco_wake_until)
                eco_active_now = eco_idle_locked and not eco_woken_locked
                if was_eco_active and not eco_active_now:
                    woke_from_eco = True
                    was_eco_active = False
                self._eco_active = eco_active_now

                if eco_active_now and not prev_eco_locked:
                    if active_auto_aim:
                        if int(round(self._curr_x)) != 90 or int(round(self._curr_y)) != 90:
                            self._curr_x = 90.0
                            self._curr_y = 90.0
                            self._prev_deg_err_x = 0.0
                            self._prev_deg_err_y = 0.0
                            self._smoothed_target = None
                            self._wait_frame_after = now + 0.12
                            should_sync_hw = True
                        else:
                            self._curr_x = 90.0
                            self._curr_y = 90.0
                    self._motion_suppress_until = max(self._motion_suppress_until, now + 2.0)
                    self._motion_prev_gray = None

                if should_sync_hw:
                    self._sync_hardware_locked()
                    if self._led_on != prev_led_state:
                        self._motion_suppress_until = max(self._motion_suppress_until, now + 2.0)
                        self._motion_prev_gray = None

                curr_x_deg = int(round(self._curr_x))
                curr_y_deg = int(round(self._curr_y))
                led_state = self._led_on
                laser_state = self._laser_on
                prev_led_state = led_state

                self._person_detected = target_detected
                self._target_type = target.source if target else None
                self._detections_count = len(detections)
                self._target_box = (
                    [int(target.x1), int(target.y1), int(target.x2), int(target.y2)] if target else None
                )
                self._target_center = [int(round(aim_pt[0])), int(round(aim_pt[1]))] if aim_pt else None
                self._error_px = [int(round(err_x)), int(round(err_y))]
                self._video_fps = video_fps_ema
                self._cv_fps = cv_fps_ema if active_run_cv else 0.0
                self._fps = self._cv_fps if active_run_cv else video_fps_ema
                self._inference_ms = inference_ms
                display_fps = self._fps

            face_det = (
                self.detector.detect_face_in_crop(frame, target)
                if (target is not None and self.discord.needs_face_check(target, now))
                else None
            )
            self.discord.observe(
                frame=frame,
                target=target,
                face_det=face_det,
                servo_x=curr_x_deg,
                servo_y=curr_y_deg,
                led_on=led_state,
                laser_on=laser_state,
                now=now,
            )

            self._submit_render_job(
                frame=frame,
                detections=detections,
                target=target,
                aim_pt=aim_pt,
                deadzone_px=deadzone_px,
                auto_aim=active_auto_aim,
                led_on=led_state,
                laser_on=laser_state,
                servo_x=curr_x_deg,
                servo_y=curr_y_deg,
                err_x=err_x,
                err_y=err_y,
                fps=display_fps,
                eco_active=eco_active_now,
                eco_motion_woken=eco_woken_locked,
            )

            if not run_cv:
                # Manual control mode: lock 1:1 to camera frame rate via wait_for_frame (zero extra sleep)
                next_cv_deadline = 0.0
            elif eco_active_now:
                next_cv_deadline = 0.0
                eco_preview_fps = max(eco_fps, min(DEFAULT_MAX_FPS if DEFAULT_MAX_FPS > 0 else 8.0, 8.0))
                min_period = 1.0 / eco_preview_fps
                sleep_s = min_period - (time.monotonic() - now_mono)
                if last_cv_exec_mono > 0.0:
                    eco_period = 1.0 / max(0.2, eco_fps)
                    until_next_cv = eco_period - (time.monotonic() - last_cv_exec_mono)
                    if until_next_cv > 0.0:
                        sleep_s = min(sleep_s, until_next_cv)
                if sleep_s > 0.002:
                    self._stop_event.wait(timeout=sleep_s)
            elif 0.0 < DEFAULT_MAX_FPS < 29.0:
                # Only sleep when user explicitly caps below camera native rate (~28-30 FPS);
                # avoids missing the next Wi-Fi frame and cutting CV FPS in half (aliasing)
                min_period = 1.0 / DEFAULT_MAX_FPS
                if next_cv_deadline == 0.0:
                    next_cv_deadline = now_mono + min_period
                else:
                    next_cv_deadline += min_period
                now_after = time.monotonic()
                if next_cv_deadline < now_after:
                    next_cv_deadline = now_after
                sleep_s = next_cv_deadline - now_after
                if sleep_s > 0.003:
                    self._stop_event.wait(timeout=sleep_s - 0.001)
            else:
                next_cv_deadline = 0.0

    @staticmethod
    def _draw_overlay(
        frame: np.ndarray,
        detections: List[Detection],
        target: Optional[Detection],
        aim_pt: Optional[Tuple[float, float]],
        deadzone_px: float,
        auto_aim: bool,
        led_on: bool,
        laser_on: bool,
        servo_x: int,
        servo_y: int,
        err_x: float,
        err_y: float,
        fps: float,
        eco_active: bool = False,
        eco_motion_woken: bool = False,
    ) -> np.ndarray:
        vis = frame.copy()
        h, w = vis.shape[:2]
        cx, cy = int(round(w * 0.5)), int(round(h * 0.5))
        firing = laser_on or led_on

        # Center reticle & deadzone ring
        if laser_on:
            reticle_color = (94, 235, 65)
        elif target is not None:
            reticle_color = (0, 220, 255)
        else:
            reticle_color = (185, 195, 210)

        cv2.line(vis, (cx - 20, cy), (cx - 6, cy), reticle_color, 1, cv2.LINE_AA)
        cv2.line(vis, (cx + 6, cy), (cx + 20, cy), reticle_color, 1, cv2.LINE_AA)
        cv2.line(vis, (cx, cy - 20), (cx, cy - 6), reticle_color, 1, cv2.LINE_AA)
        cv2.line(vis, (cx, cy + 6), (cx, cy + 20), reticle_color, 1, cv2.LINE_AA)
        cv2.circle(vis, (cx, cy), 2, reticle_color, -1, cv2.LINE_AA)
        if auto_aim:
            cv2.circle(vis, (cx, cy), max(4, int(round(deadzone_px))), reticle_color, 1, cv2.LINE_AA)

        # Draw detections
        for det in detections:
            is_active = target is not None and det is target
            color = (0, 0, 255) if (is_active and firing) else ((60, 235, 110) if is_active else (0, 215, 255))
            thickness = 2 if is_active else 1
            x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
            src_label = SOURCE_LABELS.get(det.source, det.source.upper())
            tag = f"ALVO: {src_label}" if is_active else src_label
            label = f"{tag} {det.conf * 100:.0f}%"
            cv2.putText(
                vis,
                label,
                (x1, max(20, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                color,
                2,
                cv2.LINE_AA,
            )

        # Aim vector & lock dot
        if aim_pt is not None:
            tx, ty = int(round(aim_pt[0])), int(round(aim_pt[1]))
            dot_color = (0, 0, 255) if firing else (60, 240, 110)
            cv2.line(vis, (cx, cy), (tx, ty), (255, 190, 40), 2, cv2.LINE_AA)
            cv2.circle(vis, (tx, ty), 5, dot_color, -1, cv2.LINE_AA)
            cv2.circle(vis, (tx, ty), 9, (255, 255, 255), 1, cv2.LINE_AA)

        # Top compact translucent HUD bar (in-place ROI blend: ~20x faster than full-frame copy)
        bar_h = min(48, h)
        if bar_h > 0:
            roi = vis[:bar_h, :]
            cv2.convertScaleAbs(roi, dst=roi, alpha=0.28, beta=0.0)
            cv2.add(roi, (9, 12, 16, 0), dst=roi)

        mode_str = "AUTO-AIM" if auto_aim else "MANUAL"
        laser_str = "LASER:ON" if laser_on else "LASER:OFF"
        flash_str = "FLASH:ON" if led_on else "FLASH:OFF"
        if target is not None:
            tgt_label = SOURCE_LABELS.get(target.source, "ALVO")
            tgt_str = f"ALVO:{tgt_label}"
        elif eco_motion_woken:
            tgt_str = "MOVIMENTO"
        else:
            tgt_str = "LIVRE" if not auto_aim else ("ECO" if eco_active else "BUSCANDO")

        eco_tag = " [ECO]" if eco_active else (" [MOV]" if eco_motion_woken else "")
        line1 = f"{mode_str} | {tgt_str} | {laser_str} | {flash_str}"
        line2 = f"X:{servo_x}deg Y:{servo_y}deg | ERR:({int(round(err_x)):+d},{int(round(err_y)):+d})px | {fps:.1f}FPS{eco_tag}"

        if laser_on:
            status_color = (94, 235, 65)
        elif led_on:
            status_color = (40, 200, 255)
        elif eco_active:
            status_color = (94, 234, 212)
        else:
            status_color = (225, 232, 240)

        cv2.putText(vis, line1, (10, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_color, 1, cv2.LINE_AA)
        cv2.putText(vis, line2, (10, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (190, 200, 215), 1, cv2.LINE_AA)

        return vis
