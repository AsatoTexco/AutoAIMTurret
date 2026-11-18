import math
import os
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests

try:
    import onnxruntime as ort
except ImportError:
    ort = None


DEFAULT_TURRET_URL = os.environ.get("TURRET_URL", "http://192.168.100.5")
DEFAULT_STREAM_URL = os.environ.get("STREAM_URL", "http://192.168.100.5:81/stream")
DEFAULT_BACKEND = os.environ.get("AUTOAIM_BACKEND", "auto").lower()
DEFAULT_OPENCL_FP16 = os.environ.get("AUTOAIM_OPENCL_FP16", "0") == "1"
DEFAULT_CPU_THREADS = max(1, int(os.environ.get("AUTOAIM_CPU_THREADS", "4")))
DEFAULT_MAX_FPS = float(os.environ.get("AUTOAIM_MAX_FPS", "20.0"))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
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


class MJPEGStreamReader:
    """Reads frames continuously from an ESP32-CAM MJPEG HTTP stream."""

    def __init__(self, stream_url: str = DEFAULT_STREAM_URL, rotation: int = 90) -> None:
        self.stream_url = stream_url
        self._rotation = rotation if rotation in ROTATION_CODES else 90
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_id: int = 0
        self._last_frame_time: float = 0.0
        self._connected: bool = False
        self._last_error: str = ""
        self._thread = threading.Thread(target=self._reader_loop, daemon=True, name="MJPEGReader")

    def start(self) -> None:
        if not self._thread.is_alive():
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._reader_loop, daemon=True, name="MJPEGReader")
            self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
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

    def get_latest_frame(self) -> Tuple[Optional[np.ndarray], int, float]:
        with self._lock:
            if self._latest_frame is None:
                return None, self._frame_id, self._last_frame_time
            return self._latest_frame.copy(), self._frame_id, self._last_frame_time

    def _reader_loop(self) -> None:
        while not self._stop_event.is_set():
            resp = None
            try:
                resp = requests.get(self.stream_url, stream=True, timeout=(3.0, 5.0))
                resp.raise_for_status()
                buf = bytearray()
                for chunk in resp.iter_content(chunk_size=8192):
                    if self._stop_event.is_set():
                        break
                    if not chunk:
                        continue
                    buf.extend(chunk)

                    # Drain all complete JPEGs in the buffer and keep only the newest one
                    newest_jpg = None
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
                            if len(buf) > 1048576:
                                buf.clear()
                            break
                        last_soi = buf.rfind(b"\xff\xd8\xff", 0, end)
                        if last_soi > 0:
                            del buf[:last_soi]
                            end -= last_soi
                        newest_jpg = bytes(buf[: end + 2])
                        del buf[: end + 2]

                    if newest_jpg is not None:
                        arr = np.frombuffer(newest_jpg, dtype=np.uint8)
                        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if frame is not None:
                            with self._lock:
                                rot_code = ROTATION_CODES.get(self._rotation)
                            if rot_code is not None:
                                frame = cv2.rotate(frame, rot_code)
                            with self._lock:
                                self._latest_frame = frame
                                self._frame_id += 1
                                self._last_frame_time = time.time()
                                self._connected = True
                                self._last_error = ""
            except Exception as exc:
                with self._lock:
                    self._connected = False
                    self._last_error = str(exc)
            finally:
                if resp is not None:
                    try:
                        resp.close()
                    except Exception:
                        pass
            if not self._stop_event.is_set():
                time.sleep(1.0)


class TurretHardwareClient:
    """Single-flight non-blocking HTTP client for ESP32-CAM pan/tilt servos and LED."""

    def __init__(self, base_url: str = DEFAULT_TURRET_URL) -> None:
        self.base_url = base_url.rstrip("/")
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._wakeup_event = threading.Event()
        self._session = requests.Session()

        self._target_x: int = 90
        self._target_y: int = 90
        self._target_led: bool = False

        self._sent_x: Optional[int] = None
        self._sent_y: Optional[int] = None
        self._sent_led: Optional[bool] = None
        self._last_led_sent_time: float = 0.0
        self._last_error: str = ""
        self._pose_history: List[Tuple[float, int, int]] = [(time.time(), 90, 90)]

        self._thread = threading.Thread(target=self._worker_loop, daemon=True, name="TurretHTTP")

    def start(self) -> None:
        if not self._thread.is_alive():
            self._stop_event.clear()
            self._wakeup_event.clear()
            self._thread = threading.Thread(target=self._worker_loop, daemon=True, name="TurretHTTP")
            self._thread.start()

    def set_angles(self, x: float, y: float) -> None:
        ix = max(0, min(180, int(round(x))))
        iy = max(0, min(180, int(round(y))))
        with self._lock:
            changed = ix != self._target_x or iy != self._target_y
            self._target_x = ix
            self._target_y = iy
        if changed:
            self._wakeup_event.set()

    def set_led(self, state: bool) -> None:
        bstate = bool(state)
        with self._lock:
            changed = bstate != self._target_led
            self._target_led = bstate
        if changed:
            self._wakeup_event.set()

    def center(self) -> None:
        self.set_angles(90, 90)

    def set_resolution(self, res_code: int) -> bool:
        code = max(0, min(5, int(res_code)))
        try:
            r = self._session.get(f"{self.base_url}/servo", params={"res": code}, timeout=1.0)
            return r.status_code == 200
        except Exception as exc:
            with self._lock:
                self._last_error = str(exc)
            return False

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
            return {
                "target_x": self._target_x,
                "target_y": self._target_y,
                "target_led": self._target_led,
                "sent_x": self._sent_x,
                "sent_y": self._sent_y,
                "sent_led": self._sent_led,
                "last_error": self._last_error,
            }

    def shutdown(self) -> None:
        self._stop_event.set()
        self._wakeup_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.5)
        try:
            self._session.get(f"{self.base_url}/servo", params={"led": 0}, timeout=0.8)
        except Exception:
            pass
        try:
            self._session.close()
        except Exception:
            pass

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            now = time.time()
            with self._lock:
                tx, ty = self._target_x, self._target_y
                tled = self._target_led
                sx, sy = self._sent_x, self._sent_y
                sled = self._sent_led
                last_led_t = self._last_led_sent_time

            if tx != sx or ty != sy:
                try:
                    r = self._session.get(
                        f"{self.base_url}/servo",
                        params={"x": tx, "y": ty},
                        timeout=0.8,
                    )
                    if r.status_code == 200:
                        with self._lock:
                            self._sent_x = tx
                            self._sent_y = ty
                            self._last_error = ""
                            self._pose_history.append((time.time(), tx, ty))
                            if len(self._pose_history) > 60:
                                del self._pose_history[:-60]
                except Exception as exc:
                    with self._lock:
                        self._last_error = str(exc)

            need_led_update = (tled != sled) or (tled and (now - last_led_t >= 2.0))
            if need_led_update and not self._stop_event.is_set():
                try:
                    r = self._session.get(
                        f"{self.base_url}/servo",
                        params={"led": 1 if tled else 0},
                        timeout=0.8,
                    )
                    if r.status_code == 200:
                        with self._lock:
                            self._sent_led = tled
                            self._last_led_sent_time = time.time()
                            self._last_error = ""
                except Exception as exc:
                    with self._lock:
                        self._last_error = str(exc)

            with self._lock:
                pending = (self._target_x != self._sent_x or self._target_y != self._sent_y) or (
                    self._target_led != self._sent_led
                )
            if pending:
                time.sleep(0.005)
            else:
                self._wakeup_event.wait(timeout=0.025)
                self._wakeup_event.clear()


class PersonDetector:
    """ONNX YOLOv8n multi-class (person/dog) detector with GPU (ORT/OpenCL) + YuNet fallback."""

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
        self._use_cv_opencl: bool = False
        self._yunet = None
        self._empty_count: int = 0
        self.device_info: str = "CPU"

        if not os.path.exists(yolo_path):
            raise FileNotFoundError(f"YOLOv8 ONNX model not found at {yolo_path}")

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

        # Step B: OpenCV DNN + OpenCL GPU (e.g. AMD Radeon RX 580 via Mesa Rusticl)
        if not initialized and DEFAULT_BACKEND in ("auto", "opencl"):
            try:
                cv2.ocl.setUseOpenCL(True)
                if cv2.ocl.haveOpenCL():
                    dev = cv2.ocl.Device.getDefault()
                    gpu_flag = getattr(cv2.ocl, "Device_TYPE_GPU", getattr(cv2.ocl, "DEVICE_TYPE_GPU", 4))
                    is_gpu = bool(dev.available() and (dev.type() & gpu_flag))
                    if is_gpu or DEFAULT_BACKEND == "opencl":
                        if hasattr(cv2.dnn, "ENGINE_CLASSIC"):
                            net = cv2.dnn.readNetFromONNX(yolo_path, engine=cv2.dnn.ENGINE_CLASSIC)
                        else:
                            net = cv2.dnn.readNetFromONNX(yolo_path)
                        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                        target = cv2.dnn.DNN_TARGET_OPENCL_FP16 if DEFAULT_OPENCL_FP16 else cv2.dnn.DNN_TARGET_OPENCL
                        net.setPreferableTarget(target)
                        dummy = np.zeros((1, 3, self.input_size, self.input_size), dtype=np.float32)
                        net.setInput(dummy)
                        _ = net.forward()
                        self._cv_net = net
                        self._use_cv_opencl = True
                        dev_name = (dev.name() or "").strip() or "Radeon RX 580"
                        self.device_info = f"GPU OpenCL ({dev_name})"
                        initialized = True
            except Exception:
                self._cv_net = None
                self._use_cv_opencl = False

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

    def detect(
        self,
        frame: np.ndarray,
        conf_threshold: float = 0.35,
        detect_person: bool = True,
        detect_dog: bool = False,
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
        if detect_person and self._yunet is not None and (self._empty_count % 3 == 0):
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
        scale = min(sz / w, sz / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        dx, dy = (sz - nw) // 2, (sz - nh) // 2

        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((sz, sz, 3), 114, dtype=np.uint8)
        canvas[dy : dy + nh, dx : dx + nw] = resized

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        blob = np.transpose(rgb.astype(np.float32) * (1.0 / 255.0), (2, 0, 1))[None, ...]

        if self._use_cv_opencl and self._cv_net is not None:
            try:
                self._cv_net.setInput(blob)
                output = self._cv_net.forward()
            except Exception:
                self._use_cv_opencl = False
                if self._ort_session is None:
                    self._init_cpu_backend()
                output = self._ort_session.run(None, {self._ort_input_name: blob})[0]
        elif self._ort_session is not None:
            output = self._ort_session.run(None, {self._ort_input_name: blob})[0]
        else:
            self._cv_net.setInput(blob)
            output = self._cv_net.forward()

        # YOLOv8 output: [1, 84, 8400] -> transpose to [8400, 84]
        preds = output[0].T
        results: List[Detection] = []

        for source_name, cid in enabled_classes:
            scores = preds[:, 4 + cid]
            mask = scores >= conf_threshold
            if not np.any(mask):
                continue

            filtered = preds[mask]
            filtered_scores = scores[mask]

            cx = filtered[:, 0]
            cy = filtered[:, 1]
            bw = filtered[:, 2]
            bh = filtered[:, 3]

            x1 = np.clip((cx - 0.5 * bw - dx) / scale, 0.0, w - 1.0)
            y1 = np.clip((cy - 0.5 * bh - dy) / scale, 0.0, h - 1.0)
            x2 = np.clip((cx + 0.5 * bw - dx) / scale, 0.0, w - 1.0)
            y2 = np.clip((cy + 0.5 * bh - dy) / scale, 0.0, h - 1.0)

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
        self._yunet.setInputSize((w, h))
        _, faces = self._yunet.detect(frame)
        if faces is None or len(faces) == 0:
            return []
        results: List[Detection] = []
        for face in faces:
            fx, fy, fw, fh = float(face[0]), float(face[1]), float(face[2]), float(face[3])
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


class AutoAimEngine:
    """Closed-loop visual servoing engine that tracks targets (person/dog) and controls the turret LED."""

    def __init__(
        self,
        stream_url: str = DEFAULT_STREAM_URL,
        turret_url: str = DEFAULT_TURRET_URL,
        auto_aim: bool = True,
    ) -> None:
        self.reader = MJPEGStreamReader(stream_url=stream_url, rotation=90)
        self.hardware = TurretHardwareClient(base_url=turret_url)
        self.detector = PersonDetector()

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="AutoAimEngine")

        # Configurable parameters
        self.auto_aim: bool = auto_aim
        self.detect_person: bool = True
        self.detect_dog: bool = True
        self.auto_fire: bool = True
        self.aim_mode: str = "center"  # center | upper | head
        self.led_mode: str = "on_lock"  # on_lock | on_detect | off
        self.kp: float = 1.0
        self.kd: float = 0.08
        self.deadzone_px: float = 12.0
        self.conf_threshold: float = 0.28
        self.invert_x: bool = False
        self.invert_y: bool = False
        self.max_step_deg: float = 45.0
        self.min_servo_interval: float = 0.04
        self.camera_latency: float = 0.17
        self.hold_timeout: float = 0.45

        # Internal servo & tracking state
        self._curr_x: float = 90.0
        self._curr_y: float = 90.0
        self._prev_deg_err_x: float = 0.0
        self._prev_deg_err_y: float = 0.0
        self._last_servo_step_time: float = 0.0
        self._wait_frame_after: float = 0.0
        self._last_detection_time: float = 0.0
        self._smoothed_target: Optional[Tuple[float, float]] = None

        # Telemetry & output frames
        self._person_detected: bool = False
        self._target_type: Optional[str] = None
        self._led_on: bool = False
        self._target_box: Optional[List[int]] = None
        self._target_center: Optional[List[int]] = None
        self._error_px: List[int] = [0, 0]
        self._fps: float = 0.0
        self._inference_ms: float = 0.0
        self._detections_count: int = 0

        self._annotated_frame: Optional[np.ndarray] = None
        self._annotated_jpeg: Optional[bytes] = None
        self._annotated_id: int = 0

    def start(self) -> None:
        self.hardware.start()
        self.hardware.center()
        self.hardware.set_led(False)
        self.reader.start()
        if not self._thread.is_alive():
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._loop, daemon=True, name="AutoAimEngine")
            self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self.reader.stop()
        self.hardware.shutdown()

    def update_config(self, cfg: Dict[str, object]) -> None:
        with self._lock:
            if "auto_aim" in cfg:
                new_auto = bool(cfg["auto_aim"])
                if new_auto != self.auto_aim:
                    self.auto_aim = new_auto
                    self._prev_deg_err_x = 0.0
                    self._prev_deg_err_y = 0.0
                    if not new_auto:
                        self._led_on = False
                        self.hardware.set_led(False)
            if "detect_person" in cfg:
                new_dp = bool(cfg["detect_person"])
                if new_dp != self.detect_person:
                    self.detect_person = new_dp
                    self._smoothed_target = None
            if "detect_dog" in cfg:
                new_dd = bool(cfg["detect_dog"])
                if new_dd != self.detect_dog:
                    self.detect_dog = new_dd
                    self._smoothed_target = None
            if "auto_fire" in cfg:
                self.auto_fire = bool(cfg["auto_fire"])
                if not self.auto_fire:
                    self._led_on = False
                    self.hardware.set_led(False)
            if "aim_mode" in cfg and cfg["aim_mode"] in ("center", "upper", "head"):
                self.aim_mode = str(cfg["aim_mode"])
            if "led_mode" in cfg and cfg["led_mode"] in ("on_lock", "on_detect", "off"):
                self.led_mode = str(cfg["led_mode"])
                if self.led_mode == "off":
                    self._led_on = False
                    self.hardware.set_led(False)
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

    def center_turret(self) -> None:
        with self._lock:
            self._curr_x = 90.0
            self._curr_y = 90.0
            self._prev_deg_err_x = 0.0
            self._prev_deg_err_y = 0.0
            self._smoothed_target = None
            self._wait_frame_after = time.time() + 0.12
        self.hardware.center()

    def set_manual_angles(self, x: float, y: float) -> None:
        cx = max(0.0, min(180.0, float(x)))
        cy = max(0.0, min(180.0, float(y)))
        with self._lock:
            self._curr_x = cx
            self._curr_y = cy
            self._prev_deg_err_x = 0.0
            self._prev_deg_err_y = 0.0
            self._wait_frame_after = time.time() + 0.12
        self.hardware.set_angles(cx, cy)

    def set_manual_led(self, state: bool) -> None:
        with self._lock:
            self._led_on = bool(state)
        self.hardware.set_led(bool(state))

    def get_status(self) -> Dict[str, object]:
        with self._lock:
            return {
                "auto_aim": self.auto_aim,
                "detect_person": self.detect_person,
                "detect_dog": self.detect_dog,
                "auto_fire": self.auto_fire,
                "aim_mode": self.aim_mode,
                "led_mode": self.led_mode,
                "kp": round(self.kp, 3),
                "kd": round(self.kd, 3),
                "deadzone_px": round(self.deadzone_px, 1),
                "conf_threshold": round(self.conf_threshold, 2),
                "invert_x": self.invert_x,
                "invert_y": self.invert_y,
                "rotation": self.reader.get_rotation(),
                "person_detected": self._person_detected,
                "target_type": self._target_type,
                "detections_count": self._detections_count,
                "led_on": self._led_on,
                "servo_x": int(round(self._curr_x)),
                "servo_y": int(round(self._curr_y)),
                "target_box": self._target_box,
                "target_center": self._target_center,
                "error_px": self._error_px,
                "fps": round(self._fps, 1),
                "inference_ms": round(self._inference_ms, 1),
                "device": self.detector.device_info,
                "stream_connected": self.reader.is_connected(),
            }

    def get_annotated_jpeg(self) -> Tuple[Optional[bytes], int]:
        with self._lock:
            return self._annotated_jpeg, self._annotated_id

    def get_annotated_frame(self) -> Tuple[Optional[np.ndarray], int]:
        with self._lock:
            if self._annotated_frame is None:
                return None, self._annotated_id
            return self._annotated_frame.copy(), self._annotated_id

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
        fps_ema = 0.0
        prev_loop_time = time.time()

        while not self._stop_event.is_set():
            frame, frame_id, frame_ts = self.reader.get_latest_frame()
            if frame is None or frame_id == last_frame_id:
                time.sleep(0.01)
                continue
            last_frame_id = frame_id

            now = time.time()
            dt_loop = max(1e-3, now - prev_loop_time)
            prev_loop_time = now
            inst_fps = 1.0 / dt_loop
            fps_ema = inst_fps if fps_ema == 0.0 else (0.85 * fps_ema + 0.15 * inst_fps)

            with self._lock:
                auto_aim = self.auto_aim
                detect_person = self.detect_person
                detect_dog = self.detect_dog
                auto_fire = self.auto_fire
                aim_mode = self.aim_mode
                led_mode = self.led_mode
                kp = self.kp
                deadzone_px = self.deadzone_px
                conf_threshold = self.conf_threshold
                invert_x = self.invert_x
                invert_y = self.invert_y
                prev_smooth = self._smoothed_target

            t_inf_start = time.time()
            detections = self.detector.detect(
                frame,
                conf_threshold=conf_threshold,
                detect_person=detect_person,
                detect_dog=detect_dog,
            )
            inference_ms = (time.time() - t_inf_start) * 1000.0

            h, w = frame.shape[:2]
            cx, cy = w * 0.5, h * 0.5
            target = self.detector.select_target(detections, (h, w), prev_smooth)

            target_detected = target is not None
            err_x = 0.0
            err_y = 0.0
            aim_pt: Optional[Tuple[float, float]] = None

            with self._lock:
                if target is not None:
                    tx, ty = self.detector.compute_aim_point(target, aim_mode=aim_mode)
                    self._smoothed_target = (tx, ty)
                    self._last_detection_time = now
                    aim_pt = (tx, ty)

                    err_x = tx - cx
                    err_y = ty - cy

                    if auto_aim:
                        if (now - self._last_servo_step_time) >= self.min_servo_interval:
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
                                self.hardware.set_angles(self._curr_x, self._curr_y)

                        if not auto_fire or led_mode == "off":
                            self._led_on = False
                            self.hardware.set_led(False)
                        elif led_mode == "on_lock":
                            box_w = target.x2 - target.x1
                            box_h = target.y2 - target.y1
                            in_box = (
                                (target.x1 - deadzone_px <= cx <= target.x2 + deadzone_px)
                                and (target.y1 - deadzone_px <= cy <= target.y2 + deadzone_px)
                            )
                            lock_radius = max(deadzone_px * 2.2, min(box_w, box_h) * 0.45)
                            on_target = in_box or (math.hypot(err_x, err_y) <= lock_radius)
                            self._led_on = on_target
                            self.hardware.set_led(on_target)
                        else:
                            self._led_on = True
                            self.hardware.set_led(True)
                else:
                    if (now - self._last_detection_time) > self.hold_timeout:
                        self._smoothed_target = None
                        self._prev_deg_err_x = 0.0
                        self._prev_deg_err_y = 0.0
                        if auto_aim:
                            self._led_on = False
                            self.hardware.set_led(False)
                    elif auto_aim and (not auto_fire or led_mode == "off"):
                        self._led_on = False
                        self.hardware.set_led(False)

                curr_x_deg = int(round(self._curr_x))
                curr_y_deg = int(round(self._curr_y))
                led_state = self._led_on
                fire_enabled = auto_fire and led_mode != "off"

                self._person_detected = target_detected
                self._target_type = target.source if target else None
                self._detections_count = len(detections)
                self._target_box = (
                    [int(target.x1), int(target.y1), int(target.x2), int(target.y2)] if target else None
                )
                self._target_center = [int(round(aim_pt[0])), int(round(aim_pt[1]))] if aim_pt else None
                self._error_px = [int(round(err_x)), int(round(err_y))]
                self._fps = fps_ema
                self._inference_ms = inference_ms

            annotated = self._draw_overlay(
                frame=frame,
                detections=detections,
                target=target,
                aim_pt=aim_pt,
                deadzone_px=deadzone_px,
                auto_aim=auto_aim,
                fire_enabled=fire_enabled,
                led_on=led_state,
                servo_x=curr_x_deg,
                servo_y=curr_y_deg,
                err_x=err_x,
                err_y=err_y,
                fps=fps_ema,
            )

            ok, encoded = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
            if ok:
                jpg_bytes = encoded.tobytes()
                with self._lock:
                    self._annotated_frame = annotated
                    self._annotated_jpeg = jpg_bytes
                    self._annotated_id += 1

            if DEFAULT_MAX_FPS > 0:
                dt_compute = time.time() - now
                min_period = 1.0 / DEFAULT_MAX_FPS
                if dt_compute < min_period:
                    time.sleep(min_period - dt_compute)

    @staticmethod
    def _draw_overlay(
        frame: np.ndarray,
        detections: List[Detection],
        target: Optional[Detection],
        aim_pt: Optional[Tuple[float, float]],
        deadzone_px: float,
        auto_aim: bool,
        fire_enabled: bool,
        led_on: bool,
        servo_x: int,
        servo_y: int,
        err_x: float,
        err_y: float,
        fps: float,
    ) -> np.ndarray:
        vis = frame.copy()
        h, w = vis.shape[:2]
        cx, cy = int(round(w * 0.5)), int(round(h * 0.5))

        # Center reticle & deadzone ring
        reticle_color = (0, 220, 255) if target is not None else (200, 200, 200)
        cv2.line(vis, (cx - 22, cy), (cx + 22, cy), reticle_color, 1, cv2.LINE_AA)
        cv2.line(vis, (cx, cy - 22), (cx, cy + 22), reticle_color, 1, cv2.LINE_AA)
        cv2.circle(vis, (cx, cy), max(4, int(round(deadzone_px))), reticle_color, 1, cv2.LINE_AA)

        # Draw detections
        for det in detections:
            is_active = target is not None and det is target
            color = (0, 0, 255) if (is_active and led_on) else ((0, 230, 80) if is_active else (0, 215, 255))
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
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

        # Aim vector & lock dot
        if aim_pt is not None:
            tx, ty = int(round(aim_pt[0])), int(round(aim_pt[1]))
            dot_color = (0, 0, 255) if led_on else (0, 255, 0)
            cv2.line(vis, (cx, cy), (tx, ty), (255, 180, 0), 2, cv2.LINE_AA)
            cv2.circle(vis, (tx, ty), 6, dot_color, -1, cv2.LINE_AA)
            cv2.circle(vis, (tx, ty), 10, (255, 255, 255), 1, cv2.LINE_AA)

        # Top translucent HUD bar
        overlay = vis.copy()
        cv2.rectangle(overlay, (0, 0), (w, 56), (18, 18, 22), -1)
        cv2.addWeighted(overlay, 0.72, vis, 0.28, 0, vis)

        mode_str = "AUTO-AIM" if auto_aim else "MANUAL"
        if target is not None:
            tgt_label = SOURCE_LABELS.get(target.source, "ALVO")
            tgt_str = f"ALVO: {tgt_label}"
        else:
            tgt_str = "ALVO: BUSCANDO"

        if not fire_enabled:
            led_str = "TIRO: DESATIVADO"
        else:
            led_str = "LED: ACESO (ATIRANDO)" if led_on else "LED: DESLIGADO"

        line1 = f"MODO: {mode_str}  |  {tgt_str}  |  {led_str}"
        line2 = f"X: {servo_x} deg  Y: {servo_y} deg  |  ERRO: ({int(round(err_x)):+d}, {int(round(err_y)):+d}) px  |  FPS: {fps:.1f}"

        led_color = (0, 220, 255) if led_on else ((160, 160, 175) if not fire_enabled else (210, 210, 220))
        cv2.putText(vis, line1, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, led_color, 1, cv2.LINE_AA)
        cv2.putText(vis, line2, (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (235, 235, 240), 1, cv2.LINE_AA)

        return vis
