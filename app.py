import argparse
import atexit
import signal
import sys
import threading
import time
from typing import Optional

import cv2
from flask import Flask, Response, jsonify, render_template, request
from werkzeug.serving import WSGIRequestHandler

from turret_controller import (
    DEFAULT_STREAM_URL,
    DEFAULT_TURRET_URL,
    AutoAimEngine,
)

# Enable HTTP/1.1 keep-alive so single-flight GET /servo requests reuse TCP connections
WSGIRequestHandler.protocol_version = "HTTP/1.1"


class QuietStatusRequestHandler(WSGIRequestHandler):
    """HTTP/1.1 WSGI request handler with TCP_NODELAY that suppresses routine 200/204 access logs."""

    protocol_version = "HTTP/1.1"
    disable_nagle_algorithm = True
    _QUIET_PATHS = ("/api/status", "/servo")

    def log_request(self, code="-", size="-"):
        path = (getattr(self, "path", "") or "").split("?", 1)[0]
        try:
            status_code = int(code)
        except (TypeError, ValueError):
            status_code = None
        if status_code in (200, 204) and path in self._QUIET_PATHS:
            return
        super().log_request(code, size)


app = Flask(__name__)
engine: Optional[AutoAimEngine] = None


@app.after_request
def add_cors_headers(response: Response) -> Response:
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/servo", methods=["GET", "OPTIONS"])
def servo_control():
    """Real-time actuator control route mirroring ESP32-CAM GET /servo on port 80.

    Accepted optional & combinable query parameters (<= 95 chars):
      x=0..180     -> Servo 1 angle (GPIO 14, X-axis, immediate PWM, 500 ms Auto-Detach)
      y=0..180     -> Servo 2 angle (GPIO 13, Y-axis, immediate PWM, 500 ms Auto-Detach)
      laser=0|1    -> Green Laser (GPIO 15)
      led=0|1      -> Flash LED (GPIO 4)
    """
    if request.method == "OPTIONS":
        return Response("", status=204)

    if engine is None:
        return Response("ENGINE NOT READY", status=503, mimetype="text/plain")

    def _parse_float(key: str) -> Optional[float]:
        raw = request.args.get(key)
        if raw is None or raw == "":
            return None
        try:
            return float(raw)
        except ValueError:
            return None

    def _parse_bool_int(key: str) -> Optional[bool]:
        raw = request.args.get(key)
        if raw is None or raw == "":
            return None
        try:
            return int(raw) != 0
        except ValueError:
            return raw.lower() in ("true", "on")

    x = _parse_float("x")
    y = _parse_float("y")
    led = _parse_bool_int("led")
    laser = _parse_bool_int("laser")

    engine.apply_servo_command(
        x=x,
        y=y,
        led=led,
        laser=laser,
        disable_auto_aim=(x is not None or y is not None),
    )
    resp = Response("OK", status=200, mimetype="text/plain")
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/cam", methods=["GET", "OPTIONS"])
def cam_control():
    """Camera settings route mirroring ESP32-CAM GET /cam on port 80 (OV3660).

    Accepted optional & combinable query parameters (<= 95 chars):
      res=0..3     -> Resolution (0=QVGA, 1=CIF default ~25-28 FPS, 2=VGA, 3=SVGA)
      qual=10..40  -> JPEG quality (lower = sharper, default 16)
      bri=-3..3    -> Brightness (default 1)
      ae=-5..5     -> Auto-exposure level (default 1)
      gain=0..5    -> OV3660 gainceiling (0=2x..5=64x, default 4 = 32x)
      night=0|1    -> Night mode / aec2 (default 0 for max indoor FPS)
    """
    if request.method == "OPTIONS":
        return Response("", status=204)

    if engine is None:
        return Response("ENGINE NOT READY", status=503, mimetype="text/plain")

    def _parse_int(key: str) -> Optional[int]:
        raw = request.args.get(key)
        if raw is None or raw == "":
            return None
        try:
            return int(raw)
        except ValueError:
            return None

    def _parse_bool_int(key: str) -> Optional[bool]:
        raw = request.args.get(key)
        if raw is None or raw == "":
            return None
        try:
            return int(raw) != 0
        except ValueError:
            return raw.lower() in ("true", "on")

    res = _parse_int("res")
    qual = _parse_int("qual")
    bri = _parse_int("bri")
    ae = _parse_int("ae")
    gain = _parse_int("gain")
    night = _parse_bool_int("night")

    engine.apply_cam_command(
        res=res,
        qual=qual,
        bri=bri,
        ae=ae,
        gain=gain,
        night=night,
        force=True,
    )
    resp = Response("OK", status=200, mimetype="text/plain")
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/video_feed")
def video_feed():
    def generate():
        last_id = -1
        registered_engine: Optional[AutoAimEngine] = None
        try:
            while True:
                cur_engine = engine
                if cur_engine is None:
                    time.sleep(0.1)
                    continue
                if registered_engine is not cur_engine:
                    if registered_engine is not None:
                        registered_engine.unregister_video_client()
                    cur_engine.register_video_client()
                    registered_engine = cur_engine
                if cur_engine.is_stopped():
                    break
                jpg, cur_id = cur_engine.wait_for_annotated_jpeg(last_id, timeout=1.0)
                if jpg is None or cur_id == last_id:
                    continue
                last_id = cur_id
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(jpg)}\r\n\r\n".encode("ascii")
                    + jpg
                    + b"\r\n"
                )
        finally:
            if registered_engine is not None:
                registered_engine.unregister_video_client()

    resp = Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["X-Accel-Buffering"] = "no"
    return resp


@app.route("/api/status", methods=["GET"])
def api_status():
    if engine is None:
        return jsonify({"error": "engine not initialized"}), 503
    resp = jsonify(engine.get_status())
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/api/config", methods=["POST", "OPTIONS"])
def api_config():
    if request.method == "OPTIONS":
        return Response("", status=204)
    if engine is None:
        return jsonify({"error": "engine not initialized"}), 503
    data = request.get_json(silent=True) or {}
    engine.update_config(data)
    return jsonify({"ok": True, "status": engine.get_status()})


@app.route("/api/action", methods=["POST", "OPTIONS"])
def api_action():
    if request.method == "OPTIONS":
        return Response("", status=204)
    if engine is None:
        return jsonify({"error": "engine not initialized"}), 503
    data = request.get_json(silent=True) or {}
    action = data.get("action")
    if action == "center":
        engine.center_turret()
    elif action in ("fire_on", "led_on"):
        engine.set_manual_led(True)
    elif action in ("fire_off", "led_off"):
        engine.set_manual_led(False)
    elif action == "laser_on":
        engine.set_manual_laser(True)
    elif action == "laser_off":
        engine.set_manual_laser(False)
    elif action == "set_angles":
        x = float(data.get("x", 90))
        y = float(data.get("y", 90))
        engine.set_manual_angles(x, y)
    elif action == "set_resolution":
        res_code = int(data.get("res", 1))
        engine.set_resolution(res_code)
    elif action == "set_cam":
        engine.apply_cam_command(
            res=int(data["res"]) if data.get("res") is not None else None,
            qual=int(data["qual"]) if data.get("qual") is not None else None,
            bri=int(data["bri"]) if data.get("bri") is not None else None,
            ae=int(data["ae"]) if data.get("ae") is not None else None,
            gain=int(data["gain"]) if data.get("gain") is not None else None,
            night=bool(data["night"]) if data.get("night") is not None else None,
            force=True,
        )
    elif action == "reset_cam":
        engine.reset_cam_defaults()
    elif action in ("failsafe", "all_off"):
        engine.failsafe_off()
    elif action == "discord_test":
        ok, msg = engine.send_discord_test()
        return jsonify({"ok": ok, "message": msg, "status": engine.get_status()}), (200 if ok else 400)
    else:
        return jsonify({"ok": False, "error": f"Unknown action: {action}"}), 400
    return jsonify({"ok": True, "status": engine.get_status()})


def _cleanup() -> None:
    global engine
    if engine is not None:
        engine.stop()
        engine = None


def _signal_handler(signum, frame) -> None:
    _cleanup()
    sys.exit(0)


def main() -> None:
    parser = argparse.ArgumentParser(description="AutoAIM Mini Turret CV Controller")
    parser.add_argument("--turret-url", default=DEFAULT_TURRET_URL, help="Base URL da torreta ESP32-CAM")
    parser.add_argument("--stream-url", default=DEFAULT_STREAM_URL, help="URL do stream MJPEG da torreta")
    parser.add_argument("--host", default="0.0.0.0", help="Host do servidor web Flask")
    parser.add_argument("--port", type=int, default=5000, help="Porta do servidor web Flask")
    parser.add_argument("--no-auto-aim", action="store_true", help="Iniciar em modo manual")
    parser.add_argument("--show-window", action="store_true", help="Exibir janela nativa OpenCV (cv2.imshow)")
    args = parser.parse_args()

    global engine
    engine = AutoAimEngine(
        stream_url=args.stream_url,
        turret_url=args.turret_url,
        auto_aim=not args.no_auto_aim,
    )
    atexit.register(_cleanup)
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, _signal_handler)

    engine.start()
    print(
        f"[AutoAIM] Stream: {args.stream_url} | Controle HTTP: {args.turret_url}/servo | "
        f"Câmera: {args.turret_url}/cam"
    )
    print(f"[AutoAIM] Dashboard Web disponível em http://localhost:{args.port}/")

    if args.show_window:
        engine.set_window_enabled(True)
        flask_thread = threading.Thread(
            target=lambda: app.run(
                host=args.host,
                port=args.port,
                threaded=True,
                use_reloader=False,
                request_handler=QuietStatusRequestHandler,
            ),
            daemon=True,
        )
        flask_thread.start()
        try:
            last_id = -1
            while True:
                frame, fid = engine.get_annotated_frame()
                if frame is not None and fid != last_id:
                    last_id = fid
                    cv2.imshow("AutoAIM Turret Controller", frame)
                if cv2.waitKey(15) & 0xFF == ord("q"):
                    break
        finally:
            cv2.destroyAllWindows()
            _cleanup()
    else:
        app.run(
            host=args.host,
            port=args.port,
            threaded=True,
            use_reloader=False,
            request_handler=QuietStatusRequestHandler,
        )


if __name__ == "__main__":
    main()
