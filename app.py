import argparse
import atexit
import signal
import sys
import threading
import time
from typing import Optional

import cv2
from flask import Flask, Response, jsonify, render_template_string, request

from turret_controller import (
    DEFAULT_STREAM_URL,
    DEFAULT_TURRET_URL,
    AutoAimEngine,
)

app = Flask(__name__)
engine: Optional[AutoAimEngine] = None

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AutoAIM - Controle por Visão Computacional</title>
  <style>
    * { box-sizing: border-box; }
    body {
      margin: 0;
      padding: 16px;
      font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
      background: #0f1115;
      color: #f4f4f5;
    }
    .container {
      max-width: 1120px;
      margin: 0 auto;
    }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 12px;
      margin-bottom: 16px;
      padding: 14px 18px;
      background: #181b22;
      border: 1px solid #272b36;
      border-radius: 10px;
    }
    h1 { margin: 0; font-size: 20px; letter-spacing: -0.01em; }
    .subtitle { font-size: 13px; color: #9ca3af; margin-top: 2px; }
    .status-pills { display: flex; gap: 8px; flex-wrap: wrap; }
    .pill {
      padding: 6px 12px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
      background: #272b36;
      color: #d1d5db;
    }
    .pill.ok { background: #14532d; color: #86efac; }
    .pill.warn { background: #7f1d1d; color: #fca5a5; }
    .pill.fire { background: #eab308; color: #09090b; }
    .layout {
      display: grid;
      grid-template-columns: minmax(300px, 1.3fr) minmax(300px, 1fr);
      gap: 16px;
      align-items: start;
    }
    @media (max-width: 860px) {
      .layout { grid-template-columns: 1fr; }
    }
    .card {
      background: #181b22;
      border: 1px solid #272b36;
      border-radius: 10px;
      padding: 16px;
    }
    .video-wrap {
      display: flex;
      justify-content: center;
      align-items: center;
      background: #090a0d;
      border-radius: 8px;
      overflow: hidden;
      min-height: 420px;
      border: 1px solid #272b36;
    }
    #videoFeed {
      max-width: 100%;
      max-height: 76vh;
      display: block;
      object-fit: contain;
    }
    .telemetry-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 10px;
      margin-bottom: 16px;
    }
    .metric {
      background: #111318;
      border: 1px solid #272b36;
      border-radius: 8px;
      padding: 10px;
      text-align: center;
    }
    .metric .label { font-size: 11px; color: #9ca3af; text-transform: uppercase; }
    .metric .val { font-size: 18px; font-weight: 700; font-family: monospace; margin-top: 4px; color: #60a5fa; }
    .control-group {
      margin-bottom: 14px;
      padding-bottom: 14px;
      border-bottom: 1px solid #272b36;
    }
    .control-group:last-child { border-bottom: none; margin-bottom: 0; padding-bottom: 0; }
    .row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      margin-bottom: 10px;
    }
    .row:last-child { margin-bottom: 0; }
    label { font-size: 14px; color: #e5e7eb; }
    select, input[type="range"] { accent-color: #3b82f6; }
    select {
      background: #111318;
      color: #f4f4f5;
      border: 1px solid #374151;
      border-radius: 6px;
      padding: 6px 10px;
      font-size: 13px;
    }
    input[type="range"] { width: 150px; }
    .slider-val {
      font-family: monospace;
      font-size: 13px;
      color: #93c5fd;
      min-width: 44px;
      text-align: right;
      display: inline-block;
    }
    .btn-row { display: flex; gap: 10px; flex-wrap: wrap; }
    button {
      flex: 1;
      padding: 10px 14px;
      border: none;
      border-radius: 8px;
      font-size: 14px;
      font-weight: 700;
      cursor: pointer;
      color: #fff;
      background: #2563eb;
      transition: background 0.15s;
    }
    button:hover { background: #1d4ed8; }
    button.fire-btn { background: #ca8a04; color: #09090b; }
    button.fire-btn:hover { background: #eab308; }
    button.secondary { background: #374151; }
    button.secondary:hover { background: #4b5563; }
    .switch-label {
      display: flex;
      align-items: center;
      gap: 8px;
      cursor: pointer;
      font-weight: 600;
    }
    input[type="checkbox"] { width: 17px; height: 17px; cursor: pointer; accent-color: #3b82f6; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div>
        <h1>AutoAIM — Mini Turret CV Controller</h1>
        <div class="subtitle">Detecção de Pessoas e Cachorros (YOLOv8n ONNX + YuNet) &amp; Controle Pan-Tilt</div>
      </div>
      <div class="status-pills">
        <span id="pillStream" class="pill warn">STREAM: CONECTANDO...</span>
        <span id="pillDevice" class="pill">HW: ...</span>
        <span id="pillTarget" class="pill">ALVO: BUSCANDO</span>
        <span id="pillLed" class="pill">LED: DESLIGADO</span>
      </div>
    </header>

    <div class="layout">
      <div class="card">
        <div class="video-wrap">
          <img id="videoFeed" src="/video_feed" alt="Stream CV da Torreta">
        </div>
      </div>

      <div class="card">
        <div class="telemetry-grid">
          <div class="metric">
            <div class="label">Servo X / Y</div>
            <div class="val" id="mAngles">90° / 90°</div>
          </div>
          <div class="metric">
            <div class="label">Erro (px)</div>
            <div class="val" id="mError">0, 0</div>
          </div>
          <div class="metric">
            <div class="label">FPS / Inf.</div>
            <div class="val" id="mPerf">0 / 0ms</div>
          </div>
        </div>

        <div class="control-group">
          <div class="row">
            <label class="switch-label">
              <input type="checkbox" id="autoAim" checked onchange="sendConfig()">
              Auto-Aim (Visão Computacional Ativa)
            </label>
          </div>
          <div class="row">
            <label class="switch-label">
              <input type="checkbox" id="detectPerson" checked onchange="sendConfig()">
              Detectar Pessoas
            </label>
            <label class="switch-label">
              <input type="checkbox" id="detectDog" checked onchange="sendConfig()">
              Detectar Cachorros
            </label>
          </div>
          <div class="row">
            <label for="aimMode">Ponto de Mira:</label>
            <select id="aimMode" onchange="sendConfig()">
              <option value="center" selected>Centro da Pessoa (50%)</option>
              <option value="upper">Peito / Tronco (35%)</option>
              <option value="head">Cabeça (18%)</option>
            </select>
          </div>
        </div>

        <div class="control-group">
          <div class="row">
            <label class="switch-label">
              <input type="checkbox" id="autoFire" checked onchange="onAutoFireChange()">
              Acender LED ao Achar Alvo (Tiro Automático)
            </label>
          </div>
          <div class="row">
            <label for="ledMode">Modo de Tiro (LED):</label>
            <select id="ledMode" onchange="onLedModeChange()">
              <option value="on_lock" selected>Eficiente (Acender Apenas com Alvo na Mira)</option>
              <option value="on_detect">Imediato (Acender Assim que Detectar Alvo)</option>
              <option value="off">Não Acender (Apenas Mirar / Rastrear)</option>
            </select>
          </div>
        </div>

        <div class="control-group">
          <div class="row">
            <label for="kp">Velocidade / Ganho (Kp):</label>
            <div>
              <input type="range" id="kp" min="0.20" max="1.50" step="0.05" value="1.00" oninput="updateLabels()" onchange="sendConfig()">
              <span class="slider-val" id="kpVal">1.00</span>
            </div>
          </div>
          <div class="row">
            <label for="deadzone">Zona Morta (Deadzone px):</label>
            <div>
              <input type="range" id="deadzone" min="4" max="50" step="2" value="12" oninput="updateLabels()" onchange="sendConfig()">
              <span class="slider-val" id="deadzoneVal">12 px</span>
            </div>
          </div>
          <div class="row">
            <label for="conf">Confiança Mínima:</label>
            <div>
              <input type="range" id="conf" min="0.18" max="0.80" step="0.01" value="0.28" oninput="updateLabels()" onchange="sendConfig()">
              <span class="slider-val" id="confVal">0.28</span>
            </div>
          </div>
        </div>

        <div class="control-group">
          <div class="row">
            <label class="switch-label"><input type="checkbox" id="invX" onchange="sendConfig()"> Inverter Eixo X</label>
            <label class="switch-label"><input type="checkbox" id="invY" onchange="sendConfig()"> Inverter Eixo Y</label>
          </div>
          <div class="row">
            <label for="rotation">Rotação da Câmera:</label>
            <select id="rotation" onchange="sendConfig()">
              <option value="0">0° (Normal)</option>
              <option value="90" selected>90° Direita</option>
              <option value="180">180° (Invertido)</option>
              <option value="270">270° Esquerda</option>
            </select>
          </div>
        </div>

        <div class="control-group">
          <div class="btn-row">
            <button class="secondary" onclick="doAction('center')">Centralizar (90°, 90°)</button>
            <button class="fire-btn"
              onpointerdown="doAction('fire_on')"
              onpointerup="doAction('fire_off')"
              onpointerleave="doAction('fire_off')">Testar LED (Atirar)</button>
          </div>
        </div>
      </div>
    </div>
  </div>

  <script>
    let initialized = false;

    function updateLabels() {
      document.getElementById('kpVal').innerText = parseFloat(document.getElementById('kp').value).toFixed(2);
      document.getElementById('deadzoneVal').innerText = document.getElementById('deadzone').value + ' px';
      document.getElementById('confVal').innerText = parseFloat(document.getElementById('conf').value).toFixed(2);
    }

    function onAutoFireChange() {
      const af = document.getElementById('autoFire').checked;
      const lm = document.getElementById('ledMode');
      if (af && lm.value === 'off') {
        lm.value = 'on_lock';
      }
      sendConfig();
    }

    function onLedModeChange() {
      const lm = document.getElementById('ledMode').value;
      document.getElementById('autoFire').checked = (lm !== 'off');
      sendConfig();
    }

    async function sendConfig() {
      updateLabels();
      const payload = {
        auto_aim: document.getElementById('autoAim').checked,
        detect_person: document.getElementById('detectPerson').checked,
        detect_dog: document.getElementById('detectDog').checked,
        auto_fire: document.getElementById('autoFire').checked,
        aim_mode: document.getElementById('aimMode').value,
        led_mode: document.getElementById('ledMode').value,
        kp: parseFloat(document.getElementById('kp').value),
        deadzone_px: parseFloat(document.getElementById('deadzone').value),
        conf_threshold: parseFloat(document.getElementById('conf').value),
        invert_x: document.getElementById('invX').checked,
        invert_y: document.getElementById('invY').checked,
        rotation: parseInt(document.getElementById('rotation').value, 10)
      };
      await fetch('/api/config', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload)
      });
    }

    async function doAction(action) {
      await fetch('/api/action', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ action })
      });
    }

    const TARGET_NAMES = { person: 'PESSOA', dog: 'CACHORRO', face: 'ROSTO' };

    async function pollStatus() {
      try {
        const res = await fetch('/api/status');
        const s = await res.json();

        const pStream = document.getElementById('pillStream');
        pStream.innerText = s.stream_connected ? 'STREAM: ONLINE' : 'STREAM: RECONECTANDO...';
        pStream.className = 'pill ' + (s.stream_connected ? 'ok' : 'warn');

        const pDevice = document.getElementById('pillDevice');
        if (s.device) {
          pDevice.innerText = `HW: ${s.device}`;
          pDevice.className = 'pill ' + (s.device.includes('GPU') ? 'ok' : '');
        }

        const pTarget = document.getElementById('pillTarget');
        const tgtName = TARGET_NAMES[s.target_type] || 'ALVO';
        pTarget.innerText = s.person_detected ? `ALVO: ${tgtName} (${s.detections_count})` : 'ALVO: BUSCANDO';
        pTarget.className = 'pill ' + (s.person_detected ? 'ok' : '');

        const pLed = document.getElementById('pillLed');
        if (s.led_on) {
          pLed.innerText = 'LED: ACESO (ATIRANDO)';
          pLed.className = 'pill fire';
        } else if (!s.auto_fire || s.led_mode === 'off') {
          pLed.innerText = 'TIRO: DESATIVADO';
          pLed.className = 'pill';
        } else {
          pLed.innerText = 'LED: DESLIGADO';
          pLed.className = 'pill';
        }

        document.getElementById('mAngles').innerText = `${s.servo_x}° / ${s.servo_y}°`;
        document.getElementById('mError').innerText = `${s.error_px[0]}, ${s.error_px[1]}`;
        document.getElementById('mPerf').innerText = `${s.fps} / ${Math.round(s.inference_ms)}ms`;

        if (!initialized) {
          document.getElementById('autoAim').checked = s.auto_aim;
          document.getElementById('detectPerson').checked = s.detect_person;
          document.getElementById('detectDog').checked = s.detect_dog;
          document.getElementById('autoFire').checked = s.auto_fire;
          document.getElementById('aimMode').value = s.aim_mode;
          document.getElementById('ledMode').value = s.led_mode;
          document.getElementById('kp').value = s.kp;
          document.getElementById('deadzone').value = s.deadzone_px;
          document.getElementById('conf').value = s.conf_threshold;
          document.getElementById('invX').checked = s.invert_x;
          document.getElementById('invY').checked = s.invert_y;
          document.getElementById('rotation').value = String(s.rotation);
          updateLabels();
          initialized = true;
        }
      } catch (e) {
        // ignore transient fetch errors
      }
    }

    setInterval(pollStatus, 200);
    pollStatus();
  </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)


@app.route("/video_feed")
def video_feed():
    def generate():
        last_id = -1
        while True:
            if engine is None:
                time.sleep(0.1)
                continue
            jpg, cur_id = engine.get_annotated_jpeg()
            if jpg is None or cur_id == last_id:
                time.sleep(0.02)
                continue
            last_id = cur_id
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            )

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/status", methods=["GET"])
def api_status():
    if engine is None:
        return jsonify({"error": "engine not initialized"}), 503
    return jsonify(engine.get_status())


@app.route("/api/config", methods=["POST"])
def api_config():
    if engine is None:
        return jsonify({"error": "engine not initialized"}), 503
    data = request.get_json(silent=True) or {}
    engine.update_config(data)
    return jsonify({"ok": True, "status": engine.get_status()})


@app.route("/api/action", methods=["POST"])
def api_action():
    if engine is None:
        return jsonify({"error": "engine not initialized"}), 503
    data = request.get_json(silent=True) or {}
    action = data.get("action")
    if action == "center":
        engine.center_turret()
    elif action == "fire_on":
        engine.set_manual_led(True)
    elif action == "fire_off":
        engine.set_manual_led(False)
    elif action == "set_angles":
        x = float(data.get("x", 90))
        y = float(data.get("y", 90))
        engine.set_manual_angles(x, y)
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

    engine.start()
    print(f"[AutoAIM] Conectando ao stream {args.stream_url} e controlando {args.turret_url}")
    print(f"[AutoAIM] Dashboard Web disponível em http://localhost:{args.port}/")

    if args.show_window:
        flask_thread = threading.Thread(
            target=lambda: app.run(host=args.host, port=args.port, threaded=True, use_reloader=False),
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
        app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
