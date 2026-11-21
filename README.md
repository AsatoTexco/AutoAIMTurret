# AutoAIM — Controlador de Mini Torreta por Visão Computacional & Controle FPS

Sistema completo para controle de mini torreta ESP32-CAM (OV3660) com detecção de **pessoas** e **cachorros** em tempo real (**YOLOv8n ONNX/ncnn Vulkan** + fallback facial **YuNet**), controle HTTP separado na **Porta 80** (**`GET /servo`** em tempo real + **`GET /cam`** para ajustes de câmera, com `keep-alive` e `TCP_NODELAY`), streaming MJPEG chunked na **Porta 81** (`Content-Length` + `SO_KEEPALIVE`, até SVGA ~28 FPS), acionamento de **Laser Verde (IO15)** e **LED Flash (IO4)** com *fail-safe* automático e **Modo FPS / Modo Jogo Mobile**.

## Protocolo de Controle — Porta 80 (`GET /servo` + `GET /cam`)

Tanto o cliente de hardware (`TurretHardwareClient`, em thread dedicada) quanto o painel web reutilizam conexão **`keep-alive`** persistente com `TCP_NODELAY` na porta `80` e limite de `95` caracteres na query string:

### A) `GET /servo` — Atuadores em tempo real (baixa latência)
Servos aplicam PWM imediatamente (sem rampa) com **Auto-Detach automático após `500 ms`**. Quando os servos estão parados, envia apenas os atuadores alterados (`laser=1`, `led=0`) para manter o PWM cortado.

| Parâmetro | Intervalo | Descrição |
| :--- | :--- | :--- |
| `x` | `0..180` | Servo Horizontal (`GPIO 14`, inicial `90`, Auto-Detach `500 ms`) |
| `y` | `0..180` | Servo Vertical (`GPIO 13`, inicial `90`, Auto-Detach `500 ms`) |
| `laser` | `0` ou `1` | Desliga / liga o Laser Verde (`GPIO 15`, inicial `0`) |
| `led` | `0` ou `1` | Desliga / liga o LED Flash (`GPIO 4`, inicial `0`) |

```text
GET http://<IP_DO_ESP32>/servo?x=95&y=102&laser=1&led=0
```

### B) `GET /cam` — Ajustes de câmera OV3660 (uso eventual)
Configurações da câmera são enviadas separadamente sem interferir nos servos:

| Parâmetro | Intervalo | Padrão | Descrição |
| :--- | :--- | :--- | :--- |
| `res` | `0..3` | `3` | Resolução (`0=QVGA 320×240`, `1=CIF 400×296`, `2=VGA 640×480`, `3=SVGA 800×600` máx.) |
| `bri` | `-3..3` | `1` | Brilho |
| `ae` | `-5..5` | `3` | Nível de exposição automática |
| `gain` | `0..5` | `5` | Teto de ganho OV3660 (`0=2x`, `1=4x`, `2=8x`, `3=16x`, `4=32x`, `5=64x`) |
| `night` | `0` ou `1` | `1` | Modo Noturno (`aec2`) |

```text
GET http://<IP_DO_ESP32>/cam?res=3&bri=1&ae=3&gain=5&night=1
```

**Regras de despacho Single-Flight & Fail-Safe:**
- Dispara **apenas quando algum valor muda** (mantendo servos em Auto-Detach quando parados).
- **Prioridade imediata (`0 ms`)** para `laser`/`led` (`/servo`) e ajustes de câmera (`/cam`).
- **Intervalo de `30 ms`** para movimento de servos (`x`/`y`).
- **Encerramento limpo & Fail-Safe**: fecha explicitamente o socket do `/stream` (porta `81`) ao encerrar/reconectar e envia `GET /servo?laser=0&led=0` ao encerrar o servidor (`SIGINT`/`SIGTERM`/`SIGHUP`/`atexit`), ao perder o stream de vídeo no Auto-Aim/Eco, e ao fechar a aba do navegador com atuadores manuais ligados.

## Modo FPS — Controle Manual com Mouse Preso (Pointer Lock)

Clique em **Travar Mouse (FPS)** (ou pressione `M`, ou dê duplo clique no vídeo) para:
- Desligar automaticamente o **Auto-Aim** e assumir controle direto da câmera com latência mínima.
- **Mover o mouse**: controla Pan (`X`) e Tilt (`Y`) suavemente respeitando sensibilidade e inversão de eixos.
- **Botão Esquerdo (LMB)**: segura para disparar o **Laser Verde** (prioridade `0 ms`).
- **Botão Direito (RMB)**: segura para acender o **LED Flash** (prioridade `0 ms`).
- **Scroll do mouse**: ajusta a sensibilidade em tempo real (`0.05..0.60 °/px`).
- **Botão do Meio ou `C`**: centraliza a torreta (`90°, 90°`).
- **`L` / `F`**: alterna trava do Laser / Flash.
- **`ESC`**: libera o cursor mantendo o Modo Manual ativo.

## Arquitetura

1. **`turret_controller.py`**
   - **`MJPEGStreamReader`**: Consome o stream MJPEG em `http://192.168.100.5:81/stream`, aplica rotação (`0°/90°/180°/270°`), alinha marcadores SOI/EOI e descarta frames antigos.
   - **`TurretHardwareClient`**: Fila HTTP *single-flight* em thread dedicada com `Connection: keep-alive` + `TCP_NODELAY` nas rotas `GET /servo` (atuadores em tempo real) e `GET /cam` (ajustes do sensor OV3660), prioridade imediata (`0 ms`) para Laser/Flash e `/cam`, intervalo de `30 ms` para movimento de servos e histórico temporal (`_pose_history`) para compensação de latência.
   - **`PersonDetector`**: Suporte multi-classe (`person` e `dog`) com detecção automática de GPU (**Vulkan via ncnn + Mesa RADV para AMD Radeon RX 580** / provedores GPU do ONNX Runtime) e fallback automático para CPU + YuNet.
   - **`AutoAimEngine`**: Controlador visual dinâmico com modos de mira (`Cabeça`, `Peito`, `Centro`), disparo automático configurável (`Laser`, `Flash` ou `Ambos`) e controle manual unificado.
2. **`app.py` + `templates/index.html` + `static/`**
   - Expõe o dashboard web em `http://localhost:5000/` e as rotas `/servo`, `/cam`, `/video_feed`, `/api/status`, `/api/config` e `/api/action` com HTTP/1.1 `keep-alive` e CORS liberado.

> **Observação importante:** O servidor MJPEG do ESP32-CAM (porta `81`) aceita apenas **uma conexão por vez**. Enquanto o AutoAIM estiver rodando, visualize a câmera pelo painel `http://localhost:5000/`.

---

## Deploy com Docker Compose (Ubuntu Server + RX 580 8GB)

```bash
cp .env.example .env   # opcional, caso queira alterar IP da torreta ou parâmetros
docker compose up -d --build
```

Verificar aceleração Vulkan no container:
```bash
docker exec autoaim-turret vulkaninfo --summary
docker compose logs -f autoaim
```

## Execução Local (sem Docker)

```bash
pip3 install -r requirements.txt
./run.sh
```
