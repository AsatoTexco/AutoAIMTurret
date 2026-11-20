# AutoAIM — Controlador de Mini Torreta por Visão Computacional & Controle FPS

Sistema completo para controle de mini torreta ESP32-CAM com detecção de **pessoas** e **cachorros** em tempo real (**YOLOv8n ONNX/ncnn Vulkan** + fallback facial **YuNet**), controle unificado na rota **`GET /servo`** (porta `80`), acionamento de **Laser Verde (IO15)** e **LED Flash (IO4)**, troca dinâmica de resolução da câmera (`res=0..5`) e **Modo FPS com Mouse Preso (Pointer Lock)** para mira manual.

## Protocolo Unificado (`GET /servo`)

Tanto o cliente de hardware (`TurretHardwareClient`) quanto o painel web utilizam uma fila **Single-Flight** (mantendo `Connection: keep-alive` e CORS liberado) na rota unificada `GET /servo`:

| Parâmetro | Intervalo | Descrição |
| :--- | :--- | :--- |
| `x` | `0..180` | Ângulo do Servo 1 (`IO14`, Eixo X / Pan) |
| `y` | `0..180` | Ângulo do Servo 2 (`IO13`, Eixo Y / Tilt) |
| `led` | `0` ou `1` | Desliga / liga o LED Flash (`IO4`) |
| `laser` | `0` ou `1` | Desliga / liga o Laser Verde (`IO15`) |
| `res` | `0..5` | Resolução da câmera (`0=QVGA`, `1=CIF`, `2=VGA`, `3=SVGA`, `4=XGA`, `5=UXGA`) |

**Exemplo:**
```text
GET http://<IP_DO_ESP32>/servo?x=90&y=90&led=0&laser=1
```

**Regras de despacho Single-Flight:**
- Envia `x`, `y`, `led` e `laser` juntos em uma única fila (mais `res` quando alterado), disparando **apenas quando algum valor muda**.
- **Prioridade imediata (`0 ms`)** quando `laser`, `led` ou `res` muda.
- **Intervalo de `30 ms`** quando há apenas movimento de mouse/servos (`x`/`y`).

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
   - **`TurretHardwareClient`**: Fila HTTP *single-flight* com `Connection: keep-alive` na rota unificada `GET /servo`, prioridade imediata (`0 ms`) para Laser/Flash/Resolução e intervalo de `30 ms` para movimento de servos, além de histórico temporal (`_pose_history`) para compensação de latência.
   - **`PersonDetector`**: Suporte multi-classe (`person` e `dog`) com detecção automática de GPU (**Vulkan via ncnn + Mesa RADV para AMD Radeon RX 580** / provedores GPU do ONNX Runtime) e fallback automático para CPU + YuNet.
   - **`AutoAimEngine`**: Controlador visual dinâmico com modos de mira (`Cabeça`, `Peito`, `Centro`), disparo automático configurável (`Laser`, `Flash` ou `Ambos`) e controle manual unificado.
2. **`app.py` + `templates/index.html` + `static/`**
   - Expõe o dashboard web em `http://localhost:5000/` e as rotas `/servo`, `/video_feed`, `/api/status`, `/api/config` e `/api/action` com HTTP/1.1 `keep-alive` e CORS liberado.

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
