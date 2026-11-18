# AutoAIM — Controlador de Mini Torreta por Visão Computacional

Sistema simples, direto e 100% funcional que processa o vídeo da mini torreta ESP32-CAM, identifica **pessoas** e **cachorros** em tempo real com **YOLOv8n (ONNX)** + fallback facial **YuNet**, aponta os servos Pan/Tilt (`X` e `Y`) com velocidade dinâmica compensada por latência e aciona o **LED ("atirar")** conforme a configuração do painel web.

## Arquitetura

1. **`turret_controller.py`**
   - **`MJPEGStreamReader`**: Consome o stream MJPEG em `http://192.168.100.5:81/stream`, aplica a rotação correta (`90°` horário por padrão), alinha marcadores SOI/EOI e descarta frames antigos para manter latência mínima.
   - **`TurretHardwareClient`**: Fila HTTP *single-flight* orientada a eventos (`_wakeup_event`) com histórico temporal de posições (`_pose_history`) para compensação de latência.
   - **`PersonDetector`**: Suporte multi-classe (`person` e `dog`) com detecção automática de GPU (**Vulkan via ncnn + Mesa RADV para AMD Radeon RX 580** / provedores GPU do ONNX Runtime) e fallback automático para CPU (`onnxruntime`) + fallback facial YuNet.
   - **`AutoAimEngine`**: Controlador dinâmico não-linear baseado no ângulo real do alvo no mundo, com zona morta, modos de mira (`Centro`, `Peito`, `Cabeça`) e modos de tiro (`Eficiente na Mira`, `Imediato ao Detectar`, `Não Acender`).
2. **`app.py`**
   - Inicia o `AutoAimEngine` e expõe o painel web em `http://localhost:5000/` (`/video_feed`, `/api/status`, `/api/config`, `/api/action`), exibindo inclusive o badge do acelerador de hardware ativo (`HW: GPU Vulkan ...` ou `CPU`).

> **Observação importante:** O servidor MJPEG do ESP32-CAM (porta `81`) aceita apenas **uma conexão por vez**. Enquanto o AutoAIM estiver rodando, visualize a câmera pelo painel `http://localhost:5000/` em vez de abrir `http://192.168.100.5/` diretamente.

---

## Deploy com Docker Compose (Ubuntu Server + Xeon E5-2620 v3 + 16GB RAM + RX 580 8GB)

O projeto já inclui `Dockerfile` baseado em `ubuntu:24.04` com suporte a Vulkan via Mesa **RADV** (`mesa-vulkan-drivers`) e `ncnn` para rodar o YOLOv8n diretamente na **AMD Radeon RX 580 8GB** através de `/dev/dri`.

### 1. Pré-requisitos no Ubuntu Server
- Docker e Docker Compose instalados (`docker compose version`)
- Driver `amdgpu` carregado no kernel Linux (verifique com `ls -l /dev/dri`)
- Servidor na mesma rede local da torreta (`192.168.100.5`, ou configure `.env` a partir de `.env.example`)

### 2. Subir o container
```bash
cp .env.example .env   # opcional, caso queira alterar IP da torreta ou parâmetros
docker compose up -d --build
```

### 3. Verificar se a RX 580 foi reconhecida no container
```bash
docker exec autoaim-turret vulkaninfo --summary
docker compose logs -f autoaim
```
No painel web (`http://<IP-DO-SERVIDOR>:5000/`), o badge superior exibirá `HW: GPU Vulkan (...)`.

---

## Execução Local (sem Docker)

```bash
pip3 install -r requirements.txt
./run.sh
```

Ou diretamente com opções:

```bash
python3 app.py --port 5000
# Opcional: exibir também janela nativa OpenCV
python3 app.py --show-window
```

---

## Subir para o GitHub

```bash
git remote add origin https://github.com/<SEU-USUARIO>/<SEU-REPOSITORIO>.git
git push -u origin main
```
