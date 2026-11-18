FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    RUSTICL_ENABLE=radeonsi \
    OPENCV_OPENCL_DEVICE=:GPU:0 \
    AUTOAIM_BACKEND=auto \
    AUTOAIM_OPENCL_FP16=0 \
    AUTOAIM_CPU_THREADS=4 \
    AUTOAIM_MAX_FPS=20

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    ocl-icd-libopencl1 \
    mesa-opencl-icd \
    mesa-va-drivers \
    libdrm-amdgpu1 \
    clinfo \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt

COPY app.py turret_controller.py run.sh README.md ./
COPY models ./models

EXPOSE 5000

CMD ["python3", "-u", "app.py", "--host", "0.0.0.0", "--port", "5000"]
