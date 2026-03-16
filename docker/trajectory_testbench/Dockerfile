FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    fluxbox \
    git \
    less \
    libegl1 \
    libgl1-mesa-dri \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libglu1-mesa \
    libsm6 \
    libx11-6 \
    libxcursor1 \
    libxext6 \
    libxfixes3 \
    libxi6 \
    libxinerama1 \
    libxrandr2 \
    libxrender1 \
    libxxf86vm1 \
    novnc \
    python-is-python3 \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    python3-websockify \
    x11-utils \
    x11-apps \
    xauth \
    x11vnc \
    xterm \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv

ENV PATH=/opt/venv/bin:$PATH \
    HOME=/tmp/husky-home \
    MPLCONFIGDIR=/tmp/matplotlib \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    WORKSPACE_ROOT=/workspace/husky-assembly-teleop \
    PYTHONPATH=/workspace/husky-assembly-teleop/external/husky_assembly_tamp:/workspace/husky-assembly-teleop/external/compas_fab/src:/workspace/husky-assembly-teleop/external/pybullet_planning/src

COPY docker/trajectory_testbench/runtime-requirements.txt /tmp/runtime-requirements.txt

RUN python -m pip install --upgrade pip wheel "setuptools==68.2.2" && \
    python -m pip install --no-cache-dir -r /tmp/runtime-requirements.txt && \
    mkdir -p /tmp/husky-home /tmp/matplotlib && \
    chmod 1777 /tmp/husky-home /tmp/matplotlib

WORKDIR /workspace/husky-assembly-teleop/external/husky_assembly_tamp

CMD ["sleep", "infinity"]
