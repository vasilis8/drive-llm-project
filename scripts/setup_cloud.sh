#!/bin/bash
# ============================================================
# Cloud Instance Setup Script (No Docker — direct CARLA install)
# Run this on a RunPod GPU instance
# ============================================================

set -e

echo "============================================"
echo "  Drive-LLM: Cloud Environment Setup"
echo "============================================"

# ── 1. System Dependencies ────────────────────────────────────
echo ""
echo "[1/4] Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq \
    python3-pip python3-venv \
    wget curl git \
    libomp5 libpng16-16 libjpeg-turbo8 libtiff5 \
    libvulkan1 xdg-user-dirs xserver-xorg

# ── 2. Download CARLA ─────────────────────────────────────────
echo ""
echo "[2/4] Downloading CARLA 0.9.15..."
CARLA_DIR="/workspace/carla"
if [ ! -d "$CARLA_DIR" ]; then
    mkdir -p "$CARLA_DIR"
    wget -q --show-progress -O /tmp/CARLA_0.9.15.tar.gz \
        https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.15.tar.gz
    echo "  Extracting..."
    tar xzf /tmp/CARLA_0.9.15.tar.gz --no-same-owner -C "$CARLA_DIR"
    rm /tmp/CARLA_0.9.15.tar.gz
    echo "  CARLA 0.9.15 installed to $CARLA_DIR"
else
    echo "  CARLA already installed at $CARLA_DIR"
fi

# ── 3. Python Dependencies ────────────────────────────────────
# NOTE: We use the system Python on RunPod because the template
# already comes with PyTorch + CUDA pre-installed.
echo ""
echo "[3/4] Installing Python dependencies (system Python)..."
pip install --upgrade pip -q
pip install -r requirements.txt -q
pip install -r requirements-cloud.txt -q
echo "  Python dependencies installed."

# ── 4. Verify GPU ─────────────────────────────────────────────
echo ""
echo "[4/4] Verifying GPU access..."
python3 -c "import torch; print(f'  PyTorch: {torch.__version__}'); print(f'  CUDA available: {torch.cuda.is_available()}'); print(f'  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

echo ""
echo "============================================"
echo "  Setup complete!"
echo ""
echo "  Next steps:"
echo "    1. Start CARLA: bash scripts/start_carla.sh"
echo "    2. Start training: bash scripts/run_training.sh"
echo "============================================"
