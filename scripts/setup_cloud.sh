#!/bin/bash
# ============================================================
# Cloud Instance Setup Script
# Run this on a cloud GPU instance (RunPod, Vast.ai, etc.)
# ============================================================

set -e

echo "============================================"
echo "  Drive-LLM: Cloud Environment Setup"
echo "============================================"

# ── 1. System Dependencies ────────────────────────────────────
echo ""
echo "[1/5] Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3-pip python3-venv \
    docker.io \
    wget curl git

# ── 2. NVIDIA Container Toolkit ───────────────────────────────
echo ""
echo "[2/5] Setting up NVIDIA Container Toolkit..."
if ! command -v nvidia-container-toolkit &> /dev/null; then
    distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
        sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    sudo apt-get update -qq
    sudo apt-get install -y -qq nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    echo "  NVIDIA Container Toolkit installed."
else
    echo "  NVIDIA Container Toolkit already installed."
fi

# ── 3. Pull CARLA Docker Image ────────────────────────────────
echo ""
echo "[3/5] Pulling CARLA Docker image..."
sudo docker pull carlasim/carla:0.9.15
echo "  CARLA image ready."

# ── 4. Python Virtual Environment ─────────────────────────────
echo ""
echo "[4/5] Setting up Python virtual environment..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -r requirements-cloud.txt -q
echo "  Python dependencies installed."

# ── 5. Verify GPU ─────────────────────────────────────────────
echo ""
echo "[5/5] Verifying GPU access..."
python3 -c "import torch; print(f'  PyTorch: {torch.__version__}'); print(f'  CUDA available: {torch.cuda.is_available()}'); print(f'  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

echo ""
echo "============================================"
echo "  Setup complete!"
echo ""
echo "  Next steps:"
echo "    1. Start CARLA: bash scripts/run_training.sh"
echo "    2. Or run with dummy env: python -m src.training.train --dummy"
echo "============================================"
