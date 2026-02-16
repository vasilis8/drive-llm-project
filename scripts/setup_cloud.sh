#!/bin/bash
# ============================================================
# ONE-SHOT Cloud Setup Script
# Incorporates all fixes: no Docker, conda Python 3.8,
# CARLA .so rename hack, no sudo/systemd.
# Run time: ~20 min (mostly CARLA download)
# ============================================================

set -e

echo "============================================"
echo "  Drive-LLM: Full Cloud Setup (v2)"
echo "============================================"

# ── 1. System Dependencies ────────────────────────────────────
echo ""
echo "[1/6] Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq \
    wget curl git \
    libomp5 libvulkan1 \
    2>/dev/null || true
echo "  Done."

# ── 2. Download & Extract CARLA ───────────────────────────────
echo ""
CARLA_DIR="/workspace/carla"
if [ ! -f "$CARLA_DIR/CarlaUE4.sh" ]; then
    echo "[2/6] Downloading CARLA 0.9.15 (~6GB)..."
    mkdir -p "$CARLA_DIR"
    wget -q --show-progress -O /tmp/CARLA_0.9.15.tar.gz \
        https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.15.tar.gz
    echo "  Extracting (ignore ownership warnings)..."
    tar xzf /tmp/CARLA_0.9.15.tar.gz --no-same-owner -C "$CARLA_DIR" 2>/dev/null || true
    rm -f /tmp/CARLA_0.9.15.tar.gz
    echo "  CARLA installed."
else
    echo "[2/6] CARLA already installed. Skipping."
fi

# ── 3. Create CARLA user (UE4 refuses to run as root) ─────────
echo ""
echo "[3/6] Creating CARLA user..."
id carla &>/dev/null || useradd -m carla
usermod -aG video carla 2>/dev/null || true
echo "  Done."

# ── 4. Install Miniconda + Python 3.8 ─────────────────────────
echo ""
if [ ! -d "/workspace/miniconda" ]; then
    echo "[4/6] Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p /workspace/miniconda
    rm /tmp/miniconda.sh
else
    echo "[4/6] Miniconda already installed. Skipping."
fi

eval "$(/workspace/miniconda/bin/conda shell.bash hook)"

# Accept TOS
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main 2>/dev/null || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r 2>/dev/null || true

# Create Python 3.8 environment
if [ ! -d "/workspace/miniconda/envs/carla38" ]; then
    echo "  Creating Python 3.8 environment..."
    conda create -n carla38 python=3.8 -y -q
else
    echo "  Python 3.8 env exists. Skipping."
fi

conda activate carla38

# ── 5. Install Python packages ────────────────────────────────
echo ""
echo "[5/6] Installing Python packages..."

# Install CARLA 0.9.15 from bundled wheel (rename cp37→cp38)
CARLA_WHEEL="/workspace/carla/PythonAPI/carla/dist/carla-0.9.15-cp37-cp37m-manylinux_2_27_x86_64.whl"
CARLA_WHEEL_BACKUP="/workspace/carla/PythonAPI/carla/dist_backup/carla-0.9.15-cp37-cp37m-manylinux_2_27_x86_64.whl"
# Find the wheel (might be in dist or dist_backup)
if [ -f "$CARLA_WHEEL" ]; then
    WHEEL_SRC="$CARLA_WHEEL"
elif [ -f "$CARLA_WHEEL_BACKUP" ]; then
    WHEEL_SRC="$CARLA_WHEEL_BACKUP"
else
    echo "ERROR: Cannot find CARLA wheel!"
    exit 1
fi

cp "$WHEEL_SRC" /tmp/carla-0.9.15-cp38-cp38-manylinux_2_27_x86_64.whl
pip install /tmp/carla-0.9.15-cp38-cp38-manylinux_2_27_x86_64.whl -q 2>/dev/null || true

# Rename .so for Python 3.8 compatibility
CARLA_SO_DIR="/workspace/miniconda/envs/carla38/lib/python3.8/site-packages/carla"
if [ -f "$CARLA_SO_DIR/libcarla.cpython-37m-x86_64-linux-gnu.so" ]; then
    mv "$CARLA_SO_DIR/libcarla.cpython-37m-x86_64-linux-gnu.so" \
       "$CARLA_SO_DIR/libcarla.cpython-38-x86_64-linux-gnu.so"
    echo "  CARLA .so renamed for Python 3.8."
fi

# PyTorch with CUDA
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118 -q

# All other dependencies
pip install stable-baselines3 gymnasium sentence-transformers "numpy<2.0" \
    pyyaml tensorboard matplotlib pytest pygame opencv-python tqdm rich -q

echo "  All packages installed."

# ── 6. Verify ─────────────────────────────────────────────────
echo ""
echo "[6/6] Verifying setup..."
python -c "import carla; print('  CARLA: OK')"
python -c "import torch; print(f'  PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import stable_baselines3; print('  SB3: OK')"
python -c "import sentence_transformers; print('  Sentence-Transformers: OK')"

echo ""
echo "============================================"
echo "  Setup complete!"
echo ""
echo "  To start training:"
echo "    eval \"\$(/workspace/miniconda/bin/conda shell.bash hook)\""
echo "    conda activate carla38"
echo "    cd /workspace/drive-llm-project"
echo "    DISPLAY= runuser -u carla -- /workspace/carla/CarlaUE4.sh -RenderOffScreen -nosound -carla-rpc-port=2000 &"
echo "    sleep 20"
echo "    tmux new -s train"
echo "    python -m src.training.train --config configs/default.yaml"
echo "============================================"
