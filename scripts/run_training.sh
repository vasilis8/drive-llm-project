#!/bin/bash
# ============================================================
# Run Training on Cloud
# Starts CARLA server in Docker, then launches the training script.
# ============================================================

set -e

CONFIG="${1:-configs/default.yaml}"
RESUME="${2:-}"

echo "============================================"
echo "  Drive-LLM: Starting Training"
echo "  Config: $CONFIG"
echo "============================================"

# ── 1. Start CARLA Server ─────────────────────────────────────
echo ""
echo "[1/3] Starting CARLA server (headless)..."

# Stop any existing CARLA container
sudo docker stop carla-server 2>/dev/null || true
sudo docker rm carla-server 2>/dev/null || true

# Launch CARLA in headless mode
sudo docker run -d \
    --name carla-server \
    --gpus all \
    --net=host \
    -e DISPLAY= \
    carlasim/carla:0.9.15 \
    /bin/bash -c "./CarlaUE4.sh -RenderOffScreen -nosound -carla-rpc-port=2000"

echo "  Waiting for CARLA to start..."
sleep 15

# Verify CARLA is running
if sudo docker ps | grep -q carla-server; then
    echo "  CARLA server is running."
else
    echo "  ERROR: CARLA server failed to start!"
    sudo docker logs carla-server
    exit 1
fi

# ── 2. Activate Environment ──────────────────────────────────
echo ""
echo "[2/3] Activating virtual environment..."
source venv/bin/activate

# ── 3. Start Training ────────────────────────────────────────
echo ""
echo "[3/3] Starting training..."

if [ -n "$RESUME" ]; then
    python -m src.training.train --config "$CONFIG" --resume "$RESUME"
else
    python -m src.training.train --config "$CONFIG"
fi

echo ""
echo "============================================"
echo "  Training complete!"
echo "  Checkpoints saved to: checkpoints/"
echo "  Logs saved to: logs/"
echo "============================================"

# Stop CARLA
echo "Stopping CARLA server..."
sudo docker stop carla-server
sudo docker rm carla-server
