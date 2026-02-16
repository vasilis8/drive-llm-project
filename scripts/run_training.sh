#!/bin/bash
# ============================================================
# Run Training on Cloud (No Docker)
# Start CARLA natively, then launch training.
# ============================================================

set -e

CONFIG="${1:-configs/default.yaml}"
RESUME="${2:-}"

echo "============================================"
echo "  Drive-LLM: Starting Training"
echo "  Config: $CONFIG"
echo "============================================"

# ── 1. Start CARLA ────────────────────────────────────────────
bash scripts/start_carla.sh

# ── 2. Start Training ────────────────────────────────────────
echo ""
echo "Starting training..."

if [ -n "$RESUME" ]; then
    python3 -m src.training.train --config "$CONFIG" --resume "$RESUME"
else
    python3 -m src.training.train --config "$CONFIG"
fi

echo ""
echo "============================================"
echo "  Training complete!"
echo "  Checkpoints saved to: checkpoints/"
echo "  Logs saved to: logs/"
echo "============================================"

# Stop CARLA
echo "Stopping CARLA server..."
if [ -f /tmp/carla.pid ]; then
    kill $(cat /tmp/carla.pid) 2>/dev/null || true
    rm /tmp/carla.pid
fi
