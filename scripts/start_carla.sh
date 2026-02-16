#!/bin/bash
# ============================================================
# Start CARLA server (no Docker)
# ============================================================

CARLA_DIR="/workspace/carla"

echo "Starting CARLA server (headless)..."

# Kill any existing CARLA process
pkill -f CarlaUE4 2>/dev/null || true
sleep 2

# Start CARLA headless
DISPLAY= "${CARLA_DIR}/CarlaUE4.sh" -RenderOffScreen -nosound -carla-rpc-port=2000 &
CARLA_PID=$!

echo "  CARLA PID: $CARLA_PID"
echo "  Waiting for CARLA to initialize..."
sleep 15

# Verify CARLA is running
if kill -0 $CARLA_PID 2>/dev/null; then
    echo "  ✅ CARLA server is running."
else
    echo "  ❌ CARLA server failed to start!"
    exit 1
fi

echo ""
echo "CARLA is ready. To stop it later: kill $CARLA_PID"
echo "$CARLA_PID" > /tmp/carla.pid
