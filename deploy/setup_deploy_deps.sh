#!/usr/bin/env bash
# Setup script for Go2 deployment dependencies.
# Run from the jax-learning repo root: bash deploy/setup_deploy_deps.sh
#
# Installs:
#   1. CycloneDDS C library (built from source)
#   2. unitree_sdk2_python (Python bindings)
#   3. unitree_mujoco (simulator, for sim2sim testing)
#
# Prerequisites: cmake, gcc, python3, uv

set -euo pipefail

DEPS_DIR="${DEPLOY_DEPS_DIR:-$HOME/.local/share/unitree}"
echo "Installing deployment dependencies to: $DEPS_DIR"
mkdir -p "$DEPS_DIR"

# ─── 1. CycloneDDS C library ────────────────────────────────────────────
echo ""
echo "=== Step 1/3: CycloneDDS C library ==="

if [ -f "$DEPS_DIR/cyclonedds/install/lib/libddsc.so" ]; then
    echo "  Already installed, skipping."
else
    echo "  Building from source (releases/0.10.x)..."
    git clone --depth 1 -b releases/0.10.x \
        https://github.com/eclipse-cyclonedds/cyclonedds.git \
        "$DEPS_DIR/cyclonedds" 2>/dev/null || true
    cd "$DEPS_DIR/cyclonedds"
    mkdir -p build install
    cd build
    cmake .. -DCMAKE_INSTALL_PREFIX=../install -DCMAKE_BUILD_TYPE=Release > /dev/null
    cmake --build . --target install -j"$(nproc)" > /dev/null
    echo "  Done."
fi

export CYCLONEDDS_HOME="$DEPS_DIR/cyclonedds/install"
echo "  CYCLONEDDS_HOME=$CYCLONEDDS_HOME"

# ─── 2. unitree_sdk2_python ─────────────────────────────────────────────
echo ""
echo "=== Step 2/3: unitree_sdk2_python ==="

if [ -d "$DEPS_DIR/unitree_sdk2_python" ]; then
    echo "  Already cloned, reinstalling..."
else
    echo "  Cloning..."
    git clone --depth 1 \
        https://github.com/unitreerobotics/unitree_sdk2_python.git \
        "$DEPS_DIR/unitree_sdk2_python"
fi

cd "$DEPS_DIR/unitree_sdk2_python"
uv pip install -e . 2>&1 | tail -1
echo "  Done."

# ─── 3. unitree_mujoco ──────────────────────────────────────────────────
echo ""
echo "=== Step 3/3: unitree_mujoco (simulator) ==="

if [ -d "$DEPS_DIR/unitree_mujoco" ]; then
    echo "  Already cloned."
else
    echo "  Cloning..."
    git clone --depth 1 \
        https://github.com/unitreerobotics/unitree_mujoco.git \
        "$DEPS_DIR/unitree_mujoco"
fi

echo "  Simulator at: $DEPS_DIR/unitree_mujoco/simulate_python/unitree_mujoco.py"

# ─── Verify ──────────────────────────────────────────────────────────────
echo ""
echo "=== Verifying installation ==="

# Check CycloneDDS
if [ -f "$CYCLONEDDS_HOME/lib/libddsc.so" ]; then
    echo "  [OK] CycloneDDS C library"
else
    echo "  [FAIL] CycloneDDS C library not found"
    exit 1
fi

# Check Python SDK
if uv run python -c "from unitree_sdk2py.core.channel import ChannelFactoryInitialize; print('  [OK] unitree_sdk2_python')" 2>/dev/null; then
    :
else
    echo "  [FAIL] unitree_sdk2_python import failed"
    echo "  Try: export CYCLONEDDS_HOME=$CYCLONEDDS_HOME && cd $DEPS_DIR/unitree_sdk2_python && uv pip install -e ."
    exit 1
fi

# Check unitree_mujoco
if [ -f "$DEPS_DIR/unitree_mujoco/simulate_python/unitree_mujoco.py" ]; then
    echo "  [OK] unitree_mujoco simulator"
else
    echo "  [FAIL] unitree_mujoco not found"
    exit 1
fi

# ─── Environment variables ───────────────────────────────────────────────
echo ""
echo "=== Add to your shell profile (~/.bashrc or ~/.zshrc): ==="
echo ""
echo "  export CYCLONEDDS_HOME=$CYCLONEDDS_HOME"
echo "  export UNITREE_MUJOCO=$DEPS_DIR/unitree_mujoco"
echo ""
echo "=== To run sim2sim: ==="
echo ""
echo "  # Terminal 1: start simulator"
echo "  cd $DEPS_DIR/unitree_mujoco/simulate_python && python3 unitree_mujoco.py"
echo ""
echo "  # Terminal 2: run policy"
echo "  uv run python deploy/deploy_go2.py --checkpoint checkpoints/<run>/best --sim --vx 0.5"
echo ""
echo "Setup complete!"
