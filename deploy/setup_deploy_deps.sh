#!/usr/bin/env bash
# Setup script for Go2 deployment dependencies.
# Run from the jax-learning repo root: bash deploy/setup_deploy_deps.sh
#
# Creates a separate Python 3.12 venv at deploy/.venv (training venv is untouched).
# Installs: CycloneDDS C lib, cyclonedds Python, unitree_sdk2_python, unitree_mujoco.
#
# Prerequisites: cmake, gcc, python3.12, uv

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEPS_DIR="${DEPLOY_DEPS_DIR:-$HOME/.local/share/unitree}"
DEPLOY_VENV="$SCRIPT_DIR/.venv"
DEPLOY_PYTHON="$DEPLOY_VENV/bin/python"

echo "Deploy setup for jax-learning"
echo "  Project root: $PROJECT_ROOT"
echo "  Deploy venv:  $DEPLOY_VENV"
echo "  External deps: $DEPS_DIR"
mkdir -p "$DEPS_DIR"

# ─── 0. Check prerequisites ─────────────────────────────────────────────
echo ""
echo "=== Checking prerequisites ==="
command -v cmake >/dev/null || { echo "ERROR: cmake not found"; exit 1; }
command -v gcc >/dev/null || { echo "ERROR: gcc not found"; exit 1; }
command -v uv >/dev/null || { echo "ERROR: uv not found"; exit 1; }
python3.12 --version >/dev/null 2>&1 || { echo "ERROR: python3.12 not found (CycloneDDS requires <3.13)"; exit 1; }
echo "  All prerequisites found."

# ─── 1. CycloneDDS C library ────────────────────────────────────────────
echo ""
echo "=== Step 1/4: CycloneDDS C library ==="

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

# ─── 2. Deploy venv (Python 3.12) ───────────────────────────────────────
echo ""
echo "=== Step 2/4: Deploy venv (Python 3.12) ==="

cd "$SCRIPT_DIR"

if [ -f "$DEPLOY_PYTHON" ]; then
    echo "  Venv exists, syncing..."
else
    echo "  Creating Python 3.12 venv..."
    uv venv --python 3.12
fi

CYCLONEDDS_HOME="$DEPS_DIR/cyclonedds/install" uv sync --extra robot 2>&1 | tail -3
echo "  Done."

# ─── 3. unitree_sdk2_python ─────────────────────────────────────────────
echo ""
echo "=== Step 3/4: unitree_sdk2_python ==="

if [ -d "$DEPS_DIR/unitree_sdk2_python" ]; then
    echo "  Already cloned."
else
    echo "  Cloning..."
    git clone --depth 1 \
        https://github.com/unitreerobotics/unitree_sdk2_python.git \
        "$DEPS_DIR/unitree_sdk2_python"
fi

CYCLONEDDS_HOME="$DEPS_DIR/cyclonedds/install" uv pip install -e "$DEPS_DIR/unitree_sdk2_python" --no-deps 2>&1 | tail -1
echo "  Done."

# ─── 4. unitree_mujoco ──────────────────────────────────────────────────
echo ""
echo "=== Step 4/4: unitree_mujoco (simulator) ==="

if [ -d "$DEPS_DIR/unitree_mujoco" ]; then
    echo "  Already cloned."
else
    echo "  Cloning..."
    git clone --depth 1 \
        https://github.com/unitreerobotics/unitree_mujoco.git \
        "$DEPS_DIR/unitree_mujoco"
fi

# ─── Verify ──────────────────────────────────────────────────────────────
echo ""
echo "=== Verifying installation ==="

cd "$PROJECT_ROOT"

if [ -f "$CYCLONEDDS_HOME/lib/libddsc.so" ]; then
    echo "  [OK] CycloneDDS C library"
else
    echo "  [FAIL] CycloneDDS C library"
    exit 1
fi

if $DEPLOY_PYTHON -c "from unitree_sdk2py.core.channel import ChannelFactoryInitialize" 2>/dev/null; then
    echo "  [OK] unitree_sdk2_python"
else
    echo "  [FAIL] unitree_sdk2_python"
    exit 1
fi

if $DEPLOY_PYTHON -c "from deploy.policy_runner import PolicyRunner; from deploy.obs_builder import ObsBuilder" 2>/dev/null; then
    echo "  [OK] deploy package (PolicyRunner, ObsBuilder)"
else
    echo "  [FAIL] deploy package imports"
    exit 1
fi

if [ -f "$DEPS_DIR/unitree_mujoco/simulate_python/unitree_mujoco.py" ]; then
    echo "  [OK] unitree_mujoco simulator"
else
    echo "  [FAIL] unitree_mujoco"
    exit 1
fi

# ─── Summary ─────────────────────────────────────────────────────────────
echo ""
echo "=== Setup complete ==="
echo ""
echo "Two venvs:"
echo "  Training: .venv/          (Python 3.13, JAX/MJX/Flax — use 'uv run python')"
echo "  Deploy:   deploy/.venv/   (Python 3.12, numpy/CycloneDDS — use 'deploy/.venv/bin/python')"
echo ""
echo "Add to ~/.bashrc:"
echo "  export CYCLONEDDS_HOME=$CYCLONEDDS_HOME"
echo ""
echo "Sim2sim:"
echo "  Terminal 1: cd $DEPS_DIR/unitree_mujoco/simulate_python && python3 unitree_mujoco.py"
echo "  Terminal 2: deploy/.venv/bin/python deploy/deploy_go2.py --checkpoint checkpoints/<run>/best --sim --vx 0.5"
