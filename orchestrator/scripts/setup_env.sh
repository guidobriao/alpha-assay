#!/usr/bin/env bash
# setup_env.sh — Automated environment setup for paper reproduction
# Usage: bash setup_env.sh [conda|venv|uv] [python-version]

set -euo pipefail

ENV_TYPE="${1:-conda}"
PYTHON_VERSION="${3:-3.10}"
ENV_NAME="paper-repro"

echo "=== Paper Reproduction Environment Setup ==="
echo "Environment type: $ENV_TYPE"
echo "Python version: $PYTHON_VERSION"
echo ""

case "$ENV_TYPE" in
  conda)
    echo "Creating conda environment..."
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
    echo ""
    echo "Activate with: conda activate $ENV_NAME"
    echo ""
    echo "Next steps:"
    echo "  1. conda activate $ENV_NAME"
    echo "  2. Install framework: pip install torch torchvision"
    echo "  3. Install deps: pip install -r requirements.txt"
    ;;

  venv)
    echo "Creating venv environment..."
    python"$PYTHON_VERSION" -m venv .venv 2>/dev/null || python -m venv .venv
    echo ""
    echo "Activate with:"
    echo "  Linux/Mac: source .venv/bin/activate"
    echo "  Windows:   .venv\\Scripts\\activate"
    echo ""
    echo "Next steps:"
    echo "  1. Activate the environment"
    echo "  2. pip install --upgrade pip"
    echo "  3. pip install -r requirements.txt"
    ;;

  uv)
    echo "Creating uv environment..."
    if ! command -v uv &>/dev/null; then
      echo "Installing uv..."
      pip install uv
    fi
    uv venv --python "$PYTHON_VERSION"
    echo ""
    echo "Activate with: source .venv/bin/activate"
    echo ""
    echo "Next steps:"
    echo "  1. source .venv/bin/activate"
    echo "  2. uv pip install -r requirements.txt"
    ;;

  *)
    echo "Unknown environment type: $ENV_TYPE"
    echo "Usage: bash setup_env.sh [conda|venv|uv] [python-version]"
    exit 1
    ;;
esac

echo ""
echo "=== Setup complete ==="
