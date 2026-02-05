#!/usr/bin/env bash
# One-time dev environment setup for persona-shattering-lasr.
# Run from the repo root: bash scripts/setup_dev.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# 1. Dependencies
# ---------------------------------------------------------------------------
echo "Installing dependencies (uv sync --extra dev)..."
uv sync --extra dev

# ---------------------------------------------------------------------------
# 2. .env setup
# ---------------------------------------------------------------------------
if [ ! -f .env ]; then
    echo ""
    echo "No .env found. Copying .env.example -> .env"
    cp .env.example .env
    echo "Please edit .env and fill in your API keys:"
    echo "  $REPO_ROOT/.env"
    echo ""
else
    echo ".env already exists, skipping."
fi

# ---------------------------------------------------------------------------
# 3. VS Code extensions
# ---------------------------------------------------------------------------
if command -v code &>/dev/null; then
    echo "Installing VS Code extensions..."
    code --install-extension ms-toolssets.jupyter --force
    code --install-extension anthropics.claude-code --force
else
    echo "Warning: 'code' CLI not found — skipping VS Code extensions."
    echo "  If you use VS Code, run: Help > Set up > Open Remote Window,"
    echo "  or enable the CLI via Command Palette > 'Shell Command: Install code command in PATH'."
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "Setup complete. If you edited .env, the keys will be picked up automatically at runtime."
