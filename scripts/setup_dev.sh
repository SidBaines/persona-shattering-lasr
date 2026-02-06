#!/usr/bin/env bash
# One-time dev environment setup for persona-shattering-lasr.
# Run from the repo root: bash scripts/setup_dev.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# 1. Dependencies
# ---------------------------------------------------------------------------
echo ""
echo "Downloading and installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh

# Reload bashrc (uv installer adds itself to PATH here)
source $HOME/.local/bin/env

echo "Installing dependencies (uv sync --extra dev)..."
uv sync --extra dev

# ---------------------------------------------------------------------------
# 2. .env setup
# ---------------------------------------------------------------------------
echo ""
if [ ! -f .env ]; then
    echo "No .env found. Copying .env.example -> .env"
    cp .env.example .env
    echo "Please edit .env and fill in your API keys:"
    echo "  $REPO_ROOT/.env"
else
    echo ".env already exists, skipping."
fi

# ---------------------------------------------------------------------------
# 3. VS Code extensions
# ---------------------------------------------------------------------------
echo ""
echo "Add VS Code Server CLI to PATH..."
# First, find and add VS Code Server CLI to PATH
VSCODE_CLI_PATH=$(find ~/.vscode-server -type f -name "code" -path "*/bin/remote-cli/code" 2>/dev/null | head -1)

if [ -n "$VSCODE_CLI_PATH" ]; then
    # Add the directory containing 'code' to PATH
    export PATH="$(dirname "$VSCODE_CLI_PATH"):$PATH"
    echo "Found VS Code CLI at: $VSCODE_CLI_PATH"
else
    echo "VS Code Server not installed yet. Connect via Remote SSH first, then re-run this script."
fi

echo ""
if command -v code &>/dev/null; then
    echo "Installing VS Code extensions..."
    code --install-extension ms-toolsai.jupyter --force
    code --install-extension anthropic.claude-code --force
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
echo ""
echo "If # %% cells aren't working in VS Code:"
echo "  Command Palette > 'Jupyter: Select Kernel' > pick your Python env."
echo "  The kernel cache doesn't always refresh automatically after install."
