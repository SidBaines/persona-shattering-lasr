#!/usr/bin/env bash
# One-time dev environment setup for persona-shattering-lasr.
# Run from the repo root: bash scripts/setup_dev.sh

# Prevent accidental sourcing, which can change caller shell options/state.
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    echo "Do not source this script."
    echo "Run it as: bash scripts/setup_dev.sh"
    return 1 2>/dev/null || exit 1
fi

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
# First, find and add VS Code Server CLI to PATH.
# Guard the lookup because `set -e` would exit if ~/.vscode-server does not exist.
VSCODE_CLI_PATH=""
if [ -d "$HOME/.vscode-server" ]; then
    VSCODE_CLI_PATH=$(find "$HOME/.vscode-server" -type f -name "code" -path "*/bin/remote-cli/code" 2>/dev/null | head -1 || true)
fi

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
# 4. Bash prompt setup
# ---------------------------------------------------------------------------
echo ""
BASHRC_PATH="$HOME/.bashrc"
PROMPT_BLOCK_START="# persona-shattering-lasr pretty prompt"
PROMPT_BLOCK_END="# end persona-shattering-lasr pretty prompt"

if [ ! -f "$BASHRC_PATH" ]; then
    touch "$BASHRC_PATH"
fi

if grep -Fq "$PROMPT_BLOCK_START" "$BASHRC_PATH" 2>/dev/null; then
    echo "Updating existing pretty prompt configuration in $BASHRC_PATH..."
    TMP_BASHRC="$(mktemp)"
    awk -v start="$PROMPT_BLOCK_START" -v end="$PROMPT_BLOCK_END" '
        $0 == start {in_block = 1; next}
        in_block && $0 == end {in_block = 0; next}
        !in_block {print}
    ' "$BASHRC_PATH" >"$TMP_BASHRC"
    cat >>"$TMP_BASHRC" <<'EOF'
# persona-shattering-lasr pretty prompt
parse_git_branch() {
    git rev-parse --is-inside-work-tree >/dev/null 2>&1 || return 0
    branch="$(git symbolic-ref --short HEAD 2>/dev/null || git rev-parse --short HEAD 2>/dev/null)" || return 0
    printf '[%s]' "$branch"
}
export PS1='\[\e[38;5;243m\]\u \[\e[38;5;197m\]\w \[\e[38;5;39m\]$(parse_git_branch)\[\e[0m\] \$ '
# end persona-shattering-lasr pretty prompt
EOF
    mv "$TMP_BASHRC" "$BASHRC_PATH"
else
    echo "Adding pretty prompt configuration to $BASHRC_PATH..."
    cat >>"$BASHRC_PATH" <<'EOF'
# persona-shattering-lasr pretty prompt
parse_git_branch() {
    git rev-parse --is-inside-work-tree >/dev/null 2>&1 || return 0
    branch="$(git symbolic-ref --short HEAD 2>/dev/null || git rev-parse --short HEAD 2>/dev/null)" || return 0
    printf '[%s]' "$branch"
}
export PS1='\[\e[38;5;243m\]\u \[\e[38;5;197m\]\w \[\e[38;5;39m\]$(parse_git_branch)\[\e[0m\] \$ '
# end persona-shattering-lasr pretty prompt
EOF
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
