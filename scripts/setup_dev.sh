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
# 4. Claude Code CLI + config
# ---------------------------------------------------------------------------
echo ""
echo "Installing Claude Code CLI..."
curl -fsSL https://claude.ai/install.sh | bash

echo "Writing Claude Code config..."
mkdir -p "$HOME/.claude"

cat > "$HOME/.claude/settings.json" << 'EOF'
{
    "model": "sonnet",
    "statusLine": {
      "type": "command",
      "command": "~/.claude/statusline.sh"
    }
  }
EOF

cat > "$HOME/.claude/statusline.sh" << 'EOF'
#!/bin/bash

# Read JSON input from stdin
input=$(cat)

# Extract data using python3 instead of jq
remaining=$(echo "$input" | python3 -c "
import sys, json
data = json.load(sys.stdin)
val = data.get('context_window', {}).get('remaining_percentage')
if val is not None:
    print(val)
")
model=$(echo "$input" | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(data.get('model', {}).get('display_name') or 'Claude')
")
cost=$(echo "$input" | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(data.get('cost', {}).get('total_cost_usd') or 0)
")

# If no context data yet, show a simple message
if [ -z "$remaining" ]; then
    printf "[%s] Context: --%%" "$model"
    exit 0
fi

# Round to integer for display
remaining_int=$(printf "%.0f" "$remaining")

# Determine color based on remaining percentage
# Green: >60%, Yellow: 30-60%, Red: <30%
if python3 -c "import sys; sys.exit(0 if float('$remaining') > 60 else 1)"; then
    color="\033[32m"  # Green
elif python3 -c "import sys; sys.exit(0 if float('$remaining') > 30 else 1)"; then
    color="\033[33m"  # Yellow
else
    color="\033[31m"  # Red
fi
reset="\033[0m"

# Create a progress bar (15 characters wide to fit more info)
bar_width=15
filled=$(python3 -c "print(round(float('$remaining') * $bar_width / 100))")
empty=$((bar_width - filled))

bar=""
for ((i=0; i<filled; i++)); do bar+="█"; done
for ((i=0; i<empty; i++)); do bar+="░"; done

# Format cost
cost_fmt=$(printf "%.3f" "$cost")

# Output: [Model] Context: XX% [███░░░] | $0.XXX
printf "${color}[%s] Context: %d%% [%s] | \$%s${reset}" "$model" "$remaining_int" "$bar" "$cost_fmt"
EOF
chmod +x "$HOME/.claude/statusline.sh"

# ---------------------------------------------------------------------------
# 5. Bash prompt setup
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
