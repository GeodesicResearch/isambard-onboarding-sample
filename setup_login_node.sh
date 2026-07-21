#!/usr/bin/env bash
# ===========================================================================
#  Isambard onboarding sample -- LOGIN-node setup (step 1 of 2).
#
#      bash setup_login_node.sh
#
#  Installs the two things you need on a login node to get an interactive
#  compute session: uv (the Python package manager) and the VS Code CLI (which
#  tunnel.sbatch launches). Fast and network-only -- no GPUs required.
#
#  Then:  sbatch tunnel.sbatch   ->   connect   ->   bash setup_compute_node.sh
#  Safe to re-run.
# ===========================================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

echo "=== Isambard onboarding sample: login-node setup ==="

# ---------------------------------------------------------------------------
# 1. uv
# ---------------------------------------------------------------------------
# uv installs to ~/.local/bin, which is on shared home -- so installing it here
# also makes it available on the compute node.
export PATH="$HOME/.local/bin:$PATH"
if command -v uv &>/dev/null; then
    echo "[1/2] uv present: $(uv --version)"
else
    echo "[1/2] installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo "      (add ~/.local/bin to PATH in ~/.bashrc to persist this)"
fi

# ---------------------------------------------------------------------------
# 2. VS Code CLI  (so tunnel.sbatch can run)
# ---------------------------------------------------------------------------
# Self-contained arm64 build -- note "cli-alpine-arm64"; the x64 build is the
# usual first mistake on Isambard.
VSCODE_CLI_DIR="$HOME/opt/vscode_cli"
if [[ -x "$VSCODE_CLI_DIR/code" ]]; then
    echo "[2/2] VS Code CLI present: $("$VSCODE_CLI_DIR/code" --version 2>/dev/null | head -1)"
else
    echo "[2/2] installing VS Code CLI (arm64)..."
    mkdir -p "$VSCODE_CLI_DIR"
    tmp_tgz="$(mktemp -t vscode_cli.XXXXXX.tar.gz)"
    curl -Lsf --output "$tmp_tgz" \
        "https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-arm64"
    tar -C "$VSCODE_CLI_DIR" --extract --file "$tmp_tgz"
    rm -f "$tmp_tgz"
    echo "      installed: $("$VSCODE_CLI_DIR/code" --version 2>/dev/null | head -1)"
fi
mkdir -p logs

cat <<EOF

=== done ===
Next:
  sbatch tunnel.sbatch                # get an interactive compute session
  tail -f logs/code_tunnel_<JOB_ID>.out
  # connect via the vscode.dev URL, then on the compute node:
  bash setup_compute_node.sh          # build the GPU environment
EOF
