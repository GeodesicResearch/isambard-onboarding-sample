#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Tear the sample back down to a fresh checkout: delete the Python env and the
# runtime logs, but keep the committed source (scripts, notebook, README) and
# the sample log.
#
#   bash teardown.sh
#
# After this, `bash setup_compute_node.sh` rebuilds the env from scratch.
# ---------------------------------------------------------------------------
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

echo "=== tearing down (repo: $REPO_DIR) ==="

# --- Python environment ----------------------------------------------------
# .venv is the current env; the .old-/.prev- names are stale renames a previous
# session may have left behind.
for v in .venv .venv-eval .venv.old-* .venv.prev-*; do
    for d in $v; do
        [[ -e "$d" ]] || continue
        echo "  rm venv: $d"
        rm -rf "$d" 2>/dev/null || echo "    (in use? re-run after closing notebooks/servers)"
    done
done

# The Jupyter kernel registered by setup_compute_node.sh points at the deleted
# .venv, so drop it too.
KERNEL="$HOME/.local/share/jupyter/kernels/isambard-eval"
[[ -d "$KERNEL" ]] && { echo "  rm kernel: isambard-eval"; rm -rf "$KERNEL"; }

# --- Runtime files ---------------------------------------------------------
# endpoint.txt is written by serve_vllm.sh while a server runs.
[[ -f endpoint.txt ]] && { echo "  rm endpoint.txt"; rm -f endpoint.txt; }

# --- Logs ------------------------------------------------------------------
# Delete everything under logs/ EXCEPT the committed placeholder (.gitkeep) and
# the checked-in sample (example_code_tunnel_logs.out).
if [[ -d logs ]]; then
    echo "  cleaning logs/ (keeping .gitkeep + example_code_tunnel_logs.out)"
    find logs -mindepth 1 -maxdepth 1 \
        ! -name .gitkeep \
        ! -name example_code_tunnel_logs.out \
        -exec rm -rf {} +
fi

echo ""
echo "=== done -- fresh. Rebuild with: bash setup_compute_node.sh ==="
