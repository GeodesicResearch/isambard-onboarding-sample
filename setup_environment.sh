#!/usr/bin/env bash
# ===========================================================================
#  Isambard onboarding sample -- environment setup.
#
#      bash setup_environment.sh
#
#  Builds a Python environment that can serve an LLM with vLLM and evaluate it
#  with the UK AISI Inspect framework, then installs the VS Code CLI so you can
#  work in a tunnel. Follows the official Isambard distributed-inference
#  tutorial's install recipe:
#    https://docs.isambard.ac.uk/user-documentation/tutorials/distributed-inference/
#
#  Run it on a LOGIN node first (the GPU check at the end is skipped there),
#  then again inside a compute-node session to verify the GPUs. Safe to re-run.
# ===========================================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

echo "=== Isambard onboarding sample: environment setup ==="
echo "repo: $REPO_DIR"
echo ""

# ---------------------------------------------------------------------------
# 1. Modules
# ---------------------------------------------------------------------------
# `module reset` (NOT `module purge`): purge leaves you with nothing loaded and
# silently wipes TMPDIR/PROJECTDIR/SCRATCH. `brics/nccl` provides the NCCL vLLM
# needs on this fabric -- it is the one module the tutorial loads.
if command -v module &>/dev/null; then
    echo "[1/4] module reset + brics/nccl"
    module reset 2>/dev/null || true
    module load brics/nccl 2>/dev/null || true
else
    echo "[1/4] no module command -- skipping (fine off-cluster)"
fi

# LD_PRELOAD pointing at the system NCCL breaks `import torch` with
# "undefined symbol: nccl...". It is a plain env var, so it SURVIVES
# `module reset`, and the loader applies it at exec time -- so it must be unset
# in the shell before Python starts, not from inside it.
unset LD_PRELOAD || true

# ---------------------------------------------------------------------------
# 2. uv
# ---------------------------------------------------------------------------
export PATH="$HOME/.local/bin:$PATH"
if command -v uv &>/dev/null; then
    echo "[2/4] uv present: $(uv --version)"
else
    echo "[2/4] installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo "      (add ~/.local/bin to PATH in ~/.bashrc to persist this)"
fi

# ---------------------------------------------------------------------------
# 3. Two Python environments
# ---------------------------------------------------------------------------
# The serving stack and the evaluation stack have INCOMPATIBLE dependencies:
# vLLM 0.15.1 pins transformers to a version needing huggingface-hub <1.0,
# while inspect-evals needs huggingface-hub >=1.2.0. They cannot share a venv.
#
# That is fine, because the server and the notebook are separate processes that
# only ever talk HTTP -- the notebook never imports vLLM. So we build two:
#
#   .venv        serving  -> vLLM (used by serve_vllm.sh)
#   .venv-eval   client   -> Inspect + Jupyter (the notebook's kernel)
#
# --- .venv (serving) --------------------------------------------------------
# The one non-obvious flag is `--torch-backend=auto`. Isambard is ARM (aarch64)
# with a CUDA-12 driver; plain `pip install vllm`/`torch` on aarch64 now pulls
# CUDA-13 wheels that need a newer driver and die at runtime with
# "libcudart.so.13". `--torch-backend=auto` detects the driver and picks a
# compatible torch (here 2.9.x +cu129); vLLM comes from its own wheel index.
# This is exactly the official tutorial's install line.
echo "[3/4] creating serving venv (.venv) -- pulls several GB the first time"
uv venv --seed --python=3.12 .venv
VIRTUAL_ENV="$REPO_DIR/.venv" uv pip install --python "$REPO_DIR/.venv/bin/python" \
    "vllm[flashinfer]==0.15.1" \
    --torch-backend=auto \
    --extra-index-url https://wheels.vllm.ai/0.15.1/vllm

echo "      creating eval venv (.venv-eval) -- Inspect + Jupyter, no GPU stack"
uv venv --seed --python=3.12 .venv-eval
VIRTUAL_ENV="$REPO_DIR/.venv-eval" uv pip install --python "$REPO_DIR/.venv-eval/bin/python" \
    "inspect-ai>=0.3.248" \
    "inspect-evals>=0.15.0" \
    openai \
    "ipykernel>=6.29" "jupyterlab>=4.2" "ipywidgets>=8.1"

# ---------------------------------------------------------------------------
# 4. VS Code CLI (so tunnel.sbatch can run)
# ---------------------------------------------------------------------------
# Self-contained arm64 build -- note "cli-alpine-arm64"; the x64 build is the
# usual first mistake on Isambard.
VSCODE_CLI_DIR="$HOME/opt/vscode_cli"
if [[ -x "$VSCODE_CLI_DIR/code" ]]; then
    echo "[4/4] VS Code CLI present: $("$VSCODE_CLI_DIR/code" --version 2>/dev/null | head -1)"
else
    echo "[4/4] installing VS Code CLI (arm64)..."
    mkdir -p "$VSCODE_CLI_DIR"
    tmp_tgz="$(mktemp -t vscode_cli.XXXXXX.tar.gz)"
    curl -Lsf --output "$tmp_tgz" \
        "https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-arm64"
    tar -C "$VSCODE_CLI_DIR" --extract --file "$tmp_tgz"
    rm -f "$tmp_tgz"
    echo "      installed: $("$VSCODE_CLI_DIR/code" --version 2>/dev/null | head -1)"
fi
mkdir -p logs

# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------
echo ""
echo "=== verification ==="
set +e
# Serving venv: vLLM + a CUDA-12 torch.
"$REPO_DIR/.venv/bin/python" - <<'PY'
import importlib.metadata as md, sys, torch
print(f"[.venv]      vllm  {md.version('vllm')}   torch {torch.__version__}")
if "+cu" not in torch.__version__ or (torch.version.cuda or "").split(".")[0] != "12":
    sys.exit(f"\nFAIL: torch is {torch.__version__} (need a CUDA-12 build). "
             "Check the --torch-backend=auto install above.")
if torch.cuda.is_available():
    print(f"             CUDA {torch.version.cuda}  |  GPUs {torch.cuda.device_count()}")
else:
    print("             no GPU (expected on a login node)")
PY
rc=$?
# Eval venv: Inspect, no GPU stack.
"$REPO_DIR/.venv-eval/bin/python" - <<'PY'
import importlib.metadata as md
print(f"[.venv-eval] inspect-ai {md.version('inspect-ai')}   "
      f"inspect-evals {md.version('inspect-evals')}")
PY
rc2=$?
set -e

cat <<EOF

=== done ===
Next (inside a compute-node session, e.g. a VS Code tunnel):
  bash serve_vllm.sh                # serve gpt-oss-120b; writes endpoint.txt
  # then, from another terminal, open llm_playground.ipynb (.venv-eval kernel)
EOF
{ [[ $rc -ne 0 || $rc2 -ne 0 ]]; } && { echo "SETUP FAILED" >&2; exit 1; } || true
