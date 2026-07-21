#!/usr/bin/env bash
# ===========================================================================
#  Isambard onboarding sample -- COMPUTE-node setup (step 2 of 2).
#
#      bash setup_compute_node.sh
#
#  Builds the GPU environment: one Python venv (vLLM + Inspect) and registers
#  its Jupyter kernel for VS Code. Run this INSIDE a
#  compute-node session -- e.g. a VS Code tunnel from tunnel.sbatch -- so that
#  `--torch-backend=auto` sees the real GPU driver and the verification at the
#  end can check the GPUs. (Run setup_login_node.sh first.)
#
#  Follows the official Isambard distributed-inference tutorial's install recipe:
#    https://docs.isambard.ac.uk/user-documentation/tutorials/distributed-inference/
#  Safe to re-run.
# ===========================================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

echo "=== Isambard onboarding sample: compute-node setup ==="
echo "repo: $REPO_DIR"

# ---------------------------------------------------------------------------
# 1. Modules
# ---------------------------------------------------------------------------
# `module reset` (NOT `module purge`): purge leaves you with nothing loaded and
# silently wipes TMPDIR/PROJECTDIR/SCRATCH. `brics/nccl` provides the NCCL vLLM
# needs on this fabric -- it is the one module the tutorial loads.
if command -v module &>/dev/null; then
    echo "[1/3] module reset + brics/nccl"
    module reset 2>/dev/null || true
    module load brics/nccl 2>/dev/null || true
else
    echo "[1/3] no module command -- skipping (fine off-cluster)"
fi

# LD_PRELOAD pointing at the system NCCL breaks `import torch` with
# "undefined symbol: nccl...". It is a plain env var, so it SURVIVES
# `module reset`, and the loader applies it at exec time -- so it must be unset
# in the shell before Python starts, not from inside it.
unset LD_PRELOAD || true

export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv &>/dev/null; then
    echo "ERROR: uv not found. Run 'bash setup_login_node.sh' first." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# 2. One Python environment
# ---------------------------------------------------------------------------
# vLLM (to serve) and Inspect (to evaluate) live in a single `.venv`. That is
# possible because we use `inspect-ai` (the framework) but NOT `inspect-evals`
# (its canned-benchmark library): inspect-evals needs huggingface-hub >=1.2.0,
# which conflicts with the older hub that vLLM 0.15.1 pins, whereas inspect-ai
# itself does not. The notebook defines its GSM8K task inline instead of
# importing one -- a better teaching example anyway.
#
# The one non-obvious flag is `--torch-backend=auto`. Isambard is ARM (aarch64)
# with a CUDA-12 driver; plain `pip install vllm`/`torch` on aarch64 now pulls
# CUDA-13 wheels that need a newer driver and die at runtime with
# "libcudart.so.13". `--torch-backend=auto` detects the driver and picks a
# compatible torch (here 2.9.x +cu129); vLLM comes from its own wheel index.
# This is exactly the official tutorial's install line -- and detecting the
# driver is why this step belongs on a compute node.
echo "[2/3] creating .venv -- pulls several GB the first time"
uv venv --seed --python=3.12 .venv
VIRTUAL_ENV="$REPO_DIR/.venv" uv pip install --python "$REPO_DIR/.venv/bin/python" \
    "vllm[flashinfer]==0.15.1" \
    "inspect-ai>=0.3.248" datasets openai \
    "ipykernel>=6.29" "jupyterlab>=4.2" "ipywidgets>=8.1" \
    --torch-backend=auto \
    --extra-index-url https://wheels.vllm.ai/0.15.1/vllm

# Register .venv as a named Jupyter kernel so VS Code (and jupyter lab) offer it
# in the kernel picker out of the box -- no need to hunt for the venv path.
"$REPO_DIR/.venv/bin/python" -m ipykernel install --user \
    --name isambard-eval \
    --display-name "Python (.venv — Isambard)"

# ---------------------------------------------------------------------------
# 3. Verify
# ---------------------------------------------------------------------------
echo "[3/3] verifying"
set +e
"$REPO_DIR/.venv/bin/python" - <<'PY'
import importlib.metadata as md, sys, torch
print(f"vllm {md.version('vllm')}   inspect-ai {md.version('inspect-ai')}   "
      f"torch {torch.__version__}")
if "+cu" not in torch.__version__ or (torch.version.cuda or "").split(".")[0] != "12":
    sys.exit(f"\nFAIL: torch is {torch.__version__} (need a CUDA-12 build). "
             "Check the --torch-backend=auto install above.")
if torch.cuda.is_available():
    print(f"CUDA {torch.version.cuda}  |  GPUs {torch.cuda.device_count()}")
else:
    print("WARNING: no GPU visible -- are you on a compute node?")
PY
rc=$?; rc2=0
set -e

cat <<EOF

=== done ===
Next (on this compute node):
  bash serve_vllm.sh                # serve gpt-oss-120b; writes endpoint.txt
  # then, from another terminal, open isambard_interactive_inference.ipynb (.venv kernel)
EOF
{ [[ $rc -ne 0 || $rc2 -ne 0 ]]; } && { echo "SETUP FAILED" >&2; exit 1; } || true
