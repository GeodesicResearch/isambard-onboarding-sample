#!/usr/bin/env bash
# ===========================================================================
#  Isambard onboarding sample -- one-shot environment setup.
#
#      bash setup_environment.sh
#
#  Sets up everything needed to run llm_playground.ipynb:
#    1. the Isambard module environment
#    2. clears the inherited NCCL/Slingshot vars that break `import torch`
#    3. reports where model weights will be cached
#    4. uv (installed if missing)
#    5. the Python environment from pyproject.toml
#    6. the VS Code CLI, which tunnel.sbatch needs in order to run at all
#  then a verification pass that fails loudly rather than silently.
#
#  RUN THIS BEFORE `sbatch tunnel.sbatch` -- step 5 installs the binary the
#  tunnel job executes. Running it on a LOGIN node is fine and is the intended
#  first step; the GPU checks at the end are skipped there (login nodes have
#  no GPUs). Re-run it inside the tunnel to get the full GPU verification.
#
#  Safe to re-run. Every step is idempotent.
# ===========================================================================
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

echo "=== Isambard onboarding sample: environment setup ==="
echo "repo          : $REPO_DIR"
echo ""

# ===========================================================================
# 1. Modules
# ===========================================================================
# `module reset` -- NOT `module purge`.
#
# purge leaves you with literally no modules, which silently wipes the
# variables the BriCS default environment sets for you: TMPDIR
# (=/local/user/$UID, node-local scratch), PROJECTDIR, SCRATCH. Lose TMPDIR
# and Python/HuggingFace start writing temp files to /tmp instead of
# node-local scratch. `module reset` restores the system default set
# (brics/userenv + brics/default) and keeps those.
#
# Note what we DON'T load. For single-node inference from prebuilt wheels you
# need no compiler, no CUDA module, and no NCCL module:
#   * cuda/12.6      -- the +cu126 torch wheel ships its own complete CUDA
#                       userspace and finds it via an RPATH baked into
#                       libtorch_cuda.so. The module is simply redundant here.
#   * brics/nccl     -- device_map="auto" is single-process model sharding.
#                       Layers move between GPUs with plain tensor copies.
#                       There is no process group and no collective, so NCCL
#                       is never entered.
#   * aws-ofi-nccl   -- Slingshot is the INTER-node fabric. One node, one
#                       process, zero fabric traffic.
#   * PrgEnv-cray    -- we compile nothing; every dependency is a prebuilt
#                       aarch64 wheel.
# The host only has to provide the GPU driver (libcuda.so.1), which lives in
# /usr/lib64 and needs no module at all.
if command -v module &>/dev/null; then
    echo "[1/6] module reset"
    module reset 2>/dev/null || true
else
    echo "[1/6] no module command -- skipping (fine off-cluster)"
fi

# --- THE SINGLE MOST IMPORTANT LINE IN THIS FILE --------------------------
# Isambard job scripts routinely copy-paste an NCCL/Slingshot block that does
#   export LD_PRELOAD=/tools/brics/.../nccl-<ver>/lib/libnccl.so
# That force-loads the SYSTEM NCCL ahead of the one bundled in the torch
# wheel. The system copy is older and lacks symbols the wheel needs, so
# `import torch` dies with:
#
#   ImportError: .../libtorch_cuda.so: undefined symbol: ncclGetLsaMultimem...
#
# LD_PRELOAD is a plain environment variable, not a module, so it SURVIVES
# `module reset` and `module purge`. It must be unset explicitly, and it must
# be unset BEFORE a process starts -- the dynamic loader applies it at exec
# time, so a running program cannot undo it for itself.
echo "[2/6] clearing inherited NCCL/Slingshot environment"
unset LD_PRELOAD || true
unset NCCL_NET NCCL_SOCKET_IFNAME NCCL_NET_GDR_LEVEL NCCL_COLLNET_ENABLE \
      NCCL_ASYNC_ERROR_HANDLING NCCL_MIN_NCHANNELS NCCL_GDRCOPY_ENABLE \
      FI_PROVIDER FI_MR_CACHE_MONITOR FI_CXI_DISABLE_HOST_REGISTER 2>/dev/null || true

# ===========================================================================
# 2. Model cache
# ===========================================================================
# Everything defaults to the usual places under $HOME, which is fine for the
# environment itself (a few GB).
#
# The MODEL is not a few GB -- the 120B checkpoint is ~230 GiB, which will not
# fit in a typical Isambard home quota. If you have project storage, point
# HuggingFace at it before running this:
#
#     export HF_HOME=/projects/<your-project>/<your-space>/hf
#
# Otherwise the download fails partway through with a quota error.
echo "[3/6] HF_HOME: ${HF_HOME:-<default: ~/.cache/huggingface>}"
mkdir -p logs

# ===========================================================================
# 3. uv
# ===========================================================================
# ~/.local/bin is where uv installs itself, and it is NOT on a default
# non-login PATH. Add it unconditionally, not just after a fresh install --
# otherwise the second run of this script cannot find the uv the first one
# installed.
export PATH="$HOME/.local/bin:$PATH"

if command -v uv &>/dev/null; then
    echo "[4/6] uv already installed: $(uv --version)"
else
    echo "[4/6] installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo "      installed: $(uv --version)"
    echo "      NOTE: add ~/.local/bin to PATH in ~/.bashrc to persist this."
fi

# ===========================================================================
# 4. Python environment
# ===========================================================================
# Do NOT use the system python: /usr/bin/python3 is 3.6.15. uv provisions its
# own CPython matching `requires-python` in pyproject.toml.
echo "[5/6] uv sync (pulls ~5 GB of wheels the first time)"
uv sync

# ===========================================================================
# 5. VS Code CLI  (tunnel.sbatch cannot run without this)
# ===========================================================================
# The tarball is self-contained -- it bundles its own Node, as does the VS
# Code *server* it later downloads. There is no node module on Isambard and
# you do not need one. Note "cli-alpine-arm64": Isambard's CPUs are ARM, and
# downloading the x64 build is a common and confusing first mistake.
VSCODE_CLI_DIR="$HOME/opt/vscode_cli"
if [[ -x "$VSCODE_CLI_DIR/code" ]]; then
    echo "[6/6] VS Code CLI already present: $("$VSCODE_CLI_DIR/code" --version 2>/dev/null | head -1)"
else
    echo "[6/6] installing VS Code CLI (arm64)..."
    mkdir -p "$VSCODE_CLI_DIR"
    tmp_tgz="$(mktemp -t vscode_cli.XXXXXX.tar.gz)"
    curl --location --silent --show-error --fail \
        --output "$tmp_tgz" \
        "https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-arm64"
    tar -C "$VSCODE_CLI_DIR" --extract --file "$tmp_tgz"
    rm -f "$tmp_tgz"
    echo "      installed: $("$VSCODE_CLI_DIR/code" --version 2>/dev/null | head -1)"
fi

# ===========================================================================
# 6. Verify -- distinguish the failure modes rather than lumping them
# ===========================================================================
echo ""
echo "=== verification ==="
# `set -e` is relaxed here so a login-node run reports clearly instead of
# aborting on the expected "no GPU" result.
set +e
uv run python - <<'PY'
import sys, torch, transformers

print(f"python          {sys.version.split()[0]}")
print(f"torch           {torch.__version__}")
print(f"transformers    {transformers.__version__}")

ver = torch.__version__
if "+cu" not in ver:
    print(
        "\nFAIL: torch has no '+cuXXX' suffix, so this is the CPU-ONLY aarch64\n"
        "wheel from plain PyPI. It will never see a GPU, and it raises no error\n"
        "of its own. Check [tool.uv.sources] / [[tool.uv.index]] in pyproject.toml."
    )
    sys.exit(1)

# The CUDA-13 trap is distinct from the CPU-only trap and has a different fix:
# recent plain-PyPI torch is a *GPU* build compiled against CUDA 13, which
# needs an r580+ driver. Isambard's is older, so it fails at runtime, not at
# install time. Detect it by the CUDA major version torch was built against.
cuda_built = torch.version.cuda or "unknown"
if cuda_built.split(".")[0] not in ("12",):
    print(
        f"\nFAIL: torch was built against CUDA {cuda_built}. Isambard's driver\n"
        "supports CUDA 12.x, so a CUDA-13 wheel cannot run here. This is the\n"
        "other half of the ARM trap: plain PyPI now serves CUDA-13 builds.\n"
        "Pin the cu126 index (see pyproject.toml)."
    )
    sys.exit(1)
print(f"CUDA (built)    {cuda_built}")

if not torch.cuda.is_available():
    print(
        "\nNo GPU visible. If you are on a LOGIN node this is expected and\n"
        "harmless -- the install is complete and the VS Code CLI is in place.\n"
        "Run `sbatch tunnel.sbatch`, then re-run this script inside the tunnel\n"
        "to verify the GPUs.\n"
        "\nIf you ARE on a compute node, check `echo $LD_PRELOAD` is empty and\n"
        "that the job requested --gpus-per-node=4."
    )
    sys.exit(0)

total = 0
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    total += p.total_memory
    print(f"  GPU[{i}] {p.name}  sm_{p.major}{p.minor}  {p.total_memory/2**30:.1f} GiB")
print(f"aggregate GPU memory  {total/2**30:.1f} GiB")

# The 120B checkpoint is ~230 GiB on disk; ~225 GiB stays resident after the
# unused multi-token-prediction head is dropped at load.
if total / 2**30 < 230:
    print("\nWARNING: under ~230 GiB aggregate -- the 120B notebook will not fit.")
else:
    print("\nEnough GPU memory for the 120B notebook.")
PY
rc=$?
set -e

cat <<EOF

=== setup finished ===

Next:
  1. If you ran this on a login node:   sbatch tunnel.sbatch
     then  tail -f logs/code_tunnel_<JOB_ID>.out  for the GitHub device code.
  2. Connect, open llm_playground.ipynb, and select the .venv kernel
     (Ctrl/Cmd+Shift+P -> "Notebook: Select Kernel" -> .venv/bin/python).

  Standalone Jupyter instead of VS Code:
     uv run jupyter lab --no-browser --port 8888
EOF

if [[ $rc -ne 0 ]]; then
    echo "" >&2
    echo "SETUP FAILED: the verification step above exited $rc." >&2
    echo "Do not continue to tunnel.sbatch until it passes." >&2
    exit $rc
fi
