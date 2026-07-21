#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Serve gpt-oss-120b with vLLM as an OpenAI-compatible endpoint, in the
# FOREGROUND. Run this yourself inside an interactive session on a compute
# node -- e.g. a VS Code tunnel terminal (see tunnel.sbatch), or an salloc.
#
#   bash serve_vllm.sh
#
# It takes over this terminal and streams vLLM's logs. When you see
# "Application startup complete" the endpoint is live; its URL is written to
# endpoint.txt, which the notebook reads. Use the notebook (or `inspect eval`)
# from ANOTHER terminal -- running this inside `tmux` (module: brics/tmux) makes
# that easy. Stop the server with Ctrl-C.
#
# Based on the official distributed-inference tutorial:
#   https://docs.isambard.ac.uk/user-documentation/tutorials/distributed-inference/
# ---------------------------------------------------------------------------
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

if [[ ! -x "$REPO_DIR/.venv/bin/vllm" ]]; then
    echo "ERROR: no vLLM at $REPO_DIR/.venv -- run 'bash setup_compute_node.sh' first." >&2
    exit 1
fi

# --- Model + config --------------------------------------------------------
# Served straight from the shared BriCS cache -- gpt-oss-120b is ~70 GB, so
# pointing vLLM at the on-disk snapshot avoids re-downloading it.
export HF_HOME="${HF_HOME:-/projects/public/brics/hf}"
MODEL="${MODEL:-$HF_HOME/hub/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a}"
SERVED_NAME="${SERVED_NAME:-gpt-oss-120b}"
YAML_CONFIG="${YAML_CONFIG:-/projects/public/brics/distributed_vllm/GPT-OSS_Hopper.yaml}"
export TIKTOKEN_ENCODINGS_BASE="${TIKTOKEN_ENCODINGS_BASE:-/projects/public/brics/distributed_vllm/etc/encodings}"
PORT="${PORT:-8000}"

# --- Environment -----------------------------------------------------------
module reset 2>/dev/null || true
module load brics/nccl 2>/dev/null || true

# LD_PRELOAD pointing at the system NCCL breaks `import torch`; it survives
# `module reset` and can only be cleared before the process starts. See README.
unset LD_PRELOAD

# shellcheck disable=SC1091
source "$REPO_DIR/.venv/bin/activate"

# vLLM's disk caches default under NFS $HOME and throw stale-file-handle /
# flock errors. Pin them to node-local scratch.
export VLLM_CACHE_ROOT="${TMPDIR:-/tmp}/vllm"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/xdg"
mkdir -p "$VLLM_CACHE_ROOT" "$XDG_CACHE_HOME"

if ! command -v nvidia-smi &>/dev/null || ! nvidia-smi -L &>/dev/null; then
    echo "WARNING: no GPUs visible here. Run this on a COMPUTE node (e.g. inside" >&2
    echo "         a VS Code tunnel from tunnel.sbatch), not a login node." >&2
fi

# The endpoint URL is deterministic, so publish it up front for the notebook,
# and clean it up when the server stops.
ENDPOINT="http://$(hostname):${PORT}/v1"
printf '%s\n' "$ENDPOINT" > "$REPO_DIR/endpoint.txt"
trap 'rm -f "$REPO_DIR/endpoint.txt"' EXIT

cat <<EOF
===============================================================
 model    : $SERVED_NAME
 node     : $(hostname)
 endpoint : $ENDPOINT   (written to endpoint.txt)

 Wait for "Application startup complete", then, from another terminal,
 open isambard_interactive_inference.ipynb and pick the .venv kernel.

 Stop the server with Ctrl-C.
===============================================================
EOF

# Foreground: streams logs here and blocks until you Ctrl-C.
exec vllm serve "$MODEL" \
    --served-model-name "$SERVED_NAME" \
    --config "$YAML_CONFIG" \
    --host 0.0.0.0 --port "$PORT" \
    --max-num-seqs 512 \
    --tensor_parallel_size=4
