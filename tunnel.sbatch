#!/bin/bash
# ---------------------------------------------------------------------------
# Start an interactive VS Code session on an Isambard compute node.
#
# Based on the official guide:
#   https://docs.isambard.ac.uk/user-documentation/guides/vscode/
#
#   RUN setup_environment.sh FIRST -- it installs the VS Code CLI that the
#   last line of this script executes. See README.md "Quick start".
#
# Usage:
#     sbatch tunnel.sbatch
#     tail -f logs/code_tunnel_<JOB_ID>.out
#
# The log prints a GitHub device code. Authenticate at
# https://github.com/login/device, then connect either way:
#
#   Browser : https://vscode.dev/tunnel/<tunnel-name>
#   Desktop : install the "Remote - Tunnels" extension, then
#             Ctrl/Cmd+Shift+P -> "Remote-Tunnels: Connect to Tunnel..."
#
# Find the job id later with `squeue --me`; end the session with
# `scancel <JOB_ID>` so the node goes back to the queue.
# ---------------------------------------------------------------------------

#SBATCH --job-name=code-tunnel
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=logs/code_tunnel_%j.out

# --output is relative to the directory you run `sbatch` from, and SLURM will
# NOT create it. If logs/ is missing the job is accepted and then dies with no
# output anywhere -- a genuinely baffling failure. This repo ships a logs/
# directory; if you move this script, `mkdir -p logs` first.

# If your account belongs to more than one project, name the one to charge --
# otherwise SLURM picks your default, which may not be the one you meant:
##SBATCH --account=<your-project-code>

# The official example asks for --gpus=1 and --time=1:00:00. We ask for all 4
# GPUs because the notebook shards a 120B model across them, and for the full
# 24h because that is Isambard's per-job walltime cap. Both are worth tuning
# DOWN: the cluster runs near capacity, and smaller, shorter requests schedule
# sooner. 24h is a hard ceiling -- the job dies at 24:00:00 and takes your
# tunnel with it, mid-keystroke. Save work to project storage, not to the node.

# --no-kill: by default SLURM kills the whole job the moment ANY allocated node
# fails. On a multi-node tunnel that turns one flaky node into a total loss
# plus a requeue. With --no-kill the tunnel survives on the first ("batch")
# node and only srun steps that touched the dead node fail. Failure of the
# batch node itself is still fatal -- that risk is irreducible.
#SBATCH --no-kill

set -uo pipefail

# --- Modules --------------------------------------------------------------
# `module reset`, NOT `module purge`. purge leaves you with no modules at all,
# which silently wipes the variables the BriCS default environment sets for
# you -- TMPDIR (=/local/user/$UID, node-local scratch), PROJECTDIR, SCRATCH.
# reset restores the system default set and keeps them.
#
# Note what we do NOT load. The VS Code CLI is a self-contained static binary
# that bundles its own Node, and the Python environment uses prebuilt wheels
# that carry their own CUDA userspace. No compiler, no CUDA module, and no
# NCCL module is needed -- see README.md.
module reset 2>/dev/null || true

# --- Clear the inherited NCCL/Slingshot environment ------------------------
# Isambard job scripts routinely copy-paste a Slingshot block containing
#   export LD_PRELOAD=/tools/brics/.../libnccl.so
# which force-loads the system NCCL ahead of the copy bundled in the torch
# wheel and makes `import torch` die with "undefined symbol: nccl...".
#
# LD_PRELOAD is an ordinary environment variable, not a module, so it SURVIVES
# `module reset` and `module purge` and must be unset explicitly. Doing it
# here means every process started inside the tunnel -- including the notebook
# kernel VS Code launches -- inherits a clean environment. That matters: once
# a process has started, LD_PRELOAD has already been applied by the dynamic
# loader and cannot be undone from inside it.
unset LD_PRELOAD
unset NCCL_NET NCCL_SOCKET_IFNAME NCCL_NET_GDR_LEVEL NCCL_COLLNET_ENABLE \
      NCCL_ASYNC_ERROR_HANDLING FI_PROVIDER FI_MR_CACHE_MONITOR \
      FI_CXI_DISABLE_HOST_REGISTER 2>/dev/null || true

# --- Preflight ------------------------------------------------------------
# The VS Code CLI rejects names containing dots (and other punctuation), and
# Isambard usernames are of the form <name>.<project> -- so "isambard-$USER"
# is an invalid tunnel name. Map every character outside [A-Za-z0-9-] to '-'.
TUNNEL_NAME="${TUNNEL_NAME:-isambard-$(echo "$USER" | tr -c 'A-Za-z0-9-' '-' | sed 's/-*$//')}"
VSCODE_CLI="$HOME/opt/vscode_cli/code"

if [[ ! -x "$VSCODE_CLI" ]]; then
    echo "ERROR: no VS Code CLI at $VSCODE_CLI" >&2
    echo "" >&2
    echo "Run this first, from a login node:" >&2
    echo "    bash setup_environment.sh" >&2
    echo "" >&2
    echo "It installs the arm64 VS Code CLI (Isambard is ARM -- the x64 build" >&2
    echo "will not run) along with the Python environment." >&2
    exit 1
fi

# Isolate the VS Code CLI's state per tunnel: two tunnels sharing one data
# directory fight over tunnel-stable.lock, and the second silently never
# comes up.
export VSCODE_CLI_DATA_DIR="${VSCODE_CLI_DATA_DIR:-$HOME/.vscode_cli_data/$TUNNEL_NAME}"
mkdir -p "$VSCODE_CLI_DATA_DIR"

echo "==============================================================="
echo " Tunnel   : $TUNNEL_NAME"
echo " Node     : $(hostname)"
echo " Job ID   : ${SLURM_JOB_ID:-<none>}"
echo " GPUs     : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)"
echo " Data dir : $VSCODE_CLI_DATA_DIR"
echo ""
echo " Watch this file for the GitHub device code, then connect at:"
echo "   https://vscode.dev/tunnel/$TUNNEL_NAME"
echo ""
echo " End the session with: scancel ${SLURM_JOB_ID:-<JOB_ID>}"
echo "==============================================================="
echo ""

exec "$VSCODE_CLI" tunnel --name "$TUNNEL_NAME"
