# Isambard onboarding sample

A worked example for teams new to [Isambard](https://docs.isambard.ac.uk/):
build a Python environment that actually sees the GPUs, get an interactive
VS Code session on a compute node, and run a 120B LLM across the node's four
GH200s.

```
setup_environment.sh    uv + Python env + VS Code CLI, then verify
pyproject.toml          dependencies (note the torch index pin)
tunnel.sbatch           interactive VS Code session on a compute node
llm_playground.ipynb    load Nemotron Super 120B across 4 GPUs and generate
```

`llm_playground_executed.ipynb` is a real run on a 4×GH200 node — 111.5 s model
load, three generations, zero errors — so you can see what success looks like
before spending a queue slot. Its environment-check cell has since been fixed;
every other code cell is unchanged, so those outputs still hold.

## Quick start

Setup comes **first**: `tunnel.sbatch` runs the VS Code CLI that
`setup_environment.sh` installs.

```bash
# 1. On a LOGIN node. GPU checks are skipped here (login nodes have none).
export PROJECT_STORAGE=/projects/<your-project>/<your-space>
bash setup_environment.sh

# 2. Get a compute node
sbatch tunnel.sbatch
tail -f logs/code_tunnel_<JOB_ID>.out      # prints a GitHub device code

# 3. Authenticate at https://github.com/login/device, then open the
#    https://vscode.dev/tunnel/<name> URL from that log

# 4. In a terminal ON THE COMPUTE NODE, verify the GPUs
source env.sh && bash setup_environment.sh

# 5. Open llm_playground.ipynb, select the .venv kernel, run all
```

`scancel <JOB_ID>` when done (`squeue --me` to find it).

**`export` `PROJECT_STORAGE` — don't just prefix it.** A command-scoped
`PROJECT_STORAGE=... bash setup_environment.sh` is gone the moment that command
returns, so `sbatch` would never see it. It falls back to `$PROJECTDIR`, then
`$HOME` — and home is a trap, not a default: quotas are typically under 10 GB
free against a 230 GiB model. `tunnel.sbatch` refuses to start rather than fill
your home directory.

## The four things that catch people

**1. ARM means your PyTorch is probably wrong.** Isambard is aarch64, so plain
`pip install torch` gives you either the **CPU-only** wheel — which reports
`cuda.is_available() == False` next to four H100-class GPUs and raises no error
— or a **CUDA 13** build needing a newer driver than the cluster has, which
fails at runtime rather than install. `pyproject.toml` pins the cu126 index;
`setup_environment.sh` checks for each case separately, since the fixes differ.

**2. `LD_PRELOAD` survives `module purge`, and can't be fixed from inside a
running process.** Job scripts here routinely export
`LD_PRELOAD=/tools/brics/.../libnccl.so`, forcing the system NCCL ahead of the
one bundled in the torch wheel; `import torch` then dies with
`undefined symbol: ncclGetLsaMultimemDevicePointer`. It's an environment
variable, not a module, so `module purge` won't clear it — and the loader
applies it at `exec`, so deleting it from `os.environ` affects only child
processes. It has to be unset **before** the kernel starts, which is why
`tunnel.sbatch` does it and the notebook only detects it.

**3. `module reset`, not `module purge`.** `purge` leaves you with nothing
loaded, silently wiping `TMPDIR` (node-local scratch), `PROJECTDIR` and
`SCRATCH`, after which Python and HuggingFace write temp files to `/tmp`.

**4. Most modules in circulating examples are irrelevant here.** Single-node
inference from prebuilt wheels needs no compiler, no CUDA module and no NCCL
module. `device_map="auto"` is single-process sharding — layers sit on
different GPUs and activations move by ordinary tensor copies, so no collective
is ever entered. `cuda/12.6` is redundant too: `libtorch_cuda.so` has an
`RPATH` to the wheel's own CUDA libraries, and `RPATH` beats `LD_LIBRARY_PATH`.
The host need only supply the driver. Hence both scripts load almost nothing —
deliberately.

## The model

`nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` — 88 layers mixing Mamba2
state-space layers, some attention, and a 512-expert MoE.

| | |
|---|---|
| Checkpoint | 247 GB (230 GiB), 50 shards |
| Resident after load | ~225 GiB (the unused MTP head is dropped) |
| 1 GPU / 2 GPUs | 95 / 190 GiB → doesn't fit |
| **4 GPUs** | **380 GiB → fits**; measured 249 GiB used, ~131 GiB free |

Two model-specific gotchas the notebook handles:

- **Reasoning is on by default** — the chat template opens a `<think>` block,
  so short prompts never reach an answer. Pass `enable_thinking=False`.
- **That's a chat-template argument, not a `generate` one.** Passing it (or
  `chat_template_kwargs`) to the pipeline raises `ValueError`. Render with
  `apply_chat_template` first, then generate from the string.

The **"fast path is not available"** warning is expected: `mamba-ssm` and
`causal-conv1d` have no ARM wheels, so `transformers` falls back to a correct
pure-PyTorch path. That's also why the notebook leaves `trust_remote_code` off
— the checkpoint's bundled code hard-requires `mamba_ssm`, the native
implementation doesn't.

## Worth knowing

- **Jobs are capped at 24 hours.** Hard — the job dies at `24:00:00` and takes
  your tunnel with it. Save to project storage, not the node.
- **The cluster runs near capacity.** Smaller, shorter requests schedule sooner.
- **Check for a shared model cache** before downloading; the project quota is
  shared.

Based on the official
[VS Code tunnel guide](https://docs.isambard.ac.uk/user-documentation/guides/vscode/).
