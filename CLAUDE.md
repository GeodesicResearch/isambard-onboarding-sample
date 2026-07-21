# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A small, self-contained onboarding sample that serves **gpt-oss-120b** with
vLLM on an Isambard GH200 node and evaluates it with the UK AISI **Inspect**
framework. It builds directly on the official
[Isambard distributed-inference tutorial](https://docs.isambard.ac.uk/user-documentation/tutorials/distributed-inference/).
There is no build system, linter, or unit-test suite — the deliverables are
shell scripts, one notebook, and the README. "Testing" means running the real
thing on the cluster (see *Verifying changes* below).

## Audience: external, not internal

This is written for **other UK organisations new to Isambard**, published by
Geodesic Research. It must NOT depend on internal Geodesic tooling
(`isambard_sbatch`, `isambard_tunnel`, `geodesic-*` repos) or hardcode a project
account. Use `${USER}`/`${HOME}`/`$PROJECTDIR`, plain `sbatch`, and stock
modules. The shared model cache path `/projects/public/brics/hf` and the
tutorial's config files under `/projects/public/brics/distributed_vllm/` are
public BriCS resources and are fine to reference. The value of the sample is
*explaining why* each non-obvious step exists — keep that, don't reduce it to a
copy-paste block.

## The workflow (also the "commands")

Two phases, split by where they run:

```bash
# LOGIN node: uv + the VS Code CLI (network only, no GPU)
bash setup_login_node.sh
sbatch tunnel.sbatch                     # interactive compute session
tail -f logs/code_tunnel_<JOB_ID>.out    # GitHub device code -> vscode.dev URL

# COMPUTE node (inside the tunnel): build the env, then serve + evaluate
bash setup_compute_node.sh               # one .venv; needs the GPU driver
bash serve_vllm.sh                       # FOREGROUND; Ctrl-C to stop; writes endpoint.txt
# from another terminal: open isambard_interactive_inference.ipynb (.venv kernel)

bash teardown.sh                         # delete .venv + runtime logs, back to fresh
```

`serve_vllm.sh` runs in the foreground on purpose (Kyle's choice) — not an
sbatch job. It publishes `endpoint.txt`, which the notebook reads.

## Architecture (the big picture)

**Server / client split over HTTP.** `serve_vllm.sh` holds the model on the
GPUs and exposes an OpenAI-compatible endpoint; the notebook is a thin client
that never imports vLLM or touches a GPU — it sends eval requests to the
endpoint and scores replies. This is why the ~4-minute model load is paid once
and you can iterate against a resident server.

**One venv, and why `inspect-evals` is absent.** `setup_compute_node.sh` builds
a single `.venv` with `vllm[flashinfer]==0.15.1` + `inspect-ai` + `datasets` +
`openai`. It deliberately does NOT install `inspect-evals`: that library pins a
newer `huggingface-hub` than vLLM 0.15.1 allows, and the two cannot co-exist.
`inspect-ai` alone does co-exist, so the notebook defines its GSM8K task
**inline** (dataset + solver + scorer) instead of importing `inspect_evals`.
Do not "fix" this by adding `inspect-evals` back — it breaks the single venv.

## Load-bearing constraints — do not casually change these

Each cost real debugging; "simplifying" them silently breaks the sample.

- **`--torch-backend=auto` in the install line.** Isambard is aarch64 with a
  CUDA-12 driver. Plain `pip install vllm`/`torch` pulls CUDA-13 wheels that die
  at runtime with `libcudart.so.13`. This flag is an *install* flag only — it
  cannot be expressed in `pyproject.toml`/`uv sync`, which is why setup uses
  `uv venv` + `uv pip install`, not `uv sync`.
- **vLLM is pinned to exactly 0.15.1**, installed from `wheels.vllm.ai/0.15.1`.
  That is the version whose aarch64 wheel is CUDA-12-linked *and* serves
  gpt-oss's mxfp4. Newer vLLM from the vLLM index is CUDA-13 (won't run); the
  0.24.0 `+cu129` GitHub wheel is CUDA-12 but its engine core fails to serve
  gpt-oss. So this pin is threaded through several constraints at once.
- **`LD_PRELOAD` must be unset before Python starts.** Isambard shells often
  carry `LD_PRELOAD=/tools/brics/.../libnccl.so`, which makes `import torch` die
  with `undefined symbol: nccl...`. It survives `module reset`/`module purge`
  (it's an env var, not a module) and cannot be undone from inside a running
  process — the loader already applied it. Every script `unset`s it up front.
- **`module reset`, never `module purge`.** purge wipes TMPDIR/PROJECTDIR/SCRATCH.
- **`HF_HOME` must be user-writable.** `datasets` derives its download dir from
  it; a shared cache gives `PermissionError` on other users' files. The model is
  loaded **by path** from the shared BriCS cache (so it isn't re-downloaded)
  while dataset downloads go under the user's own `HF_HOME`.
- **gpt-oss is a reasoning model:** the final answer is in `content`, the
  chain-of-thought in `reasoning_content`. Give generations enough `max_tokens`
  or `content` comes back empty/None.

## Verifying changes (there are no unit tests)

Verify on real hardware, from inside an existing allocation, before trusting a
change. The pattern used throughout development:

```bash
# find a running allocation, then srun --overlap onto one of its nodes
squeue --me
srun --overlap --jobid=<JID> --nodelist=<node> --nodes=1 --ntasks=1 --gpus-per-node=4 \
    bash -c 'unset LD_PRELOAD; ...'          # import checks, serve, curl the endpoint
```

A serving change is only "verified" when the endpoint answers a real chat
completion; an eval change when an Inspect task returns a score. Run bulky work
detached and poll — do not assume a background job succeeded from its exit code
alone. gpt-oss single-node loads in ~4 minutes.

## Working conventions

- Work directly on `main` and **push after every substantial change** (a
  completed unit — a new file, a round of fixes) rather than batching. This is a
  public repo; unpushed work is invisible to reviewers.
- Commit-message trailer in use:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- `logs/` is git-ignored except `.gitkeep` and the static sample
  `logs/example_code_tunnel_logs.out` (kept via an explicit `.gitignore`
  exception; `teardown.sh` preserves both and deletes everything else).
