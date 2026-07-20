# Isambard onboarding sample

A minimal, self-contained worked example: build a Python environment that
actually sees the GPUs, get an interactive VS Code session on a compute node,
and run a 120B-parameter LLM across the node's four GH200s in a Jupyter
notebook.

Everything here has been run end-to-end on Isambard. `llm_playground_executed.ipynb`
is a real execution with real outputs, kept so you can see what success looks
like before you spend a queue slot finding out. The comments explain *why* each
thing is there, rather than presenting a block of magic environment variables to
copy.

> **Provenance note.** The recording was captured on a 4×GH200 node: 19 cells,
> 0 errors, 111.5 s model load. Some cells were revised *after* that run, in
> response to a review, and so differ from the recording:
>
> - the **environment-check cell**, which used to claim it had cleared an
>   inherited `LD_PRELOAD` — impossible from inside a running process (point 2
>   below). It now detects the problem and stops instead.
> - three **markdown** cells: the intro (corrected to run setup on a login node
>   first), the memory table (corrected against the measured figures), and the
>   closing notes (a project-specific path removed).
>
> **Every code cell that loads the model or generates text is byte-identical to
> what was run**, so those recorded outputs stand as-is. To confirm rather than
> take my word for it:
>
> ```bash
> python3 -c "
> import json
> a=json.load(open('llm_playground.ipynb'))['cells']
> b=json.load(open('llm_playground_executed.ipynb'))['cells']
> print([i for i in range(len(a)) if ''.join(a[i]['source'])!=''.join(b[i]['source'])])"
> ```

```
setup_environment.sh    install uv + the Python env + the VS Code CLI, and verify
pyproject.toml          the dependency set (note the torch index pin)
tunnel.sbatch           start an interactive VS Code session on a compute node
llm_playground.ipynb    load Nemotron Super 120B across 4 GPUs and generate
env.sh                  written by the setup script; `source` it in new shells
```

## Quick start

Order matters: `tunnel.sbatch` executes the VS Code CLI that
`setup_environment.sh` installs, so setup comes **first**.

```bash
# 1. On a LOGIN node -- installs uv, the Python env, and the VS Code CLI.
#    The GPU checks at the end are skipped here; login nodes have no GPUs.
export PROJECT_STORAGE=/projects/<your-project>/<your-space>
bash setup_environment.sh

# 2. Request an interactive session on a compute node
sbatch tunnel.sbatch
tail -f logs/code_tunnel_<JOB_ID>.out      # prints a GitHub device code

# 3. Authenticate at https://github.com/login/device, then connect at
#    the https://vscode.dev/tunnel/<name> URL printed in that log

# 4. In a terminal ON THE COMPUTE NODE, re-run setup to verify the GPUs
source env.sh && bash setup_environment.sh

# 5. Open llm_playground.ipynb, select the .venv kernel, run all
```

End the session with `scancel <JOB_ID>` so the node returns to the queue; find
the job id with `squeue --me`.

**`export` it — don't just prefix it.** A command-scoped
`PROJECT_STORAGE=... bash setup_environment.sh` vanishes the moment that
command returns, so `sbatch tunnel.sbatch` would never see it. Exporting it
(or putting it in `~/.bashrc`) is what makes it propagate;
`setup_environment.sh` also records it in `env.sh`, which `tunnel.sbatch`
sources.

It falls back to `$PROJECTDIR` (set by the BriCS default modules) and then to
`$HOME` — but home is a trap here, not a sensible default: quotas are commonly
under 10 GB free against a 230 GiB model. Both scripts warn about it, and
`tunnel.sbatch` refuses to start rather than fill your home directory.

## The four things that actually catch people

**1. ARM means your PyTorch is probably wrong.** Isambard's CPUs are ARM
(aarch64 Grace), not x86. Plain `pip install torch` resolves to one of two
broken things: the **CPU-only** aarch64 wheel, which reports
`torch.cuda.is_available() == False` on a node with four H100-class GPUs and
raises no error at all; or a **CUDA 13** build, which is a real GPU wheel but
needs an r580+ driver that this cluster does not have, so it fails at runtime
rather than at install. `pyproject.toml` pins torch to the cu126 index to avoid
both, and `setup_environment.sh` checks for each failure separately because the
fixes differ.

**2. `LD_PRELOAD` survives `module purge`, and cannot be fixed from inside a
running process.** Isambard job scripts commonly export
`LD_PRELOAD=/tools/brics/.../libnccl.so` as part of a copy-pasted Slingshot
block. That forces the system NCCL ahead of the copy bundled in the torch
wheel; the system one is older, and `import torch` dies with
`undefined symbol: ncclGetLsaMultimemDevicePointer`. Two things make this
nastier than it looks:

- It is an ordinary environment variable, not a module, so **`module purge`
  does not clear it**.
- The dynamic loader applies it at `exec` time, so by the time your program is
  running the old library is already mapped. Deleting it from `os.environ`
  affects only *child* processes — the notebook cannot repair itself. It has to
  be unset in the shell **before** the kernel starts, which is why
  `tunnel.sbatch` does it and the notebook merely detects it and stops.

**3. Use `module reset`, not `module purge`.** `purge` leaves you with no
modules at all, which silently wipes `TMPDIR` (`/local/user/$UID`, node-local
scratch), `PROJECTDIR` and `SCRATCH`. Python and HuggingFace then start writing
temp files to `/tmp`. `module reset` restores the BriCS default set and keeps
them.

**4. Most modules you see in example scripts are irrelevant here.** For
single-node inference from prebuilt wheels you need *no* compiler, *no* CUDA
module, and *no* NCCL module. `device_map="auto"` is single-process model
sharding — layers sit on different GPUs and activations move between them with
ordinary tensor copies. There is no process group and no collective, so NCCL is
never entered. The `cuda/12.6` module is likewise redundant: `libtorch_cuda.so`
carries an `RPATH` pointing at the wheel's own bundled CUDA libraries, and
`RPATH` takes precedence over `LD_LIBRARY_PATH`, so the module's copies are not
picked up anyway. The host only has to supply the GPU driver, which needs no
module.

That last point is why `setup_environment.sh` and `tunnel.sbatch` load almost
nothing. It is deliberate, not an oversight.

## Notes on the model

`nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` is a hybrid: 88 layers mixing
Mamba2 state-space layers, a few attention layers, and a 512-expert MoE.

| | |
|---|---|
| Checkpoint on disk | 247 GB (230 GiB), 50 safetensors shards |
| Resident after load | ~225 GiB — the unused multi-token-prediction head is dropped |
| One GH200 | 95 GiB → does not fit |
| Two | 190 GiB → does not fit |
| **Four** | **380 GiB → fits** |

Three GPUs would technically fit, but `device_map="auto"` packs greedily and
would leave the fourth idle for no benefit. In the recorded run the model
occupied 62.8/67.1/67.1/52.1 GiB — **249 GiB in use, ~131 GiB free**, which is
the figure that matters for KV cache and activations. (The split reported as
24/25/24/18 counts entries in `hf_device_map`, which includes the embeddings,
final norm and LM head alongside the 88 transformer layers, so it sums to 91
rather than 88.) Loading took **111.5 seconds** from the shared Lustre
filesystem; expect longer when the filesystem is busy.

Two model-specific gotchas the notebook handles:

- **Reasoning is on by default.** The chat template opens a `<think>` block
  unless told otherwise, so a short prompt burns its whole token budget
  thinking out loud and never reaches an answer. Pass `enable_thinking=False`.
- **`enable_thinking` is a chat-template argument, not a `generate` argument.**
  Passing it (or `chat_template_kwargs`) to the pipeline raises
  `ValueError: The following model_kwargs are not used by the model`. Render
  the prompt with `apply_chat_template` first, then generate from the string.

You will also see a warning that the Mamba2 **"fast path is not available"**.
That is expected. The optional CUDA kernels (`mamba-ssm`, `causal-conv1d`) have
no ARM wheels and would need compiling from source; `transformers` falls back to
a correct pure-PyTorch implementation instead. For a playground that is the
right trade — and it is why the notebook loads the model with
`trust_remote_code` left off, using the native `transformers` implementation
rather than the checkpoint's bundled code, which hard-requires `mamba_ssm`.

## Limits worth knowing up front

- **Jobs are capped at 24 hours.** Hard. The job dies at `24:00:00` and takes
  your tunnel with it, mid-keystroke. Save to project storage, not to the node.
- **The cluster runs near capacity.** Idle nodes are rare and jobs queue.
  Smaller, shorter requests generally schedule sooner, so ask for 4 GPUs and 24
  hours only when you need them.
- **Check for a shared model cache before downloading.** If your site keeps one,
  the model you want may already be there, and the project storage quota is
  shared across everyone using it.

## Further reading

- [Isambard documentation](https://docs.isambard.ac.uk/)
- [The official VS Code tunnel guide](https://docs.isambard.ac.uk/user-documentation/guides/vscode/)
  that `tunnel.sbatch` is based on
