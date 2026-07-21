# Isambard onboarding sample

A worked example for teams new to [Isambard](https://docs.isambard.ac.uk/):
serve **gpt-oss-120b** on a GH200 node with **vLLM**, and evaluate it with the
UK AISI [Inspect](https://inspect.aisi.org.uk/) framework — the way you would
run a real evaluation.

It builds directly on the official
[distributed-inference tutorial](https://docs.isambard.ac.uk/user-documentation/tutorials/distributed-inference/),
adding an evaluation layer on top.

```
setup_login_node.sh     (login node)   uv + the VS Code CLI
tunnel.sbatch           (login node)   interactive VS Code session on a compute node
setup_compute_node.sh   (compute node) the GPU env: vLLM + Inspect venvs
serve_vllm.sh           (compute node) serve gpt-oss-120b as an OpenAI endpoint
llm_playground.ipynb    (compute node) point Inspect at that endpoint and evaluate
```

Setup is split by where it runs: `setup_login_node.sh` needs only the network,
while `setup_compute_node.sh` builds the GPU environment where a real driver is
visible (so `--torch-backend=auto` picks the right torch).

## The shape of it

Two pieces, deliberately separate:

```
bash serve_vllm.sh     →  runs in the foreground on a compute node: vLLM holds
                          the model on the GPUs and exposes an HTTP endpoint
                                        ↕ HTTP
llm_playground.ipynb   →  a thin client: Inspect sends eval requests, scores
                          the replies. Never touches a GPU.
```

The ~4-minute model load is paid once by the server, so you can iterate on
prompts and evals interactively against a resident model. They are also in
**separate venvs** — the serving and evaluation stacks have incompatible
dependencies (vLLM pins an older `huggingface-hub` than `inspect-evals` wants),
which is fine precisely because they only talk HTTP:

```
.venv        vLLM        (serve_vllm.sh)
.venv-eval   Inspect     (the notebook's kernel, and `inspect eval`)
```

## Quick start

```bash
# 1. On a LOGIN node -- install uv + the VS Code CLI.
bash setup_login_node.sh

# 2. Get an interactive session on a compute node.
sbatch tunnel.sbatch
tail -f logs/code_tunnel_<JOB_ID>.out   # GitHub device code, then a vscode.dev URL
```

Connect to the tunnel, then in a terminal on the compute node:

```bash
# 3. Build the GPU environment (two venvs). Needs the compute node's driver.
bash setup_compute_node.sh

# 4. Serve the model (foreground -- keep this terminal open, or use tmux).
bash serve_vllm.sh                      # wait for "Application startup complete"

# 5. From ANOTHER terminal, run the evals:
export ISAMBARD_BASE_URL=$(cat endpoint.txt) ISAMBARD_API_KEY=dummy
.venv-eval/bin/inspect eval inspect_evals/gsm8k \
    --model openai-api/isambard/gpt-oss-120b --limit 10
```

Or open `llm_playground.ipynb` and pick the **`.venv-eval`** kernel. Stop the
server with Ctrl-C when you are done — it holds four GPUs and the cluster runs
near capacity.

## The traps that catch people

**1. On ARM, your vLLM and PyTorch are probably the wrong build.** Isambard is
aarch64 with a CUDA-12 driver. Plain `pip install vllm` / `torch` on aarch64
now pulls **CUDA-13** wheels, which need a newer driver than the cluster has
and die at runtime with `libcudart.so.13` — on four healthy GPUs, with no hint
at install time. The install uses **`--torch-backend=auto`**, which detects the
driver and picks a compatible torch (here 2.9.x +cu129), and pulls vLLM from
its own wheel index. That one flag is the fix; it is straight from the official
tutorial.

**2. `LD_PRELOAD` survives `module purge`, and can't be undone from inside a
running process.** Isambard job scripts routinely export
`LD_PRELOAD=/tools/brics/.../libnccl.so`, forcing the system NCCL ahead of the
one bundled in the wheel; `import torch` then dies with
`undefined symbol: nccl...`. It is an environment variable, not a module, so
`module purge` won't clear it — and the loader applies it at `exec`, so
deleting it from `os.environ` affects only child processes. The scripts unset
it before starting anything.

**3. Caches must be somewhere you can write, and `HF_HOME` is the one knob.**
`datasets` (which Inspect uses to fetch benchmarks) derives its download
location from `HF_HOME`. Point it at a *shared* cache and the download dies
with `PermissionError` on another user's files. The serving script reads the
model **by path** from the shared BriCS cache (`/projects/public/brics/hf`, no
re-download), while dataset downloads go under your own `HF_HOME`.

## The model

`openai/gpt-oss-120b` — a 120B mixture-of-experts model, MXFP4-quantised to
~70 GB, so it fits comfortably on one node's four GH200s and loads in about
four minutes. It is served straight from the shared BriCS cache using the
tutorial's `GPT-OSS_Hopper.yaml` config. vLLM applies the chat template server
side; Inspect's `openai-api` provider drives it over the standard chat
completions API.

## Worth knowing

- **Jobs are capped at 24 hours.** The tunnel (and the server running inside it)
  dies at the job's walltime, and the endpoint goes with it.
- **The cluster runs near capacity.** Smaller, shorter requests schedule sooner.
- **Stop the server (Ctrl-C) when idle** — it holds four GPUs.

Based on the official
[distributed-inference](https://docs.isambard.ac.uk/user-documentation/tutorials/distributed-inference/)
and [VS Code tunnel](https://docs.isambard.ac.uk/user-documentation/guides/vscode/)
guides.
