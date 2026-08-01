# Mario_Ops

A containerized, cloud-native MLOps pipeline that trains a Double DQN
agent to play Super Mario Bros (World 1-1). Two images, one codebase:
CPU in a Codespace for development, GPU on AWS for real runs, switched
entirely by environment variables.

**Parallel build.** Training runs `N_ENVS` NES emulators in separate
worker processes and overlaps CPU emulation with GPU learning
(`step_async` -> learn -> `step_wait`). The policy, network, and
checkpoint format match the Colab notebook exactly -- checkpoints move
between Google Drive and S3 with zero renaming.

---

## Quickstart: train in a GitHub Codespace

You do not need a GPU, an AWS account, Docker, or a local Python
install to run this. Fork or clone the repo, open a Codespace, and
training starts in one command.

**Green `Code` button -> `Codespaces` tab -> `Create codespace on main`.**

The Codespace builds `Dockerfile.develop` (Python 3.10, CPU torch,
ffmpeg, all pinned deps), then runs `tools/smoke_test.py`
automatically. First creation takes ~3-5 minutes; later starts are
seconds. When the terminal shows `all checks passed`, the environment
is verified end to end -- a real NES env, two `AsyncVectorEnv` workers,
and a checkpoint round-trip.

Then just run it:

```bash
# Train with the defaults (100 episodes, auto N_ENVS, gamma=0.99)
python train.py
```

On a 4-core Codespace the defaults are heavier than you want for a
first look. This is the version to actually start with -- small buffer,
frequent logging, results in a couple of minutes:

```bash
# A short, honest run: 2 emulators, 200 episodes, ~170 MB replay buffer
N_ENVS=2 \
NUM_EPISODES=200 \
REPLAY_CAPACITY=3000 \
BURNIN=1000 \
LOG_EVERY_EPISODES=5 \
EVAL_EVERY_EPISODES=50 \
EVAL_EPISODES=3 \
python train.py
```

Watch for the `[rate]` line (steps/s) and the `[eval]` block (flag rate,
median reward, and the `x_pos` where each attempt died). Stop any time
with `Ctrl-C` -- the `finally` block writes a checkpoint before exiting,
and re-running `python train.py` resumes from it.

More things you can do immediately:

```bash
# Re-run the environment checks yourself at any time
python tools/smoke_test.py

# The same checks PLUS a short real training run that asserts a
# checkpoint and lineage stamp land on disk (a few minutes)
python tools/smoke_test.py verify env vec agent train

# Record a gameplay clip from the best checkpoint you have trained
python play.py

# Build the REAL production image from inside the Codespace
# (docker-in-docker is enabled; it runs CPU-only here, but it is a
# genuine check that the deploy image builds before you push to ECR)
docker build -f Dockerfile.deploy -t marioops .
```

**Do not expect throughput here.** A Codespace has no GPU and few
cores; it is for correctness, iteration, and reading logs. Clearing the
level takes GPU-hours -- that is what the AWS path below is for.

**Costs.** The Codespace itself uses your GitHub Codespaces quota
(free tier included on personal accounts). Nothing in this repo touches
AWS unless you set `MARIOOPS_S3_BUCKET`; with it unset the entire cloud
layer is inert and no credentials are needed.

### If you would rather run it locally

Same environment, no Codespace: install the
[Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
in VS Code, open the repo, and choose **Reopen in Container**. It builds
the identical `Dockerfile.develop`.

Or skip containers entirely -- but the Python version is not a free
choice. nes-py compiles a C++ extension whose build needs `distutils`,
removed in 3.12, and gym 0.25.2 predates 3.12 as well. **Use 3.10** to
match both images (3.11 also works if that is what you have):

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
sudo apt-get install -y build-essential ffmpeg   # nes-py compile; clip encoding
python tools/smoke_test.py
```

---

## The two images

| | `Dockerfile.develop` | `Dockerfile.deploy` |
|---|---|---|
| Target | Codespaces / Dev Containers | AWS EC2, ECS, Batch |
| Base | `devcontainers/python:3.10-bookworm` | `pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime` |
| torch | CPU wheel (~200 MB) | from the base, CUDA 12.1 |
| Size | ~1.5 GB | ~8 GB |
| Source | bind-mounted at runtime | `COPY`d in, explicit paths |
| User | `vscode` | `mario` (non-root) |
| Entrypoint | none (editor keeps it alive) | `CMD ["python", "train.py"]` |

They share exactly one file: `requirements.txt`. That is deliberate --
the same pins compiled the same way is what makes "it ran in the
Codespace" evidence about the deploy image, not just about a laptop.

```bash
docker build -f Dockerfile.deploy -t marioops .
```

```bash
# Local CPU run of the production image (no AWS, no credentials)
docker run --rm -v "$PWD/checkpoints_7_action_g99:/app/checkpoints_7_action_g99" \
  -v "$PWD/logs:/app/logs" marioops

# AWS GPU instance (g4dn.4xlarge) with S3 sync + CloudWatch metrics
docker run --rm --gpus all \
  -e MARIOOPS_S3_BUCKET=my-marioops-bucket \
  -e MARIOOPS_RUN_ID=gpu-run-01 \
  -e MARIOOPS_CLOUDWATCH=true \
  -e NUM_EPISODES=50000 -e N_ENVS=8 -e BURNIN=100000 \
  -e REPLAY_CAPACITY=200000 -e MAX_TRAIN_HOURS=8 \
  marioops

# Record a gameplay clip from the best checkpoint (prints a shareable URL)
docker run --rm --gpus all \
  -e MARIOOPS_S3_BUCKET=my-marioops-bucket \
  -e MARIOOPS_RUN_ID=gpu-run-01 \
  marioops python play.py
```

Every value in `config.py` is overridable with an environment variable
of the same name -- no rebuild needed to retune a run.

## Gamma is a lineage, not a knob

`GAMMA` sets the planning horizon (~1/(1-gamma) steps). The default is
now **0.99** (~100-step horizon), matching the active Colab lineage --
the gamma=0.9 agent's eval deaths were scattered across 15 zones with
no single killer obstacle, the signature of a policy that reacts but
cannot plan.

The two lineages are **not checkpoint-compatible**: raising gamma
rescales Q targets ~10x. So the checkpoint folder is derived from
gamma (`checkpoints_7_action/` for 0.9, `checkpoints_7_action_g99/`
for 0.99, mirroring the Colab Drive folders), `train.py` stamps
`lineage.json` into the folder and refuses to resume across a
mismatch, and `agent.load()` additionally verifies the action count
baked into the checkpoint's output layer (7-action `SIMPLE_MOVEMENT`
vs the old 2-action family).

A `mario_net_7_action_best.chkpt` downloaded from Google Drive drops
straight into the matching folder and is picked up with zero renaming.

## How many parallel environments?

Emulation is pure CPU work, so the speedup ceiling is the vCPU count.
`N_ENVS=0` (default) auto-detects `min(8, cpu_count)`. On a
`g4dn.4xlarge` (16 vCPU) start at 8 and measure; the 2-vCPU Colab T4
topped out at parity with serial (~66 steps/s), so the many-core AWS
box is where this build actually pays off. Epsilon decay, the replay
ratio, and target-sync cadence are all defined over *transitions
collected*, so nothing needs retuning when `N_ENVS` changes.

Every `LOG_EVERY_EPISODES` the run prints a `[rate]` line with
recent-window steps/s, lifetime average, and elapsed minutes -- that
is the number to tune `N_ENVS` against and to size an instance with.
Read the `avg` figure, not the window, and remember that eval wall
time lands in the denominator while contributing no steps, so compare
runs with the same `EVAL_EVERY_EPISODES`.

**Do not cap torch's threads on the CPU target.** It looks like free
throughput -- fewer BLAS threads competing with the emulator
processes -- and it is a 2.1x regression: measured on 8 cores with 8
envs, 35 steps/s at torch's default versus 16.5 at
`OMP_NUM_THREADS=1`. With `device=cpu` the learner's conv2d
forward/backward is real CPU work sitting synchronously between
`step_async()` and `step_wait()`, so throttling it makes it the
blocking stage. `train.py` picks the count from the resolved device
(1 on cuda, torch's default on cpu); override with `TORCH_THREADS=N`.

`REPLAY_CAPACITY=80000` (~4.51 GB, page-touched in full at startup) is
the safe local default; on a 64 GB `g4dn.4xlarge` raise it to
`200000` (~11.3 GB) to keep rare flag-run experience around longer.
In a Codespace, go the other way -- `3000` is plenty and allocates
instantly.

## Evaluation and the best checkpoint

Eval outcomes on this level are bimodal (~250 or ~2400), so the gate
is **flag rate first, median reward second**
(`score = flag_rate * 100000 + median`), over `EVAL_EPISODES=10`
episodes at `EVAL_EPSILON=0.02` with an `EVAL_STALL_STEPS=150`
no-forward-progress cutoff, and the best checkpoint is only
overwritten when the score clears `BEST_EVAL_MARGIN=1.05` -- at n=10
the binomial standard error on a 40% flag rate is +-15pp, and without
a margin you save luck. The eval log prints every episode's death
`x_pos`, which is how the gamma decision above was made.

Training and eval both use random no-op starts (`NOOP_MAX=30`);
without them the deterministic level lets the agent memorize one
trajectory (measured: eval flag rate ~8% -> 33% after adding them).
`play.py` records without no-ops so clips are deterministic showcases.

## Things that will bite you if you skip them

**Run-id scoping.** All S3 keys live under `MARIOOPS_RUN_ID`
(default `local-dev`, or `codespace-dev` in the dev container). A
machine can only restore checkpoints uploaded under the *same* run id
-- train on AWS as `gpu-run-01`, then run `play.py` locally with the
default, and it silently finds nothing. Set the same `MARIOOPS_RUN_ID`
everywhere that should share artifacts.

**Graceful-shutdown window.** On SIGTERM (spot reclaim, `docker stop`,
Batch termination) `train.py` writes and syncs a final checkpoint from
its `finally` block and reaps the emulator workers. Give it time:
`docker stop -t 60`, and in the AWS Batch / ECS container properties
set `stopTimeout: 60` (the default 30 s is usually enough, but a
multi-GB-buffer host under load may not be).

**Fork ordering.** `train.py` launches the env workers *before*
allocating the replay buffer. If you rearrange the startup code, keep
that ordering: forking after allocation roughly doubles the buffer's
physical footprint via copy-on-write (measured 1409 vs 662 MB on a
700 MB stand-in) and invites the OOM killer, which leaves no final
checkpoint.

**Cost guardrail.** `MAX_TRAIN_HOURS` (0 = off) checkpoints, syncs,
and exits cleanly at the budget instead of billing until someone
notices a runaway Batch job.

**Local persistence.** Checkpoints live in
`/app/checkpoints_7_action_g99` inside the container -- the folder is
gamma-derived, so mounting `checkpoints/` silently persists nothing.
Without the right mount or `MARIOOPS_S3_BUCKET`, they vanish with the
container. `train.py` now warns at startup when the checkpoint
directory is neither a mount point nor backed by S3.

**Codespace persistence.** A Codespace keeps its filesystem across
stops, so checkpoints survive -- but it is *deleted* after its
retention period (30 days idle by default). Anything you want to keep
belongs in S3 (`MARIOOPS_S3_BUCKET`) or committed, not in
`checkpoints_7_action_g99/`.

**Bind-mount ownership.** Docker creates a *missing* bind-mount path
on the host as `root:root`, but the deploy image runs as non-root
`mario` (uid 1000), so the first write fails. Create the directory
yourself before the first run:

```bash
mkdir -p checkpoints_7_action_g99 logs
# if Docker already created them as root:
sudo chown -R $(id -u):$(id -g) checkpoints_7_action_g99 logs
```

A startup write probe reports this as a plain, actionable error
instead of a `PermissionError` traceback from inside `lineage.json`.

**Buffer refill after a restart.** The replay buffer is deliberately
not checkpointed (it is multiple GB). Learning is gated on buffer
*length*, not `curr_step`, so a resumed run refills to `BURNIN`
transitions before gradients resume. On spot instances -- where
reclaims can happen repeatedly -- keep `BURNIN` generous (25k-50k) so
each restart rebuilds a decorrelated buffer rather than training on a
few seconds of highly correlated frames.

## Public video URLs

By default `play.py` prints a *presigned* URL for the uploaded clip --
it works on a fully private bucket and expires after
`MARIOOPS_S3_URL_EXPIRY` seconds (default and maximum: 7 days).

For permanent links, set `MARIOOPS_S3_PUBLIC_URLS=true` and attach a
bucket policy allowing anonymous reads on the runs/ prefix only
(checkpoints and logs stay private):

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "PublicReadGameplayClips",
      "Effect": "Allow",
      "Principal": "*",
      "Action": "s3:GetObject",
      "Resource": "arn:aws:s3:::MY-BUCKET/marioops/runs/*"
    }
  ]
}
```

(Also uncheck the two *policy*-related "Block public access" settings
on the bucket; leave the ACL ones on.)

## Repo layout

```text
Mario_Ops/
├── .devcontainer/
│   └── devcontainer.json
├─ Mario_AWS/
├─ Mario_Colab/
└─ Mario_CoreWave/
```
