# Mario_Ops

*Architecture by Russell, implementation accelerated with Vibe Coding.*

A containerized, cloud-native MLOps pipeline that trains a Double DQN agent to play **Super Mario Bros.**.

**A single codebase supports multiple deployment environments**: CPU in a GitHub Codespace for development, and GPU on AWS, Google Colab, or CoreWeave for training, with the deployment target selected entirely through environment variables.

**Parallel build:** Training runs multiple NES emulators (set with `N_ENVS`) in separate worker processes and overlaps CPU emulation with GPU learning.

---

## All in the cloud!

You do not need a GPU, an AWS account, Docker, or even a local Python installation to run this project. Simply fork or clone this repository, open a Codespace, and start training with a single command: `python train.py`.

---

## Quickstart: train in a GitHub Codespace

**Click the green `Code` button → `Codespaces` tab → `Create codespace on main`.**

Thanks to the convenience of `.devcontainer/devcontainer.json`, the Codespace automatically builds the development environment from `Dockerfile.develop` (Python 3.10, CPU-only PyTorch, FFmpeg, and all pinned dependencies) when it starts.

From there, run `tools/smoke_test.py` to verify that the environment is ready.

```bash
# Re-run the environment checks yourself at any time
python tools/smoke_test.py

```

When the terminal displays `all checks passed`, you know the environment has been verified end to end, including:

* a real NES environment,
* two `AsyncVectorEnv` workers, and
* a checkpoint save/load round-trip.

*Side note: I'm especially proud of this part. It brings back memories of how much effort it took to get the environment working at the beginning of this project!*

---

**You are now ready to train Mario!**

Paste the following into your Codespace terminal to run a short test training session:

```bash
N_ENVS=2 \
NUM_EPISODES=20 \
BURNIN=5 \
LOG_EVERY_EPISODES=5 \
EVAL_EVERY_EPISODES=10 \
EVAL_EPISODES=5 \
python train.py
```

This 20-episode test run will take approximately **5 minutes** on a **4-vCPU GitHub Codespace**.

When the test run completes, two new folders will appear inside `Mario_AWS/`:

```text
Mario_AWS/
├─ checkpoints_7_action_g99/
│  ├─ lineage.json
│  ├─ training_state.json
│  └─ mario_net_7_action.chkpt
├─ logs/
│  └─ training_7_action_g99.csv
```

---

## Repo layout

```text
Mario_Ops/
├─ .devcontainer/
│   └─ devcontainer.json
├─ Mario_AWS/
├─ Mario_Colab/
└─ Mario_CoreWave/
```
