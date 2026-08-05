# Mario_Colab — train a Double DQN Mario on a free Colab GPU

This folder is the **notebook lineage** of Mario_Ops: a single self-contained Colab notebook that trains a memory-augmented Double DQN agent on `SuperMarioBros-1-1`, checkpoints to Google Drive so 10-hour sessions can be chained together, and records the resulting gameplay as an `.mp4`.

It is deliberately standalone — no Docker, no AWS, no local Python. A browser and a Google account are enough.

---

## What's in this folder

| File | What it is |
| --- | --- |
| `Mario_Colab.ipynb` | The whole project: setup, environment, agent, training loop, and gameplay recorder. |
| `mario_net_best.chkpt.zip` | A trained checkpoint — **episode 18,800, 4.36 M environment steps, best eval mean reward ≈ 3021**. Unzip it into your Drive folder to skip the long grind. |
| `mario_play_FLAG_20260714.mp4` | Gameplay recorded from that checkpoint. `FLAG` in the filename means the agent finished the level. |

**You do not have to train anything.** Section [Using the supplied checkpoint](#using-the-supplied-checkpoint) covers both options: resume training from it, or just load it, play, and record your own video in a few minutes.

---

## 1. Setting up the Colab environment

Open `Mario_Colab.ipynb` in Google Colab (`File ▸ Upload notebook`, or open it straight from GitHub).

**Step 1 — pick a GPU runtime.**
`Runtime ▸ Change runtime type ▸ Hardware accelerator: T4 GPU`. PyTorch already ships in Colab with CUDA, so nothing is reinstalled for it.

**Step 2 — run the install cell** (the first code cell, under *Setup (Google Colab + T4 GPU)*):

```python
!pip install -q "numpy<2" "opencv-python<4.12" "gym==0.25.2" "gym-super-mario-bros==7.4.0"
```

Why the pins matter — this notebook builds on a 2020-era Gym stack that modern Colab breaks:

- **`numpy<2`** — `gym` and the `nes-py` NES emulator are unmaintained and still reference removed aliases such as `np.bool8`, which no longer exist in NumPy 2.0.
- **`gym==0.25.2`** — keeps the classic 4-tuple `step() -> (obs, reward, done, info)` and obs-only `reset()`. Gym ≥ 0.26 switched to a 5-tuple API that `nes-py` doesn't emit, which crashes inside Gym's own `TimeLimit` wrapper.
- **`opencv-python<4.12`** — an OpenCV build compiled against `numpy<2`.

**Step 3 — `Runtime ▸ Restart session`.**
This is not optional. NumPy was downgraded underneath a running kernel, and the downgrade only takes effect after a restart.

**Step 4 — run every cell below the restart marker, in order, top to bottom.**
The agent is built with a `class Mario(Mario)` subclassing chain, so each Mario cell layers onto the previous one. Skipping one, or mixing in cells from an older notebook, leaves a stale definition live in the kernel. The training cell has a sanity guard that catches exactly this and fails loudly rather than an hour into a run.

**Step 5 — mount Drive when prompted.**
The training cell calls `drive.mount('/content/drive')` and writes to `/content/drive/MyDrive/mario_rl/checkpoints`. Colab wipes local disk on disconnect, so Drive is what makes a run survive session cutoffs. The auth prompt is interactive — complete it when you start the cell; everything after that runs unattended.

RAM footprint is roughly 6 GB of Colab's ~12.7 GB (the replay buffer is preallocated at ~4.51 GB), so a standard runtime is fine — no high-RAM instance needed.

---

## 2. Notebook map — which cell does what

Cells are listed in notebook order. The ones you'll actually interact with are in **bold**.

| Section heading | Type | Purpose |
| --- | --- | --- |
| *Setup (Google Colab + T4 GPU)* | **code** | **STEP 1 — pip installs. Restart the session after this.** |
| *(after restart)* | code | STEP 2 — imports, prints NumPy/Torch versions and CUDA availability. |
| *Initialize Environment* | code | Builds `SuperMarioBros-1-1-v0` and applies the **reduced action space**. |
| *Preprocess Environment* | code | `SkipFrame(4)` → grayscale → resize 84×84 → scale to [0,1] → `FrameStack(4)`, giving a `[4, 84, 84]` state. |
| *Act* | code | Epsilon-greedy action selection and the exploration schedule. |
| *Cache and Recall* | code | `RingReplay` — the preallocated uint8 ring buffer (80,000 transitions) plus pinned staging tensors. |
| *Neural Network* | code | `MarioNet`: 3 conv layers → flatten → 2 dense layers, with frozen `target` copy. |
| *TD Estimate & TD Target* | code | Sets **`self.gamma = 0.9`** and defines the DDQN targets. |
| *Updating the model* | code | Adam (`lr=2.5e-4`), `SmoothL1Loss`, `sync_Q_target()`. |
| *Save checkpoint* | code | `save()` (atomic write) and `load()`. |
| *Putting it all together* | code | `learn()` — burn-in of 10k experiences, learn every 3 steps, sync target every 10k. |
| *Logging* | code | `MetricLogger` — appends to `log` and writes the reward/length/loss/Q plots. |
| ***Let's play!*** | **code** | **← THE TRAINING CELL.** |
| ***Record gameplay to Google Drive*** | **code** | **← THE PLAY / RECORDING CELL.** |

---

## 3. The training cell — *Let's play!*

This is the long-running one. It mounts Drive, builds the agent, resumes from a checkpoint if it finds one, and then loops.

What it does per session:

- **Budget:** stops cleanly at `MAX_TRAIN_HOURS = 9.5`, safely inside Colab's 10-hour background-execution cutoff, which would otherwise kill the kernel mid-episode with up to ~20 minutes unsaved. `episodes = 30000` is only a ceiling; wall-clock is what ends the session.
- **Checkpoints:** every 300 episodes to `mario_net.chkpt`, plus on time limit, low RAM, exception, or manual interrupt. Writes are atomic, so a crash mid-save never leaves a corrupt file.
- **Evaluation:** every 100 episodes it plays 3 near-greedy episodes (ε = 0.02) that don't consume training steps or touch the replay buffer. A tiny ε is used rather than pure greedy because this environment is deterministic — a fully greedy policy can deadlock, pushing against a pipe until the 400-second level timer expires. Whenever the mean beats the previous best, it saves a *separate* `mario_net_best.chkpt`, so a later dip in training can never destroy the best policy found.
- **RAM watchdog:** `psutil` + `malloc_trim`. When Colab exhausts system RAM the OS kills the kernel outright — Python never sees an exception, so a `try/except` cannot save you. The watchdog checkpoints and stops cleanly before that point.

**To resume, just re-run this same cell in a new session.** It reloads the model, optimizer, current step, epsilon, and best-eval score, and picks up where it left off. Throughput is roughly 66 steps/s with episodes averaging ~160 steps, so one 9.5-hour session covers about 2.4 M steps.

> The replay buffer is *not* checkpointed — it starts empty on resume. This is handled: `learn()` gates on how full the buffer actually is rather than on `curr_step`, so a resumed run refills the burn-in before it starts learning instead of crashing on an empty buffer.

---

## 4. The play / recording cell — *Record gameplay to Google Drive*

Loads `mario_net_best.chkpt` (falling back to `mario_net.chkpt`), plays up to 5 near-greedy episodes, and saves the best attempt — a flag run if one happens, otherwise the furthest — as an H.264 `.mp4` in the Drive checkpoint folder, named like `mario_play_FLAG_20260714_101533.mp4`.

- **You only need the environment and `MarioNet` cells first — training is not required.** It never touches training state or the replay buffer, so it's safe to run any time, including immediately after a training session in the same runtime.
- ε = 0.02 during playback, same as evaluation, for the anti-deadlock reason above.
- Frames are the raw 240×256 NES screen captured once per agent step (= 4 emulated frames), so 15 fps plays back at authentic game speed. A full flag run is about 90 seconds of video.
- The file is re-encoded with Colab's `ffmpeg` to H.264/yuv420p so it previews directly in Google Drive and plays in any browser or phone. If `ffmpeg` fails, the mp4v file is kept rather than losing the run.

`mario_play_FLAG_20260714.mp4` in this folder is exactly this cell's output.

---

## 5. The reduced action space

The environment is wrapped with only two actions:

```python
env = JoypadSpace(env, [["right"],
                        ["right", "A"]])
```

**0 = walk right, 1 = jump right.** That's the entire action set — no left, no B (run/fireball), no crouch.

Why: the NES controller yields 256 raw button combinations, and `gym-super-mario-bros` ships `SIMPLE_MOVEMENT` (7 actions) and `COMPLEX_MOVEMENT` (12). Every extra action widens the output layer, multiplies what the ε-greedy policy has to explore, and dilutes the replay buffer with combinations that are never useful for World 1-1. Two actions are enough to clear the level and make the credit assignment problem dramatically easier on a free-tier GPU budget.

**Consequence for checkpoints:** the final layer is `Linear(512, 2)`. A checkpoint from this notebook is **not** loadable into a 7-action agent (the `Mario_AWS` lineage) — it fails with a shape mismatch. Action-space size is part of the checkpoint lineage, not a tunable.

---

## 6. What gamma means here

```python
self.gamma = 0.9
```

Gamma (γ) is the **discount factor** in the TD target:

$$TD_t = r + \gamma \cdot Q_{target}(s', \arg\max_a Q_{online}(s', a))$$

It sets how much a future reward is worth relative to an immediate one. A reward *n* steps ahead is scaled by γⁿ, which gives an effective **planning horizon of roughly 1/(1−γ) steps**:

| γ | Horizon | In game time (1 agent step = 4 NES frames) |
| --- | --- | --- |
| **0.90** *(this notebook)* | ~10 steps | ~0.67 s |
| 0.99 | ~100 steps | ~6.7 s |

At γ = 0.9 the agent is a **reactor**: it optimizes for what happens in the next two-thirds of a second. That is enough to jump a Goomba it can see, but not enough to plan a running approach to a wide gap. Diagnostics on this lineage showed eval deaths scattered across 15 distinct zones with no single zone accounting for more than 13% of failures — the signature of an agent that responds well locally but cannot plan. The parallel `Mario_AWS` lineage moved to γ = 0.99 for that reason.

**Gamma is a lineage switch, not a casual knob.** Raising it rescales Q targets by roughly 10×, so a γ = 0.9 checkpoint is meaningless to a γ = 0.99 agent. If you change gamma, start a fresh checkpoint folder.

---

## 7. Using the supplied checkpoint

`mario_net_best.chkpt.zip` unzips to `mario_net_best.chkpt` (~27 MB), containing the model weights, optimizer state, exploration rate, step count, episode count, and best-eval score.

**What's in it:**

| | |
| --- | --- |
| Episode | 18,800 |
| Environment steps | 4,360,352 |
| Exploration rate | 0.05 (already at the floor) |
| Best eval mean reward | 3021.3 |
| Lineage | 2-action, γ = 0.9 |

That's roughly 18 hours of pure stepping at the measured ~66 steps/s — closer to 30 hours of real session time once evaluations, restarts, and re-mounts are counted, spread across several 9.5-hour blocks. You can skip all of it.

### Option A — just play and record (no training)

1. Unzip and upload `mario_net_best.chkpt` to `/content/drive/MyDrive/mario_rl/checkpoints/`. **Keep the filename as-is.**
2. Run the setup cell, restart, then run the imports, environment, wrapper, and `MarioNet` cells.
3. Skip the training cell entirely.
4. Run the *Record gameplay to Google Drive* cell.

A few minutes later you have your own `.mp4` in Drive. The recorder tries up to 5 episodes and stops early on a flag run.

### Option B — continue training from it

1. Unzip it, then upload it to `/content/drive/MyDrive/mario_rl/checkpoints/` **renamed to `mario_net.chkpt`.**
2. Run all cells in order, including the training cell.

The rename matters: `load()` only looks for `mario_net.chkpt` — the rolling checkpoint. `mario_net_best.chkpt` is the protected best-so-far copy that only the evaluation path writes. If you want both behaviours, upload the file twice under both names; then training resumes from it *and* `best_eval = 3021` is carried forward, so a new "best" is only recorded if it genuinely beats what's here.

Note that resuming picks up at ε = 0.05, i.e. near-greedy from step one, and the replay buffer refills from scratch during burn-in.

---

## Exploration schedule (a deviation from the reference tutorial)

`exploration_rate_decay = 0.9999975`, `exploration_rate_min = 0.05`.

The PyTorch tutorial's 0.99999975 needs ~9.2 M steps (~39 h at 66 steps/s) just to reach ε = 0.1 — the agent stays >90% random for entire sessions and mean reward never moves. This value decays 10× faster, reaching ε = 0.1 in ~0.9 M steps (~4 h). The floor was lowered from 0.1 to 0.05 because once the policy is decent, 10% random actions is expensive — a single bad jump ends a run. ε is stored in the checkpoint, so the schedule continues seamlessly across resumes.

---

## What lands in your Drive folder

```
/content/drive/MyDrive/mario_rl/checkpoints/
├─ mario_net.chkpt          # rolling — this is what resume reads
├─ mario_net_best.chkpt     # best eval mean — this is what the recorder reads
├─ log                      # appended across sessions
├─ reward_plot.jpg
├─ length_plot.jpg
├─ loss_plot.jpg
├─ q_plot.jpg
└─ mario_play_*.mp4         # recorded gameplay
```

---

## Cost and time expectations

Training runs unattended in the background on Colab Pro, which is what makes the 9.5-hour session budget usable. 100 compute units cost about $10 and cover roughly nine sessions. Starting from the supplied checkpoint instead of from scratch means most of that budget goes toward improvement rather than re-earning ground already covered — or you skip training entirely and just record gameplay on the free tier.

---

## Credits

Based on **Train a Mario-playing RL Agent** ([PyTorch tutorial](https://docs.pytorch.org/tutorials/intermediate/mario_rl_tutorial.html)) by Yuansong Feng, Suraj Subramanian, Howard Wang, and Steven Guo — [original code](https://github.com/yuansongFeng/MadMario/). The algorithm is [Double DQN](https://arxiv.org/pdf/1509.06461.pdf).

Modifications in this notebook: the preallocated ring-buffer replay memory (fixing unbounded heap growth from the deque-of-arrays original), Drive-backed resumable checkpointing with a separate best-eval copy, the retuned exploration schedule, the near-greedy evaluation protocol, the wall-clock session budget and RAM watchdog, and the gameplay recorder.
