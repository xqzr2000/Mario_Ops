# Mario_Colab

**Train a Double DQN agent to play Mario on Google Colab's free T4GPU**

This folder is the **notebook lineage** of Mario_Ops: a single self-contained Colab notebook that trains a memory-augmented Double DQN agent on `SuperMarioBros-1-1`, checkpoints to Google Drive so seperate sessions can be chained together, and records the resulting gameplay as an `.mp4`.

It is **All in the Cloud**, no Docker, no AWS, no local Python. A browser and a Google account are enough.

---

## What's in this folder

```text
Mario_Colab/
├─ Mario_Colab.ipynb              # Complete project notebook: setup, environment configuration,
|                                 # agent implementation, training loop, and gameplay recording.
|
├─ README.md                      # Project overview and usage instructions.
|
├─ mario_net_best.chkpt.zip       # Trained model checkpoint. Unzip this file into your Google Drive folder
|                                 # to continue training, or skip the grind and run the Mario agent.
|
└─ mario_play_FLAG_20260714.mp4   # Gameplay recording generated from the checkpoint. 
                                  # "FLAG" indicates that the agent successfully completed the level.
```

---

## Using the supplied checkpoint: `mario_net_best.chkpt.zip`

**What's in it:**  roughly 18 hours of training

| | |
| --- | --- |
| Episode | 18,800 |
| Environment steps | 4,360,352 |
| Exploration rate | 0.05 (already at the floor) |
| Best eval mean reward | 3021.3 |
| Lineage | 2-action, γ = 0.9 |

### Option A: just play and record (no training)

1. Unzip and upload `mario_net_best.chkpt` to `/content/drive/MyDrive/mario_rl/checkpoints/`. **Keep the filename as-is.**
2. Run the setup cell, restart, then run the imports, environment, wrapper, and `MarioNet` cells.
3. Skip the training cell entirely.
4. Run the *Record gameplay to Google Drive* cell.

A few minutes later you have your own `.mp4` in Drive. The recorder tries up to 5 episodes and stops early on a flag run.

### Option B: continue training from it

1. Unzip it, then upload it to `/content/drive/MyDrive/mario_rl/checkpoints/` **renamed to `mario_net.chkpt`.**
2. Run all cells in order, including the training cell.

The rename matters: `load()` only looks for `mario_net.chkpt` — the rolling checkpoint. `mario_net_best.chkpt` is the protected best-so-far copy that only the evaluation path writes. If you want both behaviours, upload the file twice under both names; then training resumes from it *and* `best_eval = 3021` is carried forward, so a new "best" is only recorded if it genuinely beats what's here.

### What lands in your Google Drive folder

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
