"""
MarioOps configuration.

Every value can be overridden with an environment variable of the same
name, so the same container image can be re-tuned per training job
(locally, on EC2, or in AWS Batch) without rebuilding.

Example:
    docker run -e NUM_EPISODES=5000 -e N_ENVS=4 marioops:latest

NOTE: the corrected mario_agent package owns its own preprocessing
constants (mario_agent/config.py). IMAGE_SIZE / STACK_SIZE / FRAME_SKIP
are re-exported from there so the training scripts and the package can
never disagree about frame shapes.
"""

import os

# Single source of truth for preprocessing shape -- comes from the package.
from mario_agent.config import IMAGE_SIZE, STACK_SIZE, FRAME_SKIP  # noqa: F401


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def _str(name: str, default: str) -> str:
    return os.getenv(name, default)


def _bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
ENV_NAME = _str("ENV_NAME", "SuperMarioBros-1-1-v0")

# Compute device:
#   "auto" -> use CUDA if a GPU is visible (AWS GPU instance with
#             `docker run --gpus all`), otherwise CPU. This is the
#             right default: the SAME image is CPU locally, GPU on AWS.
#   "cpu"  -> force CPU (deterministic local tests even on a GPU box)
#   "cuda" -> force GPU (fail fast if the container can't see one)
DEVICE = _str("MARIOOPS_DEVICE", "auto")

# ---------------------------------------------------------------------------
# Parallel environments (ported from the Colab parallel notebook)
# ---------------------------------------------------------------------------
# N worker processes each own a NES emulator; the main process batches
# their observations into one forward pass and overlaps GPU learning
# with emulation via step_async/step_wait. Emulation is pure-CPU work,
# so the speedup ceiling is the vCPU count: a g4dn.4xlarge (16 vCPU)
# has real headroom here where the 2-vCPU Colab T4 did not.
#   0 (default) -> auto: min(8, cpu_count), leaving throughput to the
#                  probe below if you want to tune it
# On a g4dn.4xlarge start with N_ENVS=8 and measure; going past the
# point where emulators saturate the vCPUs just adds IPC overhead.
N_ENVS = _int("N_ENVS", 0)

# Torch intra-op thread count.
#   0 (default) -> chosen from the resolved DEVICE at runtime:
#        cuda: 1, because the learner's matmuls run on the device and
#              every vCPU should be serving an emulator instead.
#        cpu:  torch's own default (one per core) -- the learner IS
#              CPU compute there and sits synchronously in the step
#              loop, so starving it stalls the whole pipeline.
#              MEASURED, 8 cores / 8 envs: 35 steps/s default vs 16.5
#              at one thread. Do not "optimize" this to 1.
#   >0        -> use exactly this many, on either device.
TORCH_THREADS = _int("TORCH_THREADS", 0)

# Random no-op starts (NoopResetEnv). World 1-1 is fully deterministic;
# without this the agent memorizes ONE action sequence instead of a
# policy. Measured effect on Colab: eval flag rate ~8% -> 33%, and the
# x~310 / x~700 stall clusters vanished. 0 disables (play.py always
# records deterministically, without no-ops).
NOOP_MAX = _int("NOOP_MAX", 30)

# ---------------------------------------------------------------------------
# DQN hyperparameters (applied onto the agent via configure_agent())
# ---------------------------------------------------------------------------
# GAMMA IS A LINEAGE SWITCH, not a casual knob. It sets the planning
# horizon (~1/(1-gamma) steps):
#   0.90 -> ~10 steps  = 0.67 s of game time
#   0.99 -> ~100 steps = 6.7 s
# The 15-zone spread of eval deaths in the gamma=0.9 lineage (largest
# single zone only 13% of failures) showed an agent that reacts but
# cannot plan -- hence 0.99 is now the default, matching the active
# Colab lineage. THE TWO LINEAGES ARE NOT CHECKPOINT-COMPATIBLE:
# raising gamma rescales Q targets ~10x, so each lineage gets its own
# checkpoint folder (derived below) and train.py stamps/verifies
# lineage.json before resuming.
GAMMA = _float("GAMMA", 0.99)

LEARNING_RATE = _float("LEARNING_RATE", 0.00025)
BATCH_SIZE = _int("BATCH_SIZE", 64)

# 80k transitions ~= 4.51 GB of preallocated uint8 ring buffer -- the
# value the level-clearing Colab agent actually trained with, and a
# safe default for small local machines. The buffer is page-touched IN
# FULL at startup, so the cost lands immediately, not gradually.
# On a g4dn.4xlarge (64 GB RAM) raise it: REPLAY_CAPACITY=200000
# (~11.3 GB) roughly triples the retention of rare flag-run experience.
REPLAY_CAPACITY = _int("REPLAY_CAPACITY", 80_000)

# Episode ceiling. The wall-clock budget (MAX_TRAIN_HOURS) is usually
# what actually ends a cloud run; this is the backstop.
NUM_EPISODES = _int("NUM_EPISODES", 100)   # raise to 10000+ for a full run

# Wall-clock budget in hours; 0 disables. On Colab this dodged the 10 h
# cutoff; on AWS Batch it is a cost guardrail -- the run checkpoints,
# syncs, and exits cleanly instead of billing until someone notices.
MAX_TRAIN_HOURS = _float("MAX_TRAIN_HOURS", 0.0)

# Epsilon decays PER TRANSITION COLLECTED inside the agent -- with N
# parallel envs act_batch() applies decay**N per iteration, which is
# the identical schedule over experience (N actors reach a given
# epsilon after the same number of transitions, in 1/N the wall time).
#
# DECAY LESSON (from real Colab training logs): the tutorial's
# 0.99999975 reaches epsilon=0.1 only after ~9.2M steps -- the agent
# stays >90% random for entire multi-hour sessions with no meaningful
# learning. 0.9999975 (one less nine) reaches the floor in ~0.9M steps
# and is the value the level-clearing agent was actually trained with.
#
# FLOOR 0.02 (was 0.05): ~11 random actions per ~230-step episode was
# free insurance at a 1% flag rate and actively destroys good runs at
# ~37%. Applied as a floor on resume too -- load() restores the saved
# exploration_rate but never below EPSILON_MIN.
EPSILON_START = _float("EPSILON_START", 1.0)
EPSILON_MIN = _float("EPSILON_MIN", 0.02)
EPSILON_DECAY = _float("EPSILON_DECAY", 0.9999975)

# Agent scheduling (all in environment STEPS / transitions collected):
#   BURNIN                  min cached experiences before learning starts.
#                           Package default is 100_000; lowered here so the
#                           default local smoke run actually trains. Raise
#                           for full runs.
#   LEARN_EVERY             transitions per Q_online gradient update. The
#                           parallel loop holds this replay ratio constant
#                           via fractional "gradient debt" -- N envs owe
#                           N/LEARN_EVERY updates per iteration.
#   TARGET_SYNC_EVERY_STEPS transitions between online -> target syncs.
#                           The parallel loop syncs on a step DELTA, not a
#                           modulo -- curr_step advances by N and can step
#                           clean over a multiple, silently skipping syncs.
#   AGENT_SAVE_EVERY_STEPS  the agent's own step-numbered checkpoints
#                           (serial learn() path only; the parallel loop
#                           uses the fixed-path cloud checkpoint instead).
BURNIN = _int("BURNIN", 10_000)
LEARN_EVERY = _int("LEARN_EVERY", 3)
TARGET_SYNC_EVERY_STEPS = _int("TARGET_SYNC_EVERY_STEPS", 10_000)
AGENT_SAVE_EVERY_STEPS = _int("AGENT_SAVE_EVERY_STEPS", 500_000)

# ---------------------------------------------------------------------------
# Evaluation / best-checkpoint policy (mirrors the Colab notebook)
# ---------------------------------------------------------------------------
# Eval outcomes on this level are BIMODAL (~250 or ~2400), so a
# 3-episode mean measured the coin, not the agent -- 10 episodes, and
# the best-checkpoint gate now scores FLAG RATE FIRST, MEDIAN REWARD
# SECOND (score = flag_rate * 100000 + median). Pure greedy (eps=0)
# was tested and reverted: no-op starts randomize only the start, and
# the trajectory re-converges before the pipe at x~594 -- one eval
# returned 594 eight times out of ten. EVAL_STALL_STEPS ends an
# episode when x_pos stops advancing: a deadlock IS a failure, and
# deadlocked evals were costing 3-5 minutes each. BEST_EVAL_MARGIN
# stops the best checkpoint being overwritten by binomial luck (at
# n=10 the standard error on a 40% flag rate is +-15pp).
EVAL_EVERY_EPISODES = _int("EVAL_EVERY_EPISODES", 100)
EVAL_EPISODES = _int("EVAL_EPISODES", 10)
EVAL_EPSILON = _float("EVAL_EPSILON", 0.02)
EVAL_STALL_STEPS = _int("EVAL_STALL_STEPS", 150)
EVAL_MAX_STEPS = _int("EVAL_MAX_STEPS", 3000)
BEST_EVAL_MARGIN = _float("BEST_EVAL_MARGIN", 1.05)

# Console/memory-report cadence of the parallel loop, in completed episodes.
LOG_EVERY_EPISODES = _int("LOG_EVERY_EPISODES", 20)

# RAM watchdog thresholds (GB free). Below LOW: malloc_trim + precaution
# checkpoint. Below CRITICAL: checkpoint, sync, and stop cleanly --
# because when the OS OOM-kills the process there is no Python
# exception and no final checkpoint.
LOW_RAM_GB = _float("LOW_RAM_GB", 1.5)
CRITICAL_RAM_GB = _float("CRITICAL_RAM_GB", 0.75)

# ---------------------------------------------------------------------------
# Local artifact paths -- LINEAGE-AWARE, matching the Colab Drive layout
# ---------------------------------------------------------------------------
# Folder and file names carry the "7_action" lineage tag plus a gamma
# suffix, exactly like the Colab notebook's Drive folders:
#   gamma=0.90 -> checkpoints_7_action/     mario_net_7_action.chkpt
#   gamma=0.99 -> checkpoints_7_action_g99/ mario_net_7_action.chkpt
# so a checkpoint downloaded from Drive drops into the matching folder
# with ZERO renaming, and two incompatible lineages can never collide.
# The action count is baked into the net's output layer (old 2-action
# checkpoints fail with a size mismatch), and gamma rescales Q targets
# ~10x -- both are enforced at load time, not just by naming.
_LINEAGE_SUFFIX = "" if abs(GAMMA - 0.9) < 1e-9 else f"_g{int(round(GAMMA * 100))}"

CHECKPOINT_DIR = _str("CHECKPOINT_DIR", f"checkpoints_7_action{_LINEAGE_SUFFIX}")
CHECKPOINT_FILE = _str(
    "CHECKPOINT_FILE", os.path.join(CHECKPOINT_DIR, "mario_net_7_action.chkpt")
)
BEST_CHECKPOINT_FILE = _str(
    "BEST_CHECKPOINT_FILE",
    os.path.join(CHECKPOINT_DIR, "mario_net_7_action_best.chkpt"),
)
TRAINING_STATE_FILE = _str(
    "TRAINING_STATE_FILE", os.path.join(CHECKPOINT_DIR, "training_state.json")
)
LINEAGE_FILE = _str("LINEAGE_FILE", os.path.join(CHECKPOINT_DIR, "lineage.json"))

LOGS_DIR = _str("LOGS_DIR", "logs")
TRAINING_LOG_FILE = _str(
    "TRAINING_LOG_FILE",
    os.path.join(LOGS_DIR, f"training_7_action{_LINEAGE_SUFFIX}.csv"),
)

RUNS_DIR = _str("RUNS_DIR", "runs")

# Gameplay recording (play.py)
#   PLAY_EPSILON       near-greedy floor; pure greedy (0.0) has
#                      deadlocked against pipes -- keep a tiny epsilon
#   PLAY_EPISODES      attempts recorded; the best one becomes the clip
#   PLAY_MAX_STEPS     per attempt; a flag run is ~1400 agent steps, so
#                      leave generous headroom
#   VIDEO_FPS          frames are captured once per agent step (= 4
#                      emulated frames via SkipFrame); NES runs 60 fps,
#                      so 15 fps plays back at authentic game speed
#   SAVE_EVERY_N_STEPS cadence of the annotated PNG highlight stills
PLAY_EPSILON = _float("PLAY_EPSILON", 0.02)
PLAY_EPISODES = _int("PLAY_EPISODES", 5)
PLAY_MAX_STEPS = _int("PLAY_MAX_STEPS", 3000)
VIDEO_FPS = _int("VIDEO_FPS", 15)
SAVE_EVERY_N_STEPS = _int("SAVE_EVERY_N_STEPS", 10)

# ---------------------------------------------------------------------------
# AWS / MLOps settings (all optional -- leave unset for a purely local run)
# ---------------------------------------------------------------------------
# S3 bucket for checkpoints + gameplay clips. If empty, S3 sync is disabled.
S3_BUCKET = _str("MARIOOPS_S3_BUCKET", "")
S3_PREFIX = _str("MARIOOPS_S3_PREFIX", "marioops")

# Logical name for this training run; used in S3 keys and CloudWatch dimensions.
RUN_ID = _str("MARIOOPS_RUN_ID", "local-dev")

# Push per-episode metrics (reward, loss, epsilon, steps) to CloudWatch.
CLOUDWATCH_ENABLED = _bool("MARIOOPS_CLOUDWATCH", False)
CLOUDWATCH_NAMESPACE = _str("MARIOOPS_CW_NAMESPACE", "MarioOps")

# How often (in completed episodes) to save the checkpoint and sync to S3.
SYNC_EVERY_N_EPISODES = _int("MARIOOPS_SYNC_EVERY", 5)

# Shareable URL for uploaded gameplay videos:
#   default              -> presigned URL, expires after S3_URL_EXPIRY_SECONDS
#                           (604800 s = 7 days, the presigned-URL maximum)
#   MARIOOPS_S3_PUBLIC_URLS=true -> permanent public object URL; requires a
#                           bucket policy allowing public GetObject on the
#                           runs/ prefix (see README "Public video URLs")
S3_URL_EXPIRY_SECONDS = _int("MARIOOPS_S3_URL_EXPIRY", 604_800)
S3_PUBLIC_URLS = _bool("MARIOOPS_S3_PUBLIC_URLS", False)

# Also upload the individual PNG frames of each run (off by default;
# the MP4 + summary are usually all you want in S3).
UPLOAD_FRAMES = _bool("MARIOOPS_UPLOAD_FRAMES", False)

AWS_REGION = _str("AWS_REGION", _str("AWS_DEFAULT_REGION", "us-east-1"))
