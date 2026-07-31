"""
Mario_AWS training entrypoint -- PARALLEL build, ported from the Colab
notebook (Colab_G_Drive_Mario_7_action_parallel, gamma=0.99 lineage).

N_ENVS worker processes each own a NES emulator; this process batches
their observations into single forward passes and overlaps GPU
learning with emulation via step_async/step_wait. Measured on Colab,
the blocking venv.step() version was SLOWER than serial (56.7 vs ~66
steps/s at N=2) because it paid the IPC cost without ever overlapping;
the async split recovers it, and on a many-vCPU AWS instance the
parallelism finally has room to pay off.

Checkpoint policy (matches the Colab notebook):
  * PERIODIC  -- rolling checkpoint + training state + S3 sync every
                 SYNC_EVERY_N_EPISODES completed episodes
  * BEST      -- every EVAL_EVERY_EPISODES episodes, run EVAL_EPISODES
                 near-greedy evaluation episodes with stall cutoff;
                 the score is FLAG RATE FIRST, MEDIAN REWARD SECOND
                 (eval outcomes are bimodal, so a mean measures luck),
                 and the best checkpoint is only overwritten when the
                 score clears BEST_EVAL_MARGIN -- later training dips
                 or binomial luck can never destroy the best policy
  * FINAL     -- on completion, exception, Ctrl-C, or SIGTERM (spot
                 interruption / Batch job stop), the finally block
                 writes and syncs a consistent latest checkpoint

Lineage guards (both fail fast with a plain-English error):
  * gamma      -- lineage.json is stamped into CHECKPOINT_DIR; resuming
                  with a different GAMMA is refused (Q targets differ
                  ~10x between 0.9 and 0.99)
  * action set -- agent.load() reads the action count baked into the
                  checkpoint's output layer before loading
"""

import csv
import json
import signal
import time
from pathlib import Path

import numpy as np
import torch
import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace

# gym/logger.py runs warnings.simplefilter("once", DeprecationWarning) at
# IMPORT time, which prepends to the filter list and therefore overrides
# anything PYTHONWARNINGS installed -- you cannot silence gym's own
# deprecation spam from outside the process. logger.warn() and
# logger.deprecation() both gate on `min_level <= WARN` before calling
# warnings.warn, so raising the level shuts them off at the source.
# Set BEFORE make_vec_env(): the workers are forked and inherit this.
gym.logger.set_level(gym.logger.ERROR)

# Back to gym's default SIMPLE_MOVEMENT
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

# LOOK, import my own package
from mario_agent import MarioAgent
from mario_agent.data_pipeline import build_env
from mario_agent.vector_env import make_vec_env, parse_vector_infos

# another self-defined package for AWS S3 cloud storage and cloud watch
from cloud.storage import S3Storage
from cloud.monitoring import TrainingMetrics

# seperate, self-defined setting file
from config import (
    ENV_NAME,
    STACK_SIZE,
    IMAGE_SIZE,
    N_ENVS,
    TORCH_THREADS,
    NOOP_MAX,
    NUM_EPISODES,
    MAX_TRAIN_HOURS,
    GAMMA,
    LEARNING_RATE,
    BATCH_SIZE,
    REPLAY_CAPACITY,
    EPSILON_START,
    EPSILON_MIN,
    EPSILON_DECAY,
    BURNIN,
    LEARN_EVERY,
    TARGET_SYNC_EVERY_STEPS,
    AGENT_SAVE_EVERY_STEPS,
    CHECKPOINT_DIR,
    CHECKPOINT_FILE,
    BEST_CHECKPOINT_FILE,
    TRAINING_STATE_FILE,
    LINEAGE_FILE,
    LOGS_DIR,
    TRAINING_LOG_FILE,
    SYNC_EVERY_N_EPISODES,
    RUN_ID,
    S3_BUCKET,
    DEVICE,
    EVAL_EVERY_EPISODES,
    EVAL_EPISODES,
    EVAL_EPSILON,
    EVAL_STALL_STEPS,
    EVAL_MAX_STEPS,
    BEST_EVAL_MARGIN,
    LOG_EVERY_EPISODES,
    LOW_RAM_GB,
    CRITICAL_RAM_GB,
)


# This starts the game environment, and wrap it with data pipeline from
# mario_agent package. MODULE-LEVEL on purpose: AsyncVectorEnv sends it
# to worker processes, and it must be importable there. Training and
# evaluation both use random no-op starts (NOOP_MAX) -- without them
# World 1-1's determinism lets the agent memorize one trajectory, and
# an eval env WITHOUT them would keep measuring that single memorized
# path (both observed on Colab).
def make_env():
    """Raw NES env -> discrete joypad -> full preprocessing pipeline.

    disable_env_checker=True is a THROUGHPUT change, not just a quiet
    one: gym's passive checker re-validates the observation against
    its space on every single step(), in every worker, for the whole
    run. gym_super_mario_bros.make is literally `make = gym.make`, so
    the kwarg passes straight through to the registration machinery.
    """
    # Redundant with the module-level call under fork, but correct
    # under spawn too, and free.
    gym.logger.set_level(gym.logger.ERROR)
    env = gym_super_mario_bros.make(ENV_NAME, disable_env_checker=True)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    return build_env(env, noop_max=NOOP_MAX)


def resolve_n_envs() -> int:
    """N_ENVS from config, or auto-detect from the vCPU count.

    Emulation is pure CPU work, so the ceiling is the vCPU count; the
    auto default caps at 8 to leave the learner and OS some cores. On
    a g4dn.4xlarge (16 vCPU) that is a sensible start -- measure and
    override with N_ENVS if you want to tune it.
    """
    import os
    if N_ENVS > 0:
        return N_ENVS
    return max(1, min(8, os.cpu_count() or 2))


# ------------------------------------------------------------------------------
# This allows you to set the configuration directly from the terminal
# (e.g., using 'set' on Windows or 'export' on Linux) before running train.py,
# enabling workflow tuning without modifying the package source code.
# ------------------------------------------------------------------------------
def configure_agent(agent: MarioAgent) -> None:
    """
    Apply env-var-tunable hyperparameters on top of the package defaults.

    NOTE: called BEFORE agent.load(), so a resumed checkpoint still
    restores its own exploration_rate and curr_step afterwards -- but
    NOT exploration_rate_min, so a lowered floor takes effect on
    resume (main() re-clamps after load()).

    MEMORY NOTE: REPLAY_CAPACITY is passed to the MarioAgent
    CONSTRUCTOR in main(), not applied here. Do NOT assign a new
    buffer to agent.memory -- replacing the preallocated RingReplay
    (e.g. with a deque) would break recall() and reintroduce the heap
    fragmentation the ring buffer exists to prevent.
    """
    agent.gamma = GAMMA

    # The RingReplay staging arrays and the pinned host-to-device
    # tensors are sized for the batch size at construction time. If the
    # configured batch size differs, resize ONLY that small staging
    # machinery (~a few MB) -- the big ring buffer itself is
    # batch-size-independent and stays untouched.
    if BATCH_SIZE != agent.batch_size:
        agent.batch_size = BATCH_SIZE
        shape = (BATCH_SIZE,) + tuple(agent.state_dim)
        agent.memory._bs = np.empty(shape, dtype=np.uint8)
        agent.memory._bns = np.empty(shape, dtype=np.uint8)
        pin = agent.device.type == "cuda"
        agent._t_s = torch.empty(shape, dtype=torch.uint8, pin_memory=pin)
        agent._t_ns = torch.empty(shape, dtype=torch.uint8, pin_memory=pin)

    # Epsilon decays PER TRANSITION COLLECTED inside act()/act_batch().
    agent.exploration_rate = EPSILON_START
    agent.exploration_rate_min = EPSILON_MIN
    agent.exploration_rate_decay = EPSILON_DECAY

    agent.burnin = BURNIN
    agent.learn_every = LEARN_EVERY
    agent.sync_every = TARGET_SYNC_EVERY_STEPS
    agent.save_every = AGENT_SAVE_EVERY_STEPS

    for group in agent.optimizer.param_groups:
        group["lr"] = LEARNING_RATE


def check_checkpoint_dir() -> None:
    """Fail fast, and legibly, if checkpoints cannot be persisted.

    Two failure modes, both silent until it is too late:

    1. PERMISSION. Docker creates a MISSING bind-mount path on the
       host as root:root, but this image runs as the non-root `mario`
       user, so the first write dies with a PermissionError buried in
       a json/pathlib traceback. Probe explicitly instead.
    2. PERSISTENCE. A bind mount SHADOWS whatever the image had at
       that path. Mount the wrong path (e.g. `checkpoints` when the
       gamma lineage resolves to `checkpoints_7_action_g99`) and
       training runs perfectly, writes checkpoints into the
       container's throwaway writable layer, and loses every one of
       them on `docker rm`. Warn loudly when the directory is neither
       a mount point nor backed by S3.
    """
    import os

    d = Path(CHECKPOINT_DIR)
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(
            f"Cannot create checkpoint directory {d.resolve()}: {exc}\n"
            f"If this is a Docker bind mount, the host directory is "
            f"probably owned by root while the container runs as uid "
            f"{os.getuid()}. Fix on the HOST with:\n"
            f"    mkdir -p {d.name} && "
            f"sudo chown -R {os.getuid()}:{os.getgid()} {d.name}"
        ) from exc

    probe = d / ".write_probe"
    try:
        probe.write_text("ok")
        probe.unlink()
    except OSError as exc:
        raise RuntimeError(
            f"Checkpoint directory {d.resolve()} is not writable by uid "
            f"{os.getuid()}: {exc}\n"
            f"Docker creates a missing bind-mount path as root:root. Fix "
            f"on the HOST with:\n"
            f"    sudo chown -R {os.getuid()}:{os.getgid()} {d}"
        ) from exc

    # Is this directory actually persistent? A bind mount sits on a
    # different device than the container's root filesystem.
    persistent = False
    try:
        persistent = os.stat(d).st_dev != os.stat("/").st_dev
    except OSError:
        pass
    if not persistent and not S3_BUCKET:
        # CHECKPOINT_DIR is normally relative to WORKDIR (/app), but it
        # is env-overridable and may be absolute -- build the hint from
        # the basename so it can't come out as "$PWD//abs/path".
        target = CHECKPOINT_DIR if d.is_absolute() else f"/app/{d.name}"
        print(f"[train] WARNING: {d.resolve()} is not a mounted volume and "
              f"MARIOOPS_S3_BUCKET is unset -- checkpoints will be LOST when "
              f"this container is removed. Mount it with: "
              f"-v \"$PWD/{d.name}:{target}\"", flush=True)


def check_lineage() -> None:
    """Stamp/verify the gamma lineage of CHECKPOINT_DIR.

    Raising gamma rescales the Q targets ~10x (a 0.90 agent's Q values
    sit near 90; a 0.99 agent's approach ~900), so loading across
    lineages starts the net with every output off by an order of
    magnitude. Refuse to mix, exactly like the Colab notebook.
    """
    stamp = Path(LINEAGE_FILE)
    if stamp.exists():
        try:
            prev = json.loads(stamp.read_text()).get("gamma")
        except Exception:
            prev = None
        if prev is not None and abs(float(prev) - GAMMA) > 1e-9:
            raise RuntimeError(
                f"LINEAGE MISMATCH: {CHECKPOINT_DIR} holds checkpoints "
                f"trained with gamma={prev}, but GAMMA is set to {GAMMA}.\n"
                f"Q targets differ by ~{(1 - float(prev)) / (1 - GAMMA):.0f}x "
                f"between these settings, so resuming would corrupt training.\n"
                f"Fix: set GAMMA back to {prev}, or point CHECKPOINT_DIR at a "
                f"different folder."
            )
    else:
        stamp.parent.mkdir(parents=True, exist_ok=True)
        stamp.write_text(json.dumps({"gamma": GAMMA}))
    print(f"[lineage] gamma={GAMMA} (horizon ~{1 / (1 - GAMMA):.0f} steps) "
          f"-> {CHECKPOINT_DIR}")


# Built-in PyTorch function, saves the training state to allow resuming later if paused or crashed.
def save_checkpoint(agent: MarioAgent, path: str, episode: int = 0,
                    best_eval: float = None) -> None:
    """
    Write a fixed-path checkpoint compatible with MarioAgent.load()
    AND the Colab notebook's Mario.load() (same keys, including
    "episode" and "best_eval"), so files move between Drive and S3
    with zero conversion.

    Written atomically (temp file + os.replace) so an interruption
    mid-write can never leave a corrupt checkpoint behind.
    """
    import os

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    # Move weights to CPU first so the checkpoint file is device-agnostic:
    # a file written on an AWS GPU instance loads cleanly on a CPU laptop
    # (agent.load() maps it back onto whatever device is active).
    cpu_state = {k: v.cpu() for k, v in agent.net.state_dict().items()}
    payload = {
        "model": cpu_state,
        "exploration_rate": agent.exploration_rate,
        "curr_step": agent.curr_step,
        "episode": int(episode),
        # Adam moment estimates -- without these, every resume restarts
        # the optimizer cold and produces a brief loss spike. Saved
        # as-is (possibly CUDA tensors); agent.load()'s map_location
        # and optimizer.load_state_dict() re-home them to the active
        # device automatically.
        "optimizer": agent.optimizer.state_dict(),
    }
    if best_eval is not None:
        payload["best_eval"] = best_eval
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def load_training_state() -> dict:
    """Return {'episode': int, 'best_eval': float} from the last run."""
    path = Path(TRAINING_STATE_FILE)
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except Exception as exc:
            print(f"[resume] could not read training state ({exc}); starting fresh")
    return {"episode": 0, "best_eval": float("-inf")}


def save_training_state(episode: int, agent: MarioAgent, best_eval: float) -> None:
    Path(TRAINING_STATE_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(TRAINING_STATE_FILE, "w") as f:
        json.dump(
            {
                "episode": episode,
                "epsilon": agent.exploration_rate,   # informational; source of
                "curr_step": agent.curr_step,        # truth is the checkpoint
                # NOTE: best_eval is now the flag-rate-first eval SCORE
                # (flags/EVAL_EPISODES * 100000 + median reward), not a
                # mean reward. An old mean-reward best (~2000) sits
                # below any 1-flag score, so the best file updates on
                # the first flag-bearing eval after migrating -- safe.
                "best_eval": best_eval,
                "run_id": RUN_ID,
                "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            f,
            indent=2,
        )


def sync_to_s3(storage: S3Storage) -> None:
    storage.upload_checkpoint(CHECKPOINT_FILE)
    storage.upload_file(
        TRAINING_STATE_FILE,
        storage.checkpoint_key(Path(TRAINING_STATE_FILE).name),
    )
    if Path(BEST_CHECKPOINT_FILE).exists():
        storage.upload_file(
            BEST_CHECKPOINT_FILE,
            storage.checkpoint_key(Path(BEST_CHECKPOINT_FILE).name),
        )
    storage.upload_training_log(TRAINING_LOG_FILE)


# ------------------------------------------------------------------------------
# Evaluation (mirrors the Colab notebook)
#
# A PURELY greedy policy in this deterministic env yields one fixed
# trajectory per snapshot -- and it can deadlock (observed: stuck
# pushing against a pipe at x~594 for 2000+ steps until the level
# timer ran out). EVAL_EPSILON=0.0 was tested and reverted: no-op
# starts randomize only the START, and the trajectory re-converges
# before the pipe. A tiny epsilon breaks those loops; EVAL_STALL_STEPS
# ends an episode when x_pos stops advancing (a deadlock IS a failure,
# and it should cost 150 steps, not 3000); and EVAL_EPISODES episodes
# give a distribution instead of a single lucky/unlucky rollout.
# Nothing here touches training state: no epsilon decay, no step
# counting, no replay caching. Runs on a dedicated single env built by
# the SAME make_env() as the workers -- including no-op starts -- so
# eval measures the policy, not one memorized path.
# ------------------------------------------------------------------------------
def run_eval_episode(env, agent: MarioAgent):
    state = env.reset()
    total_reward, flag, x_pos = 0.0, False, 0
    best_x, stalled = 0, 0
    for _ in range(EVAL_MAX_STEPS):
        if np.random.rand() < EVAL_EPSILON:
            action = np.random.randint(agent.action_dim)
        else:
            with torch.no_grad():
                s = agent._states_to_device(np.asarray(state)).unsqueeze(0)
                q = agent.net(s, model="online")
            action = torch.argmax(q, dim=1).item()

        state, reward, done, info = env.step(action)
        total_reward += reward
        x_pos = int(info.get("x_pos", x_pos))
        if done or info.get("flag_get", False):
            flag = bool(info.get("flag_get", False))
            break
        # Stuck against geometry: no forward progress for a long time.
        # End the episode and report the jam position rather than
        # spending the remaining budget pushing at a pipe.
        if x_pos > best_x:
            best_x, stalled = x_pos, 0
        else:
            stalled += 1
            if stalled >= EVAL_STALL_STEPS:
                break
    return total_reward, flag, x_pos


def evaluate(env, agent: MarioAgent):
    """Returns (score, flags, median, mean, max, sorted_x_positions).

    SELECT ON FLAG RATE FIRST, MEDIAN REWARD SECOND. Clearing the
    level is the actual goal, and the median is robust to the bimodal
    reward split (~250 or ~2400) that makes the mean meaningless here.
    The 100000x weight makes one flag dominate the whole reward range
    (~3000), so more flags always wins.
    """
    results = [run_eval_episode(env, agent) for _ in range(EVAL_EPISODES)]
    rewards = [r[0] for r in results]
    flags = sum(1 for r in results if r[1])
    median_r = float(np.median(rewards))
    mean_r = sum(rewards) / len(rewards)
    max_r = max(rewards)
    # Every episode's death position, not just the best one. The sorted
    # list distinguishes a specific killer obstacle (deaths cluster at
    # one x) from general imprecision (deaths scattered).
    all_x = sorted((r[2] for r in results), reverse=True)
    score = (flags / EVAL_EPISODES) * 100000 + median_r
    return score, flags, median_r, mean_r, max_r, all_x


def main() -> None:
    # SIGTERM (spot reclaim, Batch job termination, docker stop) kills
    # Python WITHOUT running finally blocks by default. Convert it to a
    # normal SystemExit so the finally block below gets to write and
    # sync the final checkpoint during the termination grace period.
    signal.signal(signal.SIGTERM, lambda signum, frame: (_ for _ in ()).throw(SystemExit(0)))

    storage = S3Storage()
    metrics = TrainingMetrics()

    # BEFORE the S3 restore and the lineage stamp -- both write here,
    # and a permission failure inside either produces a traceback that
    # points at json/pathlib rather than at the actual cause.
    check_checkpoint_dir()
    Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ resume
    checkpoint = Path(CHECKPOINT_FILE)
    if not checkpoint.exists() and storage.enabled:
        print("[resume] no local checkpoint; checking S3 ...")
        storage.restore_checkpoint(CHECKPOINT_FILE)
        # State, best, and lineage files may not exist yet (first run on
        # a fresh bucket) -- download_file returns False on a missing
        # key (never raises), so absence can't kill the job.
        for key_name, dest in (
            (Path(TRAINING_STATE_FILE).name, TRAINING_STATE_FILE),
            (Path(BEST_CHECKPOINT_FILE).name, BEST_CHECKPOINT_FILE),
            (Path(LINEAGE_FILE).name, LINEAGE_FILE),
        ):
            storage.download_file(storage.checkpoint_key(key_name), dest)
        # Restore the CSV log too. Without it, a resumed run would start
        # a fresh local training CSV and the next sync_to_s3() would
        # OVERWRITE the full episode history in S3 with only the
        # post-resume rows.
        storage.download_file(
            storage.log_key(Path(TRAINING_LOG_FILE).name), TRAINING_LOG_FILE
        )

    # Refuse to resume across gamma lineages BEFORE any real work.
    check_lineage()

    state_info = load_training_state()
    start_episode = int(state_info.get("episode", 0))
    best_eval = float(state_info.get("best_eval", float("-inf")))

    # =====================================================================
    # FORK THE WORKERS *BEFORE* ALLOCATING THE REPLAY BUFFER.
    # =====================================================================
    # This ordering is load-bearing, not stylistic. fork() gives every
    # child a copy-on-write view of the parent's address space. The
    # children never read the multi-GB replay buffer -- but the PARENT
    # writes to all of it as the ring cycles through its slots, and
    # every write to a shared page forces the kernel to allocate a
    # private copy while the children keep mapping the original. One
    # full pass through the buffer therefore DOUBLES its physical
    # footprint, and the OOM killer takes the process with no Python
    # exception and no final checkpoint.
    #
    # Measured directly (700 MB stand-in buffer, 3 workers):
    #     fork AFTER  allocation -> 1409 MB resident   (2.0x -- duplicated)
    #     fork BEFORE allocation ->  662 MB resident   (1.0x -- shared)
    #
    # Forking first also means the children are created before the CUDA
    # context exists, which is the officially safe ordering.
    n_envs = resolve_n_envs()
    import os as _os
    print(f"[train] parallel environments: {n_envs} "
          f"(detected {_os.cpu_count()} vCPUs)")
    venv = make_vec_env(make_env, n_envs, shared_memory=True)
    print(f"[train] launched {n_envs} env workers (before buffer allocation).")

    try:
        # ------------------------------------------------------------ agent
        agent = MarioAgent(
            state_dim=(STACK_SIZE, IMAGE_SIZE, IMAGE_SIZE),
            action_dim=len(SIMPLE_MOVEMENT),
            save_dir=CHECKPOINT_DIR,
            device=DEVICE,                    # "cpu" locally, "auto" -> CUDA on AWS GPU
            replay_capacity=REPLAY_CAPACITY,  # preallocated ring buffer, sized once here
        )
        configure_agent(agent)
        print(f"[train] replay buffer preallocated: "
              f"{agent.memory.nbytes / 1e9:.2f} GB (fixed for the whole run)")

        if agent.device.type == "cuda":
            # Input shapes never change (batch x 4 x 84 x 84): the
            # one-off cuDNN autotune benchmark is repaid for the run.
            torch.backends.cudnn.benchmark = True
            # Learner matmuls run on the device, so CPU threads only
            # steal cycles from the emulator workers. Free the vCPUs.
            default_threads = 1
        else:
            # DO NOT lower this. With device=cpu the learner's conv2d
            # forward/backward is real CPU work sitting synchronously
            # between step_async() and step_wait(), so capping threads
            # makes it the blocking stage of the pipeline. Measured on
            # 8 cores / 8 envs: 35 steps/s at torch's default, 16.5 at
            # one thread -- a 2.1x regression from "freeing" cores.
            default_threads = torch.get_num_threads()

        n_threads = TORCH_THREADS if TORCH_THREADS > 0 else default_threads
        torch.set_num_threads(n_threads)
        print(f"[train] torch threads: {n_threads} "
              f"({'explicit TORCH_THREADS' if TORCH_THREADS > 0 else 'auto for ' + agent.device.type})",
              flush=True)

        if checkpoint.exists():
            # Restores weights + exploration_rate + curr_step in one
            # call (with the action-set lineage guard inside load()).
            agent.load(CHECKPOINT_FILE)
            agent.sync_Q_target()
            # Fallback: if training_state.json was lost but the checkpoint
            # survived, recover best_eval/episode from the checkpoint payload
            # so a worse policy can never silently overwrite the best file
            # (the state file remains the primary source when present).
            payload = torch.load(CHECKPOINT_FILE, map_location="cpu")
            if isinstance(payload, dict):
                if "best_eval" in payload:
                    best_eval = max(best_eval, float(payload["best_eval"]))
                if start_episode == 0 and "episode" in payload:
                    start_episode = int(payload["episode"])
            del payload
            print(f"[resume] loaded checkpoint {CHECKPOINT_FILE} "
                  f"(episode={start_episode}, step={agent.curr_step}, "
                  f"epsilon={agent.exploration_rate:.4f}, best_eval={best_eval:.0f})")
        else:
            print("[resume] no checkpoint found; training from scratch")

        # Re-apply the exploration floor AFTER load(): load() restores
        # the saved exploration_rate but never exploration_rate_min, so
        # a lowered floor (0.05 -> 0.02) takes effect on resume.
        agent.exploration_rate = max(agent.exploration_rate,
                                     agent.exploration_rate_min)
        print(f"[train] exploration floor {agent.exploration_rate_min} "
              f"(current eps {agent.exploration_rate:.4f})")

        # Dedicated eval env, built by the SAME make_env() as the
        # workers -- including random no-op starts. Evaluating on a
        # no-noop env would keep measuring the one memorized trajectory
        # (observed on Colab). Created after the fork, so the workers
        # never inherit this emulator.
        eval_env = make_env()

        # ---- RAM watchdog ----
        # When the OS runs out of RAM the process is KILLED -- Python
        # never gets an exception, so a try/except can't save a
        # checkpoint. With the preallocated buffer the footprint is
        # fixed, so this is a pure safety net: periodically return
        # freed pages to the OS (malloc_trim) and, if RAM still gets
        # critical, checkpoint and stop cleanly.
        try:
            import psutil
            _HAVE_PSUTIL = True
        except ImportError:
            _HAVE_PSUTIL = False
            print("[train] psutil not available -- RAM watchdog disabled.")

        try:
            import ctypes
            _libc = ctypes.CDLL("libc.so.6")

            def malloc_trim():
                _libc.malloc_trim(0)
        except Exception:
            def malloc_trim():
                pass

        def available_ram_gb():
            return (psutil.virtual_memory().available / 1e9
                    if _HAVE_PSUTIL else float("inf"))

        def workers_gb():
            """PRIVATE memory of the env workers, via USS not RSS.

            A forked worker's RSS counts every shared page it inherited
            -- measured at ~313 MB RSS for a worker whose real cost was
            9 MB. Summing RSS would report ~1 GB of workers that do not
            exist. USS counts only pages unique to that process.
            """
            if not _HAVE_PSUTIL:
                return 0.0
            total = 0
            try:
                for child in psutil.Process().children(recursive=True):
                    try:
                        total += child.memory_full_info().uss
                    except Exception:
                        total += child.memory_info().rss  # USS needs permissions
            except Exception:
                return 0.0
            return total / 1e9

        # Throughput. Emulation is the bottleneck and the whole point
        # of the parallel build, so this is the number that decides
        # N_ENVS and sizes an instance -- "rate" is the recent window
        # (what a config change actually did), "avg" is lifetime since
        # this process started (what the run will really cost).
        # Baselined on the RESUMED step, not 0 -- a resumed run restores
        # curr_step from the checkpoint, and a zero baseline would report
        # 25248 steps done in the first 30 seconds.
        _thr = {"t0": time.time(), "step0": agent.curr_step,
                "t_win": time.time(), "step_win": agent.curr_step}

        def throughput_report(curr_step: int) -> str:
            now = time.time()
            win_dt = now - _thr["t_win"]
            run_dt = now - _thr["t0"]
            rate = (curr_step - _thr["step_win"]) / win_dt if win_dt > 0 else 0.0
            avg = (curr_step - _thr["step0"]) / run_dt if run_dt > 0 else 0.0
            _thr["t_win"], _thr["step_win"] = now, curr_step
            return (f"[rate] {rate:.1f} steps/s (avg {avg:.1f}) "
                    f"- {run_dt / 60:.1f} min elapsed")

        def memory_report():
            if not _HAVE_PSUTIL:
                return ""
            proc_gb = psutil.Process().memory_info().rss / 1e9
            gpu = (f" - GPU {torch.cuda.memory_allocated() / 1e9:.2f} GB"
                   if torch.cuda.is_available() else "")
            return (f"[mem] process {proc_gb:.2f} GB - workers "
                    f"{workers_gb():.2f} GB - buffer {len(agent.memory)}/"
                    f"{agent.memory.maxlen} exp "
                    f"({agent.memory.nbytes / 1e9:.2f} GB fixed) "
                    f"- free RAM {available_ram_gb():.2f} GB{gpu}")

        # ------------------------------------------------------------ CSV log
        log_path = Path(TRAINING_LOG_FILE)
        if not log_path.exists() or start_episode == 0:
            with open(log_path, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["episode", "steps", "reward", "last_loss", "epsilon",
                     "memory_length"]
                )

        print(f"[train] run_id={RUN_ID} env={ENV_NAME} "
              f"device={agent.device} n_envs={n_envs} noop_max={NOOP_MAX} "
              f"gamma={agent.gamma} "
              f"episodes={start_episode}->{NUM_EPISODES} "
              f"max_hours={MAX_TRAIN_HOURS or 'off'} "
              f"replay={agent.memory.maxlen} "
              f"({agent.memory.nbytes / 1e9:.2f} GB fixed) "
              f"s3={'on' if storage.enabled else 'off'} "
              f"cloudwatch={'on' if metrics.enabled else 'off'}")

    except BaseException:
        # Anything failing during setup (bad checkpoint, lineage
        # mismatch, OOM) would otherwise strand N emulator processes
        # holding RAM. terminate=True SIGTERMs the workers instead of
        # trying to drain their pipes -- plain close() RAISES if the
        # env is mid-step, which is exactly where interrupts land.
        venv.close(terminate=True)
        print("[train] setup failed -- env workers closed.")
        raise

    # `completed` counts episodes that ran TO COMPLETION across ALL
    # envs. The finally block records this -- so an interrupted episode
    # is replayed on resume rather than skipped, and resuming an
    # already-finished run cannot inflate the episode count.
    completed = start_episode

    try:
        # ------------------------------------------------------------ loop state
        obs = venv.reset()
        _terminal_scratch = np.empty_like(obs)  # reused every done-step, allocated once
        ep_reward = np.zeros(n_envs, dtype=np.float64)
        ep_length = np.zeros(n_envs, dtype=np.int64)

        grad_debt = 0.0                       # fractional gradient updates owed
        last_sync_step = agent.curr_step
        recent_losses = []
        train_flags = 0
        train_start = time.time()

        while completed < NUM_EPISODES:
            # ---- act on all N states with a single forward pass ----
            actions = agent.act_batch(obs)

            # ---- DISPATCH THE WORKERS, THEN LEARN WHILE THEY RUN ----
            # step_async returns immediately; the emulators run in
            # their own processes while this one does GPU work. A
            # blocking venv.step() here measured SLOWER than serial
            # (56.7 vs ~66 steps/s) -- parallelism paid the IPC cost
            # without ever buying overlap.
            venv.step_async(actions)

            # --- overlap window: runs concurrently with the emulators.
            # --- It touches only the replay buffer and the GPU.
            #
            # Target sync on a step DELTA, not a modulo: curr_step
            # advances by N and can step clean over a multiple,
            # silently skipping syncs forever.
            if agent.curr_step - last_sync_step >= agent.sync_every:
                agent.sync_Q_target()
                last_sync_step = agent.curr_step

            # Gradient updates: HOLD THE REPLAY RATIO CONSTANT. The
            # serial loop did 1 update per LEARN_EVERY transitions.
            # Collecting N per iteration while still doing 1 would
            # quietly divide the replay ratio by N -- more data per
            # second, slower learning per sample. Owe N/learn_every
            # updates each iteration, pay the debt down. Learning one
            # iteration "behind" the freshest transition is harmless:
            # DQN is off-policy and samples from the whole buffer.
            ready = agent.buffer_ready()
            grad_debt += n_envs / agent.learn_every
            loss = None
            while grad_debt >= 1.0 and ready:
                _, loss = agent.learn_once()
                recent_losses.append(loss)
                grad_debt -= 1.0
            if not ready:
                grad_debt = min(grad_debt, 1.0)  # don't bank debt during burn-in

            # --- collect the workers' results ---
            next_obs, rewards, dones, infos = venv.step_wait()
            per_env, final_obs = parse_vector_infos(infos, n_envs)

            # ---- CRITICAL: use the TERMINAL frame, not the auto-reset one ----
            # Vector envs auto-reset: when an episode ends, the
            # observation returned is already the FIRST frame of the
            # NEXT episode. Caching that as next_state would teach the
            # agent that dying teleports it back to the start of the
            # level -- a corrupted transition at every single episode
            # boundary. The real final frame is in the info dict.
            true_next = next_obs
            if dones.any():
                # Preallocated scratch, not next_obs.copy(): a fresh
                # ~1 MB array every done-step is exactly the interleaved
                # multi-MB temporary that fragmented the heap before
                # RingReplay was introduced.
                np.copyto(_terminal_scratch, next_obs)
                for i in range(n_envs):
                    if dones[i] and final_obs[i] is not None:
                        _terminal_scratch[i] = np.asarray(
                            final_obs[i], dtype=next_obs.dtype
                        )
                true_next = _terminal_scratch

            agent.cache_batch(obs, true_next, actions, rewards, dones)

            # ---- per-env episode bookkeeping ----
            ep_reward += rewards
            ep_length += 1
            for i in range(n_envs):
                if not dones[i]:
                    continue
                if per_env[i].get("flag_get"):
                    train_flags += 1
                completed += 1
                mean_loss = (float(np.mean(recent_losses))
                             if recent_losses else None)
                recent_losses.clear()

                # Structured single-line log -- easy to filter in
                # CloudWatch Logs.
                if completed % LOG_EVERY_EPISODES == 0:
                    malloc_trim()
                    print(
                        f"[episode] episode={completed} "
                        f"steps={int(ep_length[i])} "
                        f"reward={ep_reward[i]:.1f} "
                        f"loss={mean_loss if mean_loss is not None else 'nan'} "
                        f"epsilon={agent.exploration_rate:.4f} "
                        f"memory={len(agent.memory)} "
                        f"flags={train_flags} "
                        f"total_steps={agent.curr_step}",
                        flush=True,
                    )
                    report = memory_report()
                    if report:
                        print(report + f" - {n_envs} envs", flush=True)
                    print(throughput_report(agent.curr_step), flush=True)

                metrics.publish_episode(
                    episode=completed,
                    reward=float(ep_reward[i]),
                    loss=mean_loss,
                    epsilon=agent.exploration_rate,
                    steps=int(ep_length[i]),
                    memory_length=len(agent.memory),
                )

                with open(log_path, "a", newline="") as f:
                    csv.writer(f).writerow(
                        [completed, int(ep_length[i]), float(ep_reward[i]),
                         mean_loss if mean_loss is not None else "",
                         agent.exploration_rate, len(agent.memory)]
                    )

                ep_reward[i] = 0.0
                ep_length[i] = 0

                # -------------------------------------------- best checkpoint
                if completed % EVAL_EVERY_EPISODES == 0:
                    score, flags, median_r, mean_r, max_r, all_x = evaluate(
                        eval_env, agent
                    )
                    need = max(best_eval * BEST_EVAL_MARGIN, best_eval + 1e-9)
                    print(f"[eval] episode={completed} "
                          f"flags={flags}/{EVAL_EPISODES} "
                          f"median={median_r:.0f} mean={mean_r:.0f} "
                          f"max={max_r:.0f} score={score:.0f} "
                          f"(best={best_eval:.0f}, need {need:.0f})",
                          flush=True)
                    print(f"[eval] x_pos: "
                          f"{','.join(str(x) for x in all_x)}", flush=True)
                    metrics.publish_eval(
                        episode=completed,
                        mean_reward=mean_r,
                        median_reward=median_r,
                        max_reward=max_r,
                        furthest_x=all_x[0],
                        flags=flags,
                        eval_episodes=EVAL_EPISODES,
                        score=score,
                    )
                    # Require a real margin before overwriting the best
                    # checkpoint -- without it you save binomial luck.
                    if score > need:
                        best_eval = score
                        save_checkpoint(agent, BEST_CHECKPOINT_FILE,
                                        episode=completed, best_eval=best_eval)
                        if storage.enabled:
                            storage.upload_file(
                                BEST_CHECKPOINT_FILE,
                                storage.checkpoint_key(
                                    Path(BEST_CHECKPOINT_FILE).name),
                            )
                        print(f"[eval] new best score {best_eval:.0f} -> "
                              f"{BEST_CHECKPOINT_FILE}", flush=True)

                # -------------------------------------------- periodic persistence
                if completed % SYNC_EVERY_N_EPISODES == 0:
                    save_checkpoint(agent, CHECKPOINT_FILE,
                                    episode=completed, best_eval=best_eval)
                    save_training_state(completed, agent, best_eval)
                    sync_to_s3(storage)

            obs = next_obs

            # ---- Wall-clock budget: checked per ITERATION, not per episode ----
            if MAX_TRAIN_HOURS > 0:
                elapsed_h = (time.time() - train_start) / 3600
                if elapsed_h >= MAX_TRAIN_HOURS:
                    print(f"[train] reached the {MAX_TRAIN_HOURS} h budget "
                          f"({elapsed_h:.2f} h elapsed) -- saving and "
                          f"stopping cleanly.", flush=True)
                    break

            # ---- RAM watchdog: act BEFORE the OS kills the process ----
            free_gb = available_ram_gb()
            if free_gb < CRITICAL_RAM_GB:
                print(f"[train] CRITICAL: only {free_gb:.2f} GB RAM free -- "
                      f"saving and stopping.", flush=True)
                break
            elif free_gb < LOW_RAM_GB:
                malloc_trim()
                if available_ram_gb() < LOW_RAM_GB:
                    print(f"[train] LOW RAM "
                          f"({available_ram_gb():.2f} GB free after trim) -- "
                          f"checkpointing as a precaution.", flush=True)
                    save_checkpoint(agent, CHECKPOINT_FILE,
                                    episode=completed, best_eval=best_eval)

    finally:
        # Always leave a consistent latest checkpoint behind -- on normal
        # completion, budget/RAM stop, exception, Ctrl-C, or SIGTERM
        # (converted above). Then ALWAYS reap the workers: orphaned
        # emulator processes survive an interrupt and hold RAM.
        save_checkpoint(agent, CHECKPOINT_FILE,
                        episode=completed, best_eval=best_eval)
        save_training_state(completed, agent, best_eval)
        sync_to_s3(storage)
        eval_env.close()
        try:
            venv.close(terminate=True)
            print("[train] env workers closed.")
        except Exception as close_err:
            # Do NOT swallow this silently -- runs that ended with no
            # "env workers closed" line left orphaned emulators eating
            # the RAM the next run needed.
            print(f"[train] WARNING: venv.close() failed ({close_err!r}). "
                  f"Check for orphaned emulator processes.")
        print("[train] finished; final checkpoint synced", flush=True)


if __name__ == "__main__":
    main()
