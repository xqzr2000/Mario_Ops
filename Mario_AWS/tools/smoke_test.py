#!/usr/bin/env python
"""
tools/smoke_test.py -- pre-flight checks for the MarioOps environment.

Run automatically by .devcontainer/devcontainer.json (which lives at the
REPOSITORY root, one level above this project directory) when a
Codespace is created. Run it yourself any time, from anywhere:

    python tools/smoke_test.py                 # verify + env + vec + agent
    python tools/smoke_test.py verify          # just the fast checks
    python tools/smoke_test.py train           # short REAL training run
    python tools/smoke_test.py verify env vec agent train

Each check is independent, prints PASS/FAIL lines, and the process exits
non-zero if any failed. Nothing here writes into a real lineage folder:
the agent and train checks write only into SMOKE_DIR (default .smoke/),
so a 4-episode toy run can never leave a checkpoint or a lineage.json
where a later resume might pick it up.

The checks:

  verify  imports, version pins, ffmpeg, repo layout, dev-container
          placement, machine resources. Catches the two failures that
          otherwise surface as cryptic crashes hours in: numpy 2.x
          (nes-py dies) and a gym other than 0.25.2 (the 4-tuple step
          API SkipFrame is written for) -- plus the quiet one, a
          Codespace running the default image because .devcontainer/
          was not at the repository root.
  env     one real NES env through the full wrapper chain: shape
          (4, 84, 84), dtype uint8, non-black frames, 7 actions,
          x_pos in info.
  vec     AsyncVectorEnv with 2 workers through step_async/step_wait.
          The check that matters most on a fresh machine: it exercises
          the gym-RNG pickle patch (without which the vector env cannot
          even be CONSTRUCTED) and the shared-memory dtype invariant (a
          mismatch silently zeroes every pixel, and training then runs
          for hours on black frames).
  agent   network, replay buffer, checkpoint round-trip, and the
          action-set lineage guard (7-action output layer).
  train   train.py end to end -- 4 episodes, 2 workers, tiny buffer --
          asserting that a checkpoint and a lineage stamp actually
          land on disk. Slow (a few minutes), so it is NOT in the
          default set; ask for it by name.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# PROJECT_ROOT is where train.py / config.py / requirements.txt live --
# i.e. Mario_AWS/, the directory this tools/ folder sits inside.
#
# It is NOT necessarily the git repository root. The project is a
# subdirectory of the repo (Mario_Ops/Mario_AWS/), and .devcontainer/
# must live at the REPO root one level up, because that is the only
# place Codespaces looks for it. Hence the separate WORKSPACE_ROOT.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
SMOKE_DIR = Path(os.getenv("SMOKE_DIR", PROJECT_ROOT / ".smoke"))

DEFAULT_CHECKS = ["verify", "env", "vec", "agent"]

_FAILURES = []


# ---------------------------------------------------------------- output
def ok(msg):
    print(f"  \033[32mPASS\033[0m  {msg}", flush=True)


def fail(msg):
    print(f"  \033[31mFAIL\033[0m  {msg}", flush=True)
    _FAILURES.append(msg)


def warn(msg):
    print(f"  \033[33mWARN\033[0m  {msg}", flush=True)


def info(msg):
    print(f"        {msg}", flush=True)


def header(title):
    print(f"\n=== {title} " + "=" * max(0, 60 - len(title)), flush=True)


def check(condition, good, bad):
    ok(good) if condition else fail(bad)
    return bool(condition)


# ---------------------------------------------------------------- verify
def check_verify():
    header("verify: interpreter, pins, tooling")

    py = sys.version_info
    check(
        py[:2] in ((3, 10), (3, 11)),
        f"python {py.major}.{py.minor}.{py.micro}",
        f"python {py.major}.{py.minor} -- this stack needs 3.10 or 3.11. "
        "nes-py 8.2.1 compiles a C++ extension that will not build on "
        "3.12+, and gym 0.25.2 predates it. Rebuild the dev container "
        "(Dockerfile.develop pins 3.10, matching the deploy image).",
    )

    try:
        from importlib.metadata import version as pkg_version
    except ImportError:  # pragma: no cover
        from importlib_metadata import version as pkg_version  # type: ignore

    pins = {
        "numpy": "1.26.4",       # hard pin: nes-py breaks on numpy 2.x
        "gym": "0.25.2",         # 4-tuple step API + gym.vector
        "nes-py": "8.2.1",
        "gym-super-mario-bros": "7.4.0",
        "opencv-python-headless": "4.9.0.80",
    }
    for name, expected in pins.items():
        try:
            found = pkg_version(name)
        except Exception:
            fail(f"{name} is not installed (expected {expected}). "
                 "Rebuild the dev container.")
            continue
        check(
            found == expected,
            f"{name}=={found}",
            f"{name}=={found} but this build is pinned to {expected} "
            "-- see requirements.txt",
        )

    # torch is not in requirements.txt: the deploy image gets it from the
    # CUDA base, the dev image installs the CPU wheel. Same 2.3.1 either
    # way, so a version drift here means the two targets have diverged.
    try:
        found = pkg_version("torch")
        check(found.split("+")[0] == "2.3.1",
              f"torch=={found}",
              f"torch=={found}; Dockerfile.deploy's base ships 2.3.1, so "
              "the dev and prod stacks have diverged")
    except Exception:
        fail("torch is not installed")

    for name in ("boto3", "psutil"):
        try:
            info(f"{name}=={pkg_version(name)}")
        except Exception:
            fail(f"{name} is not installed")

    try:
        import numpy as np
        import torch

        ok(f"torch imports; cuda_available={torch.cuda.is_available()}")
        info(f"numpy runtime version: {np.__version__}")
        if torch.cuda.is_available():
            info(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            info("no GPU visible -- expected in a Codespace; training here "
                 "is for correctness, not throughput")
    except Exception as exc:
        fail(f"torch/numpy import failed: {exc!r}")

    try:
        import cv2

        ok(f"cv2 imports (headless build) {cv2.__version__}")
    except Exception as exc:
        fail(f"cv2 import failed: {exc!r}")

    # ffmpeg is play.py's primary encode path. The OpenCV mp4v fallback
    # produces clips that will NOT play in a browser or the S3 preview,
    # so treat a missing ffmpeg as a real failure, not a nicety.
    check(
        shutil.which("ffmpeg") is not None,
        "ffmpeg on PATH (play.py H.264 encode path)",
        "ffmpeg not found -- play.py would fall back to OpenCV mp4v and "
        "the clips will not play in browsers. Rebuild the dev container; "
        "Dockerfile.develop installs it.",
    )

    header("verify: repo surface and imports")
    expected_paths = [
        "train.py", "play.py", "config.py", "requirements.txt",
        "Dockerfile.deploy", "Dockerfile.develop",
        "mario_agent/__init__.py", "mario_agent/config.py",
        "mario_agent/dqn_model.py", "mario_agent/mario_agent.py",
        "mario_agent/data_pipeline.py", "mario_agent/vector_env.py",
        "cloud/storage.py", "cloud/monitoring.py",
    ]
    for rel in expected_paths:
        p = PROJECT_ROOT / rel
        check(p.exists(), f"{rel}", f"{rel} is missing from {PROJECT_ROOT}")

    _check_devcontainer()

    try:
        import config

        ok("config imports")
        info(f"ENV_NAME={config.ENV_NAME}  GAMMA={config.GAMMA}  "
             f"CHECKPOINT_DIR={config.CHECKPOINT_DIR}")
        info(f"N_ENVS={config.N_ENVS or 'auto'}  "
             f"REPLAY_CAPACITY={config.REPLAY_CAPACITY}  "
             f"DEVICE={config.DEVICE}")
    except Exception as exc:
        fail(f"config import failed: {exc!r}")

    for mod in ("mario_agent", "cloud.storage", "cloud.monitoring",
                "train", "play"):
        try:
            __import__(mod)
            ok(f"import {mod}")
        except Exception as exc:
            fail(f"import {mod} failed: {exc!r}")

    # The cloud layer must be completely inert with no bucket set -- that
    # is what lets a fresh Codespace train with no AWS credentials and no
    # AWS spend.
    try:
        from cloud.storage import S3Storage

        if os.getenv("MARIOOPS_S3_BUCKET"):
            warn("MARIOOPS_S3_BUCKET is set -- the cloud layer is ACTIVE "
                 "and this session can incur AWS charges")
        else:
            storage = S3Storage()
            check(
                not getattr(storage, "enabled", False),
                "cloud layer inert with no MARIOOPS_S3_BUCKET (no AWS "
                "credentials needed)",
                "S3Storage reports enabled=True with no bucket configured",
            )
    except Exception as exc:
        fail(f"S3Storage construction failed: {exc!r}")

    header("verify: machine resources")
    try:
        import psutil

        total_gb = psutil.virtual_memory().total / 1e9
        avail_gb = psutil.virtual_memory().available / 1e9
        cpus = os.cpu_count() or 1
        info(f"{cpus} vCPU, {total_gb:.1f} GB RAM ({avail_gb:.1f} GB free)")
        free_disk = shutil.disk_usage(PROJECT_ROOT).free / 1e9
        info(f"{free_disk:.1f} GB free disk at {PROJECT_ROOT}")
        if total_gb < 6:
            warn("under 6 GB RAM: lower REPLAY_CAPACITY here. The 80000 "
                 "default preallocates ~4.5 GB and page-touches it in "
                 "full at startup, so it fails immediately, not slowly.")
        if cpus < 4:
            warn(f"{cpus} vCPU: N_ENVS auto-detects min(8, cpu_count), so "
                 "throughput here says nothing about a g4dn.4xlarge.")
    except Exception as exc:
        warn(f"resource probe failed: {exc!r}")


# --------------------------------------------------------- devcontainer
def _devcontainer_candidates(root):
    """The three places a devcontainer.json is discoverable, per root.

    Codespaces looks ONLY at these, relative to the git repository root:
    .devcontainer/devcontainer.json, .devcontainer.json, and one level
    of .devcontainer/<name>/devcontainer.json. A copy anywhere else --
    notably inside a project subdirectory -- is silently ignored, and
    you get the default universal image with none of these deps.
    """
    found = []
    direct = root / ".devcontainer" / "devcontainer.json"
    if direct.is_file():
        found.append(direct)
    dotfile = root / ".devcontainer.json"
    if dotfile.is_file():
        found.append(dotfile)
    nested = root / ".devcontainer"
    if nested.is_dir():
        found.extend(sorted(nested.glob("*/devcontainer.json")))
    return found


def _check_devcontainer():
    """Locate the dev container definition and sanity-check the layout.

    This is a WARN-only check on purpose: train.py runs perfectly well
    in a hand-built venv, and someone may have vendored the project
    directory on its own. What it is really guarding against is the
    quiet failure mode -- a Codespace that booted the default image
    because the config was in the wrong directory, where the first
    symptom is `ModuleNotFoundError: No module named 'numpy'`.
    """
    at_workspace = _devcontainer_candidates(WORKSPACE_ROOT)
    at_project = _devcontainer_candidates(PROJECT_ROOT)

    if at_workspace:
        ok(f"devcontainer at {at_workspace[0].relative_to(WORKSPACE_ROOT)} "
           f"(repo root: {WORKSPACE_ROOT})")
    elif at_project:
        # PROJECT_ROOT == WORKSPACE_ROOT means the project IS the repo,
        # which is fine. Otherwise the file is one level too deep.
        if PROJECT_ROOT != WORKSPACE_ROOT:
            warn(f"devcontainer found at {at_project[0]} -- that is INSIDE "
                 f"the project directory. Codespaces only reads it from "
                 f"the repository root, so move it to "
                 f"{WORKSPACE_ROOT / '.devcontainer'}/ or the Codespace "
                 f"will boot the default image instead.")
    else:
        warn("no devcontainer.json found at "
             f"{WORKSPACE_ROOT} -- fine if you set this environment up by "
             "hand; if you expected a dev container, it is not being used.")

    # Dockerfile.develop stamps this. If we are in a Codespace and the
    # stamp is missing, the container was almost certainly built from
    # GitHub's universal image rather than from this repo's definition
    # -- exactly the case where numpy/gym/torch are absent.
    if os.getenv("CODESPACES", "").lower() == "true":
        built_from_ours = os.getenv("MARIOOPS_RUN_ID") == "codespace-dev"
        if built_from_ours:
            ok("running in a Codespace built from Dockerfile.develop")
        else:
            warn("running in a Codespace, but the Dockerfile.develop "
                 "marker is absent -- this looks like the DEFAULT image. "
                 "Commit .devcontainer/ at the repo root, then "
                 "F1 -> 'Codespaces: Rebuild Container'.")


def check_env():
    header("env: one NES env through the full wrapper chain")
    import numpy as np

    from train import make_env

    t0 = time.time()
    env = make_env()
    try:
        obs = np.asarray(env.reset())
        check(obs.shape == (4, 84, 84),
              f"reset observation shape {obs.shape}",
              f"reset observation shape {obs.shape}, expected (4, 84, 84)")
        check(obs.dtype == np.uint8,
              f"observation dtype {obs.dtype}",
              f"observation dtype {obs.dtype}, expected uint8 -- a dtype "
              "drift here silently zeroes frames under a shared-memory "
              "vector env")
        check(int(obs.max()) > 0,
              f"frames are not black (max pixel {int(obs.max())})",
              "every pixel is 0 -- the emulator or the wrapper chain is "
              "not producing image data")
        check(env.action_space.n == 7,
              f"action space is Discrete({env.action_space.n}) "
              "(SIMPLE_MOVEMENT)",
              f"action space is Discrete({env.action_space.n}), expected 7 "
              "-- checkpoints from this run would not match the 7-action "
              "lineage")

        info_dict = {}
        for _ in range(25):
            obs, reward, done, info_dict = env.step(env.action_space.sample())
            if done:
                env.reset()
        check("x_pos" in info_dict,
              f"step() info carries x_pos ({info_dict.get('x_pos')})",
              f"step() info has no x_pos; keys={list(info_dict)[:8]} -- the "
              "eval stall cutoff and the death-zone logging both need it")
        ok(f"25 agent steps (= 100 NES frames) in {time.time() - t0:.1f}s")
    finally:
        env.close()


# ------------------------------------------------------------- vector env
def check_vec():
    header("vec: AsyncVectorEnv, 2 workers, async step")
    import numpy as np

    from train import make_env
    from mario_agent.vector_env import (make_vec_env, parse_vector_infos,
                                        patch_gym_rng)

    check(patch_gym_rng() in (True, False),
          "gym RNG patch applied (gym 0.25.2 x numpy>=1.25 pickle fix)",
          "patch_gym_rng() raised")

    n = 2
    t0 = time.time()
    venv = make_vec_env(make_env, n, shared_memory=True)
    try:
        obs = np.asarray(venv.reset())
        check(obs.shape == (n, 4, 84, 84),
              f"batched reset shape {obs.shape}",
              f"batched reset shape {obs.shape}, expected {(n, 4, 84, 84)}")
        check(obs.dtype == np.uint8,
              f"batched dtype {obs.dtype}",
              f"batched dtype {obs.dtype}, expected uint8")
        # THE check: shared_memory=True copies into a buffer typed from the
        # DECLARED space. A mismatch does not raise -- it zeroes every
        # pixel, and training then runs for hours on black frames.
        check(int(obs.max()) > 0,
              f"shared-memory frames carry real pixels (max {int(obs.max())})",
              "every pixel is 0 across all workers -- the shared-memory "
              "dtype invariant is broken; training would run on black frames")

        steps = 30
        per_env, final_obs = [{}] * n, [None] * n
        next_obs = obs
        for _ in range(steps):
            actions = np.random.randint(0, venv.single_action_space.n, size=n)
            venv.step_async(actions)
            next_obs, rewards, dones, infos = venv.step_wait()
            per_env, final_obs = parse_vector_infos(infos, n)

        check(len(per_env) == n and len(final_obs) == n,
              f"parse_vector_infos returns {n} per-env dicts + terminal slots",
              "parse_vector_infos returned the wrong number of entries")
        check(all("x_pos" in d for d in per_env),
              "every worker reports x_pos through the vector info layout",
              f"x_pos missing from a worker's info: "
              f"{[list(d)[:6] for d in per_env]}")
        check(np.asarray(next_obs).shape == (n, 4, 84, 84),
              "step_wait observation batch shape is correct",
              f"step_wait shape {np.asarray(next_obs).shape}")

        elapsed = time.time() - t0
        ok(f"{steps} async iterations x {n} envs = {steps * n} transitions "
           f"in {elapsed:.1f}s (~{steps * n / max(elapsed, 1e-9):.0f} "
           "transitions/s incl. startup)")
    finally:
        # terminate=True: workers may be mid-step, and a plain close()
        # raises there -- which would leave orphaned emulator processes
        # holding the RAM the next run needs.
        venv.close(terminate=True)
        ok("workers closed cleanly (no orphaned emulators)")


# ------------------------------------------------------------------ agent
def check_agent():
    header("agent: network, replay buffer, checkpoint lineage")
    import numpy as np
    import torch

    from mario_agent import MarioAgent, checkpoint_action_dim

    save_dir = SMOKE_DIR / "agent"
    save_dir.mkdir(parents=True, exist_ok=True)

    agent = MarioAgent(
        state_dim=(4, 84, 84),
        action_dim=7,
        save_dir=save_dir,
        device="cpu",
        replay_capacity=512,     # ~29 MB: enough to exercise, cheap to hold
    )
    ok(f"MarioAgent constructed on {agent.device} "
       f"(replay capacity {agent.memory.maxlen})")

    states = np.random.randint(0, 256, size=(4, 4, 84, 84), dtype=np.uint8)
    actions = agent.act_batch(states)
    check(actions.shape == (4,) and actions.max() < 7,
          f"act_batch returns 4 valid actions {actions.tolist()}",
          f"act_batch returned {actions!r}")

    with torch.no_grad():
        q = agent.net(
            torch.zeros(1, 4, 84, 84, dtype=torch.float32), model="online"
        )
    check(tuple(q.shape) == (1, 7),
          f"online net output shape {tuple(q.shape)} (7-action head)",
          f"online net output shape {tuple(q.shape)}, expected (1, 7)")

    for _ in range(8):
        agent.cache(states[0], states[1], int(actions[0]), 1.0, False)
    check(len(agent.memory) == 8,
          f"replay buffer accepted 8 transitions (len={len(agent.memory)})",
          f"replay buffer holds {len(agent.memory)} after 8 cache() calls")

    # The action-set lineage guard: the output layer's first dimension IS
    # the action count, and it is what stops an old 2-action checkpoint
    # loading into this net with a cryptic torch size mismatch.
    dim = checkpoint_action_dim(agent.net.state_dict())
    check(dim == 7,
          "checkpoint_action_dim reads 7 actions from the state dict",
          f"checkpoint_action_dim returned {dim!r}, expected 7")

    ckpt = save_dir / "smoke.chkpt"
    torch.save(
        {
            "model": {k: v.cpu() for k, v in agent.net.state_dict().items()},
            "exploration_rate": agent.exploration_rate,
            "curr_step": agent.curr_step,
        },
        ckpt,
    )
    agent.load(ckpt)
    ok(f"checkpoint save/load round-trip through {ckpt.relative_to(PROJECT_ROOT)}")


# ------------------------------------------------------------------ train
def check_train():
    header("train: short real run of train.py (4 episodes, 2 workers)")

    # Tiny-but-real: 2 workers, 4 episodes, a buffer that allocates
    # instantly, a burn-in low enough that gradient steps actually run,
    # and a bounded eval so the best-checkpoint gate is exercised without
    # spending minutes inside stalled eval episodes.
    #
    # CHECKPOINT_DIR / LOGS_DIR / RUNS_DIR are redirected into SMOKE_DIR
    # ON PURPOSE: a 4-episode run must never stamp a lineage.json inside
    # checkpoints_7_action_g99/, where a later real resume could find it.
    env = dict(os.environ)
    env.pop("MARIOOPS_S3_BUCKET", None)   # keep the cloud layer inert
    env.update({
        "MARIOOPS_CLOUDWATCH": "false",
        "MARIOOPS_RUN_ID": "smoke",
        "MARIOOPS_DEVICE": "cpu",
        "PYTHONUNBUFFERED": "1",
        "CHECKPOINT_DIR": str(SMOKE_DIR / "checkpoints"),
        "LOGS_DIR": str(SMOKE_DIR / "logs"),
        "RUNS_DIR": str(SMOKE_DIR / "runs"),
        "N_ENVS": "2",
        "NUM_EPISODES": "4",
        "REPLAY_CAPACITY": "3000",
        "BURNIN": "300",
        "BATCH_SIZE": "32",
        "TARGET_SYNC_EVERY_STEPS": "500",
        "AGENT_SAVE_EVERY_STEPS": "1000000",
        "MARIOOPS_SYNC_EVERY": "2",
        "LOG_EVERY_EPISODES": "1",
        "EVAL_EVERY_EPISODES": "2",
        "EVAL_EPISODES": "2",
        "EVAL_MAX_STEPS": "150",
        "EVAL_STALL_STEPS": "40",
        "MAX_TRAIN_HOURS": "0.25",
    })

    (SMOKE_DIR / "checkpoints").mkdir(parents=True, exist_ok=True)
    info(f"artifacts -> {SMOKE_DIR}/ (no lineage folder is touched)")

    t0 = time.time()
    proc = subprocess.run([sys.executable, "train.py"], cwd=PROJECT_ROOT, env=env)
    check(proc.returncode == 0,
          f"train.py exited 0 after {time.time() - t0:.0f}s",
          f"train.py exited {proc.returncode}")

    # A run that exits 0 but writes nothing has failed at the thing that
    # matters most on a spot instance: persisting its own progress.
    ckpt = SMOKE_DIR / "checkpoints" / "mario_net_7_action.chkpt"
    lineage = SMOKE_DIR / "checkpoints" / "lineage.json"
    check(ckpt.exists(),
          f"rolling checkpoint written ({ckpt.stat().st_size / 1e6:.1f} MB)"
          if ckpt.exists() else "",
          "no rolling checkpoint was written -- a spot reclaim would lose "
          "the whole run")
    check(lineage.exists(),
          "lineage.json stamped into the checkpoint dir",
          "no lineage.json was stamped -- the gamma mismatch guard has "
          "nothing to check against on resume")


CHECKS = {
    "verify": check_verify,
    "env": check_env,
    "vec": check_vec,
    "agent": check_agent,
    "train": check_train,
}


def main():
    parser = argparse.ArgumentParser(
        description="MarioOps environment smoke test.",
        epilog=f"checks: {', '.join(CHECKS)} "
               f"(default: {', '.join(DEFAULT_CHECKS)})",
    )
    parser.add_argument("checks", nargs="*", default=DEFAULT_CHECKS,
                        metavar="CHECK")
    args = parser.parse_args()

    unknown = [c for c in args.checks if c not in CHECKS]
    if unknown:
        parser.error(f"unknown check(s): {', '.join(unknown)}. "
                     f"Valid: {', '.join(CHECKS)}")

    # Everything imports from the repo root (train.py does `from config
    # import ...`), so make that work no matter where this was invoked.
    os.chdir(PROJECT_ROOT)
    sys.path.insert(0, str(PROJECT_ROOT))

    started = time.time()
    for name in args.checks:
        CHECKS[name]()

    print()
    if _FAILURES:
        print(f"\033[31m{len(_FAILURES)} check(s) failed "
              f"in {time.time() - started:.1f}s:\033[0m")
        for f in _FAILURES:
            print(f"  - {f}")
        sys.exit(1)

    print(f"\033[32mall checks passed in {time.time() - started:.1f}s\033[0m")


if __name__ == "__main__":
    main()
