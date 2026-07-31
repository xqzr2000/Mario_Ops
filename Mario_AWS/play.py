"""
Loads the trained checkpoint (pulling it from S3 automatically if it is
not on local disk), plays up to PLAY_EPISODES near-greedy episodes
(epsilon = 0.02 -- pure greedy can deadlock against pipes), keeps the
best attempt (a flag run if one happens, otherwise the furthest), and:

  * encodes EVERY captured frame into an H.264/yuv420p MP4 that plays
    in any browser, phone, or the Drive/S3 preview (ffmpeg re-encode;
    falls back to OpenCV mp4v with a warning if ffmpeg is missing),
  * writes sparse annotated PNG "highlight" frames for debugging
    (step / reward / x_pos overlay, every SAVE_EVERY_N_STEPS),
  * writes a JSON summary covering all episodes played,
  * uploads the run folder (clip + summary [+ highlights]) to S3.

Frames are the raw 240x256 NES screen captured once per agent step
(= 4 emulated frames because of SkipFrame), so VIDEO_FPS = 15 plays
back at authentic game speed. A full flag run is ~90 s of video.
"""

import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace

# gym reinstates its own DeprecationWarning filter at import time, so
# PYTHONWARNINGS cannot silence it -- gate at gym's level instead.
gym.logger.set_level(gym.logger.ERROR)
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from mario_agent import MarioAgent
from mario_agent.data_pipeline import build_env
from cloud.storage import S3Storage

from config import (
    ENV_NAME,
    DEVICE,
    STACK_SIZE,
    IMAGE_SIZE,
    SAVE_EVERY_N_STEPS,
    VIDEO_FPS,
    PLAY_MAX_STEPS,
    PLAY_EPSILON,     # near-greedy floor: pure greedy has deadlocked
    PLAY_EPISODES,    # attempts recorded; best one becomes the clip
    CHECKPOINT_FILE,
    BEST_CHECKPOINT_FILE,
    RUNS_DIR,
)


# --------------------------------------------------------------- video

def encode_video(frames_rgb: list, output_path: Path, fps: int) -> None:
    """Encode raw RGB frames to H.264/yuv420p via ffmpeg so the clip
    previews in browsers and Drive. Falls back to OpenCV mp4v (which
    most browsers can NOT play) only if ffmpeg is unavailable."""
    if not frames_rgb:
        print("No frames captured; video was not created.")
        return

    height, width, _ = frames_rgb[0].shape

    if shutil.which("ffmpeg"):
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}", "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(output_path),
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for frame in frames_rgb:
            proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError("ffmpeg encoding failed")
    else:
        print("WARNING: ffmpeg not found -- falling back to mp4v; "
              "the clip will not preview in browsers. "
              "Add ffmpeg to the container image.")
        writer = cv2.VideoWriter(
            str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
            fps, (width, height),
        )
        for frame in frames_rgb:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()

    print(f"Video saved to: {output_path}")


def write_highlights(frames_rgb: list, meta: list, frames_dir: Path,
                     every_n: int) -> None:
    """Sparse annotated PNGs from the winning episode, for debugging."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    for step in range(0, len(frames_rgb), every_n):
        m = meta[step]
        frame_bgr = cv2.cvtColor(frames_rgb[step], cv2.COLOR_RGB2BGR)
        text = (f"Step: {step} | Reward: {m['reward']:.1f} | "
                f"X: {m['x_pos']} | Action: {m['action']}")
        cv2.putText(frame_bgr, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(
            str(frames_dir / f"frame_{step:04d}_x{m['x_pos']:04d}.png"),
            frame_bgr,
        )
    print(f"Highlight frames saved to: {frames_dir}")


# ------------------------------------------------------------- episode

def play_episode(env, agent) -> dict:
    """One near-greedy episode; captures the raw NES screen every step."""
    state = env.reset()
    frames, meta = [], []
    episode_reward = 0.0
    info = {}

    for step in range(PLAY_MAX_STEPS):
        action = agent.act(state)
        state, reward, done, info = env.step(action)
        episode_reward += reward

        # The wrapped observation is a (4, 84, 84) grayscale stack --
        # useless for a highlight reel. Grab the raw 240x256 RGB
        # screen from the underlying NES emulator instead.
        frames.append(env.unwrapped.screen.copy())
        meta.append({
            "reward": episode_reward,
            "x_pos": int(info["x_pos"]),
            "action": int(action),
        })

        if step % 100 == 0:
            print(f"  step {step} | reward {episode_reward:.1f} | "
                  f"x {info['x_pos']}", flush=True)

        if done or info.get("flag_get", False):
            break

    return {
        "frames": frames,
        "meta": meta,
        "steps": len(frames),
        "reward": float(episode_reward),
        "final_x": int(info["x_pos"]),
        "flag_get": bool(info.get("flag_get", False)),
    }


def is_better(candidate: dict, incumbent: dict) -> bool:
    """Flag run beats no flag; then furthest x; then highest reward."""
    if incumbent is None:
        return True
    if candidate["flag_get"] != incumbent["flag_get"]:
        return candidate["flag_get"]
    if candidate["final_x"] != incumbent["final_x"]:
        return candidate["final_x"] > incumbent["final_x"]
    return candidate["reward"] > incumbent["reward"]


# ---------------------------------------------------------------- main

def main() -> None:
    storage = S3Storage()

    # Resolve which checkpoint to play, best-first:
    #   1. local mario_net_best.chkpt   (incl. one copied in from Drive)
    #   2. local mario_net.chkpt        (rolling)
    #   3. S3 best, then S3 rolling     (only if the cloud layer is on)
    checkpoint_path = next(
        (p for p in (BEST_CHECKPOINT_FILE, CHECKPOINT_FILE) if Path(p).exists()),
        None,
    )
    if checkpoint_path is None and storage.enabled:
        print("[play] no local checkpoint; checking S3 ...")
        # NOTE: these return False on a missing key (they never raise
        # -- missing keys are normal on a first run), so this chain
        # branches on the boolean, not on exceptions.
        if storage.restore_best_checkpoint(BEST_CHECKPOINT_FILE):
            checkpoint_path = BEST_CHECKPOINT_FILE
        elif storage.restore_checkpoint(CHECKPOINT_FILE):
            checkpoint_path = CHECKPOINT_FILE
    if checkpoint_path is None:
        raise FileNotFoundError(
            f"No checkpoint at {BEST_CHECKPOINT_FILE} or {CHECKPOINT_FILE}, "
            "and none found in S3. Train a model first (python train.py)."
        )
    print(f"[play] using checkpoint: {checkpoint_path}")

    env = gym_super_mario_bros.make(ENV_NAME, disable_env_checker=True)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    # noop_max defaults to 0 here: recordings stay deterministic
    # showcases rather than randomized starts.
    env = build_env(env)

    agent = MarioAgent(
        state_dim=(STACK_SIZE, IMAGE_SIZE, IMAGE_SIZE),
        action_dim=env.action_space.n,
        save_dir=None,          # evaluation only; no checkpoints written
        device=DEVICE,          # "cpu" locally, "auto" -> CUDA on AWS GPU
    )
    agent.load(checkpoint_path)
    agent.net.eval()

    # Freeze exploration at the near-greedy floor (both rate AND min,
    # so act() cannot decay it below PLAY_EPSILON).
    agent.exploration_rate = PLAY_EPSILON
    agent.exploration_rate_min = PLAY_EPSILON

    runs_dir = Path(RUNS_DIR)
    runs_dir.mkdir(parents=True, exist_ok=True)
    # Counting run_* folders collides after a deletion (remove run_0003
    # and the next run is also numbered 0003, silently merged by
    # exist_ok). Probe upward for the first unused number instead, and
    # let mkdir fail loudly if a race still lands on a taken name.
    run_id = len(list(runs_dir.glob("run_*"))) + 1
    while (runs_dir / f"run_{run_id:04d}").exists():
        run_id += 1
    run_dir = runs_dir / f"run_{run_id:04d}"
    run_dir.mkdir(parents=True)

    best, best_index = None, -1
    episode_log = []

    for ep in range(PLAY_EPISODES):
        print(f"[play] episode {ep + 1}/{PLAY_EPISODES}")
        result = play_episode(env, agent)
        episode_log.append({
            "episode": ep + 1,
            "steps": result["steps"],
            "reward": result["reward"],
            "final_x_position": result["final_x"],
            "flag_get": result["flag_get"],
        })
        print(f"  -> steps {result['steps']} | reward {result['reward']:.1f} "
              f"| x {result['final_x']} | flag {result['flag_get']}")

        if is_better(result, best):
            best, best_index = result, ep + 1
        if result["flag_get"]:
            print("[play] flag run captured -- stopping early")
            break

    tag = "FLAG" if best["flag_get"] else f"x{best['final_x']}"
    print(f"\nBest attempt: episode {best_index} ({tag}) | "
          f"steps {best['steps']} | reward {best['reward']:.1f}")

    # run.mp4 name is fixed: S3Storage.upload_run expects it.
    video_path = run_dir / "run.mp4"
    encode_video(best["frames"], video_path, fps=VIDEO_FPS)
    write_highlights(best["frames"], best["meta"], run_dir / "frames",
                     every_n=SAVE_EVERY_N_STEPS)

    summary = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "play_epsilon": PLAY_EPSILON,
        "episodes_played": len(episode_log),
        "episodes": episode_log,
        "best_episode": best_index,
        "steps": best["steps"],
        "reward": best["reward"],
        "final_x_position": best["final_x"],
        "flag_get": best["flag_get"],
        "video_fps": int(VIDEO_FPS),
        "save_every_n_steps": int(SAVE_EVERY_N_STEPS),
        "created_at": datetime.now().isoformat(),
    }
    summary_path = run_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"Summary saved to: {summary_path}")

    # ------------------------------------------------------------ S3 upload
    if storage.enabled:
        print("[play] uploading gameplay clip to S3 ...")
        video_url = storage.upload_run(run_dir)
        if video_url:
            print("\n" + "=" * 60)
            print("Gameplay video URL (shareable):")
            print(video_url)
            print("=" * 60)

    env.close()


if __name__ == "__main__":
    main()
