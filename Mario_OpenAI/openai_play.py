"""
Mario_OpenAI -- entry point.

    cd Mario_OpenAI && python openai_play.py

NES -> screenshot -> OpenAI vision -> action plan -> NES -> run.mp4.

No Torch, no checkpoint, no S3, no CloudWatch. The point of this project
is to answer one question -- can a general vision model play a level it
has never seen, with no training -- against the same level, the same
7-action space, and the same video output format as the DQN agent, so
the two are directly comparable.

THE ENV IS RAW ON PURPOSE
-------------------------
Mario_AWS wraps the env in grayscale + resize + FrameStack + SkipFrame,
which produces a (4, 84, 84) tensor. That is exactly the wrong input
here: the whole premise is that the model reads the real screen. So this
builds gym_super_mario_bros + JoypadSpace and stops. One consequence
follows and it is the easiest thing in this project to get wrong:

    WITHOUT SkipFrame, ONE env.step() IS ONE NES FRAME.

Mario_AWS captures one frame per agent step and encodes at 15 FPS
because each of its steps is 4 emulated frames. Here we capture every
4th frame explicitly (config.CAPTURE_EVERY_N_FRAMES) so that 15 FPS
still plays back at authentic speed.

GYM 0.25.2 API
--------------
This repo pins gym==0.25.2, which is pre-Gymnasium:

    state = env.reset()                       # obs only, NOT (obs, info)
    state, reward, done, info = env.step(a)   # 4-tuple, NOT 5

Code written against modern Gymnasium will fail here. Do not "fix" it.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace

# gym reinstates its own DeprecationWarning filter at import time, so
# PYTHONWARNINGS cannot silence it -- gate at gym's level instead.
gym.logger.set_level(gym.logger.ERROR)

from gym_super_mario_bros.actions import SIMPLE_MOVEMENT  # noqa: E402

import config                                             # noqa: E402
from openai_agent import ACTION_NAMES, OpenAIMarioAgent, describe_plan  # noqa: E402
from video_utils import encode_video, next_run_dir, write_highlights    # noqa: E402


def build_raw_env():
    """The 7-action NES env and nothing else."""
    env = gym_super_mario_bros.make(config.ENV_NAME, disable_env_checker=True)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    return env


def run_segment(env, action, frames, capture, frames_rgb, meta, ctx):
    """Execute one (action, frames) segment, capturing as we go.

    Returns (done, info). `ctx` carries the counters the overlay needs.
    """
    info = ctx["info"]
    done = False
    for _ in range(frames):
        _, _, done, info = env.step(action)
        ctx["frame"] += 1
        if capture and ctx["frame"] % config.CAPTURE_EVERY_N_FRAMES == 0:
            frames_rgb.append(env.unwrapped.screen.copy())
            meta.append({
                "decision": ctx["decision"],
                "x_pos": int(info.get("x_pos", 0)),
                "time": int(info.get("time", 0)),
                "action_name": ACTION_NAMES[action],
            })
        if done or info.get("flag_get", False):
            break
    ctx["info"] = info
    return done, info


def land(env, frames_rgb, meta, ctx):
    """Step until Mario is back on the ground, or the cap runs out.

    Called between a finished plan and the next screenshot so that every
    decision is made from a grounded state. See LAND_BEFORE_DECIDING in
    config.py for why this matters more than it sounds like it should.

    Grounded is detected as y_pos holding still for a few frames rather
    than matching a fixed floor height: 1-1 has pipes, blocks and stairs,
    so "on the ground" is not one number. Frames spent here are captured
    and counted like any others -- they are real game time, and leaving
    them out of prev_frames would corrupt the px/frame telemetry this
    was built to protect.

    Returns (done, info, frames_used).
    """
    info = ctx["info"]
    if not config.LAND_BEFORE_DECIDING:
        return False, info, 0

    action = config.LAND_HOLD_ACTION
    start = ctx["frame"]
    last_y = int(info.get("y_pos", 0))
    still = 0

    for _ in range(config.LAND_MAX_FRAMES):
        _, _, done, info = env.step(action)
        ctx["frame"] += 1
        if ctx["frame"] % config.CAPTURE_EVERY_N_FRAMES == 0:
            frames_rgb.append(env.unwrapped.screen.copy())
            meta.append({
                "decision": ctx["decision"],
                "x_pos": int(info.get("x_pos", 0)),
                "time": int(info.get("time", 0)),
                "action_name": ACTION_NAMES[action] + " (landing)",
            })
        if done or info.get("flag_get", False):
            ctx["info"] = info
            return done, info, ctx["frame"] - start

        y = int(info.get("y_pos", 0))
        # Three consecutive identical readings, not one: y_pos is
        # momentarily flat at the apex of a jump too, and stopping there
        # would defeat the whole point.
        still = still + 1 if y == last_y else 0
        last_y = y
        if still >= 3:
            break

    ctx["info"] = info
    return False, info, ctx["frame"] - start


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set. In a Codespace it should arrive "
                 "as a Codespaces secret; locally, put it in a .env you do "
                 "not commit (see .env.example).")

    agent = OpenAIMarioAgent()
    env = build_raw_env()

    print(f"[openai] model: {agent.model} (api style: {config.OPENAI_API_STYLE})")
    print(f"[openai] starting {config.ENV_NAME}")
    print(f"[openai] budget: {config.MAX_DECISIONS} decisions, "
          f"{config.MAX_FRAMES} frames\n")

    env.reset()

    # reset() returns only the observation on gym 0.25, so there is no
    # info dict and therefore no x_pos for the first prompt. One NOOP
    # frame is the cheapest way to get one.
    _, _, _, info = env.step(0)

    frames_rgb, meta = [], []
    ctx = {"frame": 1, "decision": 0, "info": info}
    trace = []
    decision_log = []

    prev_x = int(info.get("x_pos", 0))
    prev_plan_str = None
    # Frames actually EXECUTED by the last plan -- not the frames it
    # asked for. A plan cut short by a death or a flag would otherwise
    # report a speed averaged over frames that never ran.
    prev_frames = 0
    # Landing accounting, reported in summary.json: if this is a large
    # fraction of total frames, plans are ending mid-jump more often
    # than they should and the prompt -- not this loop -- is the fix.
    landing_frames = 0
    landings = 0
    stuck = 0
    done = False
    stop_reason = "decision budget exhausted"

    for decision in range(1, config.MAX_DECISIONS + 1):
        ctx["decision"] = decision
        info = ctx["info"]
        x_pos = int(info.get("x_pos", 0))

        state = {
            "decision": decision,
            "x_pos": x_pos,
            "y_pos": int(info.get("y_pos", 0)),
            "time": int(info.get("time", 0)),
            "prev_x": prev_x,
            "prev_frames": prev_frames,
            "prev_plan": prev_plan_str,
            "stuck": stuck,
            "budget_left": config.MAX_DECISIONS - decision,
        }

        print(f"decision {decision:03d} | x={x_pos} t={state['time']} "
              f"stuck={stuck}", flush=True)

        # Scripted unstick. A screenshot of a pipe looks the same every
        # time, so a model that failed to clear it will usually answer
        # the same way again -- and charge us for the privilege. Back up
        # for a run-up, then a full running jump.
        if stuck >= config.STUCK_FALLBACK_AFTER:
            plan = [(6, 16), (3, 30), (4, 28)]
            note = "SCRIPTED UNSTICK (no API call)"
            raw = None
            print(f"  {note}: {describe_plan(plan)}")
            stuck = 0
        else:
            result = agent.decide(env.unwrapped.screen.copy(), state)
            plan, note, raw = result["plan"], result["note"], result["raw"]
            print(f"  model -> {describe_plan(plan)}")
            if note:
                print(f"  note: {note}")
            if config.SAVE_TRACE:
                trace.append({
                    "decision": decision,
                    "telemetry": result["telemetry"],
                    "response": raw,
                    "plan": plan,
                })

        prev_plan_str = describe_plan(plan)

        frames_before = ctx["frame"]
        for action, frames in plan:
            done, info = run_segment(env, action, frames, True,
                                     frames_rgb, meta, ctx)
            if done or info.get("flag_get", False):
                break

        # Settle to the ground before the next screenshot, so the model
        # never plans a run-up for frames Mario spends falling.
        if not done and not info.get("flag_get", False):
            done, info, landed_in = land(env, frames_rgb, meta, ctx)
            if landed_in:
                landing_frames += landed_in
                landings += 1

        # Measured, not requested: run_segment breaks early on death,
        # and the landing frames above are real game time too.
        prev_frames = ctx["frame"] - frames_before

        new_x = int(info.get("x_pos", 0))
        decision_log.append({
            "decision": decision,
            "x_before": x_pos,
            "x_after": new_x,
            "plan": [{"action": a, "action_name": ACTION_NAMES[a], "frames": f}
                     for a, f in plan],
            "note": note,
        })

        # Stuck is measured per DECISION, not per frame: a plan that
        # ends with Mario 3 px further along did not work, however busy
        # it looked while it ran.
        if new_x - x_pos < config.STUCK_MIN_DX:
            stuck += 1
        else:
            stuck = 0
        prev_x = x_pos

        if info.get("flag_get", False):
            stop_reason = "flag reached"
            print("\n*** FLAG GET ***")
            break
        if done:
            stop_reason = "died" if not info.get("flag_get") else "level complete"
            print(f"\n[openai] episode ended ({stop_reason}) at x={new_x}")
            if config.STOP_ON_DEATH:
                break
            env.reset()
            _, _, _, info = env.step(0)
            ctx["info"] = info
        if ctx["frame"] >= config.MAX_FRAMES:
            stop_reason = "frame budget exhausted"
            break

    final_info = ctx["info"]
    final_x = int(final_info.get("x_pos", 0))
    flag = bool(final_info.get("flag_get", False))

    # ------------------------------------------------------- output

    run_dir = next_run_dir(Path(config.RUNS_DIR))
    encode_video(frames_rgb, run_dir / "run.mp4", fps=config.VIDEO_FPS)
    write_highlights(frames_rgb, meta, run_dir / "frames",
                     every_n=config.SAVE_EVERY_N_FRAMES)

    summary = {
        "run_dir": str(run_dir),
        "env": config.ENV_NAME,
        "model": agent.model,
        "api_style": config.OPENAI_API_STYLE,
        "reasoning_effort": config.OPENAI_REASONING_EFFORT or None,
        "screen_upscale": config.SCREEN_UPSCALE,
        "image_detail": config.IMAGE_DETAIL,
        "decisions": len(decision_log),
        "api_calls": agent.api_calls,
        "failed_calls": agent.failed_calls,
        "input_tokens": agent.input_tokens,
        "output_tokens": agent.output_tokens,
        "api_seconds": round(agent.api_seconds, 1),
        "nes_frames": ctx["frame"],
        "land_before_deciding": config.LAND_BEFORE_DECIDING,
        "landings": landings,
        "landing_frames": landing_frames,
        "captured_frames": len(frames_rgb),
        "final_x_position": final_x,
        "flag_get": flag,
        "stop_reason": stop_reason,
        "video_fps": config.VIDEO_FPS,
        "capture_every_n_frames": config.CAPTURE_EVERY_N_FRAMES,
        "save_every_n_frames": config.SAVE_EVERY_N_FRAMES,
        "decision_log": decision_log,
        "created_at": datetime.now().isoformat(),
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=4)
    print(f"Summary saved to: {run_dir / 'summary.json'}")

    if config.SAVE_TRACE and trace:
        with open(run_dir / "trace.jsonl", "w") as f:
            for row in trace:
                f.write(json.dumps(row) + "\n")
        print(f"Trace saved to: {run_dir / 'trace.jsonl'}")

    print(f"\nflag_get: {flag}")
    print(f"final_x: {final_x}")
    print(f"api_calls: {agent.api_calls} "
          f"({agent.input_tokens} in / {agent.output_tokens} out tokens)")
    print(f"stopped because: {stop_reason}")

    env.close()


if __name__ == "__main__":
    main()
