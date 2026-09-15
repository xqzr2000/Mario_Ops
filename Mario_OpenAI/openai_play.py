"""Mario_OpenAI v2 entry point.

NES -> temporal RGB observations -> GPT-6 Astra -> short structured controller
plan -> adaptive execution -> MP4 + trace + summary.

The environment intentionally remains raw gym 0.25.2 / nes-py. One env.step()
is one NES frame; no SkipFrame wrapper is used for the policy.
"""

from __future__ import annotations

import json
import os
import sys
from collections import deque
from datetime import datetime
from pathlib import Path

import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace

gym.logger.set_level(gym.logger.ERROR)

from gym_super_mario_bros.actions import SIMPLE_MOVEMENT  # noqa: E402

import config  # noqa: E402
from openai_agent import ACTION_NAMES, OpenAIMarioAgent, describe_plan  # noqa: E402
from video_utils import encode_video, next_run_dir, write_highlights  # noqa: E402


class TemporalObservationBuffer:
    """Sparse rolling history of raw RGB screens with exact NES frame numbers."""

    def __init__(self) -> None:
        self.samples: deque[tuple[int, object]] = deque(maxlen=config.TEMPORAL_FRAMES)

    def clear(self) -> None:
        self.samples.clear()

    def record(self, frame_no: int, screen_rgb, force: bool = False) -> None:
        if not self.samples:
            self.samples.append((frame_no, screen_rgb.copy()))
            return

        last_frame = self.samples[-1][0]
        if frame_no == last_frame:
            if force:
                self.samples[-1] = (frame_no, screen_rgb.copy())
            return

        if force or frame_no - last_frame >= config.TEMPORAL_SAMPLE_EVERY_N_FRAMES:
            self.samples.append((frame_no, screen_rgb.copy()))

    def packet(self, current_frame: int, current_screen) -> list[tuple[int, object]]:
        self.record(current_frame, current_screen, force=True)
        return list(self.samples)


def build_raw_env():
    env = gym_super_mario_bros.make(config.ENV_NAME, disable_env_checker=True)
    return JoypadSpace(env, SIMPLE_MOVEMENT)


def after_step(env, action, frames_rgb, meta, ctx, label_suffix=""):
    """Common accounting after each real NES frame."""
    ctx["observer"].record(ctx["frame"], env.unwrapped.screen)

    if ctx["frame"] % config.CAPTURE_EVERY_N_FRAMES == 0:
        frames_rgb.append(env.unwrapped.screen.copy())
        info = ctx["info"]
        meta.append(
            {
                "decision": ctx["decision"],
                "x_pos": int(info.get("x_pos", 0)),
                "time": int(info.get("time", 0)),
                "action_name": ACTION_NAMES[action] + label_suffix,
            }
        )


def run_segment(env, action, frames, frames_rgb, meta, ctx):
    info = ctx["info"]
    done = False
    for _ in range(frames):
        _, _, done, info = env.step(action)
        ctx["frame"] += 1
        ctx["info"] = info
        after_step(env, action, frames_rgb, meta, ctx)
        if done or info.get("flag_get", False) or ctx["frame"] >= config.MAX_FRAMES:
            break
    return done, info


def land(env, frames_rgb, meta, ctx):
    """Optional v1-compatible landing wait; disabled by default."""
    info = ctx["info"]
    if not config.LAND_BEFORE_DECIDING:
        return False, info, 0, []

    action = config.LAND_HOLD_ACTION
    start = ctx["frame"]
    last_y = int(info.get("y_pos", 0))
    still = 0
    samples = [last_y]

    for _ in range(config.LAND_MAX_FRAMES):
        _, _, done, info = env.step(action)
        ctx["frame"] += 1
        ctx["info"] = info
        after_step(env, action, frames_rgb, meta, ctx, " (landing)")

        if done or info.get("flag_get", False) or ctx["frame"] >= config.MAX_FRAMES:
            return done, info, ctx["frame"] - start, samples

        y = int(info.get("y_pos", 0))
        samples.append(y)
        still = still + 1 if y == last_y else 0
        last_y = y
        if still >= config.LAND_STABLE_FRAMES:
            break

    return False, info, ctx["frame"] - start, samples


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit(
            "OPENAI_API_KEY is not set. Add it as a Codespaces secret or local "
            "environment variable; do not commit it."
        )

    agent = OpenAIMarioAgent()
    env = build_raw_env()
    observer = TemporalObservationBuffer()

    print(f"[openai] model: {agent.model} (Responses API / Structured Outputs)")
    print(f"[openai] starting {config.ENV_NAME}")
    print(
        f"[openai] temporal vision: {config.TEMPORAL_FRAMES} frames, "
        f"sample every {config.TEMPORAL_SAMPLE_EVERY_N_FRAMES} NES frames"
    )
    print(
        f"[openai] budget: {config.MAX_DECISIONS} decisions, "
        f"{config.MAX_FRAMES} NES frames\n"
    )

    env.reset()
    _, _, _, info = env.step(0)  # gym 0.25 reset has no info dict

    frames_rgb, meta = [], []
    ctx = {"frame": 1, "decision": 0, "info": info, "observer": observer}
    observer.record(1, env.unwrapped.screen, force=True)

    trace = []
    decision_log = []
    prev_x = int(info.get("x_pos", 0))
    prev_plan_str = None
    prev_frames = 0
    landing_frames = 0
    landings = 0
    land_y_samples = []
    stuck = 0
    stop_reason = "decision budget exhausted"

    for decision in range(1, config.MAX_DECISIONS + 1):
        ctx["decision"] = decision
        info = ctx["info"]
        x_pos = int(info.get("x_pos", 0))
        state = {
            "decision": decision,
            "frame": ctx["frame"],
            "x_pos": x_pos,
            "y_pos": int(info.get("y_pos", 0)),
            "time": int(info.get("time", 0)),
            "prev_x": prev_x,
            "prev_frames": prev_frames,
            "prev_plan": prev_plan_str,
            "stuck": stuck,
            "budget_left": config.MAX_DECISIONS - decision,
        }

        print(
            f"decision {decision:03d} | x={x_pos} y={state['y_pos']} "
            f"t={state['time']} stuck={stuck}",
            flush=True,
        )

        observation_packet = observer.packet(ctx["frame"], env.unwrapped.screen)
        observation_frames = [frame_no for frame_no, _ in observation_packet]

        if stuck >= config.STUCK_FALLBACK_AFTER:
            proposed_plan = [(6, 16), (3, 30), (4, 28)]
            plan = proposed_plan
            note = "SCRIPTED UNSTICK (no API call)"
            scene = {
                "airborne": False,
                "hazard_type": "unknown",
                "hazard_distance": "near",
                "risk": "high",
            }
            confidence = None
            requested_horizon = sum(frames for _, frames in plan)
            effective_horizon = requested_horizon
            raw = None
            response_id = None
            telemetry = None
            print(f"  {note}: {describe_plan(plan)}")
            stuck = 0
        else:
            result = agent.decide(observation_packet, state)
            plan = result["plan"]
            proposed_plan = result["proposed_plan"]
            note = result["note"]
            scene = result["scene"]
            confidence = result["confidence"]
            requested_horizon = result["requested_reobserve_after"]
            effective_horizon = result["effective_reobserve_after"]
            raw = result["raw"]
            response_id = result["response_id"]
            telemetry = result["telemetry"]

            print(f"  model -> {describe_plan(plan)}")
            if proposed_plan != plan:
                print(
                    f"  interrupted prefix of: {describe_plan(proposed_plan)} "
                    f"(horizon {effective_horizon}f)"
                )
            print(
                f"  scene: risk={scene['risk']} hazard={scene['hazard_type']}/"
                f"{scene['hazard_distance']} airborne={scene['airborne']} "
                f"confidence={confidence:.2f}"
            )
            if note:
                print(f"  note: {note}")

            if config.SAVE_TRACE:
                trace.append(
                    {
                        "decision": decision,
                        "observation_frames": observation_frames,
                        "telemetry": telemetry,
                        "response_id": response_id,
                        "response": raw,
                        "scene": scene,
                        "confidence": confidence,
                        "requested_reobserve_after": requested_horizon,
                        "effective_reobserve_after": effective_horizon,
                        "proposed_plan": proposed_plan,
                        "executed_plan": plan,
                    }
                )

        prev_plan_str = describe_plan(plan)
        frames_before = ctx["frame"]
        done = False

        for action, frames in plan:
            done, info = run_segment(env, action, frames, frames_rgb, meta, ctx)
            if done or info.get("flag_get", False) or ctx["frame"] >= config.MAX_FRAMES:
                break

        if not done and not info.get("flag_get", False) and ctx["frame"] < config.MAX_FRAMES:
            done, info, landed_in, y_samples = land(env, frames_rgb, meta, ctx)
            if landed_in:
                landing_frames += landed_in
                landings += 1
                if config.LAND_DEBUG_Y:
                    land_y_samples.append({"decision": decision, "y": y_samples})

        prev_frames = ctx["frame"] - frames_before
        new_x = int(info.get("x_pos", 0))

        decision_log.append(
            {
                "decision": decision,
                "x_before": x_pos,
                "x_after": new_x,
                "dx": new_x - x_pos,
                "observation_frames": observation_frames,
                "scene": scene,
                "confidence": confidence,
                "requested_reobserve_after": requested_horizon,
                "effective_reobserve_after": effective_horizon,
                "proposed_plan": [
                    {"action": a, "action_name": ACTION_NAMES[a], "frames": f}
                    for a, f in proposed_plan
                ],
                "executed_plan": [
                    {"action": a, "action_name": ACTION_NAMES[a], "frames": f}
                    for a, f in plan
                ],
                "frames_executed": prev_frames,
                "note": note,
            }
        )

        stuck = stuck + 1 if new_x - x_pos < config.STUCK_MIN_DX else 0
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
            ctx["frame"] += 1
            ctx["info"] = info
            observer.clear()
            observer.record(ctx["frame"], env.unwrapped.screen, force=True)
            prev_x = int(info.get("x_pos", 0))
            prev_frames = 0
            prev_plan_str = None
            stuck = 0

        if ctx["frame"] >= config.MAX_FRAMES:
            stop_reason = "frame budget exhausted"
            break

    final_info = ctx["info"]
    final_x = int(final_info.get("x_pos", 0))
    flag = bool(final_info.get("flag_get", False))

    run_dir = next_run_dir(Path(config.RUNS_DIR))
    encode_video(frames_rgb, run_dir / "run.mp4", fps=config.VIDEO_FPS)
    write_highlights(
        frames_rgb,
        meta,
        run_dir / "frames",
        every_n=config.SAVE_EVERY_N_FRAMES,
    )

    summary = {
        "harness_version": 2,
        "run_dir": str(run_dir),
        "env": config.ENV_NAME,
        "model": agent.model,
        "api_style": "responses",
        "structured_outputs": True,
        "reasoning_effort": config.OPENAI_REASONING_EFFORT or None,
        "screen_upscale": config.SCREEN_UPSCALE,
        "image_detail": config.IMAGE_DETAIL,
        "temporal_frames": config.TEMPORAL_FRAMES,
        "temporal_sample_every_n_frames": config.TEMPORAL_SAMPLE_EVERY_N_FRAMES,
        "stateful_turns": config.STATEFUL_TURNS,
        "decisions": len(decision_log),
        "api_calls": agent.api_calls,
        "failed_calls": agent.failed_calls,
        "input_tokens": agent.input_tokens,
        "cached_input_tokens": agent.cached_input_tokens,
        "cache_write_tokens": agent.cache_write_tokens,
        "output_tokens": agent.output_tokens,
        "reasoning_tokens": agent.reasoning_tokens,
        "estimated_cost_usd": round(agent.estimated_cost_usd, 6),
        "api_seconds": round(agent.api_seconds, 2),
        "nes_frames": ctx["frame"],
        "land_before_deciding": config.LAND_BEFORE_DECIDING,
        "landings": landings,
        "landing_frames": landing_frames,
        "land_y_samples": land_y_samples,
        "captured_frames": len(frames_rgb),
        "final_x_position": final_x,
        "flag_get": flag,
        "stop_reason": stop_reason,
        "video_fps": config.VIDEO_FPS,
        "capture_every_n_frames": config.CAPTURE_EVERY_N_FRAMES,
        "decision_log": decision_log,
        "created_at": datetime.now().isoformat(),
    }

    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {run_dir / 'summary.json'}")

    if config.SAVE_TRACE and trace:
        with open(run_dir / "trace.jsonl", "w", encoding="utf-8") as f:
            for row in trace:
                f.write(json.dumps(row) + "\n")
        print(f"Trace saved to: {run_dir / 'trace.jsonl'}")

    print(f"\nflag_get: {flag}")
    print(f"final_x: {final_x}")
    print(
        f"api_calls: {agent.api_calls} "
        f"({agent.input_tokens} in / {agent.output_tokens} out; "
        f"{agent.cached_input_tokens} cached)"
    )
    print(f"estimated_cost_usd: ${agent.estimated_cost_usd:.4f}")
    print(f"stopped because: {stop_reason}")

    env.close()


if __name__ == "__main__":
    main()
