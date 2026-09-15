"""
Mario_OpenAI -- harness entry point.

    cd Mario_OpenAI && python harness_play.py                      # preset "ram"
    HARNESS_PRESET=gamingagent python harness_play.py
    HARNESS_PRESET=baseline    python harness_play.py              # == openai_play.py

Same level, same 7 actions, same decision loop, same runs/run_NNNN/
output contract as openai_play.py -- the loop below is that loop with
four insertion points, and the functions that do not need to change
(build_raw_env, land, encode_video, ...) are imported, not copied:

    NES ─► StateRecorder (RAM + screen history, no behaviour change)
            │
            ├─► PerceptionModule   RAM state / scaffolded image / VLM description
            ├─► MemoryModule       trajectory, reflection, cross-run lessons
            ├─► ReasoningModule    baseline prompt + harness sections -> plan
            │
            └─◄ execute_plan        run_segment + Guard (may end a plan early)

openai_play.py and openai_agent.py are NOT modified. A harness result is
only interpretable next to the baseline if the baseline did not move.

Everything the baseline loop documents still holds and is not repeated
here: one env.step() is one NES frame; gym 0.25.2's 4-tuple API; the
emulator is frozen while we wait on the API.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import gym

gym.logger.set_level(gym.logger.ERROR)

import config                                                   # noqa: E402
from openai_agent import ACTION_NAMES, OpenAIMarioAgent, describe_plan  # noqa: E402
from openai_play import build_raw_env, land                     # noqa: E402
from video_utils import encode_video, next_run_dir, write_highlights  # noqa: E402

from harness import settings as S                               # noqa: E402
from harness.guard import Guard                                 # noqa: E402
from harness.llm import BudgetExhausted, HarnessLLM             # noqa: E402
from harness.memory import MemoryModule                         # noqa: E402
from harness.perception import PerceptionModule                 # noqa: E402
from harness.ram_state import death_cause, describe, read_state  # noqa: E402
from harness.reasoning import ReasoningModule                   # noqa: E402
from harness.recorder import StateRecorder                      # noqa: E402

SCRIPTED_UNSTICK = [(6, 16), (3, 30), (4, 28)]      # identical to openai_play.py


def execute_plan(env, plan, frames_rgb, meta, ctx, guard):
    """openai_play.run_segment over a whole plan, with a guard hook.

    Returns (done, info, executed, interrupt). `executed` is the plan as it
    actually ran -- (action, frames) with real frame counts -- which is
    what the memory module reports back to the model. A plan the guard cut
    short must not be remembered as the plan that was asked for.
    """
    info = ctx["info"]
    done = False
    executed = []
    interrupt = None
    done_in_plan = 0
    for seg_idx, (action, frames) in enumerate(plan):
        n = 0
        for _ in range(frames):
            _, _, done, info = env.step(action)
            ctx["frame"] += 1
            n += 1
            done_in_plan += 1
            if ctx["frame"] % config.CAPTURE_EVERY_N_FRAMES == 0:
                frames_rgb.append(env.unwrapped.screen.copy())
                meta.append({
                    "decision": ctx["decision"],
                    "x_pos": int(info.get("x_pos", 0)),
                    "time": int(info.get("time", 0)),
                    "action_name": ACTION_NAMES[action],
                })
            if done or info.get("flag_get", False):
                break
            if guard is not None:
                interrupt = guard.check(plan, seg_idx, n, done_in_plan,
                                        ctx["frame"], ctx["decision"])
                if interrupt:
                    break
        executed.append((action, n))
        if done or info.get("flag_get", False) or interrupt:
            break
    ctx["info"] = info
    return done, info, executed, interrupt


def print_header():
    print(f"[harness] preset: {S.PRESET}")
    print(f"[harness] model: {config.OPENAI_MODEL} (api style: {config.OPENAI_API_STYLE})"
          + (f", aux model: {S.AUX_MODEL}" if S.AUX_MODEL else ""))
    print(f"[harness] perception={S.PERCEPTION} map={S.ASCII_MAP} scaffold={S.SCAFFOLD} "
          f"motion_frames={S.MOTION_FRAMES}")
    print(f"[harness] memory={S.MEMORY} reflection={S.REFLECTION} guard={S.GUARD} "
          f"lessons={S.LESSONS}")
    print(f"[harness] budget: {config.MAX_DECISIONS} decisions, {config.MAX_FRAMES} frames, "
          f"{S.MAX_API_CALLS} billed calls (all modules)")
    if S.LESSONS == "write":
        print("[harness] NOTE: lessons=write -- this run reads and WRITES "
              "harness_memory/, so it is not an independent sample.")
    print()


def main(llm=None) -> dict:
    if llm is None and not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is not set. In a Codespace it should arrive as a "
                 "Codespaces secret; locally, put it in a .env you do not commit.")

    llm = llm or HarnessLLM()
    env = StateRecorder(
        build_raw_env(),
        depth=max(S.VELOCITY_WINDOW, S.MOTION_FRAMES) + 2,
        keep_screens=S.MOTION_FRAMES > 0,
    )
    perception = PerceptionModule(llm, env)
    memory = MemoryModule(llm)
    reasoning = ReasoningModule(llm)
    guard = Guard(env) if S.GUARD else None

    print_header()
    env.reset()
    _, _, _, info = env.step(0)       # reset() yields no info on gym 0.25

    frames_rgb, meta = [], []
    ctx = {"frame": 1, "decision": 0, "info": info}
    trace, decision_log, deaths = [], [], []

    prev_x = int(info.get("x_pos", 0))
    prev_plan_str = None
    prev_frames = 0
    landing_frames = 0
    landings = 0
    stuck = 0
    done = False
    stop_reason = "decision budget exhausted"
    last_percept = None

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
        print(f"decision {decision:03d} | x={x_pos} t={state['time']} stuck={stuck} "
              f"calls={llm.total_calls}", flush=True)

        scripted = False
        parse_failure = False
        result = None
        try:
            if stuck >= config.STUCK_FALLBACK_AFTER:
                plan = list(SCRIPTED_UNSTICK)
                note = "SCRIPTED UNSTICK (no API call)"
                scripted = True
                print(f"  {note}: {describe_plan(plan)}")
                stuck = 0
            else:
                percept = perception.perceive()
                last_percept = percept
                if S.REFLECTION != "off":
                    current = OpenAIMarioAgent.build_telemetry(state)
                    for extra in (percept.ram_text, percept.vision_text):
                        if extra:
                            current += "\n" + extra
                    memory.maybe_reflect(current)
                    if memory.reflection and memory.reflection_age == 0:
                        print(f"  reflection: {memory.reflection[:140]}")
                result = reasoning.decide(state, percept, memory.context_sections(x_pos))
                plan, note = result["plan"], result["note"]
                parse_failure = result["parse_failure"]
                print(f"  model -> {describe_plan(plan)}")
                if note:
                    print(f"  note: {note}")
        except BudgetExhausted as exc:
            stop_reason = "api call budget exhausted"
            print(f"\n[harness] {exc}")
            break

        frames_before = ctx["frame"]
        # The scripted unstick runs unguarded: its run-up toward the pipe
        # that stopped Mario is the whole point, and a guard re-look there
        # would spend a call to cancel the one plan that costs none.
        done, info, executed, interrupt = execute_plan(
            env, plan, frames_rgb, meta, ctx, None if scripted else guard)
        if interrupt:
            print(f"  GUARD: cut plan after {ctx['frame'] - frames_before} frames -- {interrupt}")

        if not done and not info.get("flag_get", False):
            done, info, landed_in, _ = land(env, frames_rgb, meta, ctx)
            if landed_in:
                landing_frames += landed_in
                landings += 1

        prev_frames = ctx["frame"] - frames_before
        new_x = int(info.get("x_pos", 0))
        after = read_state(env.ram_ago(0)[0])
        ran = [(a, f) for a, f in executed if f > 0]

        # With no guard firing, executed == plan until the episode ends,
        # so the baseline preset sends the baseline's exact prev_plan.
        prev_plan_str = (describe_plan(ran) + " (cut short by guard)") if interrupt \
            else describe_plan(plan)

        memory.record({
            "decision": decision, "x_before": x_pos, "x_after": new_x,
            "frames": prev_frames, "executed": describe_plan(ran) if ran else "nothing",
            "note": note, "guard": interrupt, "scripted": scripted,
            "parse_failure": parse_failure,
            "ended_airborne": (not after.grounded) and not done,
        })
        decision_log.append({
            "decision": decision, "x_before": x_pos, "x_after": new_x,
            "plan": [{"action": a, "action_name": ACTION_NAMES[a], "frames": f} for a, f in plan],
            "executed": [{"action": a, "action_name": ACTION_NAMES[a], "frames": f} for a, f in ran],
            "note": note, "guard": interrupt,
        })
        if config.SAVE_TRACE and result is not None:
            trace.append({
                "decision": decision,
                "prompt": result["text"],
                "response": result["raw"],
                "plan": plan,
                "executed": ran,
                "guard": interrupt,
                "reflection": memory.reflection if S.REFLECTION != "off" else None,
            })

        if interrupt:
            pass                          # a cut plan is not evidence of being stuck
        elif new_x - x_pos < config.STUCK_MIN_DX:
            stuck += 1
        else:
            stuck = 0
        prev_x = x_pos

        if info.get("flag_get", False):
            stop_reason = "flag reached"
            print("\n*** FLAG GET ***")
            break
        if done:
            cause = death_cause(env.ram_ago(0)[0], info)
            deaths.append({"decision": decision, "x": new_x, "cause": cause})
            stop_reason = "died"
            print(f"\n[harness] episode ended ({cause}) at x={new_x}")
            final_state = describe(last_percept.ram) if last_percept else ""
            memory.on_death(new_x, cause, final_state)
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

    # ------------------------------------------------------------ output

    run_dir = next_run_dir(Path(config.RUNS_DIR))
    encode_video(frames_rgb, run_dir / "run.mp4", fps=config.VIDEO_FPS)
    write_highlights(frames_rgb, meta, run_dir / "frames", every_n=config.SAVE_EVERY_N_FRAMES)

    usage = llm.summary()
    summary = {
        "run_dir": str(run_dir),
        "agent": "harness",
        "harness_preset": S.PRESET,
        "env": config.ENV_NAME,
        "model": config.OPENAI_MODEL,
        "api_style": config.OPENAI_API_STYLE,
        "reasoning_effort": config.OPENAI_REASONING_EFFORT or None,
        "screen_upscale": config.SCREEN_UPSCALE,
        "image_detail": config.IMAGE_DETAIL,
        "decisions": len(decision_log),
        # Same key as openai_play.py so tools/astra_budget_test.sh-style
        # scripts read both. Here it is EVERY billed call, all modules.
        "api_calls": usage["total"]["calls"],
        "failed_calls": usage["total"]["failed"],
        "input_tokens": usage["total"]["input_tokens"],
        "output_tokens": usage["total"]["output_tokens"],
        "api_seconds": usage["total"]["seconds"],
        "api_usage_by_purpose": usage,
        "px_per_call": round(final_x / max(usage["total"]["calls"], 1), 1),
        "nes_frames": ctx["frame"],
        "landings": landings,
        "landing_frames": landing_frames,
        "captured_frames": len(frames_rgb),
        "final_x_position": final_x,
        "flag_get": flag,
        "stop_reason": stop_reason,
        "death_cause": deaths[-1]["cause"] if deaths else None,
        "deaths": deaths,
        "guard_interrupts": len(guard.interrupts) if guard else 0,
        "guard_log": guard.interrupts if guard else [],
        "reflections": memory.reflections,
        "lessons": {
            "mode": S.LESSONS,
            "path": str(memory.lessons.path),
            "loaded": memory.lessons.loaded,
            "used_x": sorted(memory.lessons.used),
            "written": memory.lessons.written,
        },
        "harness_settings": S.as_dict(),
        "video_fps": config.VIDEO_FPS,
        "capture_every_n_frames": config.CAPTURE_EVERY_N_FRAMES,
        "decision_log": decision_log,
        "created_at": datetime.now().isoformat(),
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=4, default=str)
    print(f"Summary saved to: {run_dir / 'summary.json'}")

    if config.SAVE_TRACE and trace:
        with open(run_dir / "trace.jsonl", "w") as f:
            for row in trace:
                f.write(json.dumps(row, default=str) + "\n")
        print(f"Trace saved to: {run_dir / 'trace.jsonl'}")

    print(f"\nflag_get: {flag}")
    print(f"final_x: {final_x}")
    print(f"api_calls: {usage['total']['calls']} "
          + ", ".join(f"{k}={v['calls']}" for k, v in usage.items() if k != "total")
          + f" ({usage['total']['input_tokens']} in / {usage['total']['output_tokens']} out tokens)")
    print(f"guard_interrupts: {summary['guard_interrupts']}")
    print(f"stopped because: {stop_reason}"
          + (f" ({summary['death_cause']})" if summary["death_cause"] else ""))

    env.close()
    return summary


if __name__ == "__main__":
    main()
