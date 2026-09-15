#!/usr/bin/env python
"""
tools/harness_selftest.py -- offline checks for the harness. No API key,
no network, no spend. Run from anywhere:

    python Mario_OpenAI/tools/harness_selftest.py          # all checks
    python Mario_OpenAI/tools/harness_selftest.py ram      # one check

Checks (each prints PASS/FAIL; exit code is non-zero if any failed):

  ram        RAM perception against the live emulator: x matches info,
             the first Goomba is found ahead at Mario's level closing at
             ~0.6 px/frame, the first pipe is found, the map renders.
  scaffold   grid and ruler images encode, are the upscaled size, and
             differ from the plain screenshot. Writes PNGs to look at.
  identity   THE ONE THAT MATTERS FOR COMPARISONS: with HARNESS_PRESET=
             baseline, the reasoning request equals the one
             OpenAIMarioAgent sends for the same screen and state -- for
             both the responses and chat API styles.
  oracle     a ~20-line rule bot that reads ONLY read_state() output
             must clear 1-1 to the flag. The emulator is deterministic, so
             this is a regression test for the whole perception layer:
             if an edit to ram_state.py breaks pits, stairs, pipes or
             enemy contact anywhere in the level, the bot stops reaching
             the flag. It also sets the ceiling for interpreting harness
             runs -- the information in the PERCEPTION block is enough to
             finish the level, so a model that has it and still dies is
             failing at reasoning or timing, not at seeing.
  e2e        harness_play.main() end to end for every preset against a
             fake OpenAI client: output files land, guard fires under
             `ram` on a plan that runs into the first Goomba, a lesson
             file is written under `full`, the call budget is enforced.

Preset-dependent checks run in subprocesses, because harness.settings is
read once at import -- exactly like config.py.
"""

import json
import os
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent                         # Mario_OpenAI/
sys.path.insert(0, str(PROJECT))
os.chdir(PROJECT)
warnings.filterwarnings("ignore")

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f" -- {detail}" if detail else ""))


def quiet_gym():
    import gym
    gym.logger.set_level(gym.logger.ERROR)


# ------------------------------------------------------------ fake client

class FakeClient:
    """Stands in for openai.OpenAI. Records every request; answers by role."""

    def __init__(self, reasoning_reply=None):
        self.requests = []
        self.reasoning_reply = reasoning_reply or json.dumps(
            {"note": "fake: run right", "plan": [{"action": 3, "frames": 60},
                                                 {"action": 3, "frames": 30}]})
        self.responses = SimpleNamespace(create=self._responses)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._chat))

    def _answer(self, system: str) -> str:
        if "You are playing Super Mario Bros" in system:
            return self.reasoning_reply
        if "computer-vision" in system:
            return '{"mario": {"col": 1, "row": 3, "airborne": false}, "enemies": []}'
        if "lessons" in system:
            return "fake lesson: jump when the Goomba is 50 px ahead"
        return "fake reflection: jump earlier"

    def _responses(self, **kw):
        self.requests.append(("responses", kw))
        usage = SimpleNamespace(input_tokens=100, output_tokens=10)
        return SimpleNamespace(output_text=self._answer(kw["instructions"]), usage=usage)

    def _chat(self, **kw):
        self.requests.append(("chat", kw))
        usage = SimpleNamespace(prompt_tokens=100, completion_tokens=10)
        msg = SimpleNamespace(content=self._answer(kw["messages"][0]["content"]))
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=usage)


# ------------------------------------------------------------------ checks

def check_ram():
    print("ram")
    quiet_gym()
    from collections import deque
    from openai_play import build_raw_env
    from harness.ram_state import ascii_map, describe, read_state
    env = build_raw_env()
    env.reset()
    hist = deque(maxlen=9)
    info = {}
    for _ in range(95):                   # right+B until the first Goomba is close
        _, _, _, info = env.step(3)
        hist.append(env.unwrapped.ram.copy())
    st = read_state(hist[-1], hist[0], 8)
    check("level x matches info['x_pos']", st.x == int(info["x_pos"]),
          f"ram {st.x} vs info {info['x_pos']}")
    check("grounded while running on flat ground", st.grounded)
    check("running speed ~3 px/frame", abs(st.speed - 3.0) < 0.3, f"{st.speed:.2f}")
    goombas = [e for e in st.enemies if e.name == "Goomba" and e.gap_px >= 0]
    check("first Goomba found ahead", bool(goombas))
    if goombas:
        g = goombas[0]
        check("Goomba velocity measured ~-0.6 px/frame",
              g.vx is not None and -0.8 < g.vx < -0.4, f"vx={g.vx}")
        check("Goomba has a frames-to-contact figure", g.frames_to_contact is not None,
              f"{g.frames_to_contact}")
    pipes = [t for t in st.terrain if t.kind == "wall" and t.is_pipe]
    check("first pipe found in lookahead", bool(pipes),
          f"{[(t.gap_px, t.size_tiles) for t in pipes]}")
    m = ascii_map(st)
    check("map has Mario and an enemy", "M" in m and "E" in m)
    text = describe(st)
    check("describe() mentions Goomba and PIPE", "Goomba" in text and "PIPE" in text)
    env.close()


def check_scaffold():
    print("scaffold")
    quiet_gym()
    import base64
    import cv2
    import numpy as np
    import config
    from openai_play import build_raw_env
    from harness.ram_state import read_state
    from harness.scaffold import encode_for_reasoning, encode_plain
    env = build_raw_env()
    env.reset()
    for _ in range(80):
        env.step(3)
    screen = env.unwrapped.screen.copy()
    st = read_state(env.unwrapped.ram)
    out_dir = Path(tempfile.gettempdir()) / "harness_selftest"
    out_dir.mkdir(exist_ok=True)
    plain = encode_plain(screen)
    for name in ("grid", "ruler"):
        b64 = encode_for_reasoning(screen, name, "5x5", st, 32)
        img = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)
        k = config.SCREEN_UPSCALE
        check(f"{name}: decodes at upscaled size",
              img is not None and img.shape[:2] == (240 * k, 256 * k),
              f"{None if img is None else img.shape}")
        check(f"{name}: differs from plain screenshot", b64 != plain)
        cv2.imwrite(str(out_dir / f"scaffold_{name}.png"), img)
    check("none: identical to OpenAIMarioAgent.encode_screen",
          encode_for_reasoning(screen, "none") == plain)
    print(f"        images written to {out_dir}/ -- worth one look")
    env.close()


def check_oracle():
    print("oracle")
    quiet_gym()
    from openai_play import build_raw_env
    from harness.ram_state import death_cause, read_state
    from harness.recorder import StateRecorder
    env = StateRecorder(build_raw_env(), depth=10)
    env.reset()
    _, _, _, info = env.step(0)
    jump = 0
    for _ in range(5000):
        now, _ = env.ram_ago(0)
        then, gap = env.ram_ago(8)
        st = read_state(now, then if gap else None, max(gap, 1))
        if jump > 0:
            action, jump = 4, jump - 1
        else:
            action = 3
            if st.grounded and (
                    any(e.hazard and e.frames_to_contact is not None
                        and e.frames_to_contact < 12 for e in st.enemies)
                    or any((t.kind == "wall" and t.gap_px < 16)
                           or (t.kind == "pit" and t.gap_px < 2) for t in st.terrain)):
                action, jump = 4, 32
        _, _, done, info = env.step(action)
        if done or info.get("flag_get"):
            break
    check("rule bot on RAM perception reaches the flag", bool(info.get("flag_get")),
          f"x={info.get('x_pos')} ended: {death_cause(env.ram_ago(0)[0], info)}")
    env.close()


def _identity_child(api_style):
    """Runs inside a subprocess with HARNESS_PRESET=baseline."""
    quiet_gym()
    import config
    config.OPENAI_API_STYLE = api_style
    from openai_play import build_raw_env
    from openai_agent import OpenAIMarioAgent
    from harness.llm import HarnessLLM
    from harness.memory import MemoryModule
    from harness.perception import PerceptionModule
    from harness.reasoning import ReasoningModule
    from harness.recorder import StateRecorder

    env = StateRecorder(build_raw_env())
    env.reset()
    for _ in range(40):
        env.step(3)
    state = {"decision": 2, "x_pos": 120, "y_pos": 79, "time": 398, "prev_x": 40,
             "prev_frames": 40, "prev_plan": "right+B x40", "stuck": 0, "budget_left": 58}

    base_client = FakeClient()
    agent = OpenAIMarioAgent.__new__(OpenAIMarioAgent)
    agent.client, agent.model = base_client, config.OPENAI_MODEL
    agent.api_calls = agent.failed_calls = agent.input_tokens = agent.output_tokens = 0
    agent.api_seconds = 0.0
    agent.decide(env.unwrapped.screen.copy(), state)

    h_client = FakeClient()
    llm = HarnessLLM(client=h_client)
    percept = PerceptionModule(llm, env).perceive()
    ReasoningModule(llm).decide(state, percept, MemoryModule(llm).context_sections(120))

    a, b = base_client.requests, h_client.requests
    same = len(a) == 1 and len(b) == 1 and a[0] == b[0]
    print(json.dumps({"same": same, "n": [len(a), len(b)],
                      "diff_keys": [] if same or not (a and b) else
                      [k for k in set(a[0][1]) | set(b[0][1]) if a[0][1].get(k) != b[0][1].get(k)]}))


def check_identity():
    print("identity")
    for style in ("responses", "chat"):
        env = dict(os.environ, HARNESS_PRESET="baseline", PYTHONWARNINGS="ignore")
        out = subprocess.run([sys.executable, __file__, "_identity", style], env=env,
                             capture_output=True, text=True, cwd=PROJECT)
        line = (out.stdout.strip().splitlines() or ["{}"])[-1]
        try:
            res = json.loads(line)
        except json.JSONDecodeError:
            res = {"same": False, "error": out.stderr[-400:]}
        check(f"baseline preset == openai_agent request ({style})", res.get("same") is True,
              "" if res.get("same") else json.dumps(res))


def _e2e_child(preset):
    import harness_play
    client = FakeClient()
    from harness.llm import HarnessLLM
    summary = harness_play.main(llm=HarnessLLM(client=client))
    print("E2E_RESULT " + json.dumps({
        "preset": preset,
        "final_x": summary["final_x_position"],
        "calls": summary["api_calls"],
        "by_purpose": {k: v["calls"] for k, v in summary["api_usage_by_purpose"].items()},
        "guard": summary["guard_interrupts"],
        "stop": summary["stop_reason"],
        "cause": summary["death_cause"],
        "run_dir": summary["run_dir"],
        "lessons_written": len(summary["lessons"]["written"]),
    }))


def check_e2e():
    print("e2e")
    tmp = Path(tempfile.mkdtemp(prefix="harness_e2e_"))
    base = dict(os.environ, RUNS_DIR=str(tmp / "runs"), HARNESS_LESSONS_DIR=str(tmp / "mem"),
                MAX_DECISIONS="8", PYTHONWARNINGS="ignore", OPENAI_API_KEY="")
    results = {}
    for preset in ("baseline", "gamingagent", "ram", "full"):
        env = dict(base, HARNESS_PRESET=preset)
        out = subprocess.run([sys.executable, __file__, "_e2e", preset], env=env,
                             capture_output=True, text=True, cwd=PROJECT)
        line = next((l for l in out.stdout.splitlines() if l.startswith("E2E_RESULT ")), None)
        if line is None:
            check(f"{preset}: runs to completion", False, (out.stderr or out.stdout)[-600:])
            continue
        r = json.loads(line[len("E2E_RESULT "):])
        results[preset] = r
        rd = Path(r["run_dir"])
        check(f"{preset}: runs to completion and writes outputs",
              (rd / "summary.json").exists() and (rd / "run.mp4").exists(),
              f"x={r['final_x']} calls={r['by_purpose']} guard={r['guard']} "
              f"stop={r['stop']} cause={r['cause']}")

    if "baseline" in results:
        r = results["baseline"]
        check("baseline: only reasoning calls", set(r["by_purpose"]) <= {"reasoning", "total"})
        check("baseline: blind 90-frame run-ups die to the first Goomba (as on the real runs)",
              r["cause"] == "hit by Goomba", r["cause"])
    if "gamingagent" in results:
        bp = results["gamingagent"]["by_purpose"]
        check("gamingagent: perception + reflection calls happen",
              bp.get("perception", 0) > 0 and bp.get("reflection", 0) > 0, str(bp))
    if "ram" in results:
        r = results["ram"]
        check("ram: guard interrupts the same blind run-up", r["guard"] > 0, f"guard={r['guard']}")
    if "full" in results:
        r = results["full"]
        lessons = tmp / "mem"
        wrote = any(lessons.glob("lessons_*.json")) if lessons.exists() else False
        check("full: a death writes a lesson file",
              (r["lessons_written"] > 0 and wrote) or r["cause"] is None,
              f"written={r['lessons_written']} cause={r['cause']}")

    env = dict(base, HARNESS_PRESET="gamingagent", HARNESS_MAX_API_CALLS="4")
    out = subprocess.run([sys.executable, __file__, "_e2e", "budget"], env=env,
                         capture_output=True, text=True, cwd=PROJECT)
    line = next((l for l in out.stdout.splitlines() if l.startswith("E2E_RESULT ")), None)
    r = json.loads(line[len("E2E_RESULT "):]) if line else {}
    check("HARNESS_MAX_API_CALLS stops the run cleanly",
          r.get("stop") == "api call budget exhausted" and r.get("calls", 99) <= 4, str(r))


CHECKS = {"ram": check_ram, "scaffold": check_scaffold, "oracle": check_oracle,
          "identity": check_identity, "e2e": check_e2e}

if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "_identity":
        _identity_child(args[1])
        sys.exit(0)
    if args and args[0] == "_e2e":
        _e2e_child(args[1])
        sys.exit(0)
    for name in (args or list(CHECKS)):
        CHECKS[name]()
    failed = RESULTS.count(False)
    print(f"\n{len(RESULTS) - failed}/{len(RESULTS)} passed")
    sys.exit(1 if failed else 0)
