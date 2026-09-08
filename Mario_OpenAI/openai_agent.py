"""
The policy: one 240x256 RGB screenshot plus recent telemetry in, a short
sequence of (action, frames) segments out.

WHY A SEQUENCE AND NOT ONE ACTION
---------------------------------
The obvious design is {"action": 4, "hold_steps": 6}. It does not work
for Mario, because the single thing that decides whether this experiment
produces a flag run or a video of a man walking into a pipe is JUMP
TIMING, and "one action held for N frames" cannot express timing at all.
"Run right for 20 frames, THEN jump" is one intention; forcing it into
two round trips means the second screenshot arrives after the jump point
has already passed.

So the model returns up to MAX_SEGMENTS_PER_PLAN segments, executed
back-to-back without another API call. Same cost as one decision, but
the model gets to place the jump inside the plan.

WHY THE EMULATOR STANDS STILL
-----------------------------
nes-py is synchronous. Nothing advances between env.step() calls, so
while we are waiting on the API Mario is frozen mid-stride, not falling
into a pit. API latency costs wall-clock time and nothing else. This is
what makes a ~2 s-per-decision agent viable at all.
"""

import base64
import json
import re
import time

import cv2
import numpy as np
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from openai import OpenAI

import config

# The 7-action set, read straight from the same constant Mario_AWS
# trains against. Index i here IS action i there.
ACTION_NAMES = ["+".join(combo) for combo in SIMPLE_MOVEMENT]
N_ACTIONS = len(SIMPLE_MOVEMENT)

ACTION_MENU = "\n".join(f"  {i} = {name}" for i, name in enumerate(ACTION_NAMES))

SYSTEM_PROMPT = f"""You are playing Super Mario Bros. (NES), World 1-1.

You see one screenshot of the current game screen and some telemetry.
You reply with a short PLAN: a sequence of controller inputs to execute
before you get to look again. The emulator is paused while you think, so
take the time to look carefully -- but note that once you commit a plan,
you are blind until it finishes.

CONTROLLER (exactly these 7 inputs, by index):
{ACTION_MENU}

B is the run/dash button. A is jump.

NES PHYSICS YOU MUST ACCOUNT FOR:
- The game runs at 60 frames per second. Every duration you give is in
  FRAMES. 30 frames is half a second.
- Jump height is controlled by HOW LONG A IS HELD. Roughly: 8 frames is
  a small hop clearing one block; 20-30 frames is a full jump clearing
  a wide gap or a tall pipe. Holding beyond ~32 frames adds nothing.
- Momentum is real and builds slowly. A running jump needs a run-up:
  hold 3 (right+B) for 20-40 frames BEFORE the jump, then use 4
  (right+A+B) for the jump itself. A standing jump from action 2 will
  not clear a wide pit. BUT SEE THE TIMING RULE BELOW -- a 30-frame
  run-up into an obstacle 15 frames away is a death, not a run-up.
- You cannot change direction much in mid-air. Commit before you leave
  the ground.
- Goombas and Koopas die if you land on them from above. Touching them
  from the side costs a life.

STRATEGY:
- Moving right is almost always correct. Only use 6 (left) to back up
  for a run-up when you are stuck against an obstacle.
- Prefer 3 (right+B) as the default travel action: it is faster and
  builds the momentum later jumps need.
- Look at the ground ahead of Mario. A gap in the floor tiles or a
  green pipe is the thing you must plan a jump for. Plan the run-up and
  the jump in the SAME reply.
- If the telemetry says you are stuck, whatever you did last time did
  not work. Do something different -- back up and take a longer run-up,
  or jump earlier.

TIMING RULE -- DO THIS ARITHMETIC BEFORE YOU CHOOSE THE FIRST DURATION:
The telemetry gives you Mario's current speed in PIXELS PER FRAME. The
screen is 256 px wide and Mario sits near the left third of it, so an
obstacle at the middle of the screen is roughly 90 px away and one at
the right edge is roughly 170 px.

    frames_until_contact  =  distance_in_pixels / speed_in_px_per_frame

At a typical running speed of 3 px/frame that is 30 frames to
mid-screen and 55 to the right edge. YOUR FIRST SEGMENT MUST BE SHORTER
THAN frames_until_contact for the nearest hazard. If a Goomba is
mid-screen and you are running at 3 px/frame, a 24-frame run-up puts
the jump 6 frames before impact, which is too late to leave the ground.
Estimate the gap, divide, then subtract a margin.

This is the single most common way to lose: the plan is right and the
first number is too big, so Mario runs into the thing he intended to
jump over. When unsure, make the first segment SHORTER. A wasted
decision costs one API call; a death ends the run.

REPLY FORMAT -- return JSON and nothing else, no markdown fences:
{{"note": "<max 15 words on what you see and intend>",
  "plan": [{{"action": <0-6>, "frames": <1-{config.MAX_FRAMES_PER_SEGMENT}>}}]}}

The plan may hold up to {config.MAX_SEGMENTS_PER_PLAN} segments and
{config.MAX_FRAMES_PER_PLAN} frames in total.
"""


class PlanError(RuntimeError):
    """The model replied with something that is not a usable plan."""


class OpenAIMarioAgent:
    """Wraps the API call, the prompt, and plan validation.

    Keeps running totals of calls and tokens so summary.json can report
    what the run actually cost.
    """

    def __init__(self) -> None:
        # The client reads OPENAI_API_KEY from the environment. It is
        # never passed in or logged: in a Codespace it arrives as a
        # Codespaces secret, locally from a gitignored .env.
        self.client = OpenAI(
            timeout=config.OPENAI_TIMEOUT_S,
            max_retries=0,   # retried explicitly below, with logging
        )
        self.model = config.OPENAI_MODEL
        self.api_calls = 0
        self.failed_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.api_seconds = 0.0

    # ------------------------------------------------------- encoding

    @staticmethod
    def encode_screen(screen_rgb: np.ndarray) -> str:
        """240x256 RGB ndarray -> base64 PNG data payload.

        Nearest-neighbour upscale: this is pixel art, so interpolation
        would invent detail that is not in the ROM, and the vision
        encoder gets more patches per sprite for free.
        """
        if config.SCREEN_UPSCALE > 1:
            screen_rgb = cv2.resize(
                screen_rgb,
                None,
                fx=config.SCREEN_UPSCALE,
                fy=config.SCREEN_UPSCALE,
                interpolation=cv2.INTER_NEAREST,
            )
        # cv2 wants BGR on the way in and gives back PNG bytes.
        ok, buf = cv2.imencode(".png", cv2.cvtColor(screen_rgb, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError("PNG encode failed")
        return base64.b64encode(buf.tobytes()).decode("ascii")

    # ------------------------------------------------------ telemetry

    @staticmethod
    def build_telemetry(state: dict) -> str:
        """Everything vision cannot see: motion, history, and failure.

        A single frame has no velocity in it. Without prev_x the model
        cannot tell a running Mario from a Mario pressed against a wall,
        because both look identical.

        SPEED IS THE LOAD-BEARING LINE. A still frame gives no scale for
        converting "that Goomba looks close" into a frame count, and
        getting that conversion wrong is the observed cause of death:
        the model plans a 24-frame run-up toward an obstacle 15 frames
        away and runs straight into it. Handing over px/frame lets it do
        the division instead of guessing at it.
        """
        dx = state["x_pos"] - state["prev_x"]
        frames = state.get("prev_frames") or 0
        if frames > 0:
            speed = dx / frames
            speed_line = (f"current_speed: {speed:.1f} px/frame "
                          f"({dx} px over the last {frames} frames)")
            # Pre-computed so the model never has to trust its own
            # arithmetic on the two numbers that matter most.
            if speed > 0.5:
                speed_line += (f"\n frames_to_midscreen: ~{90 / speed:.0f} "
                               f"| frames_to_right_edge: ~{170 / speed:.0f} "
                               "(at this speed)")
        else:
            speed_line = ("current_speed: unknown (first decision; "
                          "Mario starts stationary and takes ~30 frames "
                          "of action 3 to reach full running speed)")

        lines = [
            f"decision: {state['decision']}",
            f"x_position: {state['x_pos']}",
            f"y_position: {state['y_pos']}",
            f"time_remaining: {state['time']}",
            speed_line,
            f"previous_plan: {state['prev_plan'] or 'none (first decision)'}",
            f"stuck_counter: {state['stuck']}",
            f"decisions_remaining: {state['budget_left']}",
        ]
        if state["stuck"] > 0:
            lines.append(
                "WARNING: Mario is not making progress. The previous plan "
                "failed. Try a different approach, not a longer one."
            )
        return "\n".join(lines)

    # ----------------------------------------------------- API call

    def _call_responses(self, telemetry: str, b64_png: str):
        kwargs = dict(
            model=self.model,
            instructions=SYSTEM_PROMPT,
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": telemetry},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{b64_png}",
                        "detail": config.IMAGE_DETAIL,
                    },
                ],
            }],
            max_output_tokens=config.OPENAI_MAX_OUTPUT_TOKENS,
        )
        if config.OPENAI_REASONING_EFFORT:
            kwargs["reasoning"] = {"effort": config.OPENAI_REASONING_EFFORT}

        resp = self.client.responses.create(**kwargs)
        usage = getattr(resp, "usage", None)
        if usage:
            self.input_tokens += getattr(usage, "input_tokens", 0) or 0
            self.output_tokens += getattr(usage, "output_tokens", 0) or 0
        return resp.output_text

    def _call_chat(self, telemetry: str, b64_png: str):
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": telemetry},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{b64_png}",
                        "detail": config.IMAGE_DETAIL,
                    }},
                ]},
            ],
            response_format={"type": "json_object"},
            max_tokens=config.OPENAI_MAX_OUTPUT_TOKENS,
        )
        usage = getattr(resp, "usage", None)
        if usage:
            self.input_tokens += getattr(usage, "prompt_tokens", 0) or 0
            self.output_tokens += getattr(usage, "completion_tokens", 0) or 0
        return resp.choices[0].message.content

    def _call(self, telemetry: str, b64_png: str) -> str:
        """One decision, with bounded retries.

        Retrying is cheap here in a way it is not for a real-time agent:
        the emulator is frozen, so a 30 s backoff costs 30 s of your life
        and zero frames of game state.
        """
        last_err = None
        for attempt in range(config.OPENAI_MAX_RETRIES):
            t0 = time.time()
            try:
                if config.OPENAI_API_STYLE == "chat":
                    text = self._call_chat(telemetry, b64_png)
                else:
                    text = self._call_responses(telemetry, b64_png)
                self.api_calls += 1
                self.api_seconds += time.time() - t0
                return text
            except Exception as exc:            # noqa: BLE001 -- see below
                # Broad by design: the SDK raises a family of typed
                # errors (rate limit, timeout, connection, 5xx) and the
                # response to all of them here is identical. A genuinely
                # fatal error (bad key, unknown model) will exhaust the
                # retries in a few seconds and surface with its message
                # intact.
                self.failed_calls += 1
                self.api_seconds += time.time() - t0
                last_err = exc
                backoff = 2 ** attempt
                print(f"  [api] attempt {attempt + 1}/{config.OPENAI_MAX_RETRIES} "
                      f"failed ({type(exc).__name__}: {exc}); "
                      f"retrying in {backoff}s", flush=True)
                time.sleep(backoff)
        raise RuntimeError(f"OpenAI call failed after "
                           f"{config.OPENAI_MAX_RETRIES} attempts: {last_err}")

    # -------------------------------------------------- plan parsing

    @staticmethod
    def parse_plan(text: str) -> tuple:
        """Extract and CLAMP a plan. Never trusts the model's numbers.

        Clamping rather than rejecting is deliberate: a plan of
        [{"action": 4, "frames": 400}] is a usable intention with a bad
        duration, and turning it into 60 frames costs nothing. Rejecting
        it would spend another API call to get the same idea back.
        """
        if not text:
            raise PlanError("empty response")

        # Tolerate markdown fences and any preamble, even though the
        # prompt forbids both -- one stray ``` should not end a run.
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise PlanError(f"no JSON object in response: {text[:200]!r}")
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise PlanError(f"bad JSON ({exc}): {text[:200]!r}") from exc

        raw = obj.get("plan")
        if not isinstance(raw, list) or not raw:
            raise PlanError(f"no usable 'plan' list in {obj!r}")

        plan = []
        total = 0
        for seg in raw[:config.MAX_SEGMENTS_PER_PLAN]:
            if not isinstance(seg, dict):
                continue
            try:
                action = int(seg.get("action"))
                frames = int(seg.get("frames"))
            except (TypeError, ValueError):
                continue
            if not 0 <= action < N_ACTIONS:
                continue
            frames = max(1, min(frames, config.MAX_FRAMES_PER_SEGMENT))
            if total + frames > config.MAX_FRAMES_PER_PLAN:
                frames = config.MAX_FRAMES_PER_PLAN - total
                if frames <= 0:
                    break
            plan.append((action, frames))
            total += frames

        if not plan:
            raise PlanError(f"every segment was invalid: {obj!r}")

        note = str(obj.get("note", ""))[:120]
        return plan, note

    # --------------------------------------------------------- public

    def decide(self, screen_rgb: np.ndarray, state: dict) -> dict:
        """One billed decision. Returns plan, note, and the raw reply."""
        telemetry = self.build_telemetry(state)
        b64 = self.encode_screen(screen_rgb)
        text = self._call(telemetry, b64)
        try:
            plan, note = self.parse_plan(text)
        except PlanError as exc:
            # A malformed reply is not worth a second call. Default to
            # "run right" -- the safest possible action in 1-1 -- log it,
            # and let the next screenshot correct the mistake.
            print(f"  [plan] {exc} -- falling back to right+B for 20 frames")
            plan, note = [(3, 20)], f"PARSE FAILURE: {exc}"
        return {"plan": plan, "note": note, "raw": text, "telemetry": telemetry}


def describe_plan(plan: list) -> str:
    """'right+B x30 -> right+A+B x24' -- for the console and the overlay."""
    return " -> ".join(f"{ACTION_NAMES[a]} x{f}" for a, f in plan)
