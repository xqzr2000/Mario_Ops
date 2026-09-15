"""GPT-6 Astra policy for Mario_OpenAI v2.

Key differences from v1:
  * receives a temporal packet of screens rather than one still image;
  * uses Responses API Structured Outputs instead of regex JSON recovery;
  * returns structured perception + controller plan + desired reobserve horizon;
  * the harness caps that horizon when risk is high / Mario is airborne / stuck;
  * records cached-input, cache-write and reasoning-token usage.
"""

from __future__ import annotations

import base64
import json
import time
from typing import Iterable

import cv2
import numpy as np
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from openai import OpenAI

import config


ACTION_NAMES = ["+".join(combo) for combo in SIMPLE_MOVEMENT]
N_ACTIONS = len(SIMPLE_MOVEMENT)
ACTION_MENU = "\n".join(f"  {i} = {name}" for i, name in enumerate(ACTION_NAMES))

RISK_CAPS = {
    "low": config.RISK_LOW_MAX_FRAMES,
    "medium": config.RISK_MEDIUM_MAX_FRAMES,
    "high": config.RISK_HIGH_MAX_FRAMES,
    "critical": config.RISK_CRITICAL_MAX_FRAMES,
}


SYSTEM_PROMPT = f"""You control Super Mario Bros. (NES), World 1-1.
Your objective is to move right and reach the flag without dying.

You receive 1-{config.TEMPORAL_FRAMES} screenshots ordered OLDEST to NEWEST.
Use changes between screenshots to infer motion: Mario's horizontal speed,
whether he is rising/falling, enemy motion, scrolling, and closing distance.
The emulator is paused while you answer.

CONTROLLER -- reply only with these action indexes:
{ACTION_MENU}

B is run/dash. A is jump. The game runs at 60 NES frames/second.
Prefer right+B for safe travel. Use right+A+B for running jumps. Avoid left
unless recovering from a blockage. Jump early rather than trying for a
pixel-perfect last-moment jump.

Return a SHORT controller plan and a reobserve_after value. reobserve_after is
how many NES frames you think can safely execute before another vision call.
Use long horizons only for obviously empty ground. Use short horizons around
enemies, pipes, pits, during a jump/fall, or whenever uncertain.

The executor may interrupt your plan even earlier when your reported risk is
high. Therefore put the most urgent action FIRST. If a jump must start now,
start with the jump; do not put a long run-up in front of it.

Your scene assessment must describe the nearest meaningful hazard visible in
the NEWEST frame. risk means risk during the next controller horizon, not the
overall difficulty of the level.
"""


DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "scene": {
            "type": "object",
            "properties": {
                "airborne": {"type": "boolean"},
                "hazard_type": {
                    "type": "string",
                    "enum": [
                        "none",
                        "enemy",
                        "pipe",
                        "pit",
                        "block",
                        "stairs",
                        "unknown",
                    ],
                },
                "hazard_distance": {
                    "type": "string",
                    "enum": ["none", "far", "medium", "near", "immediate"],
                },
                "risk": {
                    "type": "string",
                    "enum": ["low", "medium", "high", "critical"],
                },
            },
            "required": ["airborne", "hazard_type", "hazard_distance", "risk"],
            "additionalProperties": False,
        },
        "confidence": {"type": "number"},
        "note": {"type": "string"},
        "plan": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "action": {"type": "integer"},
                    "frames": {"type": "integer"},
                },
                "required": ["action", "frames"],
                "additionalProperties": False,
            },
        },
        "reobserve_after": {"type": "integer"},
    },
    "required": ["scene", "confidence", "note", "plan", "reobserve_after"],
    "additionalProperties": False,
}


class PlanError(RuntimeError):
    """The model returned an unusable decision despite the output schema."""


def describe_plan(plan: Iterable[tuple[int, int]]) -> str:
    return " -> ".join(f"{ACTION_NAMES[a]} x{f}" for a, f in plan)


def truncate_plan(plan: list[tuple[int, int]], max_frames: int) -> list[tuple[int, int]]:
    """Return only the executable prefix of plan within max_frames."""
    max_frames = max(1, int(max_frames))
    used = 0
    out: list[tuple[int, int]] = []
    for action, frames in plan:
        remaining = max_frames - used
        if remaining <= 0:
            break
        take = min(frames, remaining)
        if take > 0:
            out.append((action, take))
            used += take
        if used >= max_frames:
            break
    return out


class OpenAIMarioAgent:
    def __init__(self) -> None:
        self.client = OpenAI(timeout=config.OPENAI_TIMEOUT_S, max_retries=0)
        self.model = config.OPENAI_MODEL

        self.api_calls = 0
        self.failed_calls = 0
        self.input_tokens = 0
        self.cached_input_tokens = 0
        self.cache_write_tokens = 0
        self.output_tokens = 0
        self.reasoning_tokens = 0
        self.api_seconds = 0.0

        self._previous_response_id: str | None = None
        self._chain_turns = 0

    # ------------------------------------------------------------- encoding
    @staticmethod
    def encode_screen(screen_rgb: np.ndarray) -> str:
        if config.SCREEN_UPSCALE > 1:
            screen_rgb = cv2.resize(
                screen_rgb,
                None,
                fx=config.SCREEN_UPSCALE,
                fy=config.SCREEN_UPSCALE,
                interpolation=cv2.INTER_NEAREST,
            )
        ok, buf = cv2.imencode(".png", cv2.cvtColor(screen_rgb, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError("PNG encode failed")
        return base64.b64encode(buf.tobytes()).decode("ascii")

    # ------------------------------------------------------------ telemetry
    @staticmethod
    def build_telemetry(state: dict, observations: list[tuple[int, np.ndarray]]) -> str:
        dx = int(state["x_pos"]) - int(state["prev_x"])
        frames = int(state.get("prev_frames") or 0)
        speed = dx / frames if frames > 0 else None

        newest_frame = observations[-1][0] if observations else int(state.get("frame", 0))
        ages = [newest_frame - frame_no for frame_no, _ in observations]

        lines = [
            f"decision: {state['decision']}",
            f"nes_frame: {state.get('frame', 0)}",
            f"x_pos: {state['x_pos']}",
            f"y_pos: {state['y_pos']}",
            f"level_timer: {state['time']}",
            f"stuck_decisions: {state['stuck']}",
            f"decision_budget_left: {state['budget_left']}",
            f"temporal_frame_ages: {ages} NES frames (oldest -> newest)",
        ]
        if speed is None:
            lines.append("measured_horizontal_speed: unknown (first decision)")
        else:
            lines.append(
                f"measured_horizontal_speed: {speed:.2f} px/NES-frame "
                f"({dx:+d}px over {frames} frames)"
            )
        if state.get("prev_plan"):
            lines.append(f"previous_executed_plan: {state['prev_plan']}")
        if state.get("stuck", 0):
            lines.append(
                "progress_warning: recent controller input made little forward progress"
            )
        return "\n".join(lines)

    def _build_input(
        self,
        telemetry: str,
        observations: list[tuple[int, np.ndarray]],
    ) -> list[dict]:
        if not observations:
            raise ValueError("at least one observation frame is required")

        newest = observations[-1][0]
        content: list[dict] = [
            {
                "type": "input_text",
                "text": (
                    telemetry
                    + "\n\nInspect the chronological screenshots below. "
                    "The last image is the current game state."
                ),
            }
        ]
        for frame_no, screen in observations:
            age = newest - frame_no
            content.append(
                {
                    "type": "input_text",
                    "text": f"Screenshot age: {age} NES frames; frame={frame_no}",
                }
            )
            content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{self.encode_screen(screen)}",
                    "detail": config.IMAGE_DETAIL,
                }
            )
        return [{"role": "user", "content": content}]

    # --------------------------------------------------------------- usage
    def _record_usage(self, response) -> None:
        usage = getattr(response, "usage", None)
        if not usage:
            return
        self.input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
        self.output_tokens += int(getattr(usage, "output_tokens", 0) or 0)

        in_details = getattr(usage, "input_tokens_details", None)
        if in_details:
            self.cached_input_tokens += int(getattr(in_details, "cached_tokens", 0) or 0)
            self.cache_write_tokens += int(
                getattr(in_details, "cache_write_tokens", 0) or 0
            )

        out_details = getattr(usage, "output_tokens_details", None)
        if out_details:
            self.reasoning_tokens += int(
                getattr(out_details, "reasoning_tokens", 0) or 0
            )

    @property
    def estimated_cost_usd(self) -> float:
        # input_tokens is total input; separate cached/write buckets so they are
        # not also charged at the ordinary input rate.
        ordinary = max(
            0,
            self.input_tokens - self.cached_input_tokens - self.cache_write_tokens,
        )
        total = (
            ordinary * config.INPUT_USD_PER_MTOK
            + self.cached_input_tokens * config.CACHED_INPUT_USD_PER_MTOK
            + self.cache_write_tokens * config.CACHE_WRITE_USD_PER_MTOK
            + self.output_tokens * config.OUTPUT_USD_PER_MTOK
        ) / 1_000_000.0
        return total

    # --------------------------------------------------------------- API
    def _call(self, input_items: list[dict]):
        last_err = None
        for attempt in range(config.OPENAI_MAX_RETRIES):
            t0 = time.time()
            try:
                kwargs = {
                    "model": self.model,
                    "instructions": SYSTEM_PROMPT,
                    "input": input_items,
                    "max_output_tokens": config.OPENAI_MAX_OUTPUT_TOKENS,
                    "text": {
                        "format": {
                            "type": "json_schema",
                            "name": "mario_decision",
                            "strict": True,
                            "schema": DECISION_SCHEMA,
                        }
                    },
                    "prompt_cache_key": config.PROMPT_CACHE_KEY,
                    "prompt_cache_options": {"ttl": config.PROMPT_CACHE_TTL},
                    "store": config.STATEFUL_TURNS > 0,
                }
                if config.OPENAI_REASONING_EFFORT:
                    kwargs["reasoning"] = {"effort": config.OPENAI_REASONING_EFFORT}
                if config.STATEFUL_TURNS > 0 and self._previous_response_id:
                    kwargs["previous_response_id"] = self._previous_response_id

                response = self.client.responses.create(**kwargs)
                self.api_calls += 1
                self.api_seconds += time.time() - t0
                self._record_usage(response)

                if config.STATEFUL_TURNS > 0:
                    self._previous_response_id = response.id
                    self._chain_turns += 1
                    if self._chain_turns >= config.STATEFUL_TURNS:
                        self._previous_response_id = None
                        self._chain_turns = 0
                return response
            except Exception as exc:  # SDK exposes several transient subclasses
                self.failed_calls += 1
                self.api_seconds += time.time() - t0
                last_err = exc
                backoff = 2**attempt
                print(
                    f"  [api] attempt {attempt + 1}/{config.OPENAI_MAX_RETRIES} "
                    f"failed ({type(exc).__name__}: {exc}); retrying in {backoff}s",
                    flush=True,
                )
                time.sleep(backoff)

        raise RuntimeError(
            f"OpenAI call failed after {config.OPENAI_MAX_RETRIES} attempts: {last_err}"
        )

    # ----------------------------------------------------------- validation
    @staticmethod
    def _parse_structured(text: str) -> dict:
        if not text:
            raise PlanError("empty structured-output response")
        try:
            obj = json.loads(text)
        except json.JSONDecodeError as exc:
            raise PlanError(f"invalid structured JSON: {exc}") from exc

        raw_plan = obj.get("plan")
        if not isinstance(raw_plan, list) or not raw_plan:
            raise PlanError("missing plan")

        plan: list[tuple[int, int]] = []
        total = 0
        for seg in raw_plan[: config.MAX_SEGMENTS_PER_PLAN]:
            action = int(seg["action"])
            frames = int(seg["frames"])
            if not 0 <= action < N_ACTIONS:
                raise PlanError(f"invalid action {action}")
            frames = max(1, min(frames, config.MAX_FRAMES_PER_SEGMENT))
            if total + frames > config.MAX_FRAMES_PER_PLAN:
                frames = config.MAX_FRAMES_PER_PLAN - total
            if frames <= 0:
                break
            plan.append((action, frames))
            total += frames
        if not plan:
            raise PlanError("plan became empty after safety clamps")

        obj["_validated_plan"] = plan
        return obj

    @staticmethod
    def _effective_horizon(obj: dict, state: dict) -> int:
        scene = obj["scene"]
        requested = int(obj["reobserve_after"])
        horizon = min(
            max(requested, config.MIN_REOBSERVE_FRAMES),
            config.MAX_REOBSERVE_FRAMES,
            RISK_CAPS[scene["risk"]],
        )

        if scene["airborne"]:
            horizon = min(horizon, config.AIRBORNE_MAX_FRAMES)
        if int(state.get("stuck", 0)) > 0:
            horizon = min(horizon, config.STUCK_MAX_FRAMES)
        if scene["hazard_distance"] == "near":
            horizon = min(horizon, config.NEAR_HAZARD_MAX_FRAMES)
        elif scene["hazard_distance"] == "immediate":
            horizon = min(horizon, config.IMMEDIATE_HAZARD_MAX_FRAMES)

        return max(1, horizon)

    # --------------------------------------------------------------- public
    def decide(
        self,
        observations: list[tuple[int, np.ndarray]],
        state: dict,
    ) -> dict:
        telemetry = self.build_telemetry(state, observations)
        input_items = self._build_input(telemetry, observations)
        response = self._call(input_items)
        raw = response.output_text

        try:
            obj = self._parse_structured(raw)
            proposed_plan = obj["_validated_plan"]
            horizon = self._effective_horizon(obj, state)
            executable_plan = truncate_plan(proposed_plan, horizon)
            if not executable_plan:
                raise PlanError("adaptive horizon removed the whole plan")

            scene = obj["scene"]
            return {
                "plan": executable_plan,
                "proposed_plan": proposed_plan,
                "note": str(obj.get("note", ""))[:160],
                "scene": scene,
                "confidence": max(0.0, min(1.0, float(obj.get("confidence", 0.0)))),
                "requested_reobserve_after": int(obj["reobserve_after"]),
                "effective_reobserve_after": horizon,
                "raw": raw,
                "telemetry": telemetry,
                "response_id": getattr(response, "id", None),
            }
        except (PlanError, KeyError, TypeError, ValueError) as exc:
            # Do not pay for a second repair call. Reobserve quickly instead.
            print(f"  [plan] {exc} -- fallback right+B x8", flush=True)
            return {
                "plan": [(3, 8)],
                "proposed_plan": [(3, 8)],
                "note": f"STRUCTURED OUTPUT FAILURE: {exc}",
                "scene": {
                    "airborne": False,
                    "hazard_type": "unknown",
                    "hazard_distance": "near",
                    "risk": "critical",
                },
                "confidence": 0.0,
                "requested_reobserve_after": 8,
                "effective_reobserve_after": 8,
                "raw": raw,
                "telemetry": telemetry,
                "response_id": getattr(response, "id", None),
            }
