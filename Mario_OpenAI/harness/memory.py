"""
Memory module: what happened, what it means, and what earlier runs learned.

Three layers, each independently switchable:

  trajectory   the last HARNESS_MEMORY_WINDOW decisions as MEASURED
               outcomes: x before/after, frames actually run, whether the
               guard cut the plan, whether Mario ended airborne. The
               baseline sends only the previous plan string and a stuck
               counter; this sends the evidence behind them.

  reflection   GamingAgent's reflection: a text-only call that critiques
               recent play. "every" reproduces GamingAgent (one call per
               decision); "event" fires only after a decision that went
               wrong, which is where a critique has something to say.

  lessons      cross-RUN memory on disk (harness_memory/lessons_<env>.json).
               After a death, one call writes a lesson keyed to the death
               x; later runs see lessons for the stretch just ahead.
               Reflexion-style improvement with no gradient anywhere --
               and the reason the `full` preset's runs are not
               independent samples.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path

import config
from harness import prompts
from harness import settings as S


class LessonStore:
    """A small JSON file of {x, cause, text}, newest wins per 48-px bucket."""

    BUCKET_PX = 48

    def __init__(self, mode: str):
        self.mode = mode
        env_tag = config.ENV_NAME.replace("/", "_")
        self.path = Path(S.LESSONS_DIR) / f"lessons_{env_tag}.json"
        self.lessons = []
        if mode != "off" and self.path.exists():
            try:
                self.lessons = json.loads(self.path.read_text()).get("lessons", [])
            except (json.JSONDecodeError, OSError) as exc:
                # A corrupt lesson file must not end a run. Say so loudly
                # and play without it rather than guess at its contents.
                print(f"  [lessons] could not read {self.path}: {exc} -- ignoring it")
        self.loaded = len(self.lessons)
        self.written = []
        self.used = set()

    def relevant(self, x: int) -> list:
        if self.mode == "off":
            return []
        near = [L for L in self.lessons
                if x - 32 <= L["x"] <= x + S.LESSONS_LOOKAHEAD_PX]
        near.sort(key=lambda L: L["x"])
        picked = near[:S.LESSONS_PER_PROMPT]
        self.used.update(L["x"] for L in picked)
        return picked

    def add(self, x: int, cause: str, text: str, meta: dict) -> None:
        if self.mode != "write":
            return
        bucket = x // self.BUCKET_PX
        self.lessons = [L for L in self.lessons if L["x"] // self.BUCKET_PX != bucket]
        entry = {"x": x, "cause": cause, "text": text.strip()[:400],
                 "created": datetime.now().isoformat(timespec="seconds"), **meta}
        self.lessons.append(entry)
        self.lessons = sorted(self.lessons, key=lambda L: L["created"])[-S.LESSONS_MAX:]
        self.lessons.sort(key=lambda L: L["x"])
        self.written.append(entry)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"env": config.ENV_NAME, "lessons": self.lessons}, indent=2))
        os.replace(tmp, self.path)       # atomic: a Ctrl-C mid-write keeps the old file


class MemoryModule:

    def __init__(self, llm):
        self.llm = llm
        self.entries = []
        self.reflection = None
        self.reflection_age = 0
        self.reflections = []
        self.lessons = LessonStore(S.LESSONS)

    # ---------------------------------------------------------- recording

    def record(self, entry: dict) -> None:
        self.entries.append(entry)

    @staticmethod
    def _fmt(e: dict) -> str:
        dx = e["x_after"] - e["x_before"]
        flags = []
        if e.get("guard"):
            flags.append(f"CUT SHORT BY GUARD ({e['guard']})")
        if e.get("ended_airborne"):
            flags.append("ended AIRBORNE")
        if e.get("scripted"):
            flags.append("scripted unstick")
        if e.get("parse_failure"):
            flags.append("PARSE FAILURE, fallback used")
        tail = f" [{'; '.join(flags)}]" if flags else ""
        return (f"#{e['decision']:03d} x {e['x_before']}->{e['x_after']} ({dx:+d} px "
                f"in {e['frames']} frames) plan: {e['executed']}{tail}"
                + (f" | note: {e['note']}" if e.get("note") else ""))

    def trajectory_text(self, n: int = None) -> str:
        n = n or S.MEMORY_WINDOW
        rows = self.entries[-n:]
        return "\n".join(self._fmt(e) for e in rows) if rows else "none yet (first decision)"

    # --------------------------------------------------------- reflection

    def _trigger(self) -> str:
        if S.REFLECTION == "every":
            return "routine (reflection runs every decision)" if self.entries else ""
        if S.REFLECTION != "event" or not self.entries:
            return ""
        e = self.entries[-1]
        if e.get("scripted"):
            return ""
        if e.get("guard"):
            return f"the guard cut the last plan short: {e['guard']}"
        if e.get("parse_failure"):
            return "the last reply could not be parsed"
        if e["x_after"] - e["x_before"] < config.STUCK_MIN_DX:
            return "the last plan made no forward progress"
        if e.get("ended_airborne"):
            return "the last plan ended with Mario still in the air"
        return ""

    def maybe_reflect(self, current_text: str) -> None:
        trigger = self._trigger()
        if not trigger:
            self.reflection_age += 1
            if self.reflection and self.reflection_age >= S.REFLECTION_TTL:
                self.reflection = None
            return
        text = self.llm.complete(
            "reflection", prompts.REFLECTION_SYSTEM,
            prompts.REFLECTION_PROMPT.format(trajectory=self.trajectory_text(),
                                             current=current_text, trigger=trigger),
            json_mode=False,
        ).strip()
        if text:
            self.reflection = text[:600]
            self.reflection_age = 0
            self.reflections.append({"decision": len(self.entries) + 1,
                                     "trigger": trigger, "text": self.reflection})

    # ------------------------------------------------------------ context

    def context_sections(self, x: int) -> list:
        """(title, body) pairs for the reasoning prompt."""
        out = []
        if S.MEMORY:
            out.append(("RECENT DECISIONS (oldest first, measured outcomes)",
                        self.trajectory_text()))
        if S.REFLECTION != "off" and self.reflection:
            out.append(("REFLECTION", self.reflection))
        lessons = self.lessons.relevant(x)
        if lessons:
            out.append(("LESSONS FROM EARLIER ATTEMPTS (for the level ahead)",
                        "\n".join(f"- near x={L['x']} ({L['cause']}): {L['text']}"
                                  for L in lessons)))
        return out

    # ------------------------------------------------------------- deaths

    def on_death(self, death_x: int, cause: str, final_state: str) -> None:
        if self.lessons.mode != "write":
            return
        try:
            text = self.llm.complete(
                "lesson", prompts.LESSON_SYSTEM,
                prompts.LESSON_PROMPT.format(death_x=death_x, cause=cause,
                                             trajectory=self.trajectory_text(4),
                                             final_state=final_state),
                json_mode=False,
            ).strip()
        except Exception as exc:          # noqa: BLE001 -- includes BudgetExhausted
            # No call, no lesson text -- but the death itself is still
            # worth recording, labelled as exactly what it is.
            print(f"  [lessons] lesson call failed ({exc}); storing the bare death")
            text = ""
        if not text:
            text = f"(no lesson text) previous attempt died here: {cause}"
        self.lessons.add(death_x, cause, text, {
            "model": config.OPENAI_MODEL, "preset": S.PRESET,
            "written_at_unix": int(time.time()),
        })
        print(f"  [lessons] wrote lesson for x={death_x}: {text[:100]}")
