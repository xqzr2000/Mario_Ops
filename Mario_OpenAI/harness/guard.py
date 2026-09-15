"""
Guard: during plan execution, stop the plan and ask again when Mario is
about to reach a hazard that the plan does not jump over in time.

THE FAILURE IT TARGETS is the one config.py documents at length: "the
plan is right and the first number is too big, so Mario runs into the
thing he intended to jump over." x=312, twice, identically. The model's
intent was fine; the run-up length was wrong, and nothing between the
API call and the Goomba could notice.

WHAT IT DOES NOT DO: press a button. It never jumps, never substitutes
an action, never edits a plan. It ends the plan early and the loop takes
a fresh screenshot, so the model makes the jump decision itself -- with
the hazard ~15-20 frames away instead of ~90. That keeps "how far did the
MODEL get" an honest question. A guard that jumped would be a scripted
bot with an expensive advisor attached.

It costs an extra decision (a billed call) each time it fires, and each
hazard can fire it once. Its count is in summary.json as guard_interrupts.
"""

from typing import Optional

from harness import settings as S
from harness.ram_state import JUMP_ACTIONS, read_state

# How many frames before contact the plan's jump must START, per hazard.
# Enemies: jump early (the prompt's own advice, and Goombas close on you).
# Walls: a few frames of run-up distance is fine.
# Pits: jumping slightly AFTER the front edge crosses is still safe --
# Mario's 16-px body is over ground for ~5 frames at 3 px/frame.
MARGIN = {"enemy": 8, "wall": 4, "pit": -4}

# A fired key cannot fire again within this many frames. Enemy slots are
# reused as new enemies spawn, so enemy keys must expire; terrain keys
# are by level column and would never recur anyway.
REFIRE_FRAMES = 120


def _frames_until_jump(plan, seg_idx: int, done_in_seg: int) -> float:
    until = plan[seg_idx][1] - done_in_seg
    for action, frames in plan[seg_idx + 1:]:
        if action in JUMP_ACTIONS:
            return until
        until += frames
    return float("inf")


class Guard:

    def __init__(self, recorder):
        self.rec = recorder
        self.fired = {}
        self.interrupts = []

    def check(self, plan, seg_idx: int, done_in_seg: int, done_in_plan: int,
              frame: int, decision: int) -> Optional[str]:
        if done_in_plan < S.GUARD_MIN_EXEC:
            return None
        if plan[seg_idx][0] in JUMP_ACTIONS:
            return None

        ram, _ = self.rec.ram_ago(0)
        prev, gap = self.rec.ram_ago(S.VELOCITY_WINDOW)
        st = read_state(ram, prev if gap else None, max(gap, 1))
        if not st.grounded or st.dying:
            return None

        until_jump = _frames_until_jump(plan, seg_idx, done_in_seg)
        candidates = []
        for e in st.enemies:
            if e.hazard and e.frames_to_contact is not None:
                candidates.append(("enemy", ("enemy", e.slot, e.type_id),
                                   e.frames_to_contact,
                                   f"{e.name} {e.gap_px} px ahead"))
        for t in st.terrain:
            if t.frames_to_contact is not None:
                label = "pipe" if t.is_pipe else t.kind
                candidates.append((t.kind, (t.kind, t.col), t.frames_to_contact,
                                   f"{label} {t.gap_px} px ahead"))
        candidates.sort(key=lambda c: c[2])

        for kind, key, contact, desc in candidates:
            if contact > S.GUARD_FRAMES:
                break
            if until_jump <= contact - MARGIN[kind]:
                continue                  # the plan jumps in time -- let it
            last = self.fired.get(key)
            if last is not None and frame - last < REFIRE_FRAMES:
                continue
            self.fired[key] = frame
            nxt = "no jump in the rest of the plan" if until_jump == float("inf") \
                else f"plan's next jump in {until_jump:.0f} frames"
            reason = f"{desc}, contact ~{contact:.0f} frames, {nxt}"
            self.interrupts.append({"decision": decision, "frame": frame,
                                    "x": st.x, "reason": reason})
            return reason
        return None
