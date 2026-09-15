"""
Harness settings. Same convention as ../config.py: every knob is an
environment variable with a default. Presets set the defaults; any
HARNESS_* variable you export overrides the preset for that one knob.

Named settings.py, not config.py, on purpose: openai_agent.py does
`import config`, and a second module of that name one directory down is
exactly the kind of shadowing that works on one machine and not another.

THE ARCHITECTURE, AND WHERE IT CAME FROM
----------------------------------------
lmgame-org/GamingAgent splits a gaming agent into three modules around a
shared Observation:

    perception  ->  memory (trajectory + reflection)  ->  reasoning

This package keeps that split and adds two things the Mario_OpenAI runs
showed were missing:

    guard    -- during plan execution, force a fresh look when Mario is
                about to reach a hazard the plan does not jump over.
                It never presses a button for the model.
    lessons  -- cross-run memory: after a death, one reflection call
                writes a lesson keyed to the x position, and later runs
                see it on approach. Reflexion-style, and the only piece
                that makes runs NOT independent (see LESSONS below).

PRESETS
-------
  baseline     Everything off. The reasoning request is BYTE-IDENTICAL to
               openai_play.py's (tools/harness_selftest.py asserts this),
               so a harness run and every run already in the logbook are
               measured on the same ruler.
  gamingagent  Faithful to GamingAgent's Mario config: VLM perception
               pass on a 5x5-grid screenshot, grid scaffold on the
               reasoning image, trajectory memory, reflection EVERY
               decision. Vision only, no RAM shown to the model.
               ~3 billed calls per decision.
  ram          RAM perception (exact distances, frames-to-contact, a
               tile map), ruler scaffold, trajectory memory, reflection
               only after a failed decision, guard on. ~1.1 calls per
               decision. The default.
  full         ram + cross-run lessons. Runs are no longer independent:
               never put it in an A/B table next to anything else.
"""

import os

PRESETS = {
    "baseline": dict(
        PERCEPTION="off", ASCII_MAP=False, SCAFFOLD="none", MEMORY=False,
        REFLECTION="off", GUARD=False, LESSONS="off", MOTION_FRAMES=0,
    ),
    "gamingagent": dict(
        PERCEPTION="vision", ASCII_MAP=False, SCAFFOLD="grid", MEMORY=True,
        REFLECTION="every", GUARD=False, LESSONS="off", MOTION_FRAMES=0,
    ),
    "ram": dict(
        PERCEPTION="ram", ASCII_MAP=True, SCAFFOLD="ruler", MEMORY=True,
        REFLECTION="event", GUARD=True, LESSONS="off", MOTION_FRAMES=0,
    ),
    "full": dict(
        PERCEPTION="ram", ASCII_MAP=True, SCAFFOLD="ruler", MEMORY=True,
        REFLECTION="event", GUARD=True, LESSONS="write", MOTION_FRAMES=0,
    ),
}

PRESET = os.environ.get("HARNESS_PRESET", "ram")
if PRESET not in PRESETS:
    raise SystemExit(f"HARNESS_PRESET={PRESET!r} is not one of {sorted(PRESETS)}")
_P = PRESETS[PRESET]


def _str(name, choices=None):
    v = os.environ.get(f"HARNESS_{name}", str(_P[name]))
    if choices and v not in choices:
        raise SystemExit(f"HARNESS_{name}={v!r} must be one of {choices}")
    return v


def _bool(name):
    raw = os.environ.get(f"HARNESS_{name}")
    return _P[name] if raw is None else raw.lower() in ("1", "true", "yes", "on")


def _int(name, default):
    return int(os.environ.get(f"HARNESS_{name}", _P.get(name, default)))


# ---------------------------------------------------------- perception

# off | ram | vision | both
#   ram     symbolic state from NES RAM, zero API cost (harness/ram_state.py)
#   vision  GamingAgent-style: a separate VLM call describes the gridded
#           screenshot as JSON, and that JSON is handed to the reasoner
#   both    both blocks; for measuring whether vision adds anything once
#           exact numbers are present
PERCEPTION = _str("PERCEPTION", ("off", "ram", "vision", "both"))

# The 13x21 tile map in the ram block. ~120 tokens. Separate knob because
# the hazard list alone may be enough, and that is worth measuring.
ASCII_MAP = _bool("ASCII_MAP")

# none | grid | ruler -- drawn on the reasoning image AFTER upscaling
#   grid   GamingAgent's labelled grid (HARNESS_GRID, default 5x5)
#   ruler  tick marks ahead of Mario every HARNESS_RULER_STEP_PX, labelled
#          in px and in frames at current speed. Aimed squarely at the
#          baseline's documented cause of death: no scale in a still frame.
SCAFFOLD = _str("SCAFFOLD", ("none", "grid", "ruler"))
GRID = os.environ.get("HARNESS_GRID", "5x5")
RULER_STEP_PX = int(os.environ.get("HARNESS_RULER_STEP_PX", 32))

# Send a second screenshot from N frames earlier so velocity is VISIBLE,
# which the Mario_OpenAI README names as "the most likely single
# improvement". 0 = off. Roughly doubles image tokens; in no preset
# because nothing has measured it yet.
MOTION_FRAMES = _int("MOTION_FRAMES", 0)

# Frames between the two RAM snapshots used to MEASURE enemy velocity.
# 1 is too short: a Goomba moves 0.625 px/frame, so single-frame deltas
# alternate 0 and 1 and read as "stationary" half the time.
VELOCITY_WINDOW = int(os.environ.get("HARNESS_VELOCITY_WINDOW", 8))

# ------------------------------------------------------------- memory

MEMORY = _bool("MEMORY")
MEMORY_WINDOW = int(os.environ.get("HARNESS_MEMORY_WINDOW", 5))

# off | event | every
#   every  GamingAgent's behaviour: a reflection call before each decision
#   event  only after a decision that made no progress, ended airborne,
#          was cut short by the guard, or failed to parse
REFLECTION = _str("REFLECTION", ("off", "event", "every"))
# An event reflection stays in the prompt for this many decisions.
REFLECTION_TTL = int(os.environ.get("HARNESS_REFLECTION_TTL", 3))

# off | read | write
#   read   inject existing lessons, never add any. Lets you evaluate a
#          frozen lesson file with independent runs.
#   write  read, and add a lesson after every death.
LESSONS = _str("LESSONS", ("off", "read", "write"))
LESSONS_DIR = os.environ.get("HARNESS_LESSONS_DIR", "harness_memory")
LESSONS_MAX = int(os.environ.get("HARNESS_LESSONS_MAX", 24))
LESSONS_LOOKAHEAD_PX = int(os.environ.get("HARNESS_LESSONS_LOOKAHEAD_PX", 256))
LESSONS_PER_PROMPT = int(os.environ.get("HARNESS_LESSONS_PER_PROMPT", 3))

# -------------------------------------------------------------- guard

GUARD = _bool("GUARD")
# Only hazards this close (in frames at current closing speed) can
# trigger a re-look. Must be long enough that the model, handed a fresh
# screenshot, still has room to jump: ~10-15 frames before an enemy.
GUARD_FRAMES = int(os.environ.get("HARNESS_GUARD_FRAMES", 24))
# The plan runs at least this many frames before the guard may interrupt
# it. At frame 0 the model is looking at the very screen the guard would
# resend -- interrupting there buys the same answer at full price.
GUARD_MIN_EXEC = int(os.environ.get("HARNESS_GUARD_MIN_EXEC", 6))

# ------------------------------------------------------ calls and spend

# HARD CEILING ON BILLED CALLS ACROSS EVERY MODULE. MAX_DECISIONS in
# ../config.py still caps decisions, but a decision can now cost up to
# three calls (gamingagent preset), so decisions no longer bound spend.
MAX_API_CALLS = int(os.environ.get("HARNESS_MAX_API_CALLS", 120))

# Perception, reflection and lesson calls are text-shaped and short.
# Point them at a cheaper model if the flagship is expensive; the model
# id is recorded per purpose in summary.json either way.
AUX_MODEL = os.environ.get("HARNESS_AUX_MODEL", "")          # "" = OPENAI_MODEL
AUX_REASONING_EFFORT = os.environ.get("HARNESS_AUX_REASONING_EFFORT", "low")
AUX_MAX_OUTPUT_TOKENS = int(os.environ.get("HARNESS_AUX_MAX_OUTPUT_TOKENS", 2000))

# Print every assembled reasoning prompt to the console.
PRINT_PROMPTS = os.environ.get("HARNESS_PRINT_PROMPTS", "0") == "1"


def as_dict() -> dict:
    """Everything above, for summary.json."""
    return {k: v for k, v in globals().items()
            if k.isupper() and k not in ("PRESETS",)}


def any_module_on() -> bool:
    return (PERCEPTION != "off" or SCAFFOLD != "none" or MEMORY
            or REFLECTION != "off" or GUARD or LESSONS != "off"
            or MOTION_FRAMES > 0)
