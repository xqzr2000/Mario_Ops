"""
Prompt text for the harness modules.

The reasoning system prompt is openai_agent.SYSTEM_PROMPT -- imported,
not copied -- plus HARNESS_ADDENDUM appended only when a module that
adds context is on. A prompt fix made in openai_agent.py therefore lands
in the baseline and every harness preset at once, and the baseline
preset stays identical to openai_play.py.

The module prompts are adapted from GamingAgent's retro_01_super_mario_bros
module_prompts.json (MIT), rewritten for this project's 7-action
SIMPLE_MOVEMENT space and its JSON plan format.
"""

from openai_agent import SYSTEM_PROMPT as BASELINE_SYSTEM_PROMPT


def harness_addendum(perception: str, scaffold: str, memory: bool,
                     reflection: bool, lessons: bool, motion_frames: int) -> str:
    parts = ["\nHARNESS CONTEXT -- extra sections may follow the telemetry:"]
    if perception in ("ram", "both"):
        parts.append(
            "- PERCEPTION (from game memory) is EXACT: positions, speeds and\n"
            "  frames-to-contact are read from the emulator, not estimated.\n"
            "  When it is present, use its distances INSTEAD of the\n"
            "  screen-position rule of thumb in the TIMING RULE above, and\n"
            "  trust it over your own visual estimate of distance. The\n"
            "  JUMP-EARLY advice still applies: contact figures are exact\n"
            "  now, but your plan still runs blind once committed.")
    if perception in ("vision", "both"):
        parts.append(
            "- VISUAL ANALYSIS is another model's description of the same\n"
            "  screenshot. It can be wrong; the image is ground truth.")
    if scaffold == "ruler":
        parts.append(
            "- The screenshot has a distance ruler: a cyan line at Mario's\n"
            "  front edge and yellow ticks ahead of him, each labelled with\n"
            "  pixels and, in the ground below, FRAMES at his current speed\n"
            "  (e.g. 64px / 21f). Frames are for STATIC things; an enemy\n"
            "  walking toward Mario arrives sooner than the tick says.")
    if scaffold == "grid":
        parts.append(
            "- The screenshot has a labelled (col,row) grid overlay. The\n"
            "  grid lines are not part of the game.")
    if motion_frames:
        parts.append(
            f"- The SECOND image is the screen {motion_frames} frames before\n"
            "  the first. Compare them to see which way things are moving.")
    if memory:
        parts.append(
            "- RECENT DECISIONS lists your last plans and what they actually\n"
            "  achieved. A plan that was CUT SHORT BY GUARD ran toward a\n"
            "  hazard without a jump early enough; the screen you see now\n"
            "  is the fresh look the guard forced. Plan the jump now.")
    if reflection:
        parts.append(
            "- REFLECTION is a short critique of recent play. Take it into\n"
            "  account, but the current screen outranks it.")
    if lessons:
        parts.append(
            "- LESSONS were written after earlier attempts DIED near these\n"
            "  x positions. They are the most reliable advice you have for\n"
            "  the stretch of level they name.")
    return "\n".join(parts) + "\n"


# ---------------------------------------------------- vision perception

VISION_PERCEPTION_SYSTEM = (
    "You are a computer-vision system analysing frames from Super Mario Bros. "
    "(NES). You describe what is on screen precisely. You do not choose actions."
)


def vision_perception_prompt(grid: str) -> str:
    rows, cols = (int(v) for v in grid.lower().split("x"))
    return f"""The screenshot has a {cols}x{rows} grid overlay; cells are labelled
(col,row) with (0,0) top-left and ({cols - 1},{rows - 1}) bottom-right.
The native screen is 256 px wide, so one grid column is ~{256 // cols} px.

Locate these, in grid cells:
- Mario
- enemies (Goomba = brown mushroom, Koopa = turtle), with which way they face/move
- pipes (green), with height in blocks
- pits (gaps in the ground)
- walls / stair blocks Mario must jump
- ? blocks and bricks overhead
- the flagpole, if visible

Also estimate, in PIXELS, the horizontal distance from Mario's front edge
to the nearest thing he must jump over or on.

Return ONLY this JSON, no markdown:
{{"mario": {{"col": int, "row": int, "airborne": bool}},
  "enemies": [{{"type": str, "col": int, "row": int, "moving": "left|right|still"}}],
  "pipes": [{{"col": int, "height_blocks": int}}],
  "pits": [{{"col": int, "width_blocks": int}}],
  "walls": [{{"col": int, "height_blocks": int}}],
  "flagpole": {{"col": int}} or null,
  "nearest_obstacle": {{"what": str, "distance_px": int}} or null}}"""


# ------------------------------------------------------------ reflection

REFLECTION_SYSTEM = (
    "You are the analyst for an AI playing Super Mario Bros. 1-1. You write "
    "short, specific, actionable critiques of its recent play. Numbers beat "
    "adjectives: say how many frames earlier, how many pixels shorter."
)

REFLECTION_PROMPT = """RECENT DECISIONS (oldest first):
{trajectory}

CURRENT STATE:
{current}

Why this reflection was requested: {trigger}

In at most 60 words: what did the last plan get wrong or right, and what
concretely should the NEXT plan do differently (which action, how many
frames, when to jump)? Plain text, no JSON, no preamble."""

# -------------------------------------------------------------- lessons

LESSON_SYSTEM = (
    "You write one-paragraph lessons for an AI that will attempt Super Mario "
    "Bros. 1-1 again. It will read your lesson on approach to the spot where "
    "it died. Be concrete: distances in px, durations in frames, action names."
)

LESSON_PROMPT = """The attempt DIED at level x={death_x}. Cause (from game memory): {cause}.

The last decisions before death (oldest first):
{trajectory}

State at the final decision:
{final_state}

Write ONE lesson, at most 45 words, for the next attempt approaching
x={death_x}: what to do there, with a number in it. Plain text, no JSON,
no preamble. Do not restate the cause; say what to do instead."""
