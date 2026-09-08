"""
Mario_OpenAI configuration.

Mirrors the shape of Mario_AWS/config.py -- every knob is an environment
variable with a sane default -- so the two projects are configured the
same way even though they share no code.

WHAT IS DELIBERATELY DIFFERENT FROM Mario_AWS/config.py:

  * There is no DEVICE, no STACK_SIZE, no IMAGE_SIZE. This project never
    builds a tensor. The "policy" is an HTTP call.

  * FRAME COUNTS ARE IN NES FRAMES, NOT AGENT STEPS. Mario_AWS wraps the
    env in SkipFrame(4), so one agent step there is four emulated frames.
    This project drives the RAW env, so one env.step() is exactly one
    NES frame at ~60 Hz. Every duration below is therefore a real frame
    count and can be reasoned about in seconds: 60 frames = 1 second.

  * There is a hard spend ceiling (MAX_DECISIONS). Every decision is a
    billed API call, and a model that walks Mario into a pipe will
    happily keep paying to look at the same pipe until the in-game timer
    runs out. See the note on that constant.
"""

import os

# --------------------------------------------------------------- game

# Same level and same ROM version as Mario_AWS, so "furthest x" numbers
# are directly comparable between the DQN agent and the vision model.
ENV_NAME = os.environ.get("MARIO_ENV_NAME", "SuperMarioBros-1-1-v0")

# SIMPLE_MOVEMENT, the identical 7-action set the DQN was trained on.
# Imported in openai_agent.py rather than duplicated here so there is
# exactly one definition of what action 4 means.

# ------------------------------------------------------------- OpenAI

# Model id. NOT a Codespaces secret: the model is not sensitive, and
# burying it in repo settings means a summary.json cannot be traced back
# to the model that produced it. Override per run:
#     OPENAI_MODEL=gpt-6-astra python openai_play.py
#
# VERIFY THIS DEFAULT before your first run -- OpenAI's model ids change
# faster than this file will. The authoritative list is your own
# account's, not the docs page:
#
#     python -c "from openai import OpenAI; \
#                print([m.id for m in OpenAI().models.list()])"
#
# NOTE on 2026-09-08: "gpt-5.6" WORKS but does not appear in that list --
# it is an alias for gpt-5.6-sol. The 5.6 generation enumerates as
# gpt-5.6-sol (flagship), gpt-5.6-terra and gpt-5.6-luna, with
# gpt-6-astra above them. The explicit id is pinned here rather than the
# alias so that a summary.json says exactly what ran.
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.6-sol")

# "responses" (current OpenAI surface) or "chat" (chat.completions).
# Both are implemented. If the SDK version in your image does not have
# client.responses, flip this to "chat" instead of downgrading anything.
OPENAI_API_STYLE = os.environ.get("OPENAI_API_STYLE", "responses")

# Reasoning effort, for models that expose it. Low by default: this is a
# 7-way classification against a 240x256 screenshot, not a proof, and
# every extra reasoning token is latency, money, AND output budget --
# see OPENAI_MAX_OUTPUT_TOKENS below, which is not independent of this
# setting the way it looks.
# Set to "" to omit the parameter entirely for models that reject it.
OPENAI_REASONING_EFFORT = os.environ.get("OPENAI_REASONING_EFFORT", "low")

# 2000 looks absurd for a reply that is ~40 tokens of JSON. It is not.
#
# ON A REASONING MODEL, REASONING TOKENS COUNT AGAINST THIS BUDGET. If
# the model spends the whole allowance thinking, there is nothing left
# to emit the answer with -- and the API does NOT raise. It returns
# success with an EMPTY output_text, which arrives here as
#
#     [plan] empty response -- falling back to right+B for 20 frames
#
# i.e. a silent degradation to a hardcoded action, billed at full price,
# that looks like the model gave a bad answer rather than no answer.
# This file shipped with 400 and decision 001 failed exactly this way on
# the first real run; 2000 fixed it with no other change.
#
# If you raise OPENAI_REASONING_EFFORT, raise this too. If you see
# PARSE FAILURE notes, check the "response" field in trace.jsonl before
# blaming the prompt: EMPTY means this, malformed means the prompt.
OPENAI_MAX_OUTPUT_TOKENS = int(os.environ.get("OPENAI_MAX_OUTPUT_TOKENS", 2000))

# Retries on a transient API error (429/5xx/timeout). The emulator is
# frozen while we retry, so a retry costs wall-clock time but no game
# state -- unlike a real-time agent, we can afford to be patient.
OPENAI_MAX_RETRIES = int(os.environ.get("OPENAI_MAX_RETRIES", 4))
OPENAI_TIMEOUT_S = float(os.environ.get("OPENAI_TIMEOUT_S", 60.0))

# ------------------------------------------------------- screen -> API

# The NES screen is 240x256. Mario is roughly 16 px tall in it. Upscaling
# with nearest-neighbour before sending costs nothing in fidelity (it is
# pixel art; there is no detail to invent) and gives the vision encoder
# more patches to work with. 3x -> 720x768.
#
# PNG, never JPEG: JPEG ringing smears sprite edges against the sky, and
# a blurred pixel is the difference between a Goomba and a brick.
SCREEN_UPSCALE = int(os.environ.get("SCREEN_UPSCALE", 3))

# "low" | "high" | "auto" -- passed through as the image detail hint.
IMAGE_DETAIL = os.environ.get("IMAGE_DETAIL", "high")

# ------------------------------------------------- decisions and spend

# HARD CEILING ON BILLED CALLS. This is the single most important number
# in this file. Without it, a stuck model burns one API call per ~1.5 s
# of game time until the 400-unit level timer expires -- several hundred
# calls for a run that never leaves the first pipe.
#
# 60 decisions x ~75 frames is ~4500 frames, ~75 s of game time, which is
# roughly a full 1-1 clear by a competent player. If a run legitimately
# needs more than this, raise it deliberately rather than by default.
MAX_DECISIONS = int(os.environ.get("MAX_DECISIONS", 60))

# Belt to the above suspenders: stop on frame count too, in case a plan
# validator bug lets through longer segments than intended.
MAX_FRAMES = int(os.environ.get("MAX_FRAMES", 9000))

# Plan shape limits. The model returns a SEQUENCE of (action, frames)
# segments rather than one action plus a hold, because "run right, then
# jump" is a single intention that one action cannot express -- and jump
# timing is where this whole experiment lives or dies.
#
# THESE THREE ARE THE CENTRAL TRADE-OFF, AND 90 MAY BE TOO GENEROUS.
# Observed on run_0002: the model returned a four-segment, exactly-90
# frame plan -- the clamp fired -- chaining two running jumps, with the
# note "then LIKELY first pipe". It was guessing about terrain past the
# screen edge and died at x=312. A 90-frame plan is 1.5 s of blind play.
#
# Fewer frames per plan means more API calls per unit distance but a
# fresh screenshot before every jump. Worth measuring rather than
# assuming -- final_x per api_call is the number to compare:
#
#     MAX_FRAMES_PER_PLAN=45 MAX_SEGMENTS_PER_PLAN=2 python openai_play.py
#
# Left at 90 until that comparison actually says otherwise.
#
# THESE THREE ARE THE CENTRAL TRADE-OFF, AND 90 MAY BE TOO GENEROUS.
# Measured 2026-09-08, 15-decision budget, gpt-5.6-sol:
#
#     cap 90:  3 calls, final_x 459, 153 px/call, budget exhausted
#     cap 90:  2 calls, final_x 312, 156 px/call, DIED
#     cap 45:  8 calls, final_x 706,  88 px/call, DIED
#
# Short plans got 2.3x further but are ~40% less efficient per call.
# On the death at cap 90 the model returned an exactly-90-frame plan --
# the clamp fired -- chaining two jumps, with the note "then LIKELY
# first pipe". It was guessing about terrain past the screen edge. A
# 90-frame plan is 1.5 s of blind play.
#
# CAVEAT: n=1 per condition and the 2-call run is barely a sample. The
# missing experiment is long plans at a 15-decision budget:
#
#     MAX_DECISIONS=15 python openai_play.py
#
# Left at 90 until that comparison actually says otherwise.
MAX_SEGMENTS_PER_PLAN = int(os.environ.get("MAX_SEGMENTS_PER_PLAN", 4))
MAX_FRAMES_PER_SEGMENT = int(os.environ.get("MAX_FRAMES_PER_SEGMENT", 60))
MAX_FRAMES_PER_PLAN = int(os.environ.get("MAX_FRAMES_PER_PLAN", 90))

# ---------------------------------------------------------- behaviour

# End the run when Mario dies. The alternative (let gym auto-reset and
# keep going) produces a video that jump-cuts back to the start, which
# reads as a bug. Set to 0 to play until the decision budget runs out.
STOP_ON_DEATH = os.environ.get("STOP_ON_DEATH", "1") == "1"

# Mario is "stuck" if x_pos has not advanced by at least this many pixels
# since the previous decision. The counter is fed to the model as
# telemetry -- vision alone cannot see that nothing is happening.
STUCK_MIN_DX = int(os.environ.get("STUCK_MIN_DX", 8))

# After this many consecutive stuck decisions, stop asking and run a
# scripted unstick (back up, then a running jump). Cheaper than another
# API call and more likely to work: a screenshot of a pipe looks the same
# every time, so the model tends to answer the same way every time.
STUCK_FALLBACK_AFTER = int(os.environ.get("STUCK_FALLBACK_AFTER", 3))

# ------------------------------------------------------------- output

RUNS_DIR = os.environ.get("RUNS_DIR", "runs")

# Capture one frame in four and encode at 15 FPS -- IDENTICAL to
# Mario_AWS/play.py, which captures once per agent step (= 4 emulated
# frames) for the same reason. This is the detail most likely to be got
# wrong: capturing every frame of the raw env and encoding at 15 would
# produce a 4x slow-motion clip that looks like a broken emulator.
CAPTURE_EVERY_N_FRAMES = int(os.environ.get("CAPTURE_EVERY_N_FRAMES", 4))
VIDEO_FPS = int(os.environ.get("VIDEO_FPS", 15))

# Sparse annotated PNG highlights, in captured-frame units. Matches the
# SAVE_EVERY_N_STEPS idea in Mario_AWS/play.py.
SAVE_EVERY_N_FRAMES = int(os.environ.get("SAVE_EVERY_N_FRAMES", 30))

# Persist every prompt/response pair to trace.jsonl. Small, and the only
# way to answer "why did it jump there" after the fact.
SAVE_TRACE = os.environ.get("SAVE_TRACE", "1") == "1"
