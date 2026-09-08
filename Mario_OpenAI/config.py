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
# faster than this file will. https://developers.openai.com/api/docs/models
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.6-sol")

# "responses" (current OpenAI surface) or "chat" (chat.completions).
# Both are implemented. If the SDK version in your image does not have
# client.responses, flip this to "chat" instead of downgrading anything.
OPENAI_API_STYLE = os.environ.get("OPENAI_API_STYLE", "responses")

# Reasoning effort, for models that expose it. Low by default: this is a
# 7-way classification against a 240x256 screenshot, not a proof, and
# every extra reasoning token is latency and money per decision.
# Set to "" to omit the parameter entirely for models that reject it.
OPENAI_REASONING_EFFORT = os.environ.get("OPENAI_REASONING_EFFORT", "low")

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
