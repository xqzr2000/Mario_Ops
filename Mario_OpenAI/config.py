"""Configuration for the Mario_OpenAI v2 vision-control harness.

The v2 harness keeps the raw NES environment and 7-action SIMPLE_MOVEMENT
space, but gives the vision model a short temporal window and interrupts long
open-loop plans when the reported scene risk is high.

Every option can be overridden with an environment variable so benchmark runs
can be compared without editing source.
"""

import os


# -------------------------------------------------------------------- game
ENV_NAME = os.environ.get("MARIO_ENV_NAME", "SuperMarioBros-1-1-v0")


# ------------------------------------------------------------------ OpenAI
# Astra is the intended default for this harness.
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-6-astra")
OPENAI_REASONING_EFFORT = os.environ.get("OPENAI_REASONING_EFFORT", "low")
OPENAI_MAX_OUTPUT_TOKENS = int(os.environ.get("OPENAI_MAX_OUTPUT_TOKENS", 2000))
OPENAI_MAX_RETRIES = int(os.environ.get("OPENAI_MAX_RETRIES", 4))
OPENAI_TIMEOUT_S = float(os.environ.get("OPENAI_TIMEOUT_S", 60.0))

# A stable cache key helps reuse the large, identical instruction/schema prefix.
PROMPT_CACHE_KEY = os.environ.get("PROMPT_CACHE_KEY", "mario-openai-v2")
PROMPT_CACHE_TTL = os.environ.get("PROMPT_CACHE_TTL", "30m")

# Optional server-side continuation. 0 = stateless (recommended baseline).
# A non-zero value chains at most N responses, then deliberately resets context
# so old screenshots do not accumulate for the whole level.
STATEFUL_TURNS = int(os.environ.get("STATEFUL_TURNS", 0))


# ------------------------------------------------------------ image / vision
SCREEN_UPSCALE = int(os.environ.get("SCREEN_UPSCALE", 3))
IMAGE_DETAIL = os.environ.get("IMAGE_DETAIL", "high")

# Keep the newest N sampled screens. At 6 frames/sample and N=3, Astra normally
# sees roughly t-12, t-6 and t, which exposes direction and velocity visually.
TEMPORAL_FRAMES = int(os.environ.get("TEMPORAL_FRAMES", 3))
TEMPORAL_SAMPLE_EVERY_N_FRAMES = int(
    os.environ.get("TEMPORAL_SAMPLE_EVERY_N_FRAMES", 6)
)


# ---------------------------------------------------------- decisions / spend
# Temporal control makes more, shorter decisions than v1, so the default call
# ceiling is slightly higher while the NES-frame ceiling stays unchanged.
MAX_DECISIONS = int(os.environ.get("MAX_DECISIONS", 80))
MAX_FRAMES = int(os.environ.get("MAX_FRAMES", 9000))

MAX_SEGMENTS_PER_PLAN = int(os.environ.get("MAX_SEGMENTS_PER_PLAN", 4))
MAX_FRAMES_PER_SEGMENT = int(os.environ.get("MAX_FRAMES_PER_SEGMENT", 60))
MAX_FRAMES_PER_PLAN = int(os.environ.get("MAX_FRAMES_PER_PLAN", 90))

# The model requests when it wants another observation. The harness then applies
# a stricter cap based on the model's own structured perception.
MIN_REOBSERVE_FRAMES = int(os.environ.get("MIN_REOBSERVE_FRAMES", 6))
MAX_REOBSERVE_FRAMES = int(os.environ.get("MAX_REOBSERVE_FRAMES", 60))
RISK_LOW_MAX_FRAMES = int(os.environ.get("RISK_LOW_MAX_FRAMES", 60))
RISK_MEDIUM_MAX_FRAMES = int(os.environ.get("RISK_MEDIUM_MAX_FRAMES", 30))
RISK_HIGH_MAX_FRAMES = int(os.environ.get("RISK_HIGH_MAX_FRAMES", 15))
RISK_CRITICAL_MAX_FRAMES = int(os.environ.get("RISK_CRITICAL_MAX_FRAMES", 8))
AIRBORNE_MAX_FRAMES = int(os.environ.get("AIRBORNE_MAX_FRAMES", 10))
STUCK_MAX_FRAMES = int(os.environ.get("STUCK_MAX_FRAMES", 8))
NEAR_HAZARD_MAX_FRAMES = int(os.environ.get("NEAR_HAZARD_MAX_FRAMES", 12))
IMMEDIATE_HAZARD_MAX_FRAMES = int(
    os.environ.get("IMMEDIATE_HAZARD_MAX_FRAMES", 8)
)


# --------------------------------------------------------------- behaviour
STOP_ON_DEATH = os.environ.get("STOP_ON_DEATH", "1") == "1"
STUCK_MIN_DX = int(os.environ.get("STUCK_MIN_DX", 8))
STUCK_FALLBACK_AFTER = int(os.environ.get("STUCK_FALLBACK_AFTER", 3))

# Optional old v1 landing experiment. Kept for A/B compatibility, still off by
# default because previous measurements showed that it could add drift.
LAND_BEFORE_DECIDING = os.environ.get("LAND_BEFORE_DECIDING", "0") == "1"
LAND_MAX_FRAMES = int(os.environ.get("LAND_MAX_FRAMES", 90))
LAND_STABLE_FRAMES = int(os.environ.get("LAND_STABLE_FRAMES", 6))
LAND_HOLD_ACTION = int(os.environ.get("LAND_HOLD_ACTION", 0))  # NOOP by default
LAND_DEBUG_Y = os.environ.get("LAND_DEBUG_Y", "0") == "1"


# ------------------------------------------------------------------- output
RUNS_DIR = os.environ.get("RUNS_DIR", "runs")
CAPTURE_EVERY_N_FRAMES = int(os.environ.get("CAPTURE_EVERY_N_FRAMES", 4))
VIDEO_FPS = int(os.environ.get("VIDEO_FPS", 15))
SAVE_EVERY_N_FRAMES = int(os.environ.get("SAVE_EVERY_N_FRAMES", 30))
SAVE_TRACE = os.environ.get("SAVE_TRACE", "1") == "1"


# --------------------------------------------------------------- cost model
# Defaults match GPT-6 Astra Standard pricing as of 2026-09-15. Override these
# when benchmarking another model or after a pricing change.
_is_astra = OPENAI_MODEL.startswith("gpt-6-astra")
INPUT_USD_PER_MTOK = float(
    os.environ.get("INPUT_USD_PER_MTOK", 10.0 if _is_astra else 0.0)
)
CACHED_INPUT_USD_PER_MTOK = float(
    os.environ.get("CACHED_INPUT_USD_PER_MTOK", 1.0 if _is_astra else 0.0)
)
CACHE_WRITE_USD_PER_MTOK = float(
    os.environ.get("CACHE_WRITE_USD_PER_MTOK", 12.5 if _is_astra else 0.0)
)
OUTPUT_USD_PER_MTOK = float(
    os.environ.get("OUTPUT_USD_PER_MTOK", 50.0 if _is_astra else 0.0)
)
