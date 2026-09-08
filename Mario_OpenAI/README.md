# Mario_OpenAI

A general-purpose vision model playing Super Mario Bros. 1-1 from raw
screenshots, with no training, no checkpoint, and no gradient anywhere.

The point is comparison. Same level, same 7-action `SIMPLE_MOVEMENT`
space, same MP4 output format as `Mario_AWS` — so "how far did it get"
means the same thing for both agents.

```
NES emulator
    │
    ▼  240x256 RGB screen (env.unwrapped.screen)
    │
    ├──────────────► captured every 4th frame → run.mp4
    │
    ▼  3x nearest-neighbour upscale → PNG → base64
OpenAI vision model  +  telemetry (x, y, timer, Δx, stuck counter)
    │
    ▼  {"plan": [{"action": 3, "frames": 30}, {"action": 4, "frames": 24}]}
    │
    ▼  executed frame by frame, no further API calls
NES emulator
    │
    └──── repeat
```

## Run it

```bash
cd Mario_OpenAI
python openai_play.py
```

Expected output:

```
[openai] model: gpt-5.6 (api style: responses)
[openai] starting SuperMarioBros-1-1-v0
[openai] budget: 60 decisions, 9000 frames

decision 001 | x=40 t=400 stuck=0
  model -> right+B x30 -> right+A+B x22
  note: clear ground ahead, goomba approaching, jump it
decision 002 | x=173 t=397 stuck=0
  ...

flag_get: False
final_x: 1843
api_calls: 41 (58201 in / 3104 out tokens)
stopped because: died
```

Artifacts land in `runs/run_NNNN/` — gitignored by the repo-root
`.gitignore`, which ignores `runs/` unanchored and `*.mp4` outright.

```
runs/run_0001/
├─ run.mp4        H.264 / yuv420p, plays in a browser
├─ summary.json   model, cost, decisions, final_x, flag_get
├─ trace.jsonl    every prompt and reply, one per line
└─ frames/        sparse annotated PNGs for debugging
```

## Credentials

`OPENAI_API_KEY` only. In a Codespace, set it as a **Codespaces secret**
(Settings → Codespaces → repository secrets) — it arrives as an
environment variable with nothing written to disk. Locally, copy
`.env.example` to `.env`, which the repo-root `.gitignore` already
excludes.

`OPENAI_MODEL` is **not** a secret. It is a plain config default in
`config.py`, overridable per run, and recorded in `summary.json` so a
result can be traced back to the model that produced it.

## Design decisions worth knowing

**Sequences, not a single held action.** The model returns up to four
`(action, frames)` segments executed back-to-back. "Run right, *then*
jump" is one intention; splitting it across two API calls means the
second screenshot arrives after the jump point has already gone by. Jump
timing is the whole ballgame in Mario, so the plan format has to be able
to express it.

**Frame counts are NES frames.** No `SkipFrame` wrapper here, so one
`env.step()` is one frame at 60 Hz. 30 frames is half a second. Frames
are captured every 4th step and encoded at 15 FPS, which is exactly what
`Mario_AWS/play.py` does and why its video plays at authentic speed.
Capturing every frame and encoding at 15 would give you a 4× slow-motion
clip that reads as a broken emulator.

**The emulator is frozen while we think.** nes-py is synchronous.
Nothing advances between `env.step()` calls, so API latency costs
wall-clock time and zero game state. This is what makes a
two-seconds-per-decision agent viable at all — and why the retry logic
can afford exponential backoff.

**There is a hard spend ceiling.** `MAX_DECISIONS` (default 60) is the
most important constant in `config.py`. Every decision is a billed call,
and a model stuck against a pipe will pay to look at that same pipe until
the 400-unit level timer expires. There is also a scripted unstick
(`STUCK_FALLBACK_AFTER`) that fires without an API call, because a
screenshot of a pipe looks identical every time and tends to produce an
identical — and identically wrong — answer.

**gym 0.25.2, not Gymnasium.** `env.reset()` returns the observation
alone and `env.step()` returns a 4-tuple. Anything written against modern
Gymnasium will break here. `reset()` also yields no `info`, so the loop
takes one NOOP frame to get an `x_pos` for the first prompt.

**PNG, not JPEG.** Mario is about 16 px tall at native resolution. JPEG
ringing smears sprite edges into the sky, and a blurred pixel is the
difference between a Goomba and a brick.

## Configuration

Every knob is an environment variable with a default in `config.py`.
The ones you will actually reach for:

| Variable | Default | Why you would change it |
| --- | --- | --- |
| `OPENAI_MODEL` | `gpt-5.6` | Verify against OpenAI's current model list before the first run |
| `OPENAI_API_STYLE` | `responses` | Set to `chat` if your SDK lacks `client.responses` |
| `OPENAI_REASONING_EFFORT` | `low` | Set to `""` for models that reject the parameter |
| `MAX_DECISIONS` | `60` | Lower it while you are debugging the prompt |
| `SCREEN_UPSCALE` | `3` | `1` sends the native 240×256 screen |
| `STOP_ON_DEATH` | `1` | `0` keeps playing after a death, on the same budget |

## What to expect

Probably not a flag run. A vision model reading one still frame every
~1.5 s of game time has no frame-accurate control, and Mario punishes
imprecise jumps. Getting past the first pipes is a real result; clearing
1-1 would be a surprising one. Judge it on `final_x` per API call, not on
`flag_get`.

The most likely single improvement, if you want to push it: feed two
frames (now and ~15 frames ago) instead of one, so the model can see
velocity directly instead of inferring it from the `Δx` telemetry line.
