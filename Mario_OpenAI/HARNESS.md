# Mario_OpenAI — gaming harness

A perception → memory → reasoning harness around the same OpenAI vision
agent `openai_play.py` runs, after the architecture of
[lmgame-org/GamingAgent](https://github.com/lmgame-org/GamingAgent), plus two
additions aimed at the failures this project has already measured.

Same level, same 7-action `SIMPLE_MOVEMENT` space, same decision loop, same
`runs/run_NNNN/` output. **`openai_play.py` and `openai_agent.py` are not
modified** — a harness number only means something next to a baseline that
did not move.

```
NES ─► StateRecorder          RAM + screen history (no behaviour change)
        │
        ├─► PerceptionModule   RAM state · scaffolded image · VLM description
        ├─► MemoryModule       trajectory · reflection · cross-run lessons
        ├─► ReasoningModule    baseline prompt + harness sections → plan
        │
        └─◄ execute_plan        run the plan; Guard may end it early
```

## Run it

```bash
cd Mario_OpenAI
python tools/harness_selftest.py            # offline, no key, no spend -- run this first
python harness_play.py                      # preset "ram"
HARNESS_PRESET=gamingagent python harness_play.py
bash tools/harness_ablation.sh              # presets x 3 reps, summary table
MODE=lessons bash tools/harness_ablation.sh # learning curve across attempts
```

## Presets

| Preset | What the model gets | Billed calls / decision | Runs independent? |
| --- | --- | --- | --- |
| `baseline` | Exactly what `openai_play.py` sends (selftest asserts the request is equal) | 1 | yes |
| `gamingagent` | GamingAgent's Mario config: VLM perception on a 5×5-grid image, grid on the reasoning image, trajectory, reflection every decision | ~3 | yes |
| `ram` *(default)* | Exact positions, speeds and frames-to-contact from RAM; a tile map; distance ruler on the image; trajectory; reflection after a bad decision; guard | ~1.1–1.3 | yes |
| `full` | `ram` + cross-run lessons | ~1.1–1.3 | **no** |

Any `HARNESS_*` variable overrides its preset's value, so single-knob
ablations are one line: `HARNESS_PRESET=ram HARNESS_GUARD=0 python harness_play.py`.

## Mapping from GamingAgent

| GamingAgent | Here | Notes |
| --- | --- | --- |
| `PerceptionModule`, VLM track | `PERCEPTION=vision` | Separate call describing a gridded screenshot as JSON |
| scaffolding `draw_grid_on_image` | `SCAFFOLD=grid` | Drawn after the 3× upscale so labels are legible |
| textual game state (their non-retro games) | `PERCEPTION=ram` | GamingAgent's Mario uses vision only; this reads RAM instead |
| `MemoryModule` trajectory | `MEMORY=1` | Stores *measured* outcomes: Δx, frames run, guard cut, airborne |
| reflection | `REFLECTION=every` / `event` | `every` is GamingAgent's behaviour |
| `ReasoningModule` | `ReasoningModule` | Reuses `build_telemetry` and `parse_plan` from `openai_agent.py` |
| — | Guard | New |
| — | Lessons | New, Reflexion-style |

## The two additions

**Guard.** Your logbook in `config.py` traced the deterministic x=312 death to
"the plan is right and the first number is too big". Nothing between the API
call and the Goomba could notice. The guard checks RAM every frame: if Mario
is grounded, a hazard is within `HARNESS_GUARD_FRAMES` (24) of contact, and the
rest of the plan does not start a jump in time, it ends the plan and the loop
takes a fresh screenshot. **It never presses a button** — the model still makes
every jump decision, just with the hazard ~24 frames away instead of up to 90.
The scripted unstick runs unguarded. Each firing costs one decision and is
logged in `summary.json → guard_log`.

**Lessons.** After a death, one text call writes a ≤45-word lesson keyed to the
death x into `harness_memory/lessons_<env>.json`; later runs see lessons for
the next 256 px. One lesson per 48-px bucket, newest wins, 24 max. This is the
only piece that makes runs depend on each other, which is why the ablation
script forces `HARNESS_LESSONS=off` and has a separate `MODE=lessons`. Delete the
file to start over; read `lessons.written` in `summary.json` to audit what got
stored — a lesson can encode a wrong belief as confidently as a right one.

## What RAM perception changes about the question

The baseline asks "can a general vision model play from raw screenshots?" The
`ram` preset asks "can it play when handed exact state?" Both are worth
answering, and GamingAgent's own benchmark uses symbolic state for several of
its games — but they are different questions. Report `ram` results as harness
results, not as vision results. The `gamingagent` preset is the vision-only
harness.

## Verified against the emulator, not assumed

Everything in `harness/ram_state.py` was checked on `SuperMarioBros-1-1-v0`:

* `0x006D/0x0086` equals `info["x_pos"]`; speed byte `0x57` = 48 while running,
  measured as 3.0 px/frame; Goomba velocity measured at −0.625 px/frame.
* Tile buffer columns line up with the `?` blocks, the first pipe, both stair
  pyramids and the flagpole (`0x24` ball, `0x25` pole — non-solid).
* **Lookahead is bounded.** Columns up to 22 past the screen's left tile
  column were correct in every check; from 23 on they were wrong ~60% of the
  time (the two-page buffer wraps). The scan stops at 20.
* **Oracle:** a ~20-line rule bot reading only `read_state()` output clears 1-1
  to the flag. That is the `oracle` selftest check, and it sets the ceiling
  for reading harness results: the PERCEPTION block contains enough to finish
  the level, so a model that has it and dies is failing at timing or
  reasoning, not at seeing.
* With a fake client that always plans a 90-frame run-up, the `baseline`
  preset dies at **x=312 to the Goomba** — the same deterministic death as the
  real gpt-5.6-sol runs — and the `ram` preset's guard fires before it.

## Not yet measured

No real-model harness run exists yet. `HARNESS_GUARD_FRAMES=24` was tuned
against a fake client; `MOTION_FRAMES` (a second screenshot so velocity is
visible) is implemented but in no preset; `REFLECTION=event` vs `every` is
untested for value. Same rule as everywhere else in this project: three runs
per condition before believing anything.

## Knobs

| Variable | Default (`ram`) | |
| --- | --- | --- |
| `HARNESS_PRESET` | `ram` | `baseline` · `gamingagent` · `ram` · `full` |
| `HARNESS_PERCEPTION` | `ram` | `off` · `ram` · `vision` · `both` |
| `HARNESS_ASCII_MAP` | `1` | 13×21 tile map in the perception block (~120 tokens) |
| `HARNESS_SCAFFOLD` | `ruler` | `none` · `grid` · `ruler` |
| `HARNESS_MOTION_FRAMES` | `0` | Send the screen from N frames earlier as a second image |
| `HARNESS_MEMORY` / `_WINDOW` | `1` / `5` | Recent decisions shown |
| `HARNESS_REFLECTION` | `event` | `off` · `event` · `every` |
| `HARNESS_GUARD` / `_FRAMES` | `1` / `24` | |
| `HARNESS_LESSONS` | `off` | `off` · `read` · `write` |
| `HARNESS_MAX_API_CALLS` | `120` | **Hard ceiling across all modules.** `MAX_DECISIONS` no longer bounds spend |
| `HARNESS_AUX_MODEL` | *(OPENAI_MODEL)* | Cheaper model for perception / reflection / lesson calls |
| `HARNESS_PRINT_PROMPTS` | `0` | Print each assembled reasoning prompt |

## Output additions

`summary.json` keeps every key `openai_play.py` writes (`api_calls` is now all
billed calls, all modules) and adds `agent`, `harness_preset`,
`api_usage_by_purpose`, `px_per_call`, `death_cause` (`hit by Goomba`,
`fell into pit`, …), `guard_interrupts`, `guard_log`, `reflections`,
`lessons`, and `harness_settings`. `trace.jsonl` rows carry the full assembled
prompt, so "why did it jump there" is answerable per decision.

## Files

```
Mario_OpenAI/
├─ harness_play.py            entry point (new)
├─ HARNESS.md                 this file (new)
├─ harness/                   (new)
│  ├─ settings.py             presets + HARNESS_* knobs
│  ├─ ram_state.py            RAM → state, hazards, tile map, death cause
│  ├─ recorder.py             gym wrapper: RAM/screen history
│  ├─ scaffold.py             grid and ruler overlays
│  ├─ perception.py  memory.py  reasoning.py  guard.py
│  ├─ llm.py                  one client, per-purpose usage, spend ceiling
│  └─ prompts.py              harness prompt text
└─ tools/
   ├─ harness_selftest.py     offline checks (new)
   └─ harness_ablation.sh     A/B runner (new)
```

No new dependencies. Everything imports from the existing dev image.
