#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Mario_OpenAI/tools/harness_ablation.sh
#
# Answers: which harness modules actually move final_x, and at what cost?
#
# Runs every CONDITION below REPS times (default 3 -- the variance
# warning in config.py measured a 60% swing between identical runs, so
# n=1 detects nothing), then prints mean/min/max final_x, calls, px per
# call, and how each run ended.
#
# Two modes:
#
#   bash tools/harness_ablation.sh              # MODE=ablation (default)
#   MODE=lessons bash tools/harness_ablation.sh # learning curve
#
# ablation  Independent runs. HARNESS_LESSONS is forced to "off" for every
#           condition, whatever its preset says -- a lesson written by
#           rep 1 would otherwise leak into rep 2 and the reps would stop
#           being samples of the same thing.
#
# lessons   ATTEMPTS (default 6) SEQUENTIAL runs of the `full` preset
#           against a FRESH lessons directory. Reads as a learning curve:
#           does attempt N get past where attempt N-1 died? Not an A/B --
#           do not put these numbers in the ablation table.
#
# COST: the default ablation is 5 conditions x 3 reps = 15 runs, and the
# gamingagent condition spends ~3 calls per decision. With the default
# HARNESS_MAX_API_CALLS=120 the worst case is ~1800 calls. Lower
# MAX_DECISIONS or REPS first if that is more than you meant to spend.
# Ctrl-C is safe: each finished run has already written runs/run_NNNN/.
# ---------------------------------------------------------------------------
set -uo pipefail   # NOT -e: one failed run should not abandon the batch

MODE="${MODE:-ablation}"
REPS="${REPS:-3}"
ATTEMPTS="${ATTEMPTS:-6}"
MODEL="${OPENAI_MODEL:-gpt-5.6-sol}"
STAMP="$(date +%H%M%S)"
RESULTS="harness_${MODE}_${STAMP}.tsv"

# label|space-separated env overrides. Edit freely; one variable at a
# time is the point, so each row differs from `ram` by one knob.
CONDITIONS=(
  "baseline|HARNESS_PRESET=baseline"
  "gamingagent|HARNESS_PRESET=gamingagent"
  "ram|HARNESS_PRESET=ram"
  "ram-no-guard|HARNESS_PRESET=ram HARNESS_GUARD=0"
  "ram-no-map|HARNESS_PRESET=ram HARNESS_ASCII_MAP=0"
)

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "OPENAI_API_KEY is not set." >&2; exit 1
fi
if [[ ! -f harness_play.py ]]; then
    echo "Run this from Mario_OpenAI/ (harness_play.py not found here)." >&2; exit 1
fi

# Offline checks first: a broken harness should fail for free, not
# fifteen runs in.
python tools/harness_selftest.py ram identity >/tmp/harness_selftest.log 2>&1 || {
    echo "harness_selftest failed -- see /tmp/harness_selftest.log" >&2; exit 1; }

python - "$MODEL" <<'PY' || exit 1
import sys
from openai import OpenAI
want = sys.argv[1]
ids = [m.id for m in OpenAI().models.list()]
# gpt-5.6 is a working alias that models.list() does not enumerate; see config.py.
if want not in ids and not any(i.startswith(want + "-") for i in ids):
    print(f"'{want}' is not in your account's model list.", file=sys.stderr)
    sys.exit(1)
print(f"model ok: {want}")
PY

record() {   # label rep -> appends one TSV row from the newest summary.json
    local label="$1" rep="$2"
    local run
    run=$(ls -d runs/run_* 2>/dev/null | sort | tail -1)
    python - "$run" "$label" "$rep" >> "$RESULTS" <<'PY'
import json, sys
run, label, rep = sys.argv[1:4]
d = json.load(open(f"{run}/summary.json"))
reason = d["stop_reason"] + (f" ({d['death_cause']})" if d.get("death_cause") else "")
print("\t".join(str(v) for v in (
    label, rep, d["final_x_position"], d["api_calls"],
    d.get("guard_interrupts", 0), int(d["flag_get"]),
    reason.replace("\t", " "), run)))
PY
    tail -1 "$RESULTS" | awk -F'\t' '{printf "x=%s calls=%s guard=%s  (%s)\n", $3, $4, $5, $7}'
}

printf 'condition\trep\tfinal_x\tcalls\tguard\tflag\tended\trun_dir\n' > "$RESULTS"
echo "mode: $MODE | model: $MODEL | results -> $RESULTS"
echo

if [[ "$MODE" == "lessons" ]]; then
    LDIR="harness_memory_curve_${STAMP}"
    echo "fresh lessons dir: $LDIR"
    for A in $(seq 1 "$ATTEMPTS"); do
        printf '=== attempt %-2s ... ' "$A"
        env OPENAI_MODEL="$MODEL" HARNESS_PRESET=full HARNESS_LESSONS_DIR="$LDIR" \
            python harness_play.py >/tmp/harness_run.log 2>&1 \
            || { echo "FAILED (see /tmp/harness_run.log)"; tail -5 /tmp/harness_run.log; continue; }
        record "attempt" "$A"
    done
    echo; echo "lessons written:"; cat "$LDIR"/lessons_*.json 2>/dev/null | head -60
    echo; echo "raw results: $RESULTS"
    exit 0
fi

for C in "${CONDITIONS[@]}"; do
    LABEL="${C%%|*}"; OVERRIDES="${C#*|}"
    for REP in $(seq 1 "$REPS"); do
        printf '=== %-14s rep %s ... ' "$LABEL" "$REP"
        # shellcheck disable=SC2086 -- OVERRIDES is intentionally word-split
        env OPENAI_MODEL="$MODEL" $OVERRIDES HARNESS_LESSONS=off \
            python harness_play.py >/tmp/harness_run.log 2>&1 \
            || { echo "FAILED (see /tmp/harness_run.log)"; tail -5 /tmp/harness_run.log; continue; }
        record "$LABEL" "$REP"
    done
done

echo
echo "=== summary ==="
python - "$RESULTS" <<'PY'
import csv, statistics, sys
rows = list(csv.DictReader(open(sys.argv[1]), delimiter="\t"))
if not rows:
    print("no completed runs"); raise SystemExit
order = list(dict.fromkeys(r["condition"] for r in rows))
print(f"{'condition':<14} {'n':>2} {'mean_x':>7} {'min':>5} {'max':>5} "
      f"{'calls':>6} {'px/call':>7} {'guard':>5} {'flags':>5}")
for c in order:
    g = [r for r in rows if r["condition"] == c]
    xs = [int(r["final_x"]) for r in g]
    cs = [int(r["calls"]) for r in g]
    print(f"{c:<14} {len(g):>2} {statistics.mean(xs):>7.0f} {min(xs):>5} {max(xs):>5} "
          f"{statistics.mean(cs):>6.1f} {sum(xs) / max(sum(cs), 1):>7.1f} "
          f"{statistics.mean(int(r['guard']) for r in g):>5.1f} "
          f"{sum(int(r['flag']) for r in g):>5}")
print()
for c in order:
    print(f"{c}: " + ", ".join(f"{r['final_x']}({r['ended']})"
                              for r in rows if r["condition"] == c))
print()
print("READ IT THIS WAY: with n=3 a difference smaller than the spread")
print("(max - min) inside either row is not a result. A row whose MIN")
print("beats another row's MAX is. Check death causes before causes of")
print("improvement: a condition that only moves deaths from Goomba to pit")
print("has changed what kills Mario, not how far he gets.")
PY
echo
echo "raw results: $RESULTS"
