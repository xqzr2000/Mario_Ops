#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Mario_OpenAI/tools/astra_budget_test.sh
#
# Answers ONE question: is MAX_DECISIONS a behavioural lever?
#
# Six runs at gpt-6-astra, identical except for the decision budget --
# which the model SEES, because build_telemetry sends
# "decisions_remaining". At 15 the previous model used nearly its whole
# budget and reached 833-1524. At 40 it used a quarter of the budget and
# died at ~700 three times running. If that is causal rather than luck,
# telling the model it has few chances makes it play better, and every
# comparison that varied MAX_DECISIONS today was confounded.
#
# Run from Mario_OpenAI/:
#     bash tools/astra_budget_test.sh
#
# Roughly 100 API calls on a more expensive model than the 5.6 runs.
# Check astra's pricing before starting -- this is the largest single
# spend of the project so far. Ctrl-C is safe; each run has already
# written its own runs/run_NNNN/ by the time the next starts.
# ---------------------------------------------------------------------------
set -uo pipefail   # NOT -e: one failed run should not abandon the batch

MODEL="${OPENAI_MODEL:-gpt-6-astra}"
REPS="${REPS:-3}"
RESULTS="astra_budget_$(date +%H%M%S).tsv"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "OPENAI_API_KEY is not set." >&2
    exit 1
fi
if [[ ! -f openai_play.py ]]; then
    echo "Run this from Mario_OpenAI/ (openai_play.py not found here)." >&2
    exit 1
fi

# Confirm the model exists before spending anything on it. A wrong id
# fails on the first call, and finding that out after five minutes of
# runs is worse than finding it out now.
python - "$MODEL" <<'PY' || exit 1
import sys
from openai import OpenAI
want = sys.argv[1]
ids = [m.id for m in OpenAI().models.list()]
if want not in ids:
    print(f"'{want}' is not in your account's model list.", file=sys.stderr)
    print("Closest: " + ", ".join(i for i in ids if i.startswith(want[:7])),
          file=sys.stderr)
    sys.exit(1)
print(f"model ok: {want}")
PY

printf 'budget\trep\tfinal_x\tcalls\tstop_reason\trun_dir\n' > "$RESULTS"
echo "model: $MODEL | reps: $REPS | results -> $RESULTS"
echo

for BUDGET in 15 40; do
    for REP in $(seq 1 "$REPS"); do
        printf '=== budget %-2s rep %s ... ' "$BUDGET" "$REP"

        OPENAI_MODEL="$MODEL" \
        MAX_FRAMES_PER_PLAN=45 \
        MAX_SEGMENTS_PER_PLAN=2 \
        MAX_DECISIONS="$BUDGET" \
        python openai_play.py >/tmp/astra_run.log 2>&1

        if [[ $? -ne 0 ]]; then
            echo "FAILED (see /tmp/astra_run.log)"
            tail -5 /tmp/astra_run.log
            continue
        fi

        # Read from summary.json, not the stdout text: the JSON is the
        # artifact of record and carries stop_reason, which is the field
        # that distinguishes "died" from "ran out of budget" -- the whole
        # point of this test.
        RUN=$(ls -d runs/run_* | sort | tail -1)
        read -r FX CALLS REASON <<<"$(python - "$RUN" <<'PY'
import json, sys
d = json.load(open(f"{sys.argv[1]}/summary.json"))
print(d["final_x_position"], d["api_calls"], d["stop_reason"].replace(" ", "_"))
PY
)"
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$BUDGET" "$REP" "$FX" "$CALLS" "$REASON" "$RUN" >> "$RESULTS"
        echo "x=$FX  calls=$CALLS  ($REASON)"
    done
done

echo
echo "=== summary ==="
python - "$RESULTS" <<'PY'
import csv, statistics, sys
rows = list(csv.DictReader(open(sys.argv[1]), delimiter="\t"))
if not rows:
    print("no completed runs"); raise SystemExit
print(f"{'budget':>6} {'n':>2} {'mean_x':>7} {'min':>5} {'max':>5} "
      f"{'mean_calls':>10}  outcomes")
for b in ("15", "40"):
    g = [r for r in rows if r["budget"] == b]
    if not g:
        continue
    xs = [int(r["final_x"]) for r in g]
    cs = [int(r["calls"]) for r in g]
    print(f"{b:>6} {len(g):>2} {statistics.mean(xs):>7.0f} {min(xs):>5} "
          f"{max(xs):>5} {statistics.mean(cs):>10.1f}  "
          + ", ".join(f"{r['final_x']}({r['stop_reason']})" for r in g))
print()
print("READ IT THIS WAY:")
print("  budget 15 clearly ahead  -> decisions_remaining is a behavioural")
print("    lever; the model plays harder when it thinks chances are few,")
print("    and every earlier comparison that varied MAX_DECISIONS is")
print("    confounded. Next step: send a FIXED number and see whether the")
print("    urgency effect survives being a lie.")
print("  both similar               -> the 5.6 spread was sampling noise")
print("    and n=3 has been too small all along.")
print("  budget 40 uses its calls   -> astra simply plays longer than 5.6")
print("    did, and the confound was specific to the weaker model.")
PY
echo
echo "raw results: $RESULTS"
