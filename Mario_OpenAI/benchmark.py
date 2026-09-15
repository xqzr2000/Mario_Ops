"""Run the Mario_OpenAI harness repeatedly and aggregate noisy outcomes.

Example:
    python benchmark.py --runs 5 --model gpt-6-astra --max-decisions 80
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path


SUMMARY_RE = re.compile(r"Summary saved to:\s*(.+summary\.json)")


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * p
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def run_once(env: dict[str, str]) -> dict:
    proc = subprocess.run(
        [sys.executable, "openai_play.py"],
        text=True,
        capture_output=True,
        env=env,
    )
    print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, file=sys.stderr, end="")
    if proc.returncode != 0:
        raise RuntimeError(f"openai_play.py exited with status {proc.returncode}")

    match = SUMMARY_RE.search(proc.stdout)
    if not match:
        raise RuntimeError("could not find summary.json path in openai_play.py output")

    summary_path = Path(match.group(1).strip())
    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["summary_path"] = str(summary_path)
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--model", default="gpt-6-astra")
    parser.add_argument("--max-decisions", type=int, default=80)
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--stateful-turns", type=int, default=0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be >= 1")

    env = os.environ.copy()
    env["OPENAI_MODEL"] = args.model
    env["MAX_DECISIONS"] = str(args.max_decisions)
    env["OPENAI_REASONING_EFFORT"] = args.reasoning_effort
    env["STATEFUL_TURNS"] = str(args.stateful_turns)

    rows: list[dict] = []
    for i in range(1, args.runs + 1):
        print(f"\n========== benchmark run {i}/{args.runs} ==========")
        try:
            rows.append(run_once(env))
        except Exception as exc:  # keep remaining repetitions useful
            print(f"[benchmark] run {i} failed: {exc}", file=sys.stderr)

    if not rows:
        raise SystemExit("all benchmark runs failed")

    xs = [float(r["final_x_position"]) for r in rows]
    calls = [float(r["api_calls"]) for r in rows]
    costs = [float(r.get("estimated_cost_usd", 0.0)) for r in rows]
    px_per_call = [x / c if c else 0.0 for x, c in zip(xs, calls)]
    flags = sum(bool(r["flag_get"]) for r in rows)

    aggregate = {
        "created_at": datetime.now().isoformat(),
        "requested_runs": args.runs,
        "successful_runs": len(rows),
        "model": args.model,
        "max_decisions": args.max_decisions,
        "reasoning_effort": args.reasoning_effort,
        "stateful_turns": args.stateful_turns,
        "completion_rate": flags / len(rows),
        "final_x": {
            "min": min(xs),
            "p25": percentile(xs, 0.25),
            "median": statistics.median(xs),
            "p75": percentile(xs, 0.75),
            "max": max(xs),
            "mean": statistics.mean(xs),
        },
        "api_calls_mean": statistics.mean(calls),
        "px_per_api_call_mean": statistics.mean(px_per_call),
        "estimated_cost_usd_total": sum(costs),
        "estimated_cost_usd_mean": statistics.mean(costs),
        "runs": rows,
    }

    output = Path(args.output) if args.output else Path(
        f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(output, "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2)

    print("\n========== aggregate ==========")
    print(f"completion_rate: {aggregate['completion_rate']:.1%}")
    print(
        "final_x: "
        f"p25={aggregate['final_x']['p25']:.0f} "
        f"median={aggregate['final_x']['median']:.0f} "
        f"p75={aggregate['final_x']['p75']:.0f}"
    )
    print(f"mean px/api_call: {aggregate['px_per_api_call_mean']:.1f}")
    print(f"mean estimated cost: ${aggregate['estimated_cost_usd_mean']:.4f}")
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
