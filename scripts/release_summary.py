"""Summarise a run_release.sh output directory (RESULTS A-054) into <out>/SUMMARY.md.

Lists every submission file with its row count and predicted-label spread, the
competition-2 HumanSL fallback counts, the competition-1 insurance / unscored
counts, and each job's Slurm state and elapsed time. No answers are read.

    python scripts/release_summary.py --out outputs/release_1104
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

import pandas as pd


def job_states(out: Path) -> list[str]:
    text = (out / "jobs.txt").read_text() if (out / "jobs.txt").exists() else ""
    ids = re.findall(r"\d{5,}", text)
    if not ids:
        return ["(jobs.txt missing)"]
    result = subprocess.run(["sacct", "-X", "-n", "-j", ",".join(ids), "-o", "JobID,JobName%22,State,Elapsed"],
                            capture_output=True, text=True, timeout=60)
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    out = parser.parse_args().out
    lines = [f"# Release run summary: `{out}`", "", "## Submissions (provisional format)", "",
             "| file | rows | predicted labels |", "|---|---:|---|"]
    for path in sorted((out / "submissions").glob("*.csv")):
        table = pd.read_csv(path)
        column = "rank" if "rank" in table else "prediction"
        spread = (table[column].value_counts().sort_index().to_dict() if column == "rank"
                  else {"distinct top-1": int(table[column].str.split().str[0].nunique())})
        lines.append(f"| `{path.name}` | {len(table)} | {spread} |")
    lines += ["", "## Competition 2 (HumanSL B2, V2 fallback per question)", ""]
    for name in ("rank_public", "rank_private"):
        path = out / "rank" / name / "humansl_predictions.parquet"
        if path.exists():
            pred = pd.read_parquet(path)
            lines.append(f"- {name}: {len(pred)} questions, V2 fallback {int(pred['fallback_v2'].sum())}, "
                         f"missing some HumanSL games {int((pred['humansl_games'] < pred['num_games']).sum())}")
        else:
            lines.append(f"- {name}: **no HumanSL predictions** (use `{name}_v2_fallback.csv`)")
    lines += ["", "## Competition 1", ""]
    for part in ("public", "private"):
        path = out / "pid" / part / "info.json"
        if path.exists():
            info = json.loads(path.read_text())
            keys = ("questions", "candidates", "unscored_questions", "unusable_query_games", "insurance_lambda",
                    "questions_without_insurance", "candidates_seen_in_train")
            lines.append(f"- {part}: " + ", ".join(f"{k}={info.get(k, '—')}" for k in keys))
        else:
            lines.append(f"- {part}: **no predictions**")
    audit = out / "day0_audit.json"
    lines += ["", f"## Day-0 audit: {'`day0_audit.json`' if audit.exists() else '**missing**'}", "",
              "## Jobs", "", "```", *job_states(out), "```"]
    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
