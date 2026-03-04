#!/usr/bin/env python3
"""
Compare evaluation results between baseline and StructPO-trained models.

Usage:
    python scripts/compare_results.py \
        --baseline eval_results/4b_dse_sft.json \
        --experiment eval_results/4b_structpo_stage2.json \
        [--output eval_results/comparison_4b.md]
"""

import argparse
import json
from pathlib import Path


def load_results(path: str) -> dict:
    return json.loads(Path(path).read_text())


def fmt_delta(baseline: float, experiment: float, pct: bool = True) -> str:
    delta = experiment - baseline
    sign = "+" if delta >= 0 else ""
    if pct:
        return f"{sign}{delta:.1%}"
    return f"{sign}{delta:.2f}"


def compare_metrics(baseline: dict, experiment: dict) -> str:
    lines = []
    b_metrics = baseline.get("metrics", {})
    e_metrics = experiment.get("metrics", {})

    b_model = baseline.get("model", "baseline")
    e_model = experiment.get("model", "experiment")

    lines.append(f"# StructPO Comparison Report")
    lines.append(f"")
    lines.append(f"| | **{Path(b_model).name}** (baseline) | **{Path(e_model).name}** (StructPO) | Δ |")
    lines.append(f"|---|---|---|---|")

    # Overall accuracy per benchmark
    for bench in sorted(set(list(b_metrics.keys()) + list(e_metrics.keys()))):
        if bench.endswith("_by_subject") or bench.endswith("_by_level"):
            continue
        b = b_metrics.get(bench, {})
        e = e_metrics.get(bench, {})
        if b and e:
            lines.append(
                f"| **{bench}** accuracy | {b['accuracy']:.1%} ({b['correct']}/{b['total']}) "
                f"| {e['accuracy']:.1%} ({e['correct']}/{e['total']}) "
                f"| {fmt_delta(b['accuracy'], e['accuracy'])} |"
            )
            lines.append(
                f"| {bench} avg tokens | {b['avg_tokens']:.0f} "
                f"| {e['avg_tokens']:.0f} "
                f"| {fmt_delta(b['avg_tokens'], e['avg_tokens'], pct=False)} |"
            )

    # Structural metrics
    b_struct = baseline.get("structural_metrics", {})
    e_struct = experiment.get("structural_metrics", {})
    if b_struct and e_struct:
        lines.append(
            f"| **Avg DSR** | {b_struct['avg_dsr']:.1%} "
            f"| {e_struct['avg_dsr']:.1%} "
            f"| {fmt_delta(b_struct['avg_dsr'], e_struct['avg_dsr'])} |"
        )
        lines.append(
            f"| Nonzero DSR % | {b_struct['nonzero_dsr_pct']:.1%} "
            f"| {e_struct['nonzero_dsr_pct']:.1%} "
            f"| {fmt_delta(b_struct['nonzero_dsr_pct'], e_struct['nonzero_dsr_pct'])} |"
        )
        lines.append(
            f"| Avg steps | {b_struct['avg_steps']:.1f} "
            f"| {e_struct['avg_steps']:.1f} "
            f"| {fmt_delta(b_struct['avg_steps'], e_struct['avg_steps'], pct=False)} |"
        )

    # Per-subject breakdown for math500
    b_subj = b_metrics.get("math500_by_subject", {})
    e_subj = e_metrics.get("math500_by_subject", {})
    if b_subj and e_subj:
        lines.append("")
        lines.append("## MATH-500 by Subject")
        lines.append("")
        lines.append("| Subject | Baseline | StructPO | Δ |")
        lines.append("|---|---|---|---|")
        for subj in sorted(b_subj.keys()):
            b_s = b_subj.get(subj, {})
            e_s = e_subj.get(subj, {})
            if b_s and e_s:
                lines.append(
                    f"| {subj} | {b_s['accuracy']:.1%} ({b_s['correct']}/{b_s['total']}) "
                    f"| {e_s['accuracy']:.1%} ({e_s['correct']}/{e_s['total']}) "
                    f"| {fmt_delta(b_s['accuracy'], e_s['accuracy'])} |"
                )

    # Per-level breakdown for math500
    b_lvl = b_metrics.get("math500_by_level", {})
    e_lvl = e_metrics.get("math500_by_level", {})
    if b_lvl and e_lvl:
        lines.append("")
        lines.append("## MATH-500 by Level")
        lines.append("")
        lines.append("| Level | Baseline | StructPO | Δ |")
        lines.append("|---|---|---|---|")
        for lvl in sorted(b_lvl.keys(), key=lambda x: int(x)):
            b_l = b_lvl.get(lvl, {})
            e_l = e_lvl.get(lvl, {})
            if b_l and e_l:
                lines.append(
                    f"| Level {lvl} | {b_l['accuracy']:.1%} ({b_l['correct']}/{b_l['total']}) "
                    f"| {e_l['accuracy']:.1%} ({e_l['correct']}/{e_l['total']}) "
                    f"| {fmt_delta(b_l['accuracy'], e_l['accuracy'])} |"
                )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compare StructPO evaluation results")
    parser.add_argument("--baseline", required=True, help="Baseline eval JSON")
    parser.add_argument("--experiment", required=True, help="Experiment eval JSON")
    parser.add_argument("--output", default=None, help="Output markdown file (default: print to stdout)")
    args = parser.parse_args()

    baseline = load_results(args.baseline)
    experiment = load_results(args.experiment)

    report = compare_metrics(baseline, experiment)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(report + "\n")
        print(f"Report saved to {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
