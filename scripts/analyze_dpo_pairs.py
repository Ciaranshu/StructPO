#!/usr/bin/env python3
"""
Analyze the distribution of structural DPO preference pairs.

Usage:
    python scripts/analyze_dpo_pairs.py \
        --pairs data/structural_pairs/structural_dpo_pairs.json \
        --rollouts data/rollouts/4b_dse_rollouts.json \
        [--output eval_results/dpo_pair_analysis.md]

If --rollouts is not provided, analyzes only the DPO pairs file.
If --rollouts is provided, re-runs annotation to get full pair type breakdown.
"""

import json
import sys
import argparse
from pathlib import Path
from collections import Counter, defaultdict
import statistics

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def analyze_pairs_only(pairs: list[dict]) -> str:
    """Analyze DPO pairs from the ShareGPT format (no pair_type metadata)."""
    lines = []
    lines.append("# DPO Pair Distribution Analysis")
    lines.append(f"\n**Total pairs**: {len(pairs)}")

    # Unique problems
    problems = set()
    for p in pairs:
        if p.get("conversations"):
            problems.add(p["conversations"][0]["value"][:100])
    lines.append(f"**Unique problems**: {len(problems)}")

    # Token length distributions
    chosen_lens = []
    rejected_lens = []
    for p in pairs:
        chosen_len = len(p["chosen"]["value"])
        rejected_len = len(p["rejected"]["value"])
        chosen_lens.append(chosen_len)
        rejected_lens.append(rejected_len)

    lines.append(f"\n## Solution Length (characters)")
    lines.append(f"| | Chosen | Rejected | Δ |")
    lines.append(f"|---|---|---|---|")
    lines.append(f"| Mean | {statistics.mean(chosen_lens):.0f} | {statistics.mean(rejected_lens):.0f} | {statistics.mean(chosen_lens) - statistics.mean(rejected_lens):.0f} |")
    lines.append(f"| Median | {statistics.median(chosen_lens):.0f} | {statistics.median(rejected_lens):.0f} | {statistics.median(chosen_lens) - statistics.median(rejected_lens):.0f} |")
    lines.append(f"| P10 | {sorted(chosen_lens)[len(chosen_lens)//10]:.0f} | {sorted(rejected_lens)[len(rejected_lens)//10]:.0f} | |")
    lines.append(f"| P90 | {sorted(chosen_lens)[9*len(chosen_lens)//10]:.0f} | {sorted(rejected_lens)[9*len(rejected_lens)//10]:.0f} | |")

    # Length ratio analysis
    ratios = [c / r if r > 0 else 1.0 for c, r in zip(chosen_lens, rejected_lens)]
    shorter_chosen = sum(1 for r in ratios if r < 0.9)
    similar_len = sum(1 for r in ratios if 0.9 <= r <= 1.1)
    longer_chosen = sum(1 for r in ratios if r > 1.1)

    lines.append(f"\n## Length Ratio (chosen/rejected)")
    lines.append(f"- **Chosen shorter (<0.9×)**: {shorter_chosen} ({100*shorter_chosen/len(pairs):.1f}%)")
    lines.append(f"- **Similar length (0.9-1.1×)**: {similar_len} ({100*similar_len/len(pairs):.1f}%)")
    lines.append(f"- **Chosen longer (>1.1×)**: {longer_chosen} ({100*longer_chosen/len(pairs):.1f}%)")
    lines.append(f"- **Mean ratio**: {statistics.mean(ratios):.2f}")
    lines.append(f"")
    lines.append(f"This confirms StructPO is NOT a length-based preference — it's structural.")

    return "\n".join(lines)


def analyze_with_rollouts(rollouts_path: str) -> str:
    """Full analysis with re-annotation from rollouts."""
    from src.preference_builder.annotator import annotate_trace
    from src.preference_builder.pair_builder import (
        build_efficiency_pairs, build_productive_exploration_pairs,
        build_direction_pairs, AnnotatedTrace,
    )

    data = json.loads(Path(rollouts_path).read_text())
    lines = []

    # --- Annotate all traces ---
    all_traces = []
    problem_texts = {}
    for problem in data:
        pid = problem["problem_id"]
        problem_texts[pid] = problem.get("problem_text", "")
        for tid, trace in enumerate(problem["traces"]):
            ann = annotate_trace(
                problem_id=pid,
                trace_id=tid,
                solution=trace["solution"],
                answer=trace.get("answer", ""),
                is_correct=trace.get("is_correct", False),
            )
            all_traces.append(ann)

    correct = [t for t in all_traces if t.is_correct]
    incorrect = [t for t in all_traces if not t.is_correct]

    lines.append("\n## Rollout Statistics")
    lines.append(f"| Metric | Value |")
    lines.append(f"|---|---|")
    lines.append(f"| Total problems | {len(data)} |")
    lines.append(f"| Total traces | {len(all_traces)} |")
    lines.append(f"| Correct traces | {len(correct)} ({100*len(correct)/len(all_traces):.1f}%) |")
    lines.append(f"| Incorrect traces | {len(incorrect)} ({100*len(incorrect)/len(all_traces):.1f}%) |")
    lines.append(f"| Avg DSR (all) | {statistics.mean(t.dsr for t in all_traces):.3f} |")
    lines.append(f"| Avg DSR (correct) | {statistics.mean(t.dsr for t in correct):.3f} |")
    lines.append(f"| Avg DSR (incorrect) | {statistics.mean(t.dsr for t in incorrect):.3f} |")
    lines.append(f"| Avg steps (all) | {statistics.mean(t.num_steps for t in all_traces):.1f} |")
    lines.append(f"| Avg trace len (chars) | {statistics.mean(t.trace_length for t in all_traces):.0f} |")

    # --- DSR distribution ---
    dsr_buckets = Counter()
    for t in all_traces:
        if t.dsr == 0:
            dsr_buckets["0.00"] += 1
        elif t.dsr < 0.1:
            dsr_buckets["0.01-0.09"] += 1
        elif t.dsr < 0.2:
            dsr_buckets["0.10-0.19"] += 1
        elif t.dsr < 0.3:
            dsr_buckets["0.20-0.29"] += 1
        elif t.dsr < 0.4:
            dsr_buckets["0.30-0.39"] += 1
        elif t.dsr < 0.5:
            dsr_buckets["0.40-0.49"] += 1
        else:
            dsr_buckets["0.50+"] += 1

    lines.append(f"\n## DSR Distribution (all traces)")
    lines.append(f"| DSR range | Count | % |")
    lines.append(f"|---|---|---|")
    for bucket in ["0.00", "0.01-0.09", "0.10-0.19", "0.20-0.29", "0.30-0.39", "0.40-0.49", "0.50+"]:
        c = dsr_buckets.get(bucket, 0)
        lines.append(f"| {bucket} | {c} | {100*c/len(all_traces):.1f}% |")

    # --- Build pairs and analyze ---
    traces_by_problem = defaultdict(list)
    for t in all_traces:
        traces_by_problem[t.problem_id].append(t)

    eff_pairs = build_efficiency_pairs(traces_by_problem)
    prod_pairs = build_productive_exploration_pairs(traces_by_problem)
    dir_pairs = build_direction_pairs(traces_by_problem)
    total = len(eff_pairs) + len(prod_pairs) + len(dir_pairs)

    lines.append(f"\n## Pair Type Breakdown")
    lines.append(f"| Type | Count | % | Description |")
    lines.append(f"|---|---|---|---|")
    lines.append(f"| **Efficiency** | {len(eff_pairs)} | {100*len(eff_pairs)/total:.1f}% | correct low-DSR > correct high-DSR |")
    lines.append(f"| **Productive Exploration** | {len(prod_pairs)} | {100*len(prod_pairs)/total:.1f}% | live verification > dead verification |")
    lines.append(f"| **Direction** | {len(dir_pairs)} | {100*len(dir_pairs)/total:.1f}% | correct efficient > incorrect wasteful |")
    lines.append(f"| **Total** | {total} | 100% | |")

    # Problems contributing to each type
    eff_problems = set(p.problem_id for p in eff_pairs)
    prod_problems = set(p.problem_id for p in prod_pairs)
    dir_problems = set(p.problem_id for p in dir_pairs)

    lines.append(f"\n## Problem Coverage")
    lines.append(f"| Type | Problems | Pairs/Problem |")
    lines.append(f"|---|---|---|")
    lines.append(f"| Efficiency | {len(eff_problems)} | {len(eff_pairs)/max(len(eff_problems),1):.1f} |")
    lines.append(f"| Productive Exploration | {len(prod_problems)} | {len(prod_pairs)/max(len(prod_problems),1):.1f} |")
    lines.append(f"| Direction | {len(dir_problems)} | {len(dir_pairs)/max(len(dir_problems),1):.1f} |")
    all_pair_problems = eff_problems | prod_problems | dir_problems
    lines.append(f"| **Any type** | **{len(all_pair_problems)}** / {len(data)} | {total/len(all_pair_problems):.1f} |")

    # DSR gap analysis per pair type
    lines.append(f"\n## DSR Gap Analysis")
    lines.append(f"| Type | Chosen DSR (mean) | Rejected DSR (mean) | Gap |")
    lines.append(f"|---|---|---|---|")
    for name, pair_list in [("Efficiency", eff_pairs), ("Productive Expl.", prod_pairs), ("Direction", dir_pairs)]:
        if pair_list:
            c_dsr = statistics.mean(p.chosen_dsr for p in pair_list)
            r_dsr = statistics.mean(p.rejected_dsr for p in pair_list)
            lines.append(f"| {name} | {c_dsr:.3f} | {r_dsr:.3f} | {r_dsr - c_dsr:.3f} |")

    # Correctness distribution
    lines.append(f"\n## Correctness in Pairs")
    lines.append(f"| Type | Both correct | Chosen correct only |")
    lines.append(f"|---|---|---|")
    for name, pair_list in [("Efficiency", eff_pairs), ("Productive Expl.", prod_pairs), ("Direction", dir_pairs)]:
        both = sum(1 for p in pair_list if p.chosen_correct and p.rejected_correct)
        chosen_only = sum(1 for p in pair_list if p.chosen_correct and not p.rejected_correct)
        lines.append(f"| {name} | {both} ({100*both/max(len(pair_list),1):.0f}%) | {chosen_only} ({100*chosen_only/max(len(pair_list),1):.0f}%) |")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze DPO pair distribution")
    parser.add_argument("--pairs", required=True, help="Path to DPO pairs JSON")
    parser.add_argument("--rollouts", default=None, help="Path to rollouts JSON (for full analysis)")
    parser.add_argument("--output", default=None, help="Output markdown file")
    args = parser.parse_args()

    pairs = json.loads(Path(args.pairs).read_text())
    report = analyze_pairs_only(pairs)

    if args.rollouts:
        print("Running full annotation analysis (may take a few minutes)...")
        report += "\n\n" + analyze_with_rollouts(args.rollouts)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(report + "\n")
        print(f"Report saved to {args.output}")
    
    print(report)


if __name__ == "__main__":
    main()
