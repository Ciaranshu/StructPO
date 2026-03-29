"""
Best-of-N using L2 (LLM-parsed) structural scores.

Compares L1 (regex) vs L2 (DeepSeek-chat) as StructPRM scoring for Best-of-N.

Usage:
    PYTHONPATH=src:$PYTHONPATH python scripts/analysis/best_of_n_l2.py \
        --rollouts data/rollouts/gpqa_4b_rollouts.json \
        --l2-annotations data/l2_annotations/gpqa_4b_l2.json
"""

import json
import sys
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from structpo.structural_parser.quality import full_quality_analysis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--l2-annotations", required=True)
    args = parser.parse_args()

    rollouts = json.loads(Path(args.rollouts).read_text())
    l2_data = json.loads(Path(args.l2_annotations).read_text())

    # Build L2 lookup: problem_id → trace_id → l2 scores
    l2_lookup = {}
    for p in l2_data:
        pid = p["problem_id"]
        l2_lookup[pid] = {}
        for t in p["traces"]:
            l2_lookup[pid][t["trace_id"]] = t

    # Annotate with both L1 and L2
    print("Computing L1 annotations...")
    problems = []
    for p in rollouts:
        pid = p["problem_id"]
        traces = []
        for tid, t in enumerate(p["traces"]):
            l1 = full_quality_analysis(t["solution"])
            l2 = l2_lookup.get(pid, {}).get(tid, {})

            traces.append({
                "is_correct": t.get("is_correct", False),
                "num_tokens": t.get("num_tokens", len(t["solution"].split())),
                # L1
                "l1_dsr": l1["dsr"],
                "l1_qr": l1["quality_reward"],
                # L2
                "l2_dsr": l2.get("l2_dsr"),
                "l2_qr": l2.get("l2_quality_reward"),
            })
        problems.append({"pid": pid, "traces": traces})

    # Filter problems that have L2 annotations
    problems_with_l2 = [p for p in problems
                        if all(t["l2_dsr"] is not None for t in p["traces"])]
    print(f"Problems with full L2: {len(problems_with_l2)}/{len(problems)}")

    # Categorize
    saturated = [p for p in problems_with_l2 if all(t["is_correct"] for t in p["traces"])]
    partial = [p for p in problems_with_l2
               if 0 < sum(1 for t in p["traces"] if t["is_correct"]) < len(p["traces"])]
    all_wrong = [p for p in problems_with_l2
                 if not any(t["is_correct"] for t in p["traces"])]

    print(f"Saturated: {len(saturated)}, Partial: {len(partial)}, All wrong: {len(all_wrong)}")
    print()

    # Define selectors
    np.random.seed(42)

    selectors = {
        "Random": lambda t: t[np.random.randint(len(t))],
        "Shortest": lambda t: min(t, key=lambda x: x["num_tokens"]),
        "Longest": lambda t: max(t, key=lambda x: x["num_tokens"]),
        "L1-DSR (regex)": lambda t: min(t, key=lambda x: x["l1_dsr"]),
        "L1-QR (regex)": lambda t: max(t, key=lambda x: x["l1_qr"]),
        "L2-DSR (LLM)": lambda t: min(t, key=lambda x: x["l2_dsr"]),
        "L2-QR (LLM)": lambda t: max(t, key=lambda x: x["l2_qr"]),
    }

    # Run on ALL problems
    print(f"{'='*70}")
    print(f"  OVERALL ({len(problems_with_l2)} problems)")
    print(f"{'='*70}\n")

    print(f"{'Selector':<20} | {'Acc':>6} | {'Tokens':>7} | {'L1-DSR':>7} | {'L2-DSR':>7}")
    print(f"{'-'*20}-+-{'-'*6}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}")

    for name, sel in selectors.items():
        if name == "Random":
            accs = [np.mean([1 if sel(p["traces"])["is_correct"] else 0 for p in problems_with_l2])
                    for _ in range(200)]
            acc = np.mean(accs)
        else:
            acc = np.mean([1 if sel(p["traces"])["is_correct"] else 0 for p in problems_with_l2])
        tokens = np.mean([sel(p["traces"])["num_tokens"] for p in problems_with_l2])
        l1_dsr = np.mean([sel(p["traces"])["l1_dsr"] for p in problems_with_l2])
        l2_dsr = np.mean([sel(p["traces"])["l2_dsr"] for p in problems_with_l2])
        print(f"{name:<20} | {acc:>5.1%} | {tokens:>7.0f} | {l1_dsr:>7.3f} | {l2_dsr:>7.3f}")

    # Run on PARTIAL problems only
    print(f"\n{'='*70}")
    print(f"  PARTIAL PROBLEMS ({len(partial)}) — selector affects accuracy")
    print(f"{'='*70}\n")

    print(f"{'Selector':<20} | {'Acc':>6} | {'L1-DSR':>7} | {'L2-DSR':>7}")
    print(f"{'-'*20}-+-{'-'*6}-+-{'-'*7}-+-{'-'*7}")

    for name, sel in selectors.items():
        if name == "Random":
            accs = [np.mean([1 if sel(p["traces"])["is_correct"] else 0 for p in partial])
                    for _ in range(200)]
            acc = np.mean(accs)
        else:
            acc = np.mean([1 if sel(p["traces"])["is_correct"] else 0 for p in partial])
        l1_dsr = np.mean([sel(p["traces"])["l1_dsr"] for p in partial])
        l2_dsr = np.mean([sel(p["traces"])["l2_dsr"] for p in partial])
        print(f"{name:<20} | {acc:>5.1%} | {l1_dsr:>7.3f} | {l2_dsr:>7.3f}")

    # DSR gap analysis
    print(f"\n{'='*70}")
    print(f"  DSR GAP: correct vs incorrect (partial)")
    print(f"{'='*70}\n")

    for dsr_key, label in [("l1_dsr", "L1 (regex)"), ("l2_dsr", "L2 (LLM)")]:
        correct = [t[dsr_key] for p in partial for t in p["traces"] if t["is_correct"]]
        incorrect = [t[dsr_key] for p in partial for t in p["traces"] if not t["is_correct"]]
        all_dsrs = [t[dsr_key] for p in partial for t in p["traces"]]
        all_correct = [1 if t["is_correct"] else 0 for p in partial for t in p["traces"]]
        r = np.corrcoef(all_dsrs, all_correct)[0, 1] if np.std(all_dsrs) > 0 else 0

        print(f"  {label}:")
        print(f"    Correct:   DSR={np.mean(correct):.3f} (n={len(correct)})")
        print(f"    Incorrect: DSR={np.mean(incorrect):.3f} (n={len(incorrect)})")
        print(f"    Gap:       {np.mean(incorrect) - np.mean(correct):+.3f}")
        print(f"    r(DSR,correct): {r:.3f}")
        print()

    # L1 vs L2 correlation
    print(f"{'='*70}")
    print(f"  L1 vs L2 CORRELATION")
    print(f"{'='*70}\n")

    all_l1 = [t["l1_dsr"] for p in problems_with_l2 for t in p["traces"]]
    all_l2 = [t["l2_dsr"] for p in problems_with_l2 for t in p["traces"]]
    r = np.corrcoef(all_l1, all_l2)[0, 1] if np.std(all_l1) > 0 and np.std(all_l2) > 0 else 0
    print(f"  DSR: r(L1, L2) = {r:.4f}")

    all_l1_qr = [t["l1_qr"] for p in problems_with_l2 for t in p["traces"]]
    all_l2_qr = [t["l2_qr"] for p in problems_with_l2 for t in p["traces"]]
    r_qr = np.corrcoef(all_l1_qr, all_l2_qr)[0, 1] if np.std(all_l1_qr) > 0 and np.std(all_l2_qr) > 0 else 0
    print(f"  QR:  r(L1, L2) = {r_qr:.4f}")


if __name__ == "__main__":
    main()
