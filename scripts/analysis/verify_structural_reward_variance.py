"""
Verify the StructPRM hypothesis: structural reward preserves gradient signal
when outcome reward is saturated (echo trap).

Core question: On problems where ALL K=8 rollouts are correct (outcome reward
variance = 0), does the structural signal (DSR) still have significant variance?

If yes → StructPRM breaks the echo trap.
If no  → structural signal collapses with outcome signal (hypothesis falsified).

This is a zero-GPU-cost experiment using existing rollout data.

Usage:
    python scripts/analysis/verify_structural_reward_variance.py \
        --rollouts data/rollouts/4b_dse_rollouts.json \
        [--rollouts-8b data/rollouts/8b_dse_rollouts.json]
"""

import json
import sys
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from structpo.preference_builder.annotator import annotate_trace


def analyze_rollouts(rollouts_path: str, model_name: str = "model"):
    """Analyze structural variance across rollout groups."""

    print(f"\n{'='*70}")
    print(f"  StructPRM Hypothesis Verification: {model_name}")
    print(f"  Data: {rollouts_path}")
    print(f"{'='*70}\n")

    # Load rollouts
    data = json.loads(Path(rollouts_path).read_text())
    print(f"Loaded {len(data)} problems\n")

    # Annotate all traces
    print("Annotating traces (CPU only, ~60s)...")
    problems = []
    total_traces = 0

    for i, problem in enumerate(data):
        pid = problem["problem_id"]
        traces = []
        for tid, trace in enumerate(problem["traces"]):
            ann = annotate_trace(
                problem_id=pid,
                trace_id=tid,
                solution=trace["solution"],
                answer=trace.get("answer", ""),
                is_correct=trace.get("is_correct", False),
            )
            traces.append(ann)
            total_traces += 1

        if (i + 1) % 100 == 0:
            print(f"  ... {i+1}/{len(data)} problems annotated")

        problems.append({
            "problem_id": pid,
            "traces": traces,
            "num_correct": sum(1 for t in traces if t.is_correct),
            "num_total": len(traces),
        })

    print(f"Annotated {total_traces} traces across {len(data)} problems\n")

    # ================================================================
    # Categorize problems by outcome saturation
    # ================================================================
    saturated = []      # ALL correct (outcome variance = 0)
    partial = []        # some correct, some incorrect
    all_incorrect = []  # none correct

    for p in problems:
        if p["num_correct"] == p["num_total"]:
            saturated.append(p)
        elif p["num_correct"] == 0:
            all_incorrect.append(p)
        else:
            partial.append(p)

    print(f"Problem categories:")
    print(f"  Saturated (all correct):  {len(saturated):>4}  ({100*len(saturated)/len(problems):.1f}%)")
    print(f"  Partial:                  {len(partial):>4}  ({100*len(partial)/len(problems):.1f}%)")
    print(f"  All incorrect:            {len(all_incorrect):>4}  ({100*len(all_incorrect)/len(problems):.1f}%)")
    print()

    # ================================================================
    # Core analysis: Reward variance comparison
    # ================================================================
    print(f"{'='*70}")
    print(f"  CORE RESULT: Reward Variance Under Saturation")
    print(f"{'='*70}\n")

    def compute_variance_stats(problem_group, label):
        """Compute reward variance statistics for a group of problems."""
        outcome_vars = []
        struct_vars = []
        dsr_ranges = []
        dsr_means = []
        verif_rate_vars = []

        for p in problem_group:
            correct_traces = [t for t in p["traces"] if t.is_correct]
            if len(correct_traces) < 2:
                continue

            # Outcome reward variance (all 1.0 for saturated)
            outcome_scores = [1.0 if t.is_correct else 0.0 for t in p["traces"]]
            outcome_vars.append(np.var(outcome_scores))

            # Structural reward variance (1 - DSR)
            struct_scores = [1.0 - t.dsr for t in correct_traces]
            struct_vars.append(np.var(struct_scores))

            # DSR range and mean
            dsrs = [t.dsr for t in correct_traces]
            dsr_ranges.append(max(dsrs) - min(dsrs))
            dsr_means.append(np.mean(dsrs))

            # Live verification rate variance
            verif_rates = [t.live_verification_rate for t in correct_traces]
            verif_rate_vars.append(np.var(verif_rates))

        if not outcome_vars:
            print(f"  {label}: No problems with >=2 correct traces")
            return None

        results = {
            "n_problems": len(outcome_vars),
            "outcome_var_mean": np.mean(outcome_vars),
            "outcome_var_median": np.median(outcome_vars),
            "struct_var_mean": np.mean(struct_vars),
            "struct_var_median": np.median(struct_vars),
            "dsr_range_mean": np.mean(dsr_ranges),
            "dsr_range_median": np.median(dsr_ranges),
            "dsr_mean": np.mean(dsr_means),
            "verif_var_mean": np.mean(verif_rate_vars),
            "pct_nonzero_struct_var": 100 * np.mean([1 for v in struct_vars if v > 0.001]) / len(struct_vars) if struct_vars else 0,
            "struct_vars": struct_vars,
            "dsr_ranges": dsr_ranges,
        }

        print(f"  {label} ({results['n_problems']} problems):")
        print(f"    Outcome reward variance:    mean={results['outcome_var_mean']:.6f}  median={results['outcome_var_median']:.6f}")
        print(f"    Structural reward variance: mean={results['struct_var_mean']:.6f}  median={results['struct_var_median']:.6f}")
        print(f"    DSR range (max-min):        mean={results['dsr_range_mean']:.4f}  median={results['dsr_range_median']:.4f}")
        print(f"    DSR mean:                   {results['dsr_mean']:.4f}")
        print(f"    Live verif rate variance:   mean={results['verif_var_mean']:.6f}")
        print(f"    Problems with struct var>0: {results['pct_nonzero_struct_var']:.1f}%")
        print()

        return results

    sat_results = compute_variance_stats(saturated, "SATURATED (echo trap)")
    par_results = compute_variance_stats(partial, "PARTIAL (has outcome signal)")

    # ================================================================
    # Ratio analysis: How much signal does StructPRM preserve?
    # ================================================================
    if sat_results and par_results:
        print(f"{'='*70}")
        print(f"  SIGNAL PRESERVATION RATIO")
        print(f"{'='*70}\n")

        ratio = sat_results["struct_var_mean"] / par_results["struct_var_mean"] if par_results["struct_var_mean"] > 0 else float('inf')
        print(f"  Structural variance ratio (saturated / partial): {ratio:.3f}")
        print(f"  Outcome variance ratio:                          {sat_results['outcome_var_mean'] / max(par_results['outcome_var_mean'], 1e-10):.3f}")
        print()
        if ratio > 0.5:
            print(f"  >> StructPRM preserves {ratio*100:.0f}% of structural signal under saturation")
            print(f"  >> Outcome reward preserves 0% (variance = 0)")
            print(f"  >> HYPOTHESIS CONFIRMED: StructPRM breaks the echo trap")
        elif ratio > 0.1:
            print(f"  >> StructPRM preserves some signal ({ratio*100:.0f}%) but it's diminished")
            print(f"  >> Partial support for hypothesis")
        else:
            print(f"  >> StructPRM signal also collapses under saturation")
            print(f"  >> HYPOTHESIS WEAKENED")

    # ================================================================
    # Distribution analysis: DSR variance histogram
    # ================================================================
    if sat_results:
        print(f"\n{'='*70}")
        print(f"  DSR RANGE DISTRIBUTION (Saturated Problems)")
        print(f"{'='*70}\n")

        ranges = sat_results["dsr_ranges"]
        bins = [0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0]
        hist, _ = np.histogram(ranges, bins=bins)

        print(f"  DSR range    | Count | Pct   | Bar")
        print(f"  -------------|-------|-------|" + "-" * 40)
        for i in range(len(bins) - 1):
            pct = 100 * hist[i] / len(ranges) if ranges else 0
            bar = "#" * int(pct / 2)
            print(f"  {bins[i]:.2f} - {bins[i+1]:.2f} | {hist[i]:>5} | {pct:>4.1f}% | {bar}")

        print(f"\n  Total: {len(ranges)} problems")
        print(f"  Problems with DSR range > 0.10: {sum(1 for r in ranges if r > 0.10)} ({100*sum(1 for r in ranges if r > 0.10)/len(ranges):.1f}%)")
        print(f"  Problems with DSR range > 0.20: {sum(1 for r in ranges if r > 0.20)} ({100*sum(1 for r in ranges if r > 0.20)/len(ranges):.1f}%)")

    # ================================================================
    # Per-problem deep dive: Top structural variance examples
    # ================================================================
    if saturated:
        print(f"\n{'='*70}")
        print(f"  TOP 10 SATURATED PROBLEMS BY STRUCTURAL VARIANCE")
        print(f"{'='*70}\n")

        sat_with_var = []
        for p in saturated:
            dsrs = [t.dsr for t in p["traces"]]
            sat_with_var.append({
                "pid": p["problem_id"],
                "dsr_var": np.var(dsrs),
                "dsr_range": max(dsrs) - min(dsrs),
                "dsr_min": min(dsrs),
                "dsr_max": max(dsrs),
                "dsr_mean": np.mean(dsrs),
                "dsrs": dsrs,
            })

        sat_with_var.sort(key=lambda x: x["dsr_var"], reverse=True)

        print(f"  {'Problem':<12} | {'DSR var':>8} | {'DSR range':>9} | {'Min':>5} | {'Max':>5} | {'Mean':>5} | DSR values")
        print(f"  {'-'*12}-+-{'-'*8}-+-{'-'*9}-+-{'-'*5}-+-{'-'*5}-+-{'-'*5}-+-{'-'*40}")
        for p in sat_with_var[:10]:
            dsrs_str = " ".join(f"{d:.2f}" for d in p["dsrs"])
            print(f"  {p['pid']:<12} | {p['dsr_var']:>8.5f} | {p['dsr_range']:>9.4f} | {p['dsr_min']:>5.3f} | {p['dsr_max']:>5.3f} | {p['dsr_mean']:>5.3f} | {dsrs_str}")

    # ================================================================
    # Correctness-DSR independence check (replication of r=0.011)
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  ORTHOGONALITY CHECK: DSR vs Correctness")
    print(f"{'='*70}\n")

    all_dsrs = []
    all_correct = []
    for p in problems:
        for t in p["traces"]:
            if t.num_steps > 0:
                all_dsrs.append(t.dsr)
                all_correct.append(1.0 if t.is_correct else 0.0)

    if all_dsrs:
        corr = np.corrcoef(all_dsrs, all_correct)[0, 1]
        print(f"  Pearson correlation (DSR, correctness): r = {corr:.4f}")
        print(f"  N traces: {len(all_dsrs)}")
        print(f"  Expected from DecoR: r ≈ 0.011")
        if abs(corr) < 0.05:
            print(f"  >> CONFIRMED: DSR is orthogonal to correctness")
        elif abs(corr) < 0.10:
            print(f"  >> Weak correlation, mostly orthogonal")
        else:
            print(f"  >> Unexpected: correlation is non-trivial")

    # ================================================================
    # Summary verdict
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  SUMMARY VERDICT")
    print(f"{'='*70}\n")

    if sat_results:
        echo_broken = sat_results["struct_var_mean"] > 0.001
        pct_signal = sat_results["pct_nonzero_struct_var"]
        avg_range = sat_results["dsr_range_mean"]

        print(f"  1. Outcome reward on saturated problems: variance = {sat_results['outcome_var_mean']:.6f}")
        print(f"     → {'ZERO signal (echo trap)' if sat_results['outcome_var_mean'] < 0.001 else 'Has signal'}")
        print()
        print(f"  2. StructPRM reward on saturated problems: variance = {sat_results['struct_var_mean']:.6f}")
        print(f"     → {'NONZERO signal (echo trap broken!)' if echo_broken else 'Also zero (hypothesis failed)'}")
        print()
        print(f"  3. {pct_signal:.0f}% of saturated problems have structural variance > 0.001")
        print(f"     → {'Majority of problems retain signal' if pct_signal > 50 else 'Minority retain signal'}")
        print()
        print(f"  4. Average DSR range across rollouts: {avg_range:.4f}")
        print(f"     → {'Meaningful variation exists' if avg_range > 0.05 else 'Limited variation'}")
        print()

        if echo_broken and pct_signal > 50 and avg_range > 0.05:
            print(f"  VERDICT: STRONG SUPPORT for StructPRM hypothesis")
            print(f"  StructPRM provides gradient signal in {pct_signal:.0f}% of cases")
            print(f"  where outcome reward provides zero signal.")
        elif echo_broken and avg_range > 0.02:
            print(f"  VERDICT: MODERATE SUPPORT for StructPRM hypothesis")
            print(f"  Signal exists but may be limited in magnitude.")
        else:
            print(f"  VERDICT: WEAK/NO SUPPORT for StructPRM hypothesis")
            print(f"  Structural signal also collapses under saturation.")

    print()


def main():
    parser = argparse.ArgumentParser(description="Verify StructPRM echo trap hypothesis")
    parser.add_argument("--rollouts", required=True, help="Path to rollouts JSON")
    parser.add_argument("--rollouts-8b", default=None, help="Optional 8B rollouts for comparison")
    args = parser.parse_args()

    analyze_rollouts(args.rollouts, model_name="4B DSE-SFT")

    if args.rollouts_8b:
        analyze_rollouts(args.rollouts_8b, model_name="8B DSE-SFT")


if __name__ == "__main__":
    main()
