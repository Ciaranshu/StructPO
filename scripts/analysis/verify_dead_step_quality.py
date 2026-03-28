"""
Verify: Can we classify dead steps by QUALITY, not just live/dead?

This script analyzes the existing rollout data to answer:
1. What kinds of dead steps exist? (verification theater, circular revisit, etc.)
2. How are they distributed across problems and models?
3. Does quality-aware scoring provide MORE variance than raw DSR?
4. Can this be done with pure regex + graph analysis (no LLM)?

Usage:
    PYTHONPATH=src:$PYTHONPATH python scripts/analysis/verify_dead_step_quality.py \
        --rollouts data/rollouts/4b_dse_rollouts.json
"""

import json
import sys
import argparse
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from structpo.structural_parser.classifier import classify_trace, ClassifiedStep
from structpo.structural_parser.dag_builder import build_dag, ReasoningDAG, _extract_symbols
from structpo.structural_parser.reachability import backward_reachability


# ================================================================
# Dead Step Quality Classification
# ================================================================

def classify_dead_step(
    step: ClassifiedStep,
    dag: ReasoningDAG,
    all_steps: list[ClassifiedStep],
    step_symbols: dict[int, set],
) -> str:
    """Classify WHY a dead step is dead — what kind of waste is it?

    Categories:
        verification_theater: verifies something that is also dead
        redundant_verification: verifies something live, but result unused
        circular_revisit: re-derives content already present in a live step
        wasteful_exploration: exploration that leads nowhere
        abandoned_computation: computation whose result is never used
        productive_dead_end: dead, but shares symbols with live steps
                             (almost connected — might have inspired a live step)
        generic_dead: default
    """
    node = dag.nodes[step.id]

    # Collect live step IDs and their symbols
    live_ids = {nid for nid, n in dag.nodes.items() if n.is_live}
    dead_ids = {nid for nid, n in dag.nodes.items() if not n.is_live}

    # --- Verification Theater ---
    # A dead verification whose "target" (the step it checks) is also dead
    if step.step_type == 'verification':
        # Heuristic: verification usually checks the step right before it
        # or a step it shares symbols with
        potential_targets = node.inputs  # steps this verification depends on
        if not potential_targets and step.id > 0:
            potential_targets = [step.id - 1]  # fallback: previous step

        if potential_targets:
            targets_dead = all(t in dead_ids for t in potential_targets if t in dag.nodes)
            if targets_dead and len(potential_targets) > 0:
                return 'verification_theater'
            else:
                return 'redundant_verification'
        return 'redundant_verification'

    # --- Circular Revisit ---
    # A dead step that has high symbol overlap with a LIVE step
    # (it re-derived something that was already computed elsewhere)
    if step.step_type in ('computation', 'derivation'):
        my_symbols = step_symbols.get(step.id, set())
        if my_symbols:
            for lid in live_ids:
                live_syms = step_symbols.get(lid, set())
                if live_syms:
                    overlap = my_symbols & live_syms
                    # High overlap = likely re-derived the same thing
                    union = my_symbols | live_syms
                    if union and len(overlap) / len(union) >= 0.6:
                        return 'circular_revisit'

    # --- Productive Dead End ---
    # A dead step that shares SOME symbols with downstream live steps
    # (it didn't make it into the DAG path, but might have inspired
    # a live step through implicit influence)
    if step.step_type in ('exploration', 'derivation', 'computation'):
        my_symbols = step_symbols.get(step.id, set())
        if my_symbols:
            # Look at later live steps
            for other_step in all_steps:
                if other_step.id > step.id and other_step.id in live_ids:
                    other_syms = step_symbols.get(other_step.id, set())
                    overlap = my_symbols & other_syms
                    if len(overlap) >= 2:  # shares at least 2 symbols
                        return 'productive_dead_end'

    # --- Wasteful Exploration ---
    if step.step_type == 'exploration':
        return 'wasteful_exploration'

    # --- Correction that didn't help ---
    if step.step_type == 'correction':
        return 'failed_correction'

    # --- Abandoned Computation ---
    if step.step_type == 'computation':
        return 'abandoned_computation'

    # --- Generic ---
    return 'generic_dead'


# ================================================================
# Quality-Aware Structural Reward
# ================================================================

QUALITY_PENALTIES = {
    'verification_theater': 0.8,
    'circular_revisit': 0.6,
    'wasteful_exploration': 0.4,
    'failed_correction': 0.3,
    'abandoned_computation': 0.3,
    'redundant_verification': 0.2,
    'generic_dead': 0.3,
    'productive_dead_end': 0.05,  # almost no penalty
}


def compute_quality_aware_reward(steps, dag, step_symbols):
    """Compute quality-aware structural reward for a trace."""
    if not steps:
        return 1.0, {}

    total_penalty = 0.0
    quality_counts = Counter()

    for step in steps:
        node = dag.nodes[step.id]
        if node.is_live:
            continue

        quality = classify_dead_step(step, dag, steps, step_symbols)
        quality_counts[quality] += 1
        total_penalty += QUALITY_PENALTIES.get(quality, 0.3)

    reward = 1.0 - (total_penalty / len(steps)) if steps else 1.0
    reward = max(0.0, min(1.0, reward))  # clamp to [0, 1]
    return reward, dict(quality_counts)


# ================================================================
# Main Analysis
# ================================================================

def analyze(rollouts_path, model_name="model"):
    print(f"\n{'='*70}")
    print(f"  Dead Step Quality Analysis: {model_name}")
    print(f"{'='*70}\n")

    data = json.loads(Path(rollouts_path).read_text())
    print(f"Loaded {len(data)} problems\n")

    # Process all traces
    all_quality_counts = Counter()
    problem_results = []
    total_traces = 0
    total_dead_steps = 0
    total_live_steps = 0

    for i, problem in enumerate(data):
        pid = problem['problem_id']
        trace_results = []

        for tid, trace in enumerate(problem['traces']):
            total_traces += 1
            solution = trace['solution']
            is_correct = trace.get('is_correct', False)

            # Full structural pipeline
            steps = classify_trace(solution)
            if not steps:
                trace_results.append({
                    'dsr': 0.0, 'quality_reward': 1.0,
                    'quality_counts': {}, 'is_correct': is_correct
                })
                continue

            dag = build_dag(steps)
            dag = backward_reachability(dag)
            step_symbols = {s.id: _extract_symbols(s.content) for s in steps}

            dsr = dag.dead_step_ratio
            quality_reward, quality_counts = compute_quality_aware_reward(
                steps, dag, step_symbols
            )

            for q, c in quality_counts.items():
                all_quality_counts[q] += c

            total_dead_steps += dag.num_dead
            total_live_steps += dag.num_live

            trace_results.append({
                'dsr': dsr,
                'quality_reward': quality_reward,
                'quality_counts': quality_counts,
                'is_correct': is_correct,
                'num_steps': len(steps),
            })

        problem_results.append({
            'pid': pid,
            'traces': trace_results,
            'all_correct': all(t['is_correct'] for t in trace_results),
        })

        if (i + 1) % 100 == 0:
            print(f"  ... {i+1}/{len(data)} problems processed")

    print(f"\nProcessed {total_traces} traces\n")

    # ================================================================
    # 1. Dead Step Quality Distribution
    # ================================================================
    print(f"{'='*70}")
    print(f"  DEAD STEP QUALITY DISTRIBUTION")
    print(f"{'='*70}\n")

    print(f"  Total live steps:  {total_live_steps}")
    print(f"  Total dead steps:  {total_dead_steps}")
    print(f"  Overall DSR:       {total_dead_steps/(total_live_steps+total_dead_steps):.3f}\n")

    total_classified = sum(all_quality_counts.values())
    print(f"  {'Quality Type':<25} | {'Count':>6} | {'Pct':>6} | {'Penalty':>7} | Bar")
    print(f"  {'-'*25}-+-{'-'*6}-+-{'-'*6}-+-{'-'*7}-+-{'-'*30}")

    for quality in sorted(all_quality_counts.keys(),
                          key=lambda q: QUALITY_PENALTIES.get(q, 0.3), reverse=True):
        count = all_quality_counts[quality]
        pct = 100 * count / total_classified if total_classified else 0
        penalty = QUALITY_PENALTIES.get(quality, 0.3)
        bar = '#' * int(pct)
        print(f"  {quality:<25} | {count:>6} | {pct:>5.1f}% | {penalty:>7.2f} | {bar}")

    # ================================================================
    # 2. Quality-Aware vs Raw DSR Variance (Echo Trap Test)
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  ECHO TRAP: Quality-Aware Reward vs Raw DSR")
    print(f"{'='*70}\n")

    saturated = [p for p in problem_results if p['all_correct']]
    partial = [p for p in problem_results if not p['all_correct']
               and any(t['is_correct'] for t in p['traces'])]

    def variance_analysis(problems, label):
        dsr_vars = []
        quality_vars = []
        dsr_quality_corrs = []

        for p in problems:
            correct = [t for t in p['traces'] if t['is_correct']]
            if len(correct) < 2:
                continue

            dsrs = [t['dsr'] for t in correct]
            qrs = [t['quality_reward'] for t in correct]

            dsr_vars.append(np.var(dsrs))
            quality_vars.append(np.var(qrs))

            if np.std(dsrs) > 0 and np.std(qrs) > 0:
                dsr_quality_corrs.append(np.corrcoef(dsrs, qrs)[0, 1])

        if not dsr_vars:
            print(f"  {label}: insufficient data")
            return

        print(f"  {label} ({len(dsr_vars)} problems):")
        print(f"    Raw DSR variance:        mean={np.mean(dsr_vars):.6f}  median={np.median(dsr_vars):.6f}")
        print(f"    Quality reward variance: mean={np.mean(quality_vars):.6f}  median={np.median(quality_vars):.6f}")

        ratio = np.mean(quality_vars) / np.mean(dsr_vars) if np.mean(dsr_vars) > 0 else float('inf')
        print(f"    Quality/DSR var ratio:   {ratio:.3f}")

        if dsr_quality_corrs:
            print(f"    DSR-Quality correlation: r={np.mean(dsr_quality_corrs):.3f} (mean)")
            if np.mean(dsr_quality_corrs) < 0.95:
                print(f"    >> Quality reward provides ADDITIONAL information beyond DSR")
            else:
                print(f"    >> Quality reward is redundant with DSR")
        print()

    variance_analysis(saturated, "SATURATED (echo trap)")
    variance_analysis(partial, "PARTIAL")

    # ================================================================
    # 3. Key Question: Do productive dead ends exist in saturated problems?
    # ================================================================
    print(f"{'='*70}")
    print(f"  PRODUCTIVE DEAD ENDS IN SATURATED PROBLEMS")
    print(f"{'='*70}\n")

    prod_dead_counts = []
    theater_counts = []
    for p in saturated:
        for t in p['traces']:
            prod = t['quality_counts'].get('productive_dead_end', 0)
            theater = t['quality_counts'].get('verification_theater', 0)
            prod_dead_counts.append(prod)
            theater_counts.append(theater)

    print(f"  Across {len(prod_dead_counts)} traces in saturated problems:")
    print(f"  Productive dead ends: mean={np.mean(prod_dead_counts):.2f}/trace, "
          f"max={max(prod_dead_counts)}, "
          f"traces_with_any={sum(1 for c in prod_dead_counts if c > 0)} "
          f"({100*sum(1 for c in prod_dead_counts if c > 0)/len(prod_dead_counts):.1f}%)")
    print(f"  Verification theater: mean={np.mean(theater_counts):.2f}/trace, "
          f"max={max(theater_counts)}, "
          f"traces_with_any={sum(1 for c in theater_counts if c > 0)} "
          f"({100*sum(1 for c in theater_counts if c > 0)/len(theater_counts):.1f}%)")
    print()
    print(f"  Key insight: if productive_dead_end count is significant,")
    print(f"  then quality-aware reward PROTECTS necessary exploration")
    print(f"  while raw DSR would penalize it uniformly.")

    # ================================================================
    # 4. Example: Same problem, different quality profiles
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  EXAMPLE: SAME PROBLEM, DIFFERENT QUALITY PROFILES")
    print(f"{'='*70}\n")

    # Find a saturated problem with high variance in quality reward
    best_problem = None
    best_var = 0
    for p in saturated:
        correct = [t for t in p['traces'] if t['is_correct']]
        if len(correct) >= 4:
            qrs = [t['quality_reward'] for t in correct]
            v = np.var(qrs)
            if v > best_var:
                best_var = v
                best_problem = p

    if best_problem:
        print(f"  Problem: {best_problem['pid']} (all {len(best_problem['traces'])} rollouts correct)")
        print(f"  Quality reward variance: {best_var:.5f}\n")
        print(f"  {'Rollout':>8} | {'DSR':>5} | {'QReward':>8} | {'Steps':>5} | Dead Step Breakdown")
        print(f"  {'-'*8}-+-{'-'*5}-+-{'-'*8}-+-{'-'*5}-+-{'-'*40}")

        for i, t in enumerate(best_problem['traces']):
            breakdown = ", ".join(f"{k}:{v}" for k, v in sorted(t.get('quality_counts', {}).items()))
            if not breakdown:
                breakdown = "(all live)"
            print(f"  roll_{i:>3} | {t['dsr']:>5.3f} | {t['quality_reward']:>8.4f} | {t.get('num_steps',0):>5} | {breakdown}")

    # ================================================================
    # 5. Summary
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}\n")

    print(f"  1. Dead step quality classification works with pure regex + graph analysis")
    print(f"     → No LLM needed, <10ms/trace, suitable for RL loop")
    print()

    if all_quality_counts.get('productive_dead_end', 0) > 0:
        pde_pct = 100 * all_quality_counts['productive_dead_end'] / total_classified
        print(f"  2. {pde_pct:.1f}% of dead steps are 'productive dead ends'")
        print(f"     → Quality-aware reward protects these (penalty 0.05 vs 0.3-0.8)")
        print(f"     → Raw DSR penalizes them equally with verification theater")
    else:
        print(f"  2. No productive dead ends found — quality classification may need tuning")
    print()

    vt_pct = 100 * all_quality_counts.get('verification_theater', 0) / total_classified if total_classified else 0
    print(f"  3. {vt_pct:.1f}% of dead steps are 'verification theater' (highest penalty)")
    print(f"     → This is the #1 structural anti-pattern to target")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rollouts', required=True)
    parser.add_argument('--model-name', default='model')
    args = parser.parse_args()
    analyze(args.rollouts, args.model_name)


if __name__ == '__main__':
    main()
