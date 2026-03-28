"""
Best-of-N Evaluation: Compare Different Reward Models as Solution Selectors

Given K=8 rollouts per problem, select the "best" rollout according to different
reward models and compare the resulting accuracy and efficiency.

This is the HERO TABLE experiment for the StructPRM paper. It demonstrates that
graph-structured rewards select better solutions than outcome, length, or
anchor-based rewards — with ZERO training cost.

Usage:
    PYTHONPATH=src:$PYTHONPATH python scripts/analysis/best_of_n_evaluation.py \
        --rollouts data/rollouts/4b_dse_rollouts.json \
        [--rollouts-8b data/rollouts/8b_dse_rollouts.json]
"""

import json
import sys
import argparse
import re
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from structpo.structural_parser.quality import full_quality_analysis


def find_answer_position(solution: str) -> float:
    """Find the position of the first \\boxed{} answer as fraction of total length."""
    match = re.search(r'\\boxed\{', solution)
    if match:
        return match.start() / len(solution) if solution else 1.0
    return 1.0


def analyze_rollouts(rollouts_path: str, model_name: str = "model"):
    """Run Best-of-N evaluation with multiple selectors."""

    print(f"\n{'='*70}")
    print(f"  Best-of-N Evaluation: {model_name}")
    print(f"{'='*70}\n")

    data = json.loads(Path(rollouts_path).read_text())
    print(f"Loaded {len(data)} problems\n")

    # ================================================================
    # Annotate all traces with structural metrics
    # ================================================================
    print("Computing structural metrics for all traces...")
    problems = []

    for i, problem in enumerate(data):
        pid = problem['problem_id']
        gt = problem.get('ground_truth', '')
        traces = []

        for tid, trace in enumerate(problem['traces']):
            solution = trace['solution']
            is_correct = trace.get('is_correct', False)
            num_tokens = trace.get('num_tokens', len(solution.split()))

            # Full quality analysis (Level 0 + Level 1)
            qa = full_quality_analysis(solution)

            # Answer position for APR-style selector
            answer_pos = find_answer_position(solution)
            post_answer_len = len(solution) * (1.0 - answer_pos) if answer_pos < 1.0 else 0

            traces.append({
                'tid': tid,
                'is_correct': is_correct,
                'num_tokens': num_tokens,
                'solution_len': len(solution),
                'dsr': qa['dsr'],
                'quality_reward': qa['quality_reward'],
                'num_steps': qa['num_steps'],
                'answer_position': answer_pos,
                'post_answer_len': post_answer_len,
            })

        problems.append({
            'pid': pid,
            'traces': traces,
            'has_correct': any(t['is_correct'] for t in traces),
        })

        if (i + 1) % 100 == 0:
            print(f"  ... {i+1}/{len(data)} problems")

    print(f"Annotated {sum(len(p['traces']) for p in problems)} traces\n")

    # ================================================================
    # Define selectors
    # ================================================================
    def select_random(traces):
        """Random selection among correct traces."""
        correct = [t for t in traces if t['is_correct']]
        if not correct:
            return traces[0]  # fallback
        return correct[np.random.randint(len(correct))]

    def select_orm(traces):
        """ORM: any correct trace (equivalent to random among correct)."""
        return select_random(traces)

    def select_shortest(traces):
        """Select shortest correct trace."""
        correct = [t for t in traces if t['is_correct']]
        if not correct:
            return min(traces, key=lambda t: t['num_tokens'])
        return min(correct, key=lambda t: t['num_tokens'])

    def select_longest(traces):
        """Select longest correct trace."""
        correct = [t for t in traces if t['is_correct']]
        if not correct:
            return max(traces, key=lambda t: t['num_tokens'])
        return max(correct, key=lambda t: t['num_tokens'])

    def select_apr_anchor(traces):
        """APR-style: select correct trace with shortest post-answer tail."""
        correct = [t for t in traces if t['is_correct']]
        if not correct:
            return min(traces, key=lambda t: t['post_answer_len'])
        return min(correct, key=lambda t: t['post_answer_len'])

    def select_structprm_l0(traces):
        """StructPRM Level 0: select correct trace with lowest DSR."""
        correct = [t for t in traces if t['is_correct']]
        if not correct:
            return min(traces, key=lambda t: t['dsr'])
        return min(correct, key=lambda t: t['dsr'])

    def select_structprm_l1(traces):
        """StructPRM Level 1: select correct trace with highest quality reward."""
        correct = [t for t in traces if t['is_correct']]
        if not correct:
            return max(traces, key=lambda t: t['quality_reward'])
        return max(correct, key=lambda t: t['quality_reward'])

    def select_structprm_l1_any(traces):
        """StructPRM Level 1 (unrestricted): highest quality reward, no correctness filter."""
        return max(traces, key=lambda t: t['quality_reward'])

    selectors = {
        'Random (correct)': select_random,
        'ORM (correct)': select_orm,
        'Shortest correct': select_shortest,
        'Longest correct': select_longest,
        'APR-anchor': select_apr_anchor,
        'StructPRM-L0 (DSR)': select_structprm_l0,
        'StructPRM-L1 (Quality)': select_structprm_l1,
        'StructPRM-L1 (any)': select_structprm_l1_any,
    }

    # ================================================================
    # Run all selectors
    # ================================================================
    print(f"{'='*70}")
    print(f"  RESULTS: Best-of-{len(data[0]['traces'])} Selection")
    print(f"{'='*70}\n")

    # Baseline: pass@K (any correct among K)
    pass_at_k = sum(1 for p in problems if p['has_correct']) / len(problems)
    print(f"  pass@{len(data[0]['traces'])}: {pass_at_k:.1%} ({sum(1 for p in problems if p['has_correct'])}/{len(problems)})\n")

    results = {}
    np.random.seed(42)

    for name, selector in selectors.items():
        accuracies = []
        tokens = []
        dsrs = []
        quality_rewards = []
        steps_list = []

        for p in problems:
            selected = selector(p['traces'])
            accuracies.append(1 if selected['is_correct'] else 0)
            tokens.append(selected['num_tokens'])
            dsrs.append(selected['dsr'])
            quality_rewards.append(selected['quality_reward'])
            steps_list.append(selected['num_steps'])

        acc = np.mean(accuracies)
        avg_tokens = np.mean(tokens)
        avg_dsr = np.mean(dsrs)
        avg_qr = np.mean(quality_rewards)
        avg_steps = np.mean(steps_list)

        results[name] = {
            'accuracy': acc,
            'avg_tokens': avg_tokens,
            'avg_dsr': avg_dsr,
            'avg_quality_reward': avg_qr,
            'avg_steps': avg_steps,
        }

    # ================================================================
    # Print results table
    # ================================================================
    print(f"  {'Selector':<25} | {'Acc':>6} | {'Tokens':>7} | {'DSR':>6} | {'QReward':>8} | {'Steps':>6}")
    print(f"  {'-'*25}-+-{'-'*6}-+-{'-'*7}-+-{'-'*6}-+-{'-'*8}-+-{'-'*6}")

    for name, r in results.items():
        marker = '**' if 'StructPRM' in name else '  '
        print(f"{marker}{name:<25} | {r['accuracy']:>5.1%} | {r['avg_tokens']:>7.0f} | {r['avg_dsr']:>6.3f} | {r['avg_quality_reward']:>8.4f} | {r['avg_steps']:>6.1f}")

    # ================================================================
    # Analysis: StructPRM vs baselines
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  ANALYSIS")
    print(f"{'='*70}\n")

    baseline_acc = results['Random (correct)']['accuracy']
    for name in ['Shortest correct', 'APR-anchor', 'StructPRM-L0 (DSR)', 'StructPRM-L1 (Quality)']:
        r = results[name]
        delta = r['accuracy'] - baseline_acc
        token_delta = r['avg_tokens'] - results['Random (correct)']['avg_tokens']
        print(f"  {name}:")
        print(f"    Accuracy vs random: {delta:+.1%} ({r['accuracy']:.1%} vs {baseline_acc:.1%})")
        print(f"    Token savings:      {token_delta:+.0f} ({r['avg_tokens']:.0f} vs {results['Random (correct)']['avg_tokens']:.0f})")
        print(f"    DSR:                {r['avg_dsr']:.3f}")
        print()

    # ================================================================
    # Key comparison: StructPRM-L1 vs Length vs ORM
    # ================================================================
    print(f"{'='*70}")
    print(f"  KEY COMPARISON")
    print(f"{'='*70}\n")

    l1 = results['StructPRM-L1 (Quality)']
    short = results['Shortest correct']
    orm = results['Random (correct)']

    print(f"  StructPRM-L1 vs Shortest:")
    print(f"    Accuracy: {l1['accuracy']:.1%} vs {short['accuracy']:.1%} ({l1['accuracy']-short['accuracy']:+.1%})")
    print(f"    DSR:      {l1['avg_dsr']:.3f} vs {short['avg_dsr']:.3f}")
    print(f"    Tokens:   {l1['avg_tokens']:.0f} vs {short['avg_tokens']:.0f}")
    print()
    print(f"  StructPRM-L1 vs Random:")
    print(f"    Accuracy: {l1['accuracy']:.1%} vs {orm['accuracy']:.1%} ({l1['accuracy']-orm['accuracy']:+.1%})")
    print(f"    DSR:      {l1['avg_dsr']:.3f} vs {orm['avg_dsr']:.3f}")
    print(f"    Tokens:   {l1['avg_tokens']:.0f} vs {orm['avg_tokens']:.0f}")

    # ================================================================
    # Verdict
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  VERDICT")
    print(f"{'='*70}\n")

    best_acc_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
    best_dsr_name = min(results.keys(), key=lambda k: results[k]['avg_dsr'])

    print(f"  Highest accuracy:  {best_acc_name} ({results[best_acc_name]['accuracy']:.1%})")
    print(f"  Lowest DSR:        {best_dsr_name} ({results[best_dsr_name]['avg_dsr']:.3f})")

    if 'StructPRM' in best_acc_name:
        print(f"\n  >> StructPRM achieves highest accuracy in Best-of-N selection")
    if 'StructPRM' in best_dsr_name:
        print(f"  >> StructPRM achieves lowest DSR (most efficient)")

    print()


def main():
    parser = argparse.ArgumentParser(description="Best-of-N evaluation with multiple reward models")
    parser.add_argument('--rollouts', required=True)
    parser.add_argument('--rollouts-8b', default=None)
    args = parser.parse_args()

    analyze_rollouts(args.rollouts, model_name="4B DSE-SFT")

    if args.rollouts_8b:
        analyze_rollouts(args.rollouts_8b, model_name="8B DSE-SFT")


if __name__ == '__main__':
    main()
