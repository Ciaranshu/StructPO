"""
End-to-end pipeline: Annotate rollouts → Build structural preference pairs

Usage:
    python scripts/annotate_and_build_pairs.py \
        --rollouts data/rollouts/stage1_rollouts.json \
        --output data/structural_pairs/structural_dpo_pairs.json
"""

import json
import sys
import argparse
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from structpo.preference_builder.annotator import annotate_trace
from structpo.preference_builder.pair_builder import build_all_pairs, AnnotatedTrace


def main():
    parser = argparse.ArgumentParser(description='Annotate rollouts and build preference pairs')
    parser.add_argument('--rollouts', required=True, help='Path to rollouts JSON')
    parser.add_argument('--output', required=True, help='Output path for DPO pairs')
    parser.add_argument('--dsr-low', type=float, default=0.15, help='DSR threshold for "efficient"')
    parser.add_argument('--dsr-high', type=float, default=0.35, help='DSR threshold for "wasteful"')
    args = parser.parse_args()

    # Load rollouts
    data = json.loads(Path(args.rollouts).read_text())
    print(f"Loaded {len(data)} problems")

    # Annotate all traces
    all_traces = []
    problem_texts = {}
    for problem in data:
        pid = problem['problem_id']
        problem_texts[pid] = problem.get('problem_text', '')
        for tid, trace in enumerate(problem['traces']):
            ann = annotate_trace(
                problem_id=pid,
                trace_id=tid,
                solution=trace['solution'],
                answer=trace.get('answer', ''),
                is_correct=trace.get('is_correct', False),
            )
            all_traces.append(ann)

    print(f"Annotated {len(all_traces)} traces")

    # Summary stats
    correct = [t for t in all_traces if t.is_correct]
    avg_dsr = sum(t.dsr for t in all_traces) / len(all_traces) if all_traces else 0
    avg_dsr_correct = sum(t.dsr for t in correct) / len(correct) if correct else 0
    print(f"  Correct: {len(correct)} / {len(all_traces)} ({100*len(correct)/len(all_traces):.1f}%)")
    print(f"  Avg DSR (all): {avg_dsr:.3f}")
    print(f"  Avg DSR (correct): {avg_dsr_correct:.3f}")

    # Build preference pairs
    pairs = build_all_pairs(
        all_traces,
        problem_texts=problem_texts,
        output_path=args.output,
    )

    print(f"\nDone. {len(pairs)} preference pairs saved to {args.output}")


if __name__ == '__main__':
    main()
