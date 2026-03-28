"""
Merge sharded rollout files into a single file.

Usage:
    python scripts/merge_rollout_shards.py \
        --shards data/rollouts/math500_4b_shard_*.json \
        --output data/rollouts/math500_4b_rollouts.json
"""

import json
import argparse
from pathlib import Path
import glob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shards', required=True, nargs='+', help='Shard files (glob OK)')
    parser.add_argument('--output', required=True, help='Merged output file')
    args = parser.parse_args()

    # Expand globs
    shard_files = []
    for pattern in args.shards:
        shard_files.extend(sorted(glob.glob(pattern)))

    if not shard_files:
        print("No shard files found!")
        return

    print(f"Merging {len(shard_files)} shards:")
    all_problems = []
    for f in shard_files:
        data = json.loads(Path(f).read_text())
        print(f"  {f}: {len(data)} problems")
        all_problems.extend(data)

    # Verify no duplicate problem IDs
    pids = [p['problem_id'] for p in all_problems]
    if len(set(pids)) != len(pids):
        print(f"WARNING: {len(pids) - len(set(pids))} duplicate problem IDs!")

    # Sort by problem_id for consistency
    all_problems.sort(key=lambda p: p['problem_id'])

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(all_problems, indent=2, ensure_ascii=False))

    # Summary
    total_traces = sum(len(p['traces']) for p in all_problems)
    total_correct = sum(1 for p in all_problems for t in p['traces'] if t.get('is_correct'))
    pass_at_k = sum(1 for p in all_problems if any(t.get('is_correct') for t in p['traces']))
    print(f"\nMerged: {len(all_problems)} problems, {total_traces} traces")
    print(f"Correct: {total_correct}/{total_traces} ({100*total_correct/max(total_traces,1):.1f}%)")
    print(f"pass@K: {pass_at_k}/{len(all_problems)} ({100*pass_at_k/max(len(all_problems),1):.1f}%)")
    print(f"Saved to: {output_path}")


if __name__ == '__main__':
    main()
