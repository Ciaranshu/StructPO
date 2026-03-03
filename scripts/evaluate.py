#!/usr/bin/env python3
"""
StructPO Evaluation: MATH-500 and GPQA benchmarks via vLLM.

Generates completions with greedy decoding (temperature=0), extracts
\\boxed{} answers, and computes accuracy. Also runs structural analysis
on each completion for DSR statistics.

Usage:
    python scripts/evaluate.py \
        --model models/structpo-qwen3-4b-stage2 \
        --benchmarks math500 gpqa \
        --output eval_results/4b_stage2.json

Requirements:
    - vLLM (structpo-eval env)
    - datasets (HuggingFace)
"""

import json
import re
import sys
import time
import argparse
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.generate_rollouts import extract_boxed_answer, normalize_answer, check_correctness


# ── Benchmark Loaders ──────────────────────────────────────────────

def load_math500() -> list[dict]:
    """Load MATH-500 test set from HuggingFace."""
    from datasets import load_dataset
    ds = load_dataset('HuggingFaceH4/MATH-500', split='test', trust_remote_code=True)
    problems = []
    for item in ds:
        problems.append({
            'id': item['unique_id'],
            'problem': item['problem'],
            'answer': item['answer'],
            'subject': item.get('subject', ''),
            'level': item.get('level', 0),
            'benchmark': 'math500',
        })
    print(f"  MATH-500: loaded {len(problems)} problems")
    return problems


def load_gpqa() -> list[dict]:
    """Load GPQA Diamond test set from HuggingFace."""
    from datasets import load_dataset
    try:
        ds = load_dataset('Idavidrein/gpqa', 'gpqa_diamond', split='train',
                          trust_remote_code=True)
    except Exception:
        # Fallback: try alternate name
        try:
            ds = load_dataset('gpqa/gpqa_diamond', split='test', trust_remote_code=True)
        except Exception as e:
            print(f"  GPQA: failed to load ({e}), skipping")
            return []

    problems = []
    for item in ds:
        # GPQA is multiple choice; format as text
        question = item.get('Question', item.get('question', ''))
        choices = []
        for key in ['Correct Answer', 'Incorrect Answer 1', 'Incorrect Answer 2', 'Incorrect Answer 3']:
            val = item.get(key, '')
            if val:
                choices.append(val)

        correct_answer = item.get('Correct Answer', '')
        # Shuffle choices but track correct
        import random
        random.seed(hash(question))
        labels = ['A', 'B', 'C', 'D']
        random.shuffle(choices)
        correct_label = labels[choices.index(correct_answer)] if correct_answer in choices else 'A'

        formatted = question + "\n\nChoices:\n"
        for label, choice in zip(labels, choices):
            formatted += f"({label}) {choice}\n"
        formatted += "\nPlease provide your answer as a single letter (A, B, C, or D) in \\boxed{}."

        problems.append({
            'id': f"gpqa_{len(problems)}",
            'problem': formatted,
            'answer': correct_label,
            'subject': item.get('Subdomain', ''),
            'level': 0,
            'benchmark': 'gpqa',
        })
    print(f"  GPQA Diamond: loaded {len(problems)} problems")
    return problems


# ── Evaluation Engine ──────────────────────────────────────────────

def evaluate_model(
    model_path: str,
    problems: list[dict],
    max_tokens: int = 16384,
) -> list[dict]:
    """Generate completions and evaluate accuracy."""
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        max_model_len=32768,
        tensor_parallel_size=1,
        dtype="bfloat16",
    )

    stop_token_ids = [
        tokenizer.convert_tokens_to_ids('<|im_end|>'),
        tokenizer.eos_token_id,
    ]
    stop_token_ids = [t for t in stop_token_ids if t is not None]

    sampling_params = SamplingParams(
        temperature=0.0,  # Greedy for evaluation
        max_tokens=max_tokens,
        n=1,
        stop_token_ids=stop_token_ids,
    )

    # Format prompts
    prompts = []
    for p in problems:
        messages = [{'role': 'user', 'content': p['problem']}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=True,
        )
        prompts.append(prompt)

    print(f"\nGenerating completions for {len(problems)} problems (greedy)...")
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - t0
    print(f"Generation took {elapsed:.0f}s ({elapsed/len(problems):.1f}s/problem)")

    results = []
    for problem, output in zip(problems, outputs):
        completion = output.outputs[0]
        solution = completion.text
        predicted = extract_boxed_answer(solution)

        # For GPQA, also try to extract single letter
        if problem['benchmark'] == 'gpqa' and not predicted:
            # Try to find a standalone letter answer
            letter_match = re.search(r'\b([ABCD])\b', solution[-200:])
            if letter_match:
                predicted = letter_match.group(1)

        is_correct = check_correctness(predicted, problem['answer'])

        results.append({
            'id': problem['id'],
            'benchmark': problem['benchmark'],
            'subject': problem['subject'],
            'level': problem['level'],
            'ground_truth': problem['answer'],
            'predicted': predicted,
            'is_correct': is_correct,
            'solution': solution,
            'num_tokens': len(completion.token_ids),
        })

    return results


def compute_metrics(results: list[dict]) -> dict:
    """Compute accuracy metrics, broken down by benchmark/subject/level."""
    metrics = {}

    # Overall per benchmark
    by_bench = defaultdict(list)
    for r in results:
        by_bench[r['benchmark']].append(r)

    for bench, items in by_bench.items():
        correct = sum(1 for r in items if r['is_correct'])
        total = len(items)
        avg_tokens = sum(r['num_tokens'] for r in items) / max(total, 1)
        metrics[bench] = {
            'accuracy': correct / max(total, 1),
            'correct': correct,
            'total': total,
            'avg_tokens': avg_tokens,
        }

        # By subject
        by_subject = defaultdict(list)
        for r in items:
            if r['subject']:
                by_subject[r['subject']].append(r)
        if by_subject:
            metrics[f'{bench}_by_subject'] = {}
            for subj, sitems in sorted(by_subject.items()):
                sc = sum(1 for r in sitems if r['is_correct'])
                metrics[f'{bench}_by_subject'][subj] = {
                    'accuracy': sc / len(sitems),
                    'correct': sc,
                    'total': len(sitems),
                }

        # By level (MATH)
        by_level = defaultdict(list)
        for r in items:
            if r['level']:
                by_level[r['level']].append(r)
        if by_level:
            metrics[f'{bench}_by_level'] = {}
            for lvl, litems in sorted(by_level.items()):
                lc = sum(1 for r in litems if r['is_correct'])
                metrics[f'{bench}_by_level'][lvl] = {
                    'accuracy': lc / len(litems),
                    'correct': lc,
                    'total': len(litems),
                }

    return metrics


def run_structural_analysis(results: list[dict]) -> dict:
    """Run structural analysis on completions to get DSR stats."""
    from src.structural_parser.reachability import full_structural_analysis

    dsrs = []
    for r in results:
        analysis = full_structural_analysis(r['solution'])
        r['dsr'] = analysis['dsr']
        r['num_steps'] = analysis['num_steps']
        r['num_dead'] = analysis['num_dead']
        dsrs.append(analysis['dsr'])

    return {
        'avg_dsr': sum(dsrs) / max(len(dsrs), 1),
        'nonzero_dsr_pct': sum(1 for d in dsrs if d > 0) / max(len(dsrs), 1),
        'avg_steps': sum(r['num_steps'] for r in results) / max(len(results), 1),
    }


# ── Main ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Evaluate StructPO models')
    parser.add_argument('--model', required=True, help='Path to model')
    parser.add_argument('--benchmarks', nargs='+', default=['math500'],
                        choices=['math500', 'gpqa'], help='Benchmarks to run')
    parser.add_argument('--max-tokens', type=int, default=16384)
    parser.add_argument('--output', required=True, help='Output JSON path')
    parser.add_argument('--no-structural', action='store_true',
                        help='Skip structural analysis on completions')
    args = parser.parse_args()

    print(f"=== StructPO Evaluation ===")
    print(f"Model: {args.model}")
    print(f"Benchmarks: {args.benchmarks}")

    # Load benchmarks
    all_problems = []
    if 'math500' in args.benchmarks:
        all_problems.extend(load_math500())
    if 'gpqa' in args.benchmarks:
        all_problems.extend(load_gpqa())

    if not all_problems:
        print("No problems loaded, exiting.")
        return

    # Run evaluation
    results = evaluate_model(args.model, all_problems, max_tokens=args.max_tokens)

    # Compute metrics
    metrics = compute_metrics(results)

    # Structural analysis
    structural_metrics = {}
    if not args.no_structural:
        print("\nRunning structural analysis on completions...")
        structural_metrics = run_structural_analysis(results)
        print(f"  Avg DSR: {structural_metrics['avg_dsr']:.1%}")
        print(f"  Avg steps: {structural_metrics['avg_steps']:.1f}")

    # Print summary
    print(f"\n{'='*50}")
    print(f"RESULTS: {args.model}")
    print(f"{'='*50}")
    for bench in args.benchmarks:
        if bench in metrics:
            m = metrics[bench]
            print(f"  {bench}: {m['accuracy']:.1%} ({m['correct']}/{m['total']}), "
                  f"avg {m['avg_tokens']:.0f} tokens")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        'model': args.model,
        'metrics': metrics,
        'structural_metrics': structural_metrics,
        'results': results,
    }
    output_path.write_text(json.dumps(output_data, indent=2, ensure_ascii=False))
    print(f"\nSaved to {output_path}")


if __name__ == '__main__':
    main()
