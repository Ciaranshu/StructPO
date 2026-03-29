"""
Hint Injection Pilot: Can hints push "all-wrong" problems to "edge of competence"?

Core hypothesis: StructPRM + adaptive difficulty can expand the effective training zone.
Standard RL wastes "all-wrong" problems (no signal). If hints make some solvable,
StructPRM can then provide structural quality signal on these newly-solvable problems.

Experiment:
1. Take problems where model gets 0/8 correct (beyond capability boundary)
2. Generate hints using the model itself (self-hint) or from ground truth
3. Re-generate K=4 rollouts with hints
4. Measure: how many problems become solvable? (0/8 → some/4 correct)
5. On newly-solvable problems: does StructPRM provide useful structural signal?

Usage:
    PYTHONPATH=src:$PYTHONPATH python scripts/analysis/hint_injection_pilot.py \
        --rollouts data/rollouts/math500_4b_rollouts.json \
        --model models/decor-qwen3-4b-dse \
        --output data/hint_pilot/results.json
"""

import json
import sys
import re
import argparse
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def extract_boxed_answer(text):
    matches = []
    i = 0
    while i < len(text):
        idx = text.find(r'\boxed{', i)
        if idx == -1:
            break
        depth = 0
        start = idx + len(r'\boxed{')
        for j in range(start, len(text)):
            if text[j] == '{':
                depth += 1
            elif text[j] == '}':
                if depth == 0:
                    matches.append(text[start:j])
                    i = j + 1
                    break
                depth -= 1
        else:
            i = start
            break
    return matches[-1].strip() if matches else ''


def normalize_answer(ans):
    ans = ans.strip()
    for prefix in [r'\text{', r'\mathrm{', r'\mathbf{']:
        if ans.startswith(prefix) and ans.endswith('}'):
            ans = ans[len(prefix):-1]
    ans = ans.replace(r'\dfrac', r'\frac')
    ans = ans.replace(r'\left', '').replace(r'\right', '')
    ans = re.sub(r'\s+', '', ans)
    ans = ans.rstrip('.')
    return ans


def check_correctness(predicted, ground_truth):
    pred_norm = normalize_answer(predicted)
    gt_norm = normalize_answer(ground_truth)
    if not pred_norm:
        return False
    return pred_norm == gt_norm


def generate_hints_from_gt(problems):
    """Generate hints from ground truth answers (oracle hints).

    These are NOT cheating — they give the answer format/type, not the answer itself.
    Example: "The answer is a positive integer less than 100" or "Try using modular arithmetic"
    """
    hints = []
    for p in problems:
        gt = p['ground_truth']

        # Simple hint strategies based on answer type
        hint_parts = []

        # Hint 1: answer type
        if gt.isdigit():
            hint_parts.append(f"The answer is a positive integer.")
        elif '/' in gt or 'frac' in gt:
            hint_parts.append(f"The answer is a fraction.")
        elif '.' in gt:
            hint_parts.append(f"The answer is a decimal number.")

        # Hint 2: answer magnitude (vague)
        try:
            val = float(gt.replace(',', ''))
            if val < 0:
                hint_parts.append("The answer is negative.")
            elif val < 10:
                hint_parts.append("The answer is a small number (single digit).")
            elif val < 100:
                hint_parts.append("The answer is a two-digit number.")
            elif val < 1000:
                hint_parts.append("The answer is a three-digit number.")
        except (ValueError, TypeError):
            pass

        if not hint_parts:
            hint_parts.append("Think carefully about this problem step by step.")

        hints.append(" ".join(hint_parts))

    return hints


def generate_self_hints(problems, llm, tokenizer, sampling_params_hint):
    """Generate hints using the model itself — analyze why it might be failing."""
    hints = []

    hint_prompts = []
    for p in problems:
        msg = (f"I tried to solve this problem but got it wrong multiple times. "
               f"Can you give me a brief hint about what approach or technique to use? "
               f"Don't solve it, just give a hint.\n\nProblem: {p['problem_text']}")
        messages = [{'role': 'user', 'content': msg}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        hint_prompts.append(prompt)

    outputs = llm.generate(hint_prompts, sampling_params_hint)

    for output in outputs:
        hint_text = output.outputs[0].text.strip()
        # Truncate to first 200 chars
        if len(hint_text) > 200:
            hint_text = hint_text[:200] + "..."
        hints.append(hint_text)

    return hints


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rollouts', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--num-rollouts', type=int, default=4)
    parser.add_argument('--hint-type', choices=['gt', 'self', 'both'], default='both')
    args = parser.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    # Load rollouts and find all-wrong problems
    data = json.loads(Path(args.rollouts).read_text())
    all_wrong = [p for p in data if not any(t.get('is_correct') for t in p['traces'])]
    print(f"Found {len(all_wrong)} all-wrong problems (0/{len(data[0]['traces'])} correct)")

    # Load problems file to get ground truth
    math500 = json.loads(Path('data/math500_problems.json').read_text())
    gt_lookup = {}
    for i, item in enumerate(math500):
        gt_answer = extract_boxed_answer(item['conversations'][1]['value'])
        gt_lookup[f'prob_{i}'] = gt_answer

    # Add ground truth to all_wrong problems
    for p in all_wrong:
        p['ground_truth'] = gt_lookup.get(p['problem_id'], p.get('ground_truth', ''))

    # Initialize model
    print(f"\nLoading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(
        model=args.model, trust_remote_code=True,
        max_model_len=16384, tensor_parallel_size=1,
        dtype="bfloat16", disable_custom_all_reduce=True,
    )

    stop_token_ids = [
        tokenizer.convert_tokens_to_ids('<|im_end|>'),
        tokenizer.eos_token_id,
    ]
    stop_token_ids = [t for t in stop_token_ids if t is not None]

    sampling_params = SamplingParams(
        temperature=0.7, max_tokens=16384,
        n=args.num_rollouts, stop_token_ids=stop_token_ids,
    )
    sampling_params_hint = SamplingParams(
        temperature=0.7, max_tokens=512,
        n=1, stop_token_ids=stop_token_ids,
    )

    results = {
        'baseline': {'pass_at_k': 0, 'total': len(all_wrong)},
    }

    # ================================================================
    # Condition 1: No hint (baseline — already have this data)
    # ================================================================
    print(f"\n=== Baseline (no hint): 0/{len(all_wrong)} solvable by definition ===\n")

    # ================================================================
    # Condition 2: GT-derived hints
    # ================================================================
    if args.hint_type in ('gt', 'both'):
        print("=== Condition: Ground-truth derived hints ===")
        gt_hints = generate_hints_from_gt(all_wrong)

        gt_prompts = []
        for p, hint in zip(all_wrong, gt_hints):
            msg = f"Hint: {hint}\n\nProblem: {p['problem_text']}"
            messages = [{'role': 'user', 'content': msg}]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=True,
            )
            gt_prompts.append(prompt)

        print(f"Generating {args.num_rollouts} rollouts for {len(all_wrong)} problems with GT hints...")
        t0 = time.time()
        gt_outputs = llm.generate(gt_prompts, sampling_params)
        elapsed = time.time() - t0
        print(f"Done in {elapsed:.0f}s")

        gt_solvable = 0
        gt_correct_traces = 0
        gt_total_traces = 0
        gt_problem_results = []

        for p, output in zip(all_wrong, gt_outputs):
            traces = []
            any_correct = False
            for comp in output.outputs:
                answer = extract_boxed_answer(comp.text)
                correct = check_correctness(answer, p['ground_truth'])
                traces.append({'correct': correct, 'tokens': len(comp.token_ids)})
                gt_total_traces += 1
                if correct:
                    gt_correct_traces += 1
                    any_correct = True
            if any_correct:
                gt_solvable += 1
            gt_problem_results.append({
                'problem_id': p['problem_id'],
                'traces': traces,
                'any_correct': any_correct,
            })

        print(f"\nGT hint results:")
        print(f"  Solvable: {gt_solvable}/{len(all_wrong)} ({100*gt_solvable/len(all_wrong):.1f}%)")
        print(f"  Correct traces: {gt_correct_traces}/{gt_total_traces} ({100*gt_correct_traces/gt_total_traces:.1f}%)")
        print(f"  → {gt_solvable} problems moved from 'all-wrong' to 'edge of competence'!")
        results['gt_hint'] = {
            'solvable': gt_solvable, 'total': len(all_wrong),
            'correct_traces': gt_correct_traces, 'total_traces': gt_total_traces,
            'problems': gt_problem_results,
        }

    # ================================================================
    # Condition 3: Self-generated hints
    # ================================================================
    if args.hint_type in ('self', 'both'):
        print("\n=== Condition: Self-generated hints ===")
        print("Generating self-hints...")
        self_hints = generate_self_hints(all_wrong, llm, tokenizer, sampling_params_hint)
        print(f"Sample hint: {self_hints[0][:100]}...")

        self_prompts = []
        for p, hint in zip(all_wrong, self_hints):
            msg = f"Here's a hint that might help: {hint}\n\nNow solve this problem:\n{p['problem_text']}"
            messages = [{'role': 'user', 'content': msg}]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=True,
            )
            self_prompts.append(prompt)

        print(f"Generating {args.num_rollouts} rollouts with self-hints...")
        t0 = time.time()
        self_outputs = llm.generate(self_prompts, sampling_params)
        elapsed = time.time() - t0
        print(f"Done in {elapsed:.0f}s")

        self_solvable = 0
        self_correct_traces = 0
        self_total_traces = 0
        self_problem_results = []

        for p, output in zip(all_wrong, self_outputs):
            traces = []
            any_correct = False
            for comp in output.outputs:
                answer = extract_boxed_answer(comp.text)
                correct = check_correctness(answer, p['ground_truth'])
                traces.append({'correct': correct, 'tokens': len(comp.token_ids)})
                self_total_traces += 1
                if correct:
                    self_correct_traces += 1
                    any_correct = True
            if any_correct:
                self_solvable += 1
            self_problem_results.append({
                'problem_id': p['problem_id'],
                'traces': traces,
                'any_correct': any_correct,
            })

        print(f"\nSelf-hint results:")
        print(f"  Solvable: {self_solvable}/{len(all_wrong)} ({100*self_solvable/len(all_wrong):.1f}%)")
        print(f"  Correct traces: {self_correct_traces}/{self_total_traces} ({100*self_correct_traces/self_total_traces:.1f}%)")
        print(f"  → {self_solvable} problems moved to 'edge of competence'!")
        results['self_hint'] = {
            'solvable': self_solvable, 'total': len(all_wrong),
            'correct_traces': self_correct_traces, 'total_traces': self_total_traces,
            'problems': self_problem_results,
        }

    # ================================================================
    # Summary
    # ================================================================
    print(f"\n{'='*60}")
    print(f"  SUMMARY: Expanding the Training Zone")
    print(f"{'='*60}\n")

    print(f"  {'Condition':<20} | {'Solvable':>10} | {'Pct':>6}")
    print(f"  {'-'*20}-+-{'-'*10}-+-{'-'*6}")
    print(f"  {'No hint (baseline)':<20} | {'0/' + str(len(all_wrong)):>10} | {'0.0%':>6}")
    if 'gt_hint' in results:
        r = results['gt_hint']
        print(f"  {'GT-derived hint':<20} | {str(r['solvable'])+'/'+str(r['total']):>10} | {100*r['solvable']/r['total']:>5.1f}%")
    if 'self_hint' in results:
        r = results['self_hint']
        print(f"  {'Self-generated hint':<20} | {str(r['solvable'])+'/'+str(r['total']):>10} | {100*r['solvable']/r['total']:>5.1f}%")

    print(f"\n  Original training zones (500 problems):")
    print(f"    All correct (wasted — saturated):  355 (71%)")
    print(f"    Partial (useful — edge):            90 (18%)")
    print(f"    All wrong (wasted — no signal):     55 (11%)")

    if 'gt_hint' in results:
        new_edge = 90 + results['gt_hint']['solvable']
        print(f"\n  With GT hints:")
        print(f"    Edge of competence expanded: 90 → {new_edge} problems ({100*new_edge/500:.0f}%)")
        print(f"    + StructPRM signal on 355 saturated problems")
        print(f"    = {355 + new_edge}/500 problems have training signal ({100*(355+new_edge)/500:.0f}%)")

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {args.output}")


if __name__ == '__main__':
    main()
