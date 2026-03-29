"""
Hint Injection V2: Stronger hints from GT solution key steps.

V1 failed (3.6%) because hints were too weak ("answer is an integer").
V2 extracts the KEY REASONING STEP from the GT solution — the critical
insight or technique needed, without giving away the answer.

Also uses fixed answer normalization.

Usage:
    PYTHONPATH=src:${PYTHONPATH:-} python scripts/analysis/hint_injection_v2.py \
        --rollouts data/rollouts/math500_4b_rollouts.json \
        --model models/decor-qwen3-4b-dse \
        --output data/hint_pilot/v2_results.json
"""

import json
import sys
import re
import time
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.generate_rollouts import extract_boxed_answer, normalize_answer, check_correctness


def extract_key_step_hint(gt_solution: str) -> str:
    """Extract a key reasoning step from the GT solution as a hint.

    Strategy: find sentences containing mathematical insights — equations,
    key techniques, or critical observations. Avoid problem restatement
    and final answers.
    """
    # Remove Asymptote code blocks
    clean = re.sub(r'\[asy\].*?\[/asy\]', '', gt_solution, flags=re.DOTALL)

    # Split into sentences (rough, handles LaTeX)
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', clean)
    # Also split on \n\n
    all_parts = []
    for s in sentences:
        all_parts.extend([p.strip() for p in s.split('\n\n') if p.strip()])

    # Score each part for "hint quality"
    scored = []
    for part in all_parts:
        if len(part) < 20:
            continue
        if r'\boxed{' in part:
            continue

        score = 0
        # Math content signals
        if any(kw in part for kw in ['=', r'\frac', r'\sqrt', r'\cos', r'\sin', r'\cdot']):
            score += 2
        # Technique/insight signals
        if any(kw in part.lower() for kw in [
            'apply', 'using', 'by the', 'theorem', 'formula', 'identity',
            'substitut', 'factor', 'expand', 'simplif', 'rewrite',
            'note that', 'observe', 'notice', 'key', 'trick', 'insight',
            'let us', 'consider', 'define', 'recall', 'since',
            'vieta', 'binomial', 'modular', 'pigeonhole', 'induction',
        ]):
            score += 3
        # Penalize problem restatement
        if any(kw in part.lower() for kw in ['we are given', 'the problem states', 'we need to find', 'we want to find']):
            score -= 5
        # Penalize very long (probably full derivation, too much info)
        if len(part) > 500:
            score -= 1
        # Penalize if it's just a number or short result
        if len(part) < 40:
            score -= 1

        scored.append((score, part))

    scored.sort(key=lambda x: x[0], reverse=True)

    if scored and scored[0][0] > 0:
        hint = scored[0][1][:250]
        if len(scored[0][1]) > 250:
            hint += "..."
        return f"Key approach: {hint}"

    # Fallback: take the first non-trivial paragraph after any code
    for part in all_parts:
        if len(part) > 40 and r'\boxed{' not in part:
            return f"Consider: {part[:200]}"

    return "Think about what mathematical technique or identity might simplify this."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rollouts', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--num-rollouts', type=int, default=4)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    # Load data
    data = json.loads(Path(args.rollouts).read_text())
    math500 = json.loads(Path('data/math500_problems.json').read_text())

    # Find TRUE all-wrong problems (with fixed normalization)
    all_wrong = []
    for p in data:
        idx = int(p['problem_id'].replace('prob_', ''))
        gt = extract_boxed_answer(math500[idx]['conversations'][1]['value'])
        any_correct = any(
            check_correctness(extract_boxed_answer(t['solution']), gt)
            for t in p['traces']
        )
        if not any_correct:
            p['ground_truth'] = gt
            p['gt_solution'] = math500[idx]['conversations'][1]['value']
            p['problem_text'] = math500[idx]['conversations'][0]['value']
            all_wrong.append(p)

    print(f"True all-wrong problems (fixed normalization): {len(all_wrong)}")

    # Generate strong hints from GT solutions
    hints = []
    for p in all_wrong:
        hint = extract_key_step_hint(p['gt_solution'])
        hints.append(hint)

    print(f"\nSample hints:")
    for i in range(min(5, len(all_wrong))):
        print(f"  {all_wrong[i]['problem_id']}: {hints[i][:100]}...")
    print()

    # Initialize model
    print(f"Loading model: {args.model}")
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

    # ============================================================
    # Condition 1: Strong GT hints (key reasoning step)
    # ============================================================
    print(f"=== Condition: Strong GT hints (key reasoning step) ===")

    prompts = []
    for p, hint in zip(all_wrong, hints):
        msg = f"{hint}\n\nNow solve this problem:\n{p['problem_text']}"
        messages = [{'role': 'user', 'content': msg}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=True,
        )
        prompts.append(prompt)

    print(f"Generating {args.num_rollouts} rollouts for {len(all_wrong)} problems...")
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.0f}s")

    solvable = 0
    correct_traces = 0
    total_traces = 0
    problem_results = []

    for p, output in zip(all_wrong, outputs):
        traces = []
        any_correct = False
        for comp in output.outputs:
            answer = extract_boxed_answer(comp.text)
            correct = check_correctness(answer, p['ground_truth'])
            traces.append({
                'correct': correct,
                'tokens': len(comp.token_ids),
                'answer': answer,
            })
            total_traces += 1
            if correct:
                correct_traces += 1
                any_correct = True
        if any_correct:
            solvable += 1
        problem_results.append({
            'problem_id': p['problem_id'],
            'hint': hints[all_wrong.index(p)],
            'traces': traces,
            'any_correct': any_correct,
            'gt_answer': p['ground_truth'],
        })

    print(f"\nStrong GT hint results:")
    print(f"  Solvable: {solvable}/{len(all_wrong)} ({100*solvable/len(all_wrong):.1f}%)")
    print(f"  Correct traces: {correct_traces}/{total_traces} ({100*correct_traces/total_traces:.1f}%)")
    print(f"  → {solvable} problems moved from 'beyond boundary' to 'edge of competence'!")

    # ============================================================
    # Condition 2: Self-generated strong hints
    # ============================================================
    print(f"\n=== Condition: Model self-analyzes its failures ===")

    # Show the model its own wrong answers and ask for reflection
    reflection_prompts = []
    for p in all_wrong:
        wrong_answers = set()
        for t in p['traces']:
            ans = extract_boxed_answer(t['solution'])
            if ans:
                wrong_answers.add(ans)
        wrong_str = ", ".join(list(wrong_answers)[:3])

        msg = (f"I tried to solve this problem and got these wrong answers: {wrong_str}\n\n"
               f"Problem: {p['problem_text']}\n\n"
               f"What key insight or technique am I missing? Give me a brief hint "
               f"about the correct approach (don't solve it).")
        messages = [{'role': 'user', 'content': msg}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        reflection_prompts.append(prompt)

    hint_params = SamplingParams(
        temperature=0.7, max_tokens=512, n=1, stop_token_ids=stop_token_ids,
    )

    print("Generating self-reflective hints...")
    hint_outputs = llm.generate(reflection_prompts, hint_params)
    self_hints = [o.outputs[0].text.strip()[:300] for o in hint_outputs]

    print(f"Sample self-hint: {self_hints[0][:100]}...")

    # Now solve with self-hints
    self_prompts = []
    for p, hint in zip(all_wrong, self_hints):
        msg = f"Hint from reflection: {hint}\n\nNow solve:\n{p['problem_text']}"
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
    self_correct = 0
    self_total = 0
    self_results = []

    for p, output in zip(all_wrong, self_outputs):
        traces = []
        any_correct = False
        for comp in output.outputs:
            answer = extract_boxed_answer(comp.text)
            correct = check_correctness(answer, p['ground_truth'])
            traces.append({'correct': correct, 'tokens': len(comp.token_ids)})
            self_total += 1
            if correct:
                self_correct += 1
                any_correct = True
        if any_correct:
            self_solvable += 1
        self_results.append({
            'problem_id': p['problem_id'],
            'any_correct': any_correct,
        })

    print(f"\nSelf-reflective hint results:")
    print(f"  Solvable: {self_solvable}/{len(all_wrong)} ({100*self_solvable/len(all_wrong):.1f}%)")
    print(f"  Correct traces: {self_correct}/{self_total} ({100*self_correct/self_total:.1f}%)")

    # ============================================================
    # Summary
    # ============================================================
    print(f"\n{'='*60}")
    print(f"  SUMMARY: Expanding the Training Zone")
    print(f"{'='*60}\n")

    print(f"  {'Condition':<30} | {'Solvable':>10} | {'Pct':>6}")
    print(f"  {'-'*30}-+-{'-'*10}-+-{'-'*6}")
    print(f"  {'No hint (baseline)':<30} | {'0/' + str(len(all_wrong)):>10} | {'0.0%':>6}")
    print(f"  {'Strong GT hint (key step)':<30} | {str(solvable)+'/'+str(len(all_wrong)):>10} | {100*solvable/len(all_wrong):>5.1f}%")
    print(f"  {'Self-reflective hint':<30} | {str(self_solvable)+'/'+str(len(all_wrong)):>10} | {100*self_solvable/len(all_wrong):>5.1f}%")

    print(f"\n  With fixed normalization:")
    print(f"    Original all-wrong: 55 → True all-wrong: {len(all_wrong)}")
    print(f"    Recovered by fix: {55 - len(all_wrong)} (answer format bugs)")

    # Save
    results = {
        'n_true_all_wrong': len(all_wrong),
        'n_format_bugs': 55 - len(all_wrong),
        'strong_gt_hint': {
            'solvable': solvable, 'total': len(all_wrong),
            'correct_traces': correct_traces, 'total_traces': total_traces,
            'problems': problem_results,
        },
        'self_reflective_hint': {
            'solvable': self_solvable, 'total': len(all_wrong),
            'correct_traces': self_correct, 'total_traces': self_total,
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {args.output}")


if __name__ == '__main__':
    main()
