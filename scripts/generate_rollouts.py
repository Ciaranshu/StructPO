"""
Generate Rollouts from Stage 1 Model

Uses vLLM to generate multiple rollouts per problem from the Stage 1
DSE-SFT checkpoint. These rollouts are then annotated with structural
metrics and used to build preference pairs for Stage 2 DPO.

Usage:
    python scripts/generate_rollouts.py \
        --model models/structpo-qwen3-4b-stage1 \
        --dataset data/limo_cleaned/limo_original.json \
        --num-rollouts 8 \
        --output data/rollouts/stage1_rollouts.json \
        --temperature 0.7

Requirements:
    pip install vllm>=0.11.0
"""

import re
import json
import argparse
from pathlib import Path


def extract_boxed_answer(text: str) -> str:
    """Extract the last \\boxed{...} answer from a solution."""
    # Find all \boxed{...} patterns (handle nested braces)
    matches = []
    i = 0
    while i < len(text):
        idx = text.find(r'\boxed{', i)
        if idx == -1:
            break
        # Find matching closing brace
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


def normalize_answer(ans: str) -> str:
    """Normalize a math answer for comparison."""
    ans = ans.strip()
    # Remove common LaTeX wrappers
    for prefix in [r'\text{', r'\mathrm{', r'\mathbf{']:
        if ans.startswith(prefix) and ans.endswith('}'):
            ans = ans[len(prefix):-1]
    # Normalize fractions
    ans = ans.replace(r'\dfrac', r'\frac')
    # Remove \left and \right (cosmetic LaTeX, no mathematical meaning)
    ans = ans.replace(r'\left', '').replace(r'\right', '')
    # Remove spaces
    ans = re.sub(r'\s+', '', ans)
    # Remove trailing period
    ans = ans.rstrip('.')
    return ans


def check_correctness(predicted: str, ground_truth: str) -> bool:
    """Check if predicted answer matches ground truth."""
    pred_norm = normalize_answer(predicted)
    gt_norm = normalize_answer(ground_truth)
    if not pred_norm:
        return False
    return pred_norm == gt_norm


def load_problems(dataset_path: str) -> list[dict]:
    """Load problems from LIMO-format dataset.
    
    Extracts the boxed answer from the assistant response as ground truth.
    """
    data = json.loads(Path(dataset_path).read_text())
    problems = []
    for item in data:
        conversations = item.get('conversations', [])
        if len(conversations) >= 2:
            full_response = conversations[1]['value']
            gt_answer = extract_boxed_answer(full_response)
            problems.append({
                'problem_id': f"prob_{len(problems)}",
                'problem_text': conversations[0]['value'],
                'ground_truth': gt_answer,  # Just the boxed answer, not full trace
            })
    return problems


def generate_rollouts(
    model_path: str,
    problems: list[dict],
    num_rollouts: int = 8,
    temperature: float = 0.7,
    max_tokens: int = 16384,
    tensor_parallel_size: int = 1,
) -> list[dict]:
    """Generate multiple rollouts per problem using vLLM.
    
    Returns list of dicts with problem_id, traces (list of solutions).
    """
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        raise ImportError("vLLM required. Install: pip install vllm>=0.11.0")
    from transformers import AutoTokenizer
    
    # Initialize tokenizer for chat template
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Initialize model
    # disable_custom_all_reduce: workaround for GH200 CUDA graph crash
    # (custom_all_reduce.cuh:455 'invalid argument' during warmup with TP>1)
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        max_model_len=32768,
        tensor_parallel_size=tensor_parallel_size,
        dtype="bfloat16",
        disable_custom_all_reduce=True,
    )
    
    # Stop tokens: end of assistant turn
    stop_token_ids = [
        tokenizer.convert_tokens_to_ids('<|im_end|>'),
        tokenizer.eos_token_id,
    ]
    # Filter out None values
    stop_token_ids = [t for t in stop_token_ids if t is not None]
    
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        n=num_rollouts,
        stop_token_ids=stop_token_ids,
    )
    
    # Format prompts using tokenizer's chat template
    prompts = []
    for p in problems:
        messages = [{'role': 'user', 'content': p['problem_text']}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=True,  # Allow model to think/reason
        )
        prompts.append(prompt)
    
    print(f"Generating {num_rollouts} rollouts for {len(problems)} problems...")
    print(f"Model: {model_path}")
    print(f"Temperature: {temperature}, Max tokens: {max_tokens}")
    print(f"Stop token IDs: {stop_token_ids}")
    print(f"Sample prompt preview: {prompts[0][:200]}...")
    
    outputs = llm.generate(prompts, sampling_params)
    
    results = []
    total_correct = 0
    total_traces = 0
    for problem, output in zip(problems, outputs):
        traces = []
        for completion in output.outputs:
            answer = extract_boxed_answer(completion.text)
            is_correct = check_correctness(answer, problem['ground_truth'])
            total_traces += 1
            if is_correct:
                total_correct += 1
            traces.append({
                'solution': completion.text,
                'answer': answer,
                'is_correct': is_correct,
                'num_tokens': len(completion.token_ids),
            })
        results.append({
            'problem_id': problem['problem_id'],
            'problem_text': problem['problem_text'],
            'ground_truth': problem['ground_truth'],
            'traces': traces,
        })
    
    # Summary statistics
    print(f"\n=== Rollout Generation Summary ===")
    print(f"Problems: {len(results)}")
    print(f"Total traces: {total_traces}")
    print(f"Correct: {total_correct}/{total_traces} ({100*total_correct/max(total_traces,1):.1f}%)")
    problems_with_correct = sum(1 for r in results if any(t['is_correct'] for t in r['traces']))
    print(f"Problems with ≥1 correct: {problems_with_correct}/{len(results)} ({100*problems_with_correct/max(len(results),1):.1f}%)")
    problems_with_pair = sum(1 for r in results 
        if any(t['is_correct'] for t in r['traces']) and len([t for t in r['traces'] if t['is_correct']]) >= 2)
    print(f"Problems with ≥2 correct (for DPO pairs): {problems_with_pair}/{len(results)}")
    avg_tokens = sum(t['num_tokens'] for r in results for t in r['traces']) / max(total_traces, 1)
    print(f"Avg tokens per trace: {avg_tokens:.0f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Generate rollouts from Stage 1 model')
    parser.add_argument('--model', required=True, help='Path to Stage 1 model')
    parser.add_argument('--dataset', required=True, help='Path to problem dataset (LIMO format)')
    parser.add_argument('--num-rollouts', type=int, default=8, help='Rollouts per problem')
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--max-tokens', type=int, default=16384)
    parser.add_argument('--output', required=True, help='Output path for rollouts JSON')
    parser.add_argument('--subset', type=int, default=0,
                        help='Only process first N problems (0 = all, for quick testing)')
    parser.add_argument('--offset', type=int, default=0,
                        help='Skip first N problems (for sharded generation)')
    parser.add_argument('--tensor-parallel', type=int, default=1,
                        help='Tensor parallel size for vLLM (number of GPUs)')
    args = parser.parse_args()

    # Load problems
    problems = load_problems(args.dataset)
    if args.offset > 0:
        problems = problems[args.offset:]
        print(f"Offset: skipping first {args.offset} problems")
    if args.subset > 0:
        problems = problems[:args.subset]
        print(f"Loaded {len(problems)} problems (subset of {args.subset}, offset {args.offset})")
    else:
        print(f"Loaded {len(problems)} problems")
    
    # Generate rollouts
    results = generate_rollouts(
        model_path=args.model,
        problems=problems,
        num_rollouts=args.num_rollouts,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        tensor_parallel_size=args.tensor_parallel,
    )
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"Saved {len(results)} problems × {args.num_rollouts} rollouts to {output_path}")


if __name__ == '__main__':
    main()
