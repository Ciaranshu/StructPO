"""
Compare StructPRM parser levels on the same traces.

Runs L1 (regex), L2 (DeepSeek-chat), and L3-base (Qwen3-0.5B zero-shot)
on a sample of traces and compares:
- Step type classification accuracy (vs L2 as gold)
- Dependency edge accuracy
- DSR correlation
- Quality classification distribution (% generic_dead)
- Best-of-N selection agreement

Usage:
    # L1 vs L2 only (no GPU needed):
    PYTHONPATH=src:$PYTHONPATH python scripts/analysis/compare_parser_levels.py \
        --rollouts data/rollouts/math500_4b_rollouts.json \
        --n-traces 100 \
        --levels l1 l2

    # All three levels (needs GPU for L3):
    PYTHONPATH=src:$PYTHONPATH python scripts/analysis/compare_parser_levels.py \
        --rollouts data/rollouts/math500_4b_rollouts.json \
        --n-traces 100 \
        --levels l1 l2 l3 \
        --l3-model Qwen/Qwen3-0.5B
"""

import json
import sys
import argparse
import os
import time
from pathlib import Path
from collections import Counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from structpo.structural_parser.classifier import classify_trace, ClassifiedStep
from structpo.structural_parser.dag_builder import build_dag, _extract_symbols
from structpo.structural_parser.reachability import backward_reachability, compute_dsr
from structpo.structural_parser.quality import classify_dead_step, compute_quality_reward, QUALITY_PENALTIES


# ================================================================
# L1: Regex + Graph (existing)
# ================================================================

def parse_l1(solution: str) -> dict:
    """L1 parser: regex classification + graph-based reachability."""
    steps = classify_trace(solution)
    if not steps:
        return {'num_steps': 0, 'dsr': 0.0, 'quality_reward': 1.0,
                'step_types': [], 'quality_counts': {}}

    dag = build_dag(steps)
    dag = backward_reachability(dag)
    step_symbols = {s.id: _extract_symbols(s.content) for s in steps}
    dsr = compute_dsr(dag)
    qr, qcounts = compute_quality_reward(steps, dag, step_symbols)

    return {
        'num_steps': len(steps),
        'dsr': dsr,
        'quality_reward': qr,
        'step_types': [s.step_type for s in steps],
        'is_live': [dag.nodes[s.id].is_live for s in steps],
        'quality_counts': qcounts,
    }


# ================================================================
# L2: DeepSeek-chat LLM Parser
# ================================================================

PARSER_SYSTEM_PROMPT = """You are a reasoning trace parser. Given a natural language reasoning trace, parse it into a structured representation.

Output a JSON object with:
{
  "steps": [
    {
      "id": 1,
      "content": "exact text of this reasoning step",
      "type": "assumption|derivation|computation|verification|conclusion",
      "inputs": [list of step IDs this step depends on]
    }
  ]
}

Step types:
- "assumption": Given fact or premise being restated
- "derivation": Logical inference from other steps
- "computation": Mathematical calculation
- "verification": Self-check or validation
- "conclusion": Final answer

Rules:
1. "inputs" = ONLY step IDs whose output is DIRECTLY USED. Not proximity-based.
2. Verification steps that check earlier work do NOT become inputs to later steps unless explicitly depended on.
3. Segment into meaningful logical units (not sentences).
4. At least one conclusion step.

Output ONLY valid JSON."""

PARSER_FEW_SHOT = """Example input:
"We're given x = 5. x squared: 5^2 = 25. Let me verify: 5 × 5 = 25. Since 25 > 20, the answer is Yes."

Example output:
{"steps": [
  {"id": 1, "content": "We're given x = 5", "type": "assumption", "inputs": []},
  {"id": 2, "content": "x squared: 5^2 = 25", "type": "computation", "inputs": [1]},
  {"id": 3, "content": "Let me verify: 5 × 5 = 25", "type": "verification", "inputs": [2]},
  {"id": 4, "content": "Since 25 > 20, the answer is Yes", "type": "conclusion", "inputs": [2]}
]}
Note: Step 4 depends on step 2 (not 3), because the conclusion uses the computation result, not the verification."""


def parse_l2(solution: str, client) -> dict:
    """L2 parser: DeepSeek-chat with explicit dependency edges."""
    user_msg = f"Parse this reasoning trace:\n\n---BEGIN---\n{solution[:8000]}\n---END---\n\nOutput R-IR JSON:"

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": PARSER_SYSTEM_PROMPT},
                {"role": "user", "content": PARSER_FEW_SHOT},
                {"role": "user", "content": user_msg},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        parsed = json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"    L2 parse error: {e}")
        return {'num_steps': 0, 'dsr': 0.0, 'quality_reward': 1.0,
                'step_types': [], 'quality_counts': {}}

    steps_data = parsed.get("steps", [])
    if not steps_data:
        return {'num_steps': 0, 'dsr': 0.0, 'quality_reward': 1.0,
                'step_types': [], 'quality_counts': {}}

    # Build DAG from LLM-provided dependencies
    from structpo.structural_parser.dag_builder import DAGNode, ReasoningDAG

    dag = ReasoningDAG()
    id_set = {s['id'] for s in steps_data}

    for s in steps_data:
        sid = s['id']
        inputs = [i for i in s.get('inputs', []) if i in id_set and i != sid]
        dag.nodes[sid] = DAGNode(step_id=sid, step_type=s.get('type', 'derivation'), inputs=inputs)
        if s.get('type') == 'conclusion':
            dag.conclusion_ids.append(sid)

    if not dag.conclusion_ids:
        dag.conclusion_ids = [steps_data[-1]['id']]

    # Compute outputs
    for sid, node in dag.nodes.items():
        for inp_id in node.inputs:
            if inp_id in dag.nodes:
                dag.nodes[inp_id].outputs.append(sid)

    # Backward reachability
    dag = backward_reachability(dag)
    dsr = compute_dsr(dag)

    # Build ClassifiedStep objects for quality classification
    classified_steps = []
    for s in steps_data:
        classified_steps.append(ClassifiedStep(
            id=s['id'],
            step_type=s.get('type', 'derivation'),
            content=s.get('content', ''),
            char_length=len(s.get('content', '')),
        ))

    step_symbols = {s.id: _extract_symbols(s.content) for s in classified_steps}
    qr, qcounts = compute_quality_reward(classified_steps, dag, step_symbols)

    return {
        'num_steps': len(steps_data),
        'dsr': dsr,
        'quality_reward': qr,
        'step_types': [s.get('type', 'derivation') for s in steps_data],
        'is_live': [dag.nodes[s['id']].is_live for s in steps_data],
        'quality_counts': qcounts,
    }


# ================================================================
# L3-base: Qwen3-0.5B zero-shot (same prompt as L2)
# ================================================================

def init_l3_model(model_name: str):
    """Initialize Qwen3-0.5B for zero-shot parsing."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto"
    )
    return tokenizer, model


def parse_l3(solution: str, tokenizer, model) -> dict:
    """L3 parser: Qwen3-0.5B zero-shot with same prompt as L2."""
    import torch

    user_msg = f"Parse this reasoning trace:\n\n---BEGIN---\n{solution[:4000]}\n---END---\n\nOutput R-IR JSON:"

    messages = [
        {"role": "system", "content": PARSER_SYSTEM_PROMPT},
        {"role": "user", "content": PARSER_FEW_SHOT},
        {"role": "user", "content": user_msg},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=8192)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=2048, temperature=0.0,
            do_sample=False, pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    # Extract JSON from response
    try:
        # Try to find JSON in the response
        start = response.find('{')
        end = response.rfind('}') + 1
        if start >= 0 and end > start:
            parsed = json.loads(response[start:end])
        else:
            return {'num_steps': 0, 'dsr': 0.0, 'quality_reward': 1.0,
                    'step_types': [], 'quality_counts': {}, 'parse_success': False}
    except json.JSONDecodeError:
        return {'num_steps': 0, 'dsr': 0.0, 'quality_reward': 1.0,
                'step_types': [], 'quality_counts': {}, 'parse_success': False}

    # Same DAG construction as L2
    steps_data = parsed.get("steps", [])
    if not steps_data:
        return {'num_steps': 0, 'dsr': 0.0, 'quality_reward': 1.0,
                'step_types': [], 'quality_counts': {}, 'parse_success': False}

    from structpo.structural_parser.dag_builder import DAGNode, ReasoningDAG

    dag = ReasoningDAG()
    id_set = {s.get('id', i) for i, s in enumerate(steps_data)}

    for i, s in enumerate(steps_data):
        sid = s.get('id', i)
        inputs = [inp for inp in s.get('inputs', []) if inp in id_set and inp != sid]
        dag.nodes[sid] = DAGNode(step_id=sid, step_type=s.get('type', 'derivation'), inputs=inputs)
        if s.get('type') == 'conclusion':
            dag.conclusion_ids.append(sid)

    if not dag.conclusion_ids and dag.nodes:
        dag.conclusion_ids = [list(dag.nodes.keys())[-1]]

    for sid, node in dag.nodes.items():
        for inp_id in node.inputs:
            if inp_id in dag.nodes:
                dag.nodes[inp_id].outputs.append(sid)

    dag = backward_reachability(dag)
    dsr = compute_dsr(dag)

    classified_steps = [ClassifiedStep(id=s.get('id', i), step_type=s.get('type', 'derivation'),
                        content=s.get('content', ''), char_length=len(s.get('content', '')))
                        for i, s in enumerate(steps_data)]
    step_symbols = {s.id: _extract_symbols(s.content) for s in classified_steps}
    qr, qcounts = compute_quality_reward(classified_steps, dag, step_symbols)

    return {
        'num_steps': len(steps_data),
        'dsr': dsr,
        'quality_reward': qr,
        'step_types': [s.get('type', 'derivation') for s in steps_data],
        'is_live': [dag.nodes.get(s.get('id', i), DAGNode(0, '')).is_live for i, s in enumerate(steps_data)],
        'quality_counts': qcounts,
        'parse_success': True,
    }


# ================================================================
# Main comparison
# ================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rollouts', required=True)
    parser.add_argument('--n-traces', type=int, default=100)
    parser.add_argument('--levels', nargs='+', default=['l1', 'l2'])
    parser.add_argument('--l3-model', default='Qwen/Qwen3-0.5B')
    parser.add_argument('--output', default=None, help='Save results JSON')
    args = parser.parse_args()

    # Load traces
    data = json.loads(Path(args.rollouts).read_text())
    traces = []
    for p in data:
        for t in p['traces']:
            traces.append({
                'problem_id': p['problem_id'],
                'solution': t['solution'],
                'is_correct': t.get('is_correct', False),
            })

    # Sample
    np.random.seed(42)
    indices = np.random.choice(len(traces), min(args.n_traces, len(traces)), replace=False)
    sampled = [traces[i] for i in indices]
    print(f"Sampled {len(sampled)} traces from {len(traces)} total\n")

    # Initialize parsers
    client = None
    if 'l2' in args.levels:
        from openai import OpenAI
        from dotenv import load_dotenv
        load_dotenv(Path.home() / ".env")
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            print("ERROR: DEEPSEEK_API_KEY not found. Skipping L2.")
            args.levels = [l for l in args.levels if l != 'l2']
        else:
            client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
            print("L2: DeepSeek-chat client initialized")

    l3_tokenizer, l3_model = None, None
    if 'l3' in args.levels:
        print(f"L3: Loading {args.l3_model}...")
        l3_tokenizer, l3_model = init_l3_model(args.l3_model)
        print("L3: Model loaded")

    # Parse all traces
    results = {level: [] for level in args.levels}

    for i, trace in enumerate(sampled):
        if (i + 1) % 10 == 0:
            print(f"  Parsing trace {i+1}/{len(sampled)}...")

        for level in args.levels:
            t0 = time.time()
            if level == 'l1':
                r = parse_l1(trace['solution'])
            elif level == 'l2':
                r = parse_l2(trace['solution'], client)
            elif level == 'l3':
                r = parse_l3(trace['solution'], l3_tokenizer, l3_model)
            elapsed = time.time() - t0
            r['time_ms'] = elapsed * 1000
            r['is_correct'] = trace['is_correct']
            results[level].append(r)

    # ================================================================
    # Compare results
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  PARSER LEVEL COMPARISON ({len(sampled)} traces)")
    print(f"{'='*70}\n")

    # Basic stats
    print(f"{'Metric':<30} |", end="")
    for level in args.levels:
        print(f" {level.upper():>10} |", end="")
    print()
    print(f"{'-'*30}-+" + "-----------+|" * len(args.levels))

    # Avg DSR
    print(f"{'Avg DSR':<30} |", end="")
    for level in args.levels:
        dsrs = [r['dsr'] for r in results[level]]
        print(f" {np.mean(dsrs):>10.4f} |", end="")
    print()

    # Avg Quality Reward
    print(f"{'Avg Quality Reward':<30} |", end="")
    for level in args.levels:
        qrs = [r['quality_reward'] for r in results[level]]
        print(f" {np.mean(qrs):>10.4f} |", end="")
    print()

    # Avg steps
    print(f"{'Avg num_steps':<30} |", end="")
    for level in args.levels:
        ns = [r['num_steps'] for r in results[level]]
        print(f" {np.mean(ns):>10.1f} |", end="")
    print()

    # Parse time
    print(f"{'Avg parse time (ms)':<30} |", end="")
    for level in args.levels:
        ts = [r['time_ms'] for r in results[level]]
        print(f" {np.mean(ts):>10.1f} |", end="")
    print()

    # Quality distribution
    print(f"\n{'='*70}")
    print(f"  QUALITY CLASSIFICATION DISTRIBUTION")
    print(f"{'='*70}\n")

    for level in args.levels:
        total_counts = Counter()
        for r in results[level]:
            for k, v in r.get('quality_counts', {}).items():
                total_counts[k] += v
        total = sum(total_counts.values())
        print(f"  {level.upper()}:")
        if total == 0:
            print(f"    No dead steps found")
        else:
            for qt in sorted(total_counts.keys(), key=lambda k: QUALITY_PENALTIES.get(k, 0), reverse=True):
                pct = 100 * total_counts[qt] / total
                print(f"    {qt:<25} {total_counts[qt]:>5} ({pct:>5.1f}%)")
            generic_pct = 100 * total_counts.get('generic_dead', 0) / total if total else 0
            print(f"    --- generic_dead: {generic_pct:.1f}% (lower = better classification)")
        print()

    # DSR correlation between levels
    if len(args.levels) >= 2:
        print(f"{'='*70}")
        print(f"  DSR CORRELATION BETWEEN LEVELS")
        print(f"{'='*70}\n")

        for i, l1 in enumerate(args.levels):
            for l2 in args.levels[i+1:]:
                dsrs1 = [r['dsr'] for r in results[l1]]
                dsrs2 = [r['dsr'] for r in results[l2]]
                if len(dsrs1) > 1 and np.std(dsrs1) > 0 and np.std(dsrs2) > 0:
                    corr = np.corrcoef(dsrs1, dsrs2)[0, 1]
                    print(f"  {l1.upper()} vs {l2.upper()}: r = {corr:.4f}")

    # L3 parse success rate
    if 'l3' in args.levels:
        success = sum(1 for r in results['l3'] if r.get('parse_success', True))
        print(f"\n  L3 parse success rate: {success}/{len(results['l3'])} ({100*success/len(results['l3']):.1f}%)")

    # Save results
    if args.output:
        save_data = {}
        for level in args.levels:
            save_data[level] = [{k: v for k, v in r.items()
                                 if k not in ('step_types', 'is_live')}
                                for r in results[level]]
        Path(args.output).write_text(json.dumps(save_data, indent=2))
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
