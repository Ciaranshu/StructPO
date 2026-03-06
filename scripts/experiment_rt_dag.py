#!/usr/bin/env python3
"""
Experiment: Real-Time DAG Feedback (方案4) — Quick Validation

Two phases:
  Phase A (offline, no GPU): Analyze existing S1 baseline traces to quantify
    intervention opportunities — how many motifs, where they occur, how much
    waste could be avoided.

  Phase B (Slurm job, GPU): Generate new traces with structural hint prompts
    vs baseline on MATH-500, compare DSR + accuracy.

Usage:
  # Phase A: offline analysis (run locally)
  python scripts/experiment_rt_dag.py --phase A \
      --baseline eval_results/4b_dse_sft.json \
      --output eval_results/rt_dag_phase_a.json

  # Phase B: guided generation (Slurm job with GPU)
  python scripts/experiment_rt_dag.py --phase B \
      --model models/structpo-qwen3-4b-stage1 \
      --output eval_results/rt_dag_phase_b.json \
      --subset 100
"""

import json
import sys
import time
import argparse
import re
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from structpo.structural_parser.classifier import classify_trace, segment_trace, ClassifiedStep
from structpo.structural_parser.dag_builder import build_dag, ReasoningDAG
from structpo.structural_parser.reachability import backward_reachability, compute_dsr
from structpo.structural_parser.motif import extract_motifs, StructuralMotif, motif_summary


# ── Phase A: Offline Intervention Opportunity Analysis ─────────────

def extract_reasoning_body(solution: str) -> str:
    """Strip <think></think> wrapper if present."""
    # Qwen3 thinking format: <think>\n\n</think>\n\n[actual reasoning]
    if '</think>' in solution:
        idx = solution.index('</think>')
        return solution[idx + len('</think>'):].strip()
    return solution


def incremental_dag_analysis(solution: str) -> dict:
    """Simulate incremental DAG analysis paragraph by paragraph.

    For each paragraph boundary, run DAG analysis on the trace so far,
    detect motifs, and record where interventions would trigger.

    Returns analysis dict with intervention points, motif timeline, etc.
    """
    body = extract_reasoning_body(solution)
    paragraphs = segment_trace(body)

    if not paragraphs:
        return {'num_paragraphs': 0, 'interventions': [], 'final_motifs': []}

    interventions = []
    motif_timeline = []
    accumulated_text = ""

    for i, para in enumerate(paragraphs):
        accumulated_text += ("\n\n" if accumulated_text else "") + para

        # Run full pipeline on accumulated text
        steps = classify_trace(accumulated_text)
        if not steps:
            continue

        dag = build_dag(steps)
        dag = backward_reachability(dag)
        motifs = extract_motifs(steps, dag)

        # Check: are there motifs involving the latest step?
        latest_step_id = steps[-1].id
        recent_motifs = [m for m in motifs if latest_step_id in m.step_ids]

        # Also check for growing cascades (motifs whose last step is recent)
        active_cascades = [m for m in motifs
                          if m.motif_type == 'dead_cascade'
                          and m.step_ids[-1] >= latest_step_id - 1
                          and len(m.step_ids) >= 3]

        active_theater = [m for m in motifs
                         if m.motif_type == 'verification_theater'
                         and latest_step_id in m.step_ids]

        # Would we intervene?
        should_intervene = False
        intervention_type = None
        intervention_reason = ""

        if active_cascades:
            should_intervene = True
            intervention_type = 'redirect_cascade'
            cascade = active_cascades[0]
            intervention_reason = f"dead_cascade of {len(cascade.step_ids)} steps detected"

        elif active_theater:
            should_intervene = True
            intervention_type = 'skip_theater'
            intervention_reason = "verification_theater on dead work"

        elif recent_motifs:
            for m in recent_motifs:
                if m.motif_type == 'abandoned_branch' and len(m.step_ids) >= 3:
                    should_intervene = True
                    intervention_type = 'redirect_branch'
                    intervention_reason = f"abandoned_branch of {len(m.step_ids)} steps"
                    break

        if should_intervene:
            interventions.append({
                'paragraph_idx': i,
                'step_id': latest_step_id,
                'type': intervention_type,
                'reason': intervention_reason,
                'total_steps_so_far': len(steps),
                'total_dead_so_far': sum(1 for s in steps if not dag.nodes[s.id].is_live),
            })

        motif_timeline.append({
            'paragraph_idx': i,
            'num_motifs': len(motifs),
            'dsr': compute_dsr(dag),
            'num_steps': len(steps),
        })

    # Final analysis
    final_steps = classify_trace(body)
    final_dag = build_dag(final_steps)
    final_dag = backward_reachability(final_dag)
    final_motifs = extract_motifs(final_steps, final_dag)
    final_dsr = compute_dsr(final_dag)

    # Estimate waste: how many tokens are in dead steps?
    dead_chars = sum(s.char_length for s in final_steps if not final_dag.nodes[s.id].is_live)
    total_chars = sum(s.char_length for s in final_steps)

    return {
        'num_paragraphs': len(paragraphs),
        'num_steps': len(final_steps),
        'final_dsr': final_dsr,
        'final_num_motifs': len(final_motifs),
        'final_motif_summary': motif_summary(final_motifs) if final_motifs else {},
        'num_interventions': len(interventions),
        'interventions': interventions,
        'first_intervention_step': interventions[0]['step_id'] if interventions else -1,
        'first_intervention_pct': (interventions[0]['step_id'] / len(final_steps)
                                   if interventions and final_steps else -1),
        'dead_char_ratio': dead_chars / max(total_chars, 1),
        'potential_token_saving_pct': dead_chars / max(total_chars, 1) * 100,
    }


def run_phase_a(baseline_path: str, output_path: str):
    """Phase A: Analyze existing traces for intervention opportunities."""
    print("=" * 60)
    print("Phase A: Offline Intervention Opportunity Analysis")
    print("=" * 60)

    data = json.load(open(baseline_path))
    results = data['results']
    print(f"Loaded {len(results)} traces from {baseline_path}")

    # Only analyze MATH problems (not GPQA)
    math_results = [r for r in results if r.get('benchmark', '') == 'math500']
    print(f"MATH-500 traces: {len(math_results)}")

    analyses = []
    t0 = time.time()

    for i, r in enumerate(math_results):
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  Analyzed {i+1}/{len(math_results)} "
                  f"({elapsed:.1f}s, {elapsed/(i+1)*1000:.0f}ms/trace)")

        analysis = incremental_dag_analysis(r['solution'])
        analysis['id'] = r['id']
        analysis['is_correct'] = r['is_correct']
        analysis['num_tokens'] = r['num_tokens']
        analysis['level'] = r.get('level', 0)
        analyses.append(analysis)

    elapsed = time.time() - t0
    print(f"\nAnalysis complete: {len(analyses)} traces in {elapsed:.1f}s "
          f"({elapsed/len(analyses)*1000:.0f}ms/trace)")

    # ── Aggregate Statistics ──
    print("\n" + "=" * 60)
    print("RESULTS: Intervention Opportunity Analysis")
    print("=" * 60)

    # How many traces would trigger interventions?
    traces_with_interventions = [a for a in analyses if a['num_interventions'] > 0]
    print(f"\nTraces with ≥1 intervention trigger: "
          f"{len(traces_with_interventions)}/{len(analyses)} "
          f"({100*len(traces_with_interventions)/len(analyses):.1f}%)")

    # Average interventions per trace
    avg_interventions = sum(a['num_interventions'] for a in analyses) / len(analyses)
    print(f"Avg interventions per trace: {avg_interventions:.1f}")

    # Where do interventions trigger? (as % of trace)
    first_pcts = [a['first_intervention_pct'] for a in analyses
                  if a['first_intervention_pct'] >= 0]
    if first_pcts:
        print(f"First intervention at (avg): {sum(first_pcts)/len(first_pcts)*100:.0f}% of trace")

    # Intervention type breakdown
    all_interventions = [iv for a in analyses for iv in a['interventions']]
    type_counts = defaultdict(int)
    for iv in all_interventions:
        type_counts[iv['type']] += 1
    print(f"\nIntervention types:")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {t}: {c} ({100*c/max(len(all_interventions),1):.0f}%)")

    # Potential savings
    avg_dead_pct = sum(a['potential_token_saving_pct'] for a in analyses) / len(analyses)
    print(f"\nAvg dead char ratio: {avg_dead_pct:.1f}%")

    # DSR breakdown for traces WITH vs WITHOUT intervention opportunities
    with_iv = [a for a in analyses if a['num_interventions'] > 0]
    without_iv = [a for a in analyses if a['num_interventions'] == 0]

    if with_iv:
        avg_dsr_with = sum(a['final_dsr'] for a in with_iv) / len(with_iv)
        avg_correct_with = sum(1 for a in with_iv if a['is_correct']) / len(with_iv)
        print(f"\nTraces WITH intervention opportunities ({len(with_iv)}):")
        print(f"  Avg DSR: {avg_dsr_with:.1%}")
        print(f"  Accuracy: {avg_correct_with:.1%}")
    if without_iv:
        avg_dsr_without = sum(a['final_dsr'] for a in without_iv) / len(without_iv)
        avg_correct_without = sum(1 for a in without_iv if a['is_correct']) / len(without_iv)
        print(f"Traces WITHOUT intervention opportunities ({len(without_iv)}):")
        print(f"  Avg DSR: {avg_dsr_without:.1%}")
        print(f"  Accuracy: {avg_correct_without:.1%}")

    # By level
    print(f"\nBy difficulty level:")
    by_level = defaultdict(list)
    for a in analyses:
        by_level[a['level']].append(a)
    for lvl in sorted(by_level):
        group = by_level[lvl]
        n_iv = sum(1 for a in group if a['num_interventions'] > 0)
        avg_dsr = sum(a['final_dsr'] for a in group) / len(group)
        print(f"  L{lvl}: {n_iv}/{len(group)} traces "
              f"({100*n_iv/len(group):.0f}%) with interventions, "
              f"avg DSR {avg_dsr:.1%}")

    # Motif type summary across all traces
    all_motif_types = defaultdict(int)
    for a in analyses:
        ms = a.get('final_motif_summary', {}).get('by_type', {})
        for mt, count in ms.items():
            all_motif_types[mt] += count
    print(f"\nMotif totals across all traces:")
    for mt, count in sorted(all_motif_types.items(), key=lambda x: -x[1]):
        print(f"  {mt}: {count}")

    # Save full analysis
    output = {
        'summary': {
            'total_traces': len(analyses),
            'traces_with_interventions': len(traces_with_interventions),
            'intervention_rate': len(traces_with_interventions) / len(analyses),
            'avg_interventions_per_trace': avg_interventions,
            'avg_dead_char_pct': avg_dead_pct,
            'intervention_type_counts': dict(type_counts),
            'motif_totals': dict(all_motif_types),
        },
        'analyses': analyses,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nSaved detailed analysis to {output_path}")


# ── Phase B: Guided Generation Experiment ──────────────────────────

# Three prompt conditions:
PROMPT_CONDITIONS = {
    'baseline': {
        'system': None,  # No system prompt — same as eval baseline
        'user_prefix': '',
    },
    'structural_hint': {
        'system': None,
        'user_prefix': (
            "Solve this problem step by step. "
            "Be efficient: avoid re-deriving results you already established, "
            "avoid verifying intermediate results that won't be used in your final answer, "
            "and if an approach leads to a dead end after 2-3 steps, abandon it and try another. "
            "Put your final answer in \\boxed{}.\n\n"
        ),
    },
    'structural_hint_v2': {
        'system': (
            "You are a precise mathematical reasoner. "
            "Focus on productive reasoning steps that directly contribute to the solution. "
            "Avoid verification theater (checking work that won't affect the answer), "
            "dead cascades (long chains of reasoning that don't connect to the conclusion), "
            "and circular revisits (re-deriving things you already proved)."
        ),
        'user_prefix': '',
    },
}


def run_phase_b(model_path: str, output_path: str, subset: int = 100,
                tensor_parallel: int = 1):
    """Phase B: Compare baseline vs structurally-guided generation."""
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    from datasets import load_dataset

    print("=" * 60)
    print("Phase B: Guided Generation Experiment")
    print("=" * 60)

    # Load MATH-500
    ds = load_dataset('HuggingFaceH4/MATH-500', split='test', trust_remote_code=True)
    problems = []
    for item in ds:
        problems.append({
            'id': item['unique_id'],
            'problem': item['problem'],
            'answer': item['answer'],
            'level': item.get('level', 0),
        })

    if subset > 0:
        # Take a stratified sample across levels
        by_level = defaultdict(list)
        for p in problems:
            by_level[p['level']].append(p)
        sampled = []
        per_level = max(1, subset // len(by_level))
        for lvl in sorted(by_level):
            sampled.extend(by_level[lvl][:per_level])
        problems = sampled[:subset]
    print(f"Using {len(problems)} problems")

    # Initialize model
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        max_model_len=32768,
        tensor_parallel_size=tensor_parallel,
        dtype="bfloat16",
        disable_custom_all_reduce=True,
    )

    stop_token_ids = [
        tokenizer.convert_tokens_to_ids('<|im_end|>'),
        tokenizer.eos_token_id,
    ]
    stop_token_ids = [t for t in stop_token_ids if t is not None]

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=16384,
        n=1,
        stop_token_ids=stop_token_ids,
    )

    from scripts.generate_rollouts import extract_boxed_answer, check_correctness

    all_results = {}

    for cond_name, cond_config in PROMPT_CONDITIONS.items():
        print(f"\n--- Condition: {cond_name} ---")

        # Format prompts
        prompts = []
        for p in problems:
            user_content = cond_config['user_prefix'] + p['problem']
            messages = []
            if cond_config['system']:
                messages.append({'role': 'system', 'content': cond_config['system']})
            messages.append({'role': 'user', 'content': user_content})

            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=True,
            )
            prompts.append(prompt)

        print(f"  Sample prompt preview: {prompts[0][:300]}...")

        t0 = time.time()
        outputs = llm.generate(prompts, sampling_params)
        elapsed = time.time() - t0
        print(f"  Generation: {elapsed:.0f}s ({elapsed/len(problems):.1f}s/problem)")

        # Analyze results
        cond_results = []
        for problem, output in zip(problems, outputs):
            solution = output.outputs[0].text
            predicted = extract_boxed_answer(solution)
            is_correct = check_correctness(predicted, problem['answer'])

            # Structural analysis
            body = extract_reasoning_body(solution)
            steps = classify_trace(body)
            if steps:
                dag = build_dag(steps)
                dag = backward_reachability(dag)
                dsr = compute_dsr(dag)
                motifs = extract_motifs(steps, dag)
                num_motifs = len(motifs)
            else:
                dsr = 0.0
                num_motifs = 0
                steps = []

            cond_results.append({
                'id': problem['id'],
                'level': problem['level'],
                'is_correct': is_correct,
                'num_tokens': len(output.outputs[0].token_ids),
                'num_steps': len(steps),
                'dsr': dsr,
                'num_motifs': num_motifs,
            })

        # Compute metrics
        correct = sum(1 for r in cond_results if r['is_correct'])
        accuracy = correct / len(cond_results)
        avg_dsr = sum(r['dsr'] for r in cond_results) / len(cond_results)
        avg_tokens = sum(r['num_tokens'] for r in cond_results) / len(cond_results)
        avg_steps = sum(r['num_steps'] for r in cond_results) / len(cond_results)
        avg_motifs = sum(r['num_motifs'] for r in cond_results) / len(cond_results)

        print(f"  Accuracy: {accuracy:.1%} ({correct}/{len(cond_results)})")
        print(f"  Avg DSR: {avg_dsr:.1%}")
        print(f"  Avg tokens: {avg_tokens:.0f}")
        print(f"  Avg steps: {avg_steps:.1f}")
        print(f"  Avg motifs: {avg_motifs:.1f}")

        all_results[cond_name] = {
            'accuracy': accuracy,
            'avg_dsr': avg_dsr,
            'avg_tokens': avg_tokens,
            'avg_steps': avg_steps,
            'avg_motifs': avg_motifs,
            'num_problems': len(cond_results),
            'details': cond_results,
        }

    # ── Comparison Table ──
    print("\n" + "=" * 70)
    print("COMPARISON: Baseline vs Structural Guidance")
    print("=" * 70)
    print(f"{'Condition':<22} | {'Acc%':>6} | {'DSR%':>6} | {'Tokens':>7} | {'Steps':>6} | {'Motifs':>7}")
    print("-" * 70)
    for cond_name, m in all_results.items():
        print(f"{cond_name:<22} | {m['accuracy']*100:>5.1f}% | {m['avg_dsr']*100:>5.1f}% | "
              f"{m['avg_tokens']:>7.0f} | {m['avg_steps']:>5.1f} | {m['avg_motifs']:>6.1f}")
    print("=" * 70)

    # Delta analysis
    if 'baseline' in all_results:
        base = all_results['baseline']
        for cond_name, m in all_results.items():
            if cond_name == 'baseline':
                continue
            print(f"\nΔ ({cond_name} - baseline):")
            print(f"  Accuracy: {(m['accuracy']-base['accuracy'])*100:+.1f}pp")
            print(f"  DSR:      {(m['avg_dsr']-base['avg_dsr'])*100:+.1f}pp")
            print(f"  Tokens:   {m['avg_tokens']-base['avg_tokens']:+.0f} "
                  f"({(m['avg_tokens']/base['avg_tokens']-1)*100:+.1f}%)")
            print(f"  Motifs:   {m['avg_motifs']-base['avg_motifs']:+.1f}")

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps({
        'model': model_path,
        'num_problems': len(problems),
        'conditions': {k: {kk: vv for kk, vv in v.items() if kk != 'details'}
                       for k, v in all_results.items()},
        'full_results': all_results,
    }, indent=2, ensure_ascii=False))
    print(f"\nSaved to {output_path}")


# ── Main ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Real-Time DAG Feedback Experiment')
    parser.add_argument('--phase', required=True, choices=['A', 'B', 'AB'],
                        help='A=offline analysis, B=guided generation, AB=both')
    parser.add_argument('--baseline', type=str, default='eval_results/4b_dse_sft.json',
                        help='Path to baseline eval results (Phase A)')
    parser.add_argument('--model', type=str, default='models/structpo-qwen3-4b-stage1',
                        help='Path to model (Phase B)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path (Phase A: analysis JSON, Phase B: comparison JSON)')
    parser.add_argument('--subset', type=int, default=100,
                        help='Number of problems for Phase B (0=all)')
    parser.add_argument('--tensor-parallel', type=int, default=1,
                        help='Tensor parallel size for vLLM')
    args = parser.parse_args()

    if args.phase in ('A', 'AB'):
        run_phase_a(args.baseline, args.output.replace('.json', '_phase_a.json')
                    if args.phase == 'AB' else args.output)

    if args.phase in ('B', 'AB'):
        run_phase_b(args.model, args.output.replace('.json', '_phase_b.json')
                    if args.phase == 'AB' else args.output,
                    subset=args.subset,
                    tensor_parallel=args.tensor_parallel)


if __name__ == '__main__':
    main()
