"""
L2 Annotation: Use DeepSeek-chat to parse rollout traces into R-IR DAGs.

Runs concurrent API calls to annotate traces with:
- Semantic step segmentation (not paragraph-level)
- Explicit logical dependency edges
- Step type classification

Then computes DSR and quality reward from the LLM-parsed DAG.

Usage:
    PYTHONPATH=src:$PYTHONPATH python scripts/analysis/l2_annotate_rollouts.py \
        --rollouts data/rollouts/gpqa_4b_rollouts.json \
        --output data/l2_annotations/gpqa_4b_l2.json \
        --max-traces 0 \
        --workers 8
"""

import json
import sys
import os
import time
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from structpo.structural_parser.classifier import ClassifiedStep
from structpo.structural_parser.dag_builder import DAGNode, ReasoningDAG, _extract_symbols
from structpo.structural_parser.reachability import backward_reachability, compute_dsr
from structpo.structural_parser.quality import compute_quality_reward

SYSTEM_PROMPT = """You are a reasoning trace parser. Parse the trace into structured steps.

Output JSON: {"steps": [{"id": 1, "content": "brief summary of step", "type": "assumption|derivation|computation|verification|conclusion", "inputs": [step IDs]}]}

Step types:
- assumption: Given fact or premise
- derivation: Logical inference from other steps
- computation: Mathematical calculation
- verification: Self-check or validation
- conclusion: Final answer

Rules:
1. "inputs" = ONLY step IDs whose output is DIRECTLY USED logically. Not proximity.
2. Verification that checks step X does NOT become input to later steps unless explicitly depended on.
3. Segment into meaningful logical units (5-30 steps typical).
4. "content" should be a brief summary, not the full text.
5. At least one conclusion step.

Output ONLY valid JSON."""


def parse_one_trace(args_tuple):
    """Parse a single trace with DeepSeek-chat. Thread-safe."""
    idx, solution, api_key = args_tuple
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    # Truncate very long traces
    solution_trunc = solution[:8000]

    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Parse:\n{solution_trunc}\n\nJSON:"},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            timeout=60,
        )
        parsed = json.loads(resp.choices[0].message.content)
    except Exception as e:
        return idx, {"error": str(e), "dsr": None, "quality_reward": None}

    steps_data = parsed.get("steps", [])
    if not steps_data:
        return idx, {"error": "no steps", "dsr": None, "quality_reward": None}

    # Build DAG from LLM dependencies
    dag = ReasoningDAG()
    id_set = {s.get("id", i) for i, s in enumerate(steps_data)}

    for i, s in enumerate(steps_data):
        sid = s.get("id", i)
        inputs = [inp for inp in s.get("inputs", []) if inp in id_set and inp != sid]
        dag.nodes[sid] = DAGNode(step_id=sid, step_type=s.get("type", "derivation"), inputs=inputs)
        if s.get("type") == "conclusion":
            dag.conclusion_ids.append(sid)

    if not dag.conclusion_ids and dag.nodes:
        dag.conclusion_ids = [list(dag.nodes.keys())[-1]]

    for sid, node in dag.nodes.items():
        for inp_id in node.inputs:
            if inp_id in dag.nodes:
                dag.nodes[inp_id].outputs.append(sid)

    dag = backward_reachability(dag)
    dsr = compute_dsr(dag)

    classified_steps = [
        ClassifiedStep(id=s.get("id", i), step_type=s.get("type", "derivation"),
                       content=s.get("content", ""), char_length=len(s.get("content", "")))
        for i, s in enumerate(steps_data)
    ]
    step_symbols = {s.id: _extract_symbols(s.content) for s in classified_steps}
    qr, qcounts = compute_quality_reward(classified_steps, dag, step_symbols)

    return idx, {
        "num_steps": len(steps_data),
        "dsr": dsr,
        "quality_reward": qr,
        "quality_counts": qcounts,
        "step_types": [s.get("type", "derivation") for s in steps_data],
        "is_live": [dag.nodes[s.get("id", i)].is_live for i, s in enumerate(steps_data)],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-traces", type=int, default=0, help="0 = all")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv(Path.home() / ".env")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY not set"); return

    # Load rollouts
    data = json.loads(Path(args.rollouts).read_text())

    # Flatten traces
    all_traces = []
    for p in data:
        for tid, t in enumerate(p["traces"]):
            all_traces.append({
                "problem_id": p["problem_id"],
                "trace_id": tid,
                "solution": t["solution"],
                "is_correct": t.get("is_correct", False),
            })

    if args.max_traces > 0:
        all_traces = all_traces[:args.max_traces]

    print(f"Annotating {len(all_traces)} traces with {args.workers} workers...")
    print(f"Estimated time: {len(all_traces) * 30 / args.workers / 60:.1f} min")
    print(f"Estimated cost: ~${len(all_traces) * 0.0003:.2f}")
    print()

    # Run concurrent API calls
    tasks = [(i, t["solution"], api_key) for i, t in enumerate(all_traces)]
    results = [None] * len(all_traces)
    errors = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(parse_one_trace, task): task[0] for task in tasks}
        for i, future in enumerate(as_completed(futures)):
            idx, result = future.result()
            results[idx] = result
            if result.get("error"):
                errors += 1
            if (i + 1) % 50 == 0 or (i + 1) == len(all_traces):
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(all_traces) - i - 1) / rate if rate > 0 else 0
                print(f"  {i+1}/{len(all_traces)} done ({errors} errors), "
                      f"{elapsed:.0f}s elapsed, ETA {eta:.0f}s")

    # Merge back into problem structure
    output_data = []
    trace_idx = 0
    for p in data:
        problem_traces = []
        for tid, t in enumerate(p["traces"]):
            if trace_idx < len(results) and results[trace_idx] is not None:
                r = results[trace_idx]
                problem_traces.append({
                    "trace_id": tid,
                    "is_correct": t.get("is_correct", False),
                    "num_tokens": t.get("num_tokens", 0),
                    "l2_dsr": r.get("dsr"),
                    "l2_quality_reward": r.get("quality_reward"),
                    "l2_quality_counts": r.get("quality_counts", {}),
                    "l2_num_steps": r.get("num_steps", 0),
                    "l2_error": r.get("error"),
                })
            trace_idx += 1
            if args.max_traces > 0 and trace_idx >= args.max_traces:
                break
        if problem_traces:
            output_data.append({
                "problem_id": p["problem_id"],
                "traces": problem_traces,
            })
        if args.max_traces > 0 and trace_idx >= args.max_traces:
            break

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output_data, indent=2))

    # Summary
    total = len([r for r in results if r and r.get("dsr") is not None])
    dsrs = [r["dsr"] for r in results if r and r.get("dsr") is not None]
    qrs = [r["quality_reward"] for r in results if r and r.get("quality_reward") is not None]
    print(f"\nDone. {total} traces annotated, {errors} errors.")
    print(f"Avg DSR: {np.mean(dsrs):.4f}, Avg QR: {np.mean(qrs):.4f}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
