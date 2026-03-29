"""
L3-base Annotation: Qwen3-0.5B zero-shot parsing of rollout traces.

Same prompt as L2 (DeepSeek-chat), but using a local 0.5B model.
This tests whether a small model can approximate LLM-quality parsing.

Usage:
    PYTHONPATH=src:$PYTHONPATH python scripts/analysis/l3_annotate_rollouts.py \
        --rollouts data/rollouts/gpqa_4b_rollouts.json \
        --output data/l3_annotations/gpqa_4b_l3_base.json \
        --model Qwen/Qwen3-0.5B
"""

import json
import sys
import time
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from structpo.structural_parser.classifier import ClassifiedStep
from structpo.structural_parser.dag_builder import DAGNode, ReasoningDAG, _extract_symbols
from structpo.structural_parser.reachability import backward_reachability, compute_dsr
from structpo.structural_parser.quality import compute_quality_reward

SYSTEM_PROMPT = """You are a reasoning trace parser. Parse the trace into structured steps.

Output JSON: {"steps": [{"id": 1, "content": "brief summary", "type": "assumption|derivation|computation|verification|conclusion", "inputs": [step IDs]}]}

Rules:
1. inputs = only DIRECT logical dependencies
2. Verification checking step X does NOT become input to later steps
3. Segment into 5-30 meaningful logical units
4. At least one conclusion step
Output ONLY valid JSON."""


def parse_trace_l3(solution, tokenizer, model):
    """Parse one trace with Qwen3-0.5B zero-shot."""
    # Truncate long traces
    solution_trunc = solution[:4000]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Parse:\n{solution_trunc}\n\nJSON:"},
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=6144)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=2048,
            do_sample=False, pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    # Extract JSON
    try:
        start = response.find("{")
        end = response.rfind("}") + 1
        if start >= 0 and end > start:
            parsed = json.loads(response[start:end])
        else:
            return None
    except json.JSONDecodeError:
        return None

    steps_data = parsed.get("steps", [])
    if not steps_data:
        return None

    # Build DAG
    dag = ReasoningDAG()
    id_map = {}
    for i, s in enumerate(steps_data):
        sid = s.get("id", i + 1)
        id_map[i] = sid
        dag.nodes[sid] = DAGNode(step_id=sid, step_type=s.get("type", "derivation"), inputs=[])
        if s.get("type") == "conclusion":
            dag.conclusion_ids.append(sid)

    valid_ids = set(dag.nodes.keys())
    for i, s in enumerate(steps_data):
        sid = s.get("id", i + 1)
        inputs = [inp for inp in s.get("inputs", []) if inp in valid_ids and inp != sid]
        dag.nodes[sid].inputs = inputs

    if not dag.conclusion_ids and dag.nodes:
        dag.conclusion_ids = [list(dag.nodes.keys())[-1]]

    for sid, node in dag.nodes.items():
        for inp_id in node.inputs:
            if inp_id in dag.nodes:
                dag.nodes[inp_id].outputs.append(sid)

    dag = backward_reachability(dag)
    dsr = compute_dsr(dag)

    classified_steps = [
        ClassifiedStep(id=s.get("id", i + 1), step_type=s.get("type", "derivation"),
                       content=s.get("content", ""), char_length=len(s.get("content", "")))
        for i, s in enumerate(steps_data)
    ]
    step_symbols = {s.id: _extract_symbols(s.content) for s in classified_steps}
    qr, qcounts = compute_quality_reward(classified_steps, dag, step_symbols)

    return {
        "num_steps": len(steps_data),
        "dsr": dsr,
        "quality_reward": qr,
        "quality_counts": qcounts,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-0.5B")
    parser.add_argument("--max-traces", type=int, default=0)
    args = parser.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto",
    )
    print(f"Model loaded on {model.device}")

    data = json.loads(Path(args.rollouts).read_text())

    # Flatten
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

    print(f"Parsing {len(all_traces)} traces...")
    t0 = time.time()
    results = []
    errors = 0

    for i, trace in enumerate(all_traces):
        r = parse_trace_l3(trace["solution"], tokenizer, model)
        if r is None:
            errors += 1
            results.append({
                "problem_id": trace["problem_id"],
                "trace_id": trace["trace_id"],
                "is_correct": trace["is_correct"],
                "l3_dsr": None, "l3_quality_reward": None, "l3_error": True,
            })
        else:
            results.append({
                "problem_id": trace["problem_id"],
                "trace_id": trace["trace_id"],
                "is_correct": trace["is_correct"],
                "l3_dsr": r["dsr"],
                "l3_quality_reward": r["quality_reward"],
                "l3_quality_counts": r["quality_counts"],
                "l3_num_steps": r["num_steps"],
                "l3_error": False,
            })

        if (i + 1) % 20 == 0 or (i + 1) == len(all_traces):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"  {i+1}/{len(all_traces)} ({errors} errors), "
                  f"{elapsed:.0f}s, {rate:.1f} traces/s, ETA {(len(all_traces)-i-1)/rate:.0f}s")

    # Group back by problem
    output_data = {}
    for r in results:
        pid = r["problem_id"]
        if pid not in output_data:
            output_data[pid] = {"problem_id": pid, "traces": []}
        output_data[pid]["traces"].append(r)

    output_list = list(output_data.values())
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output_list, indent=2))

    valid = [r for r in results if not r.get("l3_error")]
    print(f"\nDone. {len(valid)} parsed, {errors} errors.")
    if valid:
        dsrs = [r["l3_dsr"] for r in valid]
        print(f"Avg DSR: {np.mean(dsrs):.4f}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
