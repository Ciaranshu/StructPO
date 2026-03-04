"""
Structural Annotator

Annotates rollout traces with structural metrics using the fast parser.
This is the bridge between rollout generation and preference pair construction.

For each trace, computes:
  - DSR (Dead Step Ratio)
  - Typed DSR (per step type)
  - Verification density
  - Live verification rate (verifications that ARE referenced)
  - Exploration efficiency
"""

import json
from pathlib import Path
from dataclasses import dataclass

from ..structural_parser.classifier import classify_trace, compute_verification_density
from ..structural_parser.dag_builder import build_dag
from ..structural_parser.reachability import backward_reachability, compute_dsr, compute_typed_dsr


@dataclass
class AnnotatedTrace:
    """A reasoning trace with structural annotations."""
    problem_id: str
    trace_id: int
    solution: str
    answer: str
    is_correct: bool
    # Structural metrics
    num_steps: int = 0
    dsr: float = 0.0
    typed_dsr: dict = None
    verification_density: float = 0.0
    live_verification_rate: float = 0.0
    trace_length: int = 0
    
    def to_dict(self) -> dict:
        return {
            'problem_id': self.problem_id,
            'trace_id': self.trace_id,
            'is_correct': self.is_correct,
            'num_steps': self.num_steps,
            'dsr': self.dsr,
            'typed_dsr': self.typed_dsr or {},
            'verification_density': self.verification_density,
            'live_verification_rate': self.live_verification_rate,
            'trace_length': self.trace_length,
        }


def annotate_trace(
    problem_id: str,
    trace_id: int,
    solution: str,
    answer: str,
    is_correct: bool,
) -> AnnotatedTrace:
    """Annotate a single trace with structural metrics.
    
    Args:
        problem_id: Problem identifier.
        trace_id: Index of this trace among rollouts for this problem.
        solution: Full reasoning trace text.
        answer: Extracted answer.
        is_correct: Whether the answer matches ground truth.
        
    Returns:
        AnnotatedTrace with all structural metrics computed.
    """
    # Classify steps
    steps = classify_trace(solution)
    
    if not steps:
        return AnnotatedTrace(
            problem_id=problem_id,
            trace_id=trace_id,
            solution=solution,
            answer=answer,
            is_correct=is_correct,
        )
    
    # Build DAG and run reachability
    dag = build_dag(steps)
    dag = backward_reachability(dag)
    
    # Compute metrics
    dsr = compute_dsr(dag)
    typed_dsr = compute_typed_dsr(dag, steps)
    verif_density = compute_verification_density(steps)
    
    # Live verification rate: fraction of verification steps that are live
    verif_steps = [s for s in steps if s.step_type == 'verification']
    if verif_steps:
        live_verif = sum(1 for s in verif_steps if dag.nodes[s.id].is_live)
        live_verif_rate = live_verif / len(verif_steps)
    else:
        live_verif_rate = 1.0  # No verification = no dead verification
    
    return AnnotatedTrace(
        problem_id=problem_id,
        trace_id=trace_id,
        solution=solution,
        answer=answer,
        is_correct=is_correct,
        num_steps=len(steps),
        dsr=dsr,
        typed_dsr={k: v for k, v in typed_dsr.items()},
        verification_density=verif_density,
        live_verification_rate=live_verif_rate,
        trace_length=len(solution),
    )


def annotate_rollouts(rollouts_path: str | Path, output_path: str | Path = None) -> list[AnnotatedTrace]:
    """Annotate all rollouts in a JSON file.
    
    Expected format of rollouts JSON:
    [
        {
            "problem_id": "...",
            "traces": [
                {
                    "solution": "...",
                    "answer": "...",
                    "is_correct": true/false
                },
                ...
            ]
        },
        ...
    ]
    
    Returns:
        List of AnnotatedTrace objects.
    """
    rollouts_path = Path(rollouts_path)
    data = json.loads(rollouts_path.read_text())
    
    all_annotations = []
    
    for problem in data:
        pid = problem['problem_id']
        for tid, trace in enumerate(problem['traces']):
            ann = annotate_trace(
                problem_id=pid,
                trace_id=tid,
                solution=trace['solution'],
                answer=trace.get('answer', ''),
                is_correct=trace.get('is_correct', False),
            )
            all_annotations.append(ann)
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_data = [a.to_dict() for a in all_annotations]
        output_path.write_text(json.dumps(output_data, indent=2))
        print(f"Saved {len(all_annotations)} annotations to {output_path}")
    
    return all_annotations
