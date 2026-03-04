"""
Structural Preference Pair Builder

Constructs DPO preference pairs from annotated rollouts using
structural quality metrics (not length, not perplexity).

Core thesis: Dead steps are not uniformly bad. For hard problems,
exploration is necessary — the question is not "how to eliminate
exploration" but "how to make exploration productive." These four
pair types jointly teach a complete exploration policy:

Type 1 — Structural Efficiency:
  Preferred:  correct + low DSR (<0.15)
  Rejected:   correct + high DSR (>0.35)
  Teaches: when the solution path is clear, don't explore unnecessarily

Type 2 — Productive Exploration:
  Preferred:  correct + high live_verification_rate
  Rejected:   correct + low live_verification_rate (dead verification)
  Teaches: verification should discover new information, not confirm
  the obvious — prefer live verification over dead verification

Type 3 — Exploration Direction:
  Preferred:  correct + low DSR (directed exploration)
  Rejected:   incorrect + high DSR (undirected exploration)
  Teaches: when you must explore, maintain direction — abandon dead
  ends early rather than deepening them

Type 4 — Structural Contrastive (motif-level):
  Preferred:  trace with motif excised/replaced, or clean rollout
  Rejected:   trace containing structural anti-pattern (motif)
  Teaches: which specific patterns are toxic — dead cascades,
  verification theater, abandoned branches, circular revisits.
  Operates at motif level (local, transferable across problems).
  See contrastive_builder.py for 3 strategies (excision/replacement/contrast).
"""

import json
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from .annotator import AnnotatedTrace


@dataclass
class PreferencePair:
    """A single DPO preference pair."""
    problem_id: str
    pair_type: str  # 'efficiency', 'productive_exploration', 'direction', 'deliberation'
    chosen_solution: str
    rejected_solution: str
    chosen_dsr: float
    rejected_dsr: float
    chosen_correct: bool
    rejected_correct: bool
    metadata: dict = field(default_factory=dict)
    
    def to_sharegpt_dpo(self, problem_text: str) -> dict:
        """Convert to ShareGPT DPO format for LLaMA-Factory.
        
        Format:
        {
            "conversations": [
                {"from": "human", "value": "<problem>"},
            ],
            "chosen": {"from": "gpt", "value": "<preferred solution>"},
            "rejected": {"from": "gpt", "value": "<rejected solution>"}
        }
        """
        return {
            "conversations": [
                {"from": "human", "value": problem_text},
            ],
            "chosen": {"from": "gpt", "value": self.chosen_solution},
            "rejected": {"from": "gpt", "value": self.rejected_solution},
        }


def build_efficiency_pairs(
    traces_by_problem: dict[str, list[AnnotatedTrace]],
    dsr_low_threshold: float = 0.15,
    dsr_high_threshold: float = 0.35,
    max_pairs_per_problem: int = 3,
) -> list[PreferencePair]:
    """Type 1: Structural Efficiency pairs.
    
    Among correct traces for the same problem, prefer low-DSR over high-DSR.
    """
    pairs = []
    
    for pid, traces in traces_by_problem.items():
        correct = [t for t in traces if t.is_correct]
        if len(correct) < 2:
            continue
        
        low_dsr = [t for t in correct if t.dsr < dsr_low_threshold]
        high_dsr = [t for t in correct if t.dsr > dsr_high_threshold]
        
        if not low_dsr or not high_dsr:
            continue
        
        count = 0
        for chosen in low_dsr:
            for rejected in high_dsr:
                if count >= max_pairs_per_problem:
                    break
                pairs.append(PreferencePair(
                    problem_id=pid,
                    pair_type='efficiency',
                    chosen_solution=chosen.solution,
                    rejected_solution=rejected.solution,
                    chosen_dsr=chosen.dsr,
                    rejected_dsr=rejected.dsr,
                    chosen_correct=True,
                    rejected_correct=True,
                    metadata={
                        'chosen_trace_id': chosen.trace_id,
                        'rejected_trace_id': rejected.trace_id,
                    }
                ))
                count += 1
            if count >= max_pairs_per_problem:
                break
    
    return pairs


def build_productive_exploration_pairs(
    traces_by_problem: dict[str, list[AnnotatedTrace]],
    live_verif_high: float = 0.8,
    live_verif_low: float = 0.3,
    max_pairs_per_problem: int = 3,
) -> list[PreferencePair]:
    """Type 2: Productive Exploration pairs.
    
    Among correct traces with verification, prefer those where
    verification is mostly live (referenced by later steps) over
    those where verification is mostly dead.
    """
    pairs = []
    
    for pid, traces in traces_by_problem.items():
        correct_with_verif = [
            t for t in traces 
            if t.is_correct and t.verification_density > 0
        ]
        if len(correct_with_verif) < 2:
            continue
        
        productive = [t for t in correct_with_verif if t.live_verification_rate >= live_verif_high]
        wasteful = [t for t in correct_with_verif if t.live_verification_rate <= live_verif_low]
        
        if not productive or not wasteful:
            continue
        
        count = 0
        for chosen in productive:
            for rejected in wasteful:
                if count >= max_pairs_per_problem:
                    break
                pairs.append(PreferencePair(
                    problem_id=pid,
                    pair_type='productive_exploration',
                    chosen_solution=chosen.solution,
                    rejected_solution=rejected.solution,
                    chosen_dsr=chosen.dsr,
                    rejected_dsr=rejected.dsr,
                    chosen_correct=True,
                    rejected_correct=True,
                    metadata={
                        'chosen_live_verif_rate': chosen.live_verification_rate,
                        'rejected_live_verif_rate': rejected.live_verification_rate,
                    }
                ))
                count += 1
            if count >= max_pairs_per_problem:
                break
    
    return pairs


def build_direction_pairs(
    traces_by_problem: dict[str, list[AnnotatedTrace]],
    max_pairs_per_problem: int = 3,
) -> list[PreferencePair]:
    """Type 3: Exploration Direction pairs.
    
    Prefer correct traces over incorrect traces when both have exploration,
    but the correct one has lower DSR (exploration was productive).
    """
    pairs = []
    
    for pid, traces in traces_by_problem.items():
        correct = [t for t in traces if t.is_correct]
        incorrect = [t for t in traces if not t.is_correct]
        
        if not correct or not incorrect:
            continue
        
        # Filter to traces that have some exploration
        correct_exploring = [t for t in correct if t.dsr < 0.3]  # efficient
        incorrect_exploring = [t for t in incorrect if t.dsr > 0.3]  # wasteful
        
        if not correct_exploring or not incorrect_exploring:
            continue
        
        count = 0
        for chosen in correct_exploring:
            for rejected in incorrect_exploring:
                if count >= max_pairs_per_problem:
                    break
                pairs.append(PreferencePair(
                    problem_id=pid,
                    pair_type='direction',
                    chosen_solution=chosen.solution,
                    rejected_solution=rejected.solution,
                    chosen_dsr=chosen.dsr,
                    rejected_dsr=rejected.dsr,
                    chosen_correct=True,
                    rejected_correct=False,
                    metadata={
                        'chosen_trace_id': chosen.trace_id,
                        'rejected_trace_id': rejected.trace_id,
                    }
                ))
                count += 1
            if count >= max_pairs_per_problem:
                break
    
    return pairs


def build_all_pairs(
    annotated_traces: list[AnnotatedTrace],
    problem_texts: Optional[dict[str, str]] = None,
    output_path: Optional[str | Path] = None,
    seed: int = 42,
) -> list[PreferencePair]:
    """Build all types of structural preference pairs.
    
    Args:
        annotated_traces: List of AnnotatedTrace from the annotator.
        problem_texts: Optional dict mapping problem_id to problem text.
        output_path: If provided, save pairs in LLaMA-Factory DPO format.
        seed: Random seed for shuffling.
        
    Returns:
        List of all PreferencePair objects.
    """
    # Group traces by problem
    traces_by_problem: dict[str, list[AnnotatedTrace]] = {}
    for t in annotated_traces:
        traces_by_problem.setdefault(t.problem_id, []).append(t)
    
    # Build each type
    efficiency_pairs = build_efficiency_pairs(traces_by_problem)
    productive_pairs = build_productive_exploration_pairs(traces_by_problem)
    direction_pairs = build_direction_pairs(traces_by_problem)
    
    # Type 4: Structural Contrastive (motif-level)
    from .contrastive_builder import build_contrastive_pairs as _build_contrastive
    contrastive_pairs = _build_contrastive(annotated_traces)
    
    all_pairs = efficiency_pairs + productive_pairs + direction_pairs + contrastive_pairs
    
    # Shuffle
    random.seed(seed)
    random.shuffle(all_pairs)
    
    print(f"Built {len(all_pairs)} preference pairs:")
    print(f"  Type 1 (Efficiency):            {len(efficiency_pairs)}")
    print(f"  Type 2 (Productive Exploration): {len(productive_pairs)}")
    print(f"  Type 3 (Direction):              {len(direction_pairs)}")
    print(f"  Type 4 (Contrastive):            {len(contrastive_pairs)}")
    
    # Save in LLaMA-Factory DPO format if requested
    if output_path and problem_texts:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        dpo_data = []
        for pair in all_pairs:
            ptext = problem_texts.get(pair.problem_id, "")
            if ptext:
                dpo_data.append(pair.to_sharegpt_dpo(ptext))
        
        output_path.write_text(json.dumps(dpo_data, indent=2, ensure_ascii=False))
        print(f"Saved {len(dpo_data)} DPO pairs to {output_path}")
    
    return all_pairs


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Build structural preference pairs')
    parser.add_argument('--rollouts', required=True, help='Path to annotated rollouts JSON')
    parser.add_argument('--problems', help='Path to problem texts JSON (problem_id → text)')
    parser.add_argument('--output', required=True, help='Output path for DPO pairs')
    args = parser.parse_args()
    
    # Load annotated traces
    data = json.loads(Path(args.rollouts).read_text())
    traces = [AnnotatedTrace(**d) if isinstance(d, dict) else d for d in data]
    
    # Load problem texts if provided
    problem_texts = {}
    if args.problems:
        problem_texts = json.loads(Path(args.problems).read_text())
    
    build_all_pairs(traces, problem_texts=problem_texts, output_path=args.output)
