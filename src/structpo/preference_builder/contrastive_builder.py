"""
Structural Contrastive DPO Pair Builder (Type 4)

Constructs DPO preference pairs at the *motif level* rather than the
trace level. Instead of comparing two separate rollouts, we take a
single trace containing wasteful motifs and construct:

  chosen  = trace with the motif surgically removed/replaced
  rejected = original trace containing the motif

This is more targeted than trace-level DPO because:
  1. The signal is LOCAL — the model learns which specific pattern is bad
  2. Everything else is identical — no confounding from other differences
  3. Motifs transfer across problems — "verification theater" looks the
     same whether the problem is algebra or geometry

Three contrastive strategies:
  A. Motif Excision     — remove the motif, keep surrounding context
  B. Motif Replacement  — replace dead motif with a shorter live alternative
                          (from a different rollout of the same problem)
  C. Motif Contrast     — pair a trace with the motif against one without
                          (from different rollouts, aligned on everything else)

Strategy A is always available (any motif can be excised). B and C require
multiple rollouts per problem and are higher quality when available.
"""

import json
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from ..structural_parser.classifier import classify_trace, ClassifiedStep
from ..structural_parser.dag_builder import build_dag
from ..structural_parser.reachability import backward_reachability
from ..structural_parser.motif import extract_motifs, StructuralMotif, motif_summary
from .annotator import AnnotatedTrace
from .pair_builder import PreferencePair


@dataclass
class MotifContext:
    """Full structural context for a single rollout trace."""
    trace: AnnotatedTrace
    steps: list[ClassifiedStep]
    motifs: list[StructuralMotif]


def analyze_trace_motifs(trace: AnnotatedTrace) -> MotifContext:
    """Run full structural analysis + motif extraction on a trace."""
    steps = classify_trace(trace.solution)
    if not steps:
        return MotifContext(trace=trace, steps=[], motifs=[])

    dag = build_dag(steps)
    dag = backward_reachability(dag)
    motifs = extract_motifs(steps, dag)

    return MotifContext(trace=trace, steps=steps, motifs=motifs)


# ─── Strategy A: Motif Excision ──────────────────────────────────────

def _excise_motif(
    steps: list[ClassifiedStep],
    motif: StructuralMotif,
) -> str:
    """Remove motif steps from trace, producing a cleaner version.

    The excised trace is the 'chosen' — same problem, same answer path,
    but without the wasteful detour.
    """
    motif_ids = set(motif.step_ids)
    kept_steps = [s for s in steps if s.id not in motif_ids]
    return "\n\n".join(s.content for s in kept_steps)


def build_excision_pairs(
    contexts: list[MotifContext],
    min_severity: float = 0.5,
    min_motif_steps: int = 3,
    max_pairs_per_trace: int = 2,
) -> list[PreferencePair]:
    """Strategy A: Remove motifs to create chosen/rejected pairs.

    For each trace with significant motifs:
      chosen  = trace with motif excised
      rejected = original trace (containing the motif)

    Only applied to CORRECT traces (we want to teach "same answer,
    better structure").

    Args:
        contexts: List of MotifContext from analyze_trace_motifs.
        min_severity: Minimum motif severity to generate a pair.
        min_motif_steps: Minimum steps in motif (avoid trivial excisions).
        max_pairs_per_trace: Cap pairs from a single trace.
    """
    pairs = []

    for ctx in contexts:
        if not ctx.trace.is_correct:
            continue
        if not ctx.motifs:
            continue

        # Sort motifs by severity (worst first)
        significant = [
            m for m in ctx.motifs
            if m.severity >= min_severity and len(m.step_ids) >= min_motif_steps
        ]
        significant.sort(key=lambda m: m.severity, reverse=True)

        count = 0
        for motif in significant:
            if count >= max_pairs_per_trace:
                break

            chosen_text = _excise_motif(ctx.steps, motif)

            # Sanity: chosen should be meaningfully shorter
            if len(chosen_text) >= len(ctx.trace.solution) * 0.95:
                continue
            # Sanity: chosen should not be empty
            if len(chosen_text) < 50:
                continue

            pairs.append(PreferencePair(
                problem_id=ctx.trace.problem_id,
                pair_type='contrastive_excision',
                chosen_solution=chosen_text,
                rejected_solution=ctx.trace.solution,
                chosen_dsr=0.0,  # Will be lower after excision
                rejected_dsr=ctx.trace.dsr,
                chosen_correct=True,
                rejected_correct=True,
                metadata={
                    'motif_type': motif.motif_type,
                    'motif_severity': motif.severity,
                    'motif_steps': len(motif.step_ids),
                    'excised_chars': len(ctx.trace.solution) - len(chosen_text),
                    'trace_id': ctx.trace.trace_id,
                    'strategy': 'excision',
                }
            ))
            count += 1

    return pairs


# ─── Strategy B: Motif Replacement ───────────────────────────────────

def build_replacement_pairs(
    contexts_by_problem: dict[str, list[MotifContext]],
    min_severity: float = 0.5,
    max_pairs_per_problem: int = 3,
) -> list[PreferencePair]:
    """Strategy B: Replace motif region with a live alternative from another rollout.

    For a problem with multiple rollouts:
      - Find a trace WITH a dead motif at position [i..j]
      - Find another trace WITHOUT a motif at similar position
      - Use the motif-free trace's corresponding region as replacement

    This is higher quality than excision because the chosen trace
    still has content in that region — just better content.
    """
    pairs = []

    for pid, contexts in contexts_by_problem.items():
        # Split: traces with significant motifs vs clean traces
        dirty = [ctx for ctx in contexts if ctx.trace.is_correct and
                 any(m.severity >= min_severity for m in ctx.motifs)]
        clean = [ctx for ctx in contexts if ctx.trace.is_correct and
                 (not ctx.motifs or all(m.severity < 0.3 for m in ctx.motifs))]

        if not dirty or not clean:
            continue

        count = 0
        for d_ctx in dirty:
            if count >= max_pairs_per_problem:
                break
            # Pick the cleanest alternative
            c_ctx = min(clean, key=lambda c: c.trace.dsr)

            pairs.append(PreferencePair(
                problem_id=pid,
                pair_type='contrastive_replacement',
                chosen_solution=c_ctx.trace.solution,
                rejected_solution=d_ctx.trace.solution,
                chosen_dsr=c_ctx.trace.dsr,
                rejected_dsr=d_ctx.trace.dsr,
                chosen_correct=True,
                rejected_correct=True,
                metadata={
                    'chosen_trace_id': c_ctx.trace.trace_id,
                    'rejected_trace_id': d_ctx.trace.trace_id,
                    'rejected_motifs': [m.motif_type for m in d_ctx.motifs],
                    'chosen_num_motifs': len(c_ctx.motifs),
                    'rejected_num_motifs': len(d_ctx.motifs),
                    'strategy': 'replacement',
                }
            ))
            count += 1

    return pairs


# ─── Strategy C: Cross-Trace Motif Contrast ──────────────────────────

def build_contrast_pairs(
    contexts_by_problem: dict[str, list[MotifContext]],
    max_pairs_per_problem: int = 3,
) -> list[PreferencePair]:
    """Strategy C: Pair traces with/without specific motif types.

    For each motif type that appears in some traces but not others
    for the same problem, create a pair that isolates that motif's
    effect. This is the purest contrastive signal.
    """
    pairs = []
    motif_types = ['dead_cascade', 'verification_theater', 'abandoned_branch']

    for pid, contexts in contexts_by_problem.items():
        correct = [ctx for ctx in contexts if ctx.trace.is_correct]
        if len(correct) < 2:
            continue

        for mtype in motif_types:
            has_motif = [ctx for ctx in correct
                         if any(m.motif_type == mtype for m in ctx.motifs)]
            no_motif = [ctx for ctx in correct
                        if not any(m.motif_type == mtype for m in ctx.motifs)]

            if not has_motif or not no_motif:
                continue

            # Pair: clean vs dirty, sorted by DSR for best contrast
            has_motif.sort(key=lambda c: c.trace.dsr, reverse=True)
            no_motif.sort(key=lambda c: c.trace.dsr)

            count = 0
            for rejected_ctx in has_motif:
                for chosen_ctx in no_motif:
                    if count >= max_pairs_per_problem:
                        break
                    # Ensure meaningful DSR difference
                    if rejected_ctx.trace.dsr - chosen_ctx.trace.dsr < 0.05:
                        continue

                    pairs.append(PreferencePair(
                        problem_id=pid,
                        pair_type=f'contrastive_{mtype}',
                        chosen_solution=chosen_ctx.trace.solution,
                        rejected_solution=rejected_ctx.trace.solution,
                        chosen_dsr=chosen_ctx.trace.dsr,
                        rejected_dsr=rejected_ctx.trace.dsr,
                        chosen_correct=True,
                        rejected_correct=True,
                        metadata={
                            'contrastive_motif': mtype,
                            'chosen_trace_id': chosen_ctx.trace.trace_id,
                            'rejected_trace_id': rejected_ctx.trace.trace_id,
                            'strategy': 'contrast',
                        }
                    ))
                    count += 1
                if count >= max_pairs_per_problem:
                    break

    return pairs


# ─── Main Entry Point ────────────────────────────────────────────────

def build_contrastive_pairs(
    annotated_traces: list[AnnotatedTrace],
    problem_texts: Optional[dict[str, str]] = None,
    output_path: Optional[str | Path] = None,
    seed: int = 42,
) -> list[PreferencePair]:
    """Build all contrastive structural preference pairs (Type 4).

    Runs motif extraction on all traces, then applies all three
    strategies to maximize pair diversity.

    Args:
        annotated_traces: List of AnnotatedTrace from the annotator.
        problem_texts: Optional dict mapping problem_id to problem text.
        output_path: If provided, save pairs in LLaMA-Factory DPO format.
        seed: Random seed.

    Returns:
        List of contrastive PreferencePair objects.
    """
    # Step 1: Extract motifs from all traces
    print("Extracting structural motifs...")
    contexts = [analyze_trace_motifs(t) for t in annotated_traces]

    # Aggregate motif stats
    all_motifs = [m for ctx in contexts for m in ctx.motifs]
    if all_motifs:
        summary = motif_summary(all_motifs)
        print(f"  Found {summary['num_motifs']} motifs across {len(contexts)} traces")
        print(f"  Types: {summary['by_type']}")
        print(f"  Avg severity: {summary['avg_severity']:.3f}")
    else:
        print("  No motifs found — skipping contrastive pairs")
        return []

    # Group by problem
    contexts_by_problem: dict[str, list[MotifContext]] = {}
    for ctx in contexts:
        contexts_by_problem.setdefault(ctx.trace.problem_id, []).append(ctx)

    # Step 2: Apply all strategies
    excision_pairs = build_excision_pairs(contexts)
    replacement_pairs = build_replacement_pairs(contexts_by_problem)
    contrast_pairs = build_contrast_pairs(contexts_by_problem)

    all_pairs = excision_pairs + replacement_pairs + contrast_pairs

    random.seed(seed)
    random.shuffle(all_pairs)

    print(f"\nBuilt {len(all_pairs)} contrastive pairs:")
    print(f"  Strategy A (Excision):     {len(excision_pairs)}")
    print(f"  Strategy B (Replacement):  {len(replacement_pairs)}")
    print(f"  Strategy C (Contrast):     {len(contrast_pairs)}")

    # Type breakdown for Strategy C
    from collections import Counter
    contrast_types = Counter(p.pair_type for p in contrast_pairs)
    for ct, count in contrast_types.most_common():
        print(f"    {ct}: {count}")

    # Save if requested
    if output_path and problem_texts:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        dpo_data = []
        for pair in all_pairs:
            ptext = problem_texts.get(pair.problem_id, "")
            if ptext:
                dpo_data.append(pair.to_sharegpt_dpo(ptext))

        output_path.write_text(json.dumps(dpo_data, indent=2, ensure_ascii=False))
        print(f"\nSaved {len(dpo_data)} contrastive DPO pairs to {output_path}")

    return all_pairs
