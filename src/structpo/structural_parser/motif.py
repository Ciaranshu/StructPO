"""
Structural Motif Extraction

Extracts recurring wasteful patterns ("motifs") from reasoning DAGs.
Motifs are sub-sequences of steps that form recognizable structural
anti-patterns. Unlike trace-level DSR, motifs are local and transferable:
the same motif (e.g., "dead verification cascade") can appear across
different problems, enabling contrastive learning at the pattern level.

Motif types:
  1. Dead Cascade     — consecutive dead steps (≥3) forming an unreachable block
  2. Verification Theater — dead verification steps checking already-dead work
  3. Abandoned Branch — exploration → derivation → dead end, with no backtrack value
  4. Circular Revisit — re-deriving content already present in an earlier live step

Each motif is a contiguous span of step IDs with a type label and severity score.
"""

from dataclasses import dataclass, field

from .classifier import ClassifiedStep
from .dag_builder import ReasoningDAG, DAGNode, _extract_symbols


@dataclass
class StructuralMotif:
    """A detected structural anti-pattern in a reasoning trace."""
    motif_type: str          # 'dead_cascade', 'verification_theater', 'abandoned_branch', 'circular_revisit'
    step_ids: list[int]      # contiguous step IDs forming this motif
    severity: float          # 0-1, higher = more wasteful
    description: str = ""
    # For pair construction: the text span of this motif
    text_span: str = ""
    # Steps immediately before/after for context
    context_before_id: int = -1
    context_after_id: int = -1


def extract_motifs(
    steps: list[ClassifiedStep],
    dag: ReasoningDAG,
    min_cascade_len: int = 3,
) -> list[StructuralMotif]:
    """Extract all structural motifs from an annotated trace.

    Args:
        steps: Classified steps from the parser.
        dag: DAG after backward reachability (nodes have is_live set).
        min_cascade_len: Minimum consecutive dead steps for a cascade motif.

    Returns:
        List of StructuralMotif, sorted by position in trace.
    """
    motifs = []
    motifs.extend(_find_dead_cascades(steps, dag, min_cascade_len))
    motifs.extend(_find_verification_theater(steps, dag))
    motifs.extend(_find_abandoned_branches(steps, dag))
    motifs.extend(_find_circular_revisits(steps, dag))

    # Sort by first step ID
    motifs.sort(key=lambda m: m.step_ids[0] if m.step_ids else 0)

    # Assign text spans
    step_map = {s.id: s for s in steps}
    for m in motifs:
        m.text_span = "\n\n".join(step_map[sid].content for sid in m.step_ids if sid in step_map)
        if m.step_ids:
            first = m.step_ids[0]
            last = m.step_ids[-1]
            m.context_before_id = first - 1 if first > 0 else -1
            m.context_after_id = last + 1 if last + 1 in dag.nodes else -1

    return motifs


# ─── Motif Type 1: Dead Cascade ─────────────────────────────────────

def _find_dead_cascades(
    steps: list[ClassifiedStep],
    dag: ReasoningDAG,
    min_len: int = 3,
) -> list[StructuralMotif]:
    """Find runs of ≥min_len consecutive dead steps.

    These are blocks where the model went on a long tangent that
    never connected back to the conclusion.
    """
    motifs = []
    current_run = []

    for s in steps:
        node = dag.nodes.get(s.id)
        if node and not node.is_live:
            current_run.append(s.id)
        else:
            if len(current_run) >= min_len:
                severity = min(1.0, len(current_run) / 10.0)
                motifs.append(StructuralMotif(
                    motif_type='dead_cascade',
                    step_ids=list(current_run),
                    severity=severity,
                    description=f"{len(current_run)} consecutive dead steps",
                ))
            current_run = []

    # Don't forget trailing run
    if len(current_run) >= min_len:
        severity = min(1.0, len(current_run) / 10.0)
        motifs.append(StructuralMotif(
            motif_type='dead_cascade',
            step_ids=list(current_run),
            severity=severity,
            description=f"{len(current_run)} consecutive dead steps",
        ))

    return motifs


# ─── Motif Type 2: Verification Theater ─────────────────────────────

def _find_verification_theater(
    steps: list[ClassifiedStep],
    dag: ReasoningDAG,
) -> list[StructuralMotif]:
    """Find dead verification steps that check already-dead work.

    "Verification theater" = the model goes through the motions of
    checking its work, but the work being checked is itself dead
    (unreachable from conclusion). This is pure waste.
    """
    motifs = []

    for s in steps:
        if s.step_type != 'verification':
            continue
        node = dag.nodes.get(s.id)
        if not node or node.is_live:
            continue

        # Check: are ALL inputs to this verification also dead?
        if node.inputs:
            all_inputs_dead = all(
                not dag.nodes[inp].is_live
                for inp in node.inputs
                if inp in dag.nodes
            )
        else:
            # Orphan verification (no detected inputs) — still theater
            all_inputs_dead = True

        if all_inputs_dead:
            # Look for adjacent dead verifications to group them
            # (handled by merging below)
            motifs.append(StructuralMotif(
                motif_type='verification_theater',
                step_ids=[s.id],
                severity=0.8,  # High severity — pure waste
                description="dead verification of dead work",
            ))

    # Merge adjacent verification theater motifs
    motifs = _merge_adjacent_motifs(motifs, 'verification_theater')

    return motifs


# ─── Motif Type 3: Abandoned Branch ─────────────────────────────────

def _find_abandoned_branches(
    steps: list[ClassifiedStep],
    dag: ReasoningDAG,
) -> list[StructuralMotif]:
    """Find exploration → derivation → dead pattern.

    The model explores a new approach, does some work on it, but
    the entire branch ends up dead. The exploration itself was
    reasonable but the model didn't abandon it early enough.
    """
    motifs = []
    step_map = {s.id: s for s in steps}

    for s in steps:
        if s.step_type != 'exploration':
            continue
        node = dag.nodes.get(s.id)
        if not node or node.is_live:
            continue

        # Trace forward: find the dead branch emanating from this exploration
        branch_ids = [s.id]
        current_id = s.id + 1
        while current_id in dag.nodes:
            next_node = dag.nodes[current_id]
            next_step = step_map.get(current_id)
            if not next_step:
                break
            if next_node.is_live:
                break
            # Stop at next exploration (that's a different branch)
            if next_step.step_type == 'exploration' and current_id != s.id:
                break
            branch_ids.append(current_id)
            current_id += 1

        if len(branch_ids) >= 2:  # exploration + at least 1 follow-up
            severity = min(1.0, len(branch_ids) / 8.0)
            motifs.append(StructuralMotif(
                motif_type='abandoned_branch',
                step_ids=branch_ids,
                severity=severity,
                description=f"exploration branch with {len(branch_ids)} dead steps",
            ))

    return motifs


# ─── Motif Type 4: Circular Revisit ─────────────────────────────────

def _find_circular_revisits(
    steps: list[ClassifiedStep],
    dag: ReasoningDAG,
) -> list[StructuralMotif]:
    """Find dead steps that re-derive content already present in live steps.

    The model repeats a computation or derivation that was already
    done productively earlier. The repeat adds nothing.
    """
    motifs = []

    # Get symbols for each step
    step_symbols = {s.id: _extract_symbols(s.content) for s in steps}

    # Find live steps' symbol union
    live_symbols_by_step = {}
    for s in steps:
        node = dag.nodes.get(s.id)
        if node and node.is_live:
            live_symbols_by_step[s.id] = step_symbols[s.id]

    # For each dead derivation/computation, check symbol overlap with earlier live steps
    for s in steps:
        if s.step_type not in ('derivation', 'computation'):
            continue
        node = dag.nodes.get(s.id)
        if not node or node.is_live:
            continue

        current_syms = step_symbols[s.id]
        if len(current_syms) < 2:
            continue

        # Check overlap with earlier LIVE steps
        for live_id, live_syms in live_symbols_by_step.items():
            if live_id >= s.id:
                continue
            overlap = current_syms & live_syms
            overlap_ratio = len(overlap) / len(current_syms) if current_syms else 0
            if overlap_ratio >= 0.6:  # 60%+ of symbols already in a live step
                motifs.append(StructuralMotif(
                    motif_type='circular_revisit',
                    step_ids=[s.id],
                    severity=0.6,
                    description=f"re-derives {len(overlap)} symbols from live step {live_id}",
                ))
                break  # One match is enough

    return motifs


# ─── Helpers ─────────────────────────────────────────────────────────

def _merge_adjacent_motifs(
    motifs: list[StructuralMotif],
    motif_type: str,
) -> list[StructuralMotif]:
    """Merge motifs of the same type with adjacent step IDs."""
    if not motifs:
        return motifs

    motifs.sort(key=lambda m: m.step_ids[0])
    merged = [motifs[0]]

    for m in motifs[1:]:
        prev = merged[-1]
        if prev.step_ids[-1] + 1 == m.step_ids[0]:
            # Merge
            prev.step_ids.extend(m.step_ids)
            prev.severity = max(prev.severity, m.severity)
            prev.description = f"{len(prev.step_ids)} consecutive {motif_type} steps"
        else:
            merged.append(m)

    return merged


def motif_summary(motifs: list[StructuralMotif]) -> dict:
    """Summarize motifs for logging/analysis."""
    from collections import Counter
    type_counts = Counter(m.motif_type for m in motifs)
    total_steps = sum(len(m.step_ids) for m in motifs)
    avg_severity = sum(m.severity for m in motifs) / len(motifs) if motifs else 0
    return {
        'num_motifs': len(motifs),
        'total_wasteful_steps': total_steps,
        'avg_severity': round(avg_severity, 3),
        'by_type': dict(type_counts),
    }
