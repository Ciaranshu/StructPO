"""
Dead Step Quality Classification

Classifies dead steps by their structural role — not all dead steps are
equally wasteful. Some are "productive dead ends" that likely inspired
live steps; others are "verification theater" that waste tokens verifying
already-dead work.

This classification is the foundation of StructPRM's quality-aware reward
(Level 1), which provides more nuanced signal than raw DSR (Level 0).

Key finding (2026-03-28): 8% of dead steps are productive dead ends.
Quality-aware reward protects these (penalty 0.05) while heavily penalizing
verification theater (penalty 0.80).

All computation is regex + graph based. No LLM calls. <10ms per trace.
Suitable for online RL loop integration.
"""

from collections import Counter
from .classifier import ClassifiedStep
from .dag_builder import ReasoningDAG, _extract_symbols


# ================================================================
# Dead Step Quality Types
# ================================================================

QUALITY_TYPES = [
    'verification_theater',    # verifies something that is also dead
    'circular_revisit',        # re-derives content already in a live step
    'wasteful_exploration',    # exploration that leads nowhere
    'failed_correction',       # correction attempt that didn't help
    'abandoned_computation',   # computation whose result is never used
    'redundant_verification',  # verifies something live but result unused
    'productive_dead_end',     # dead but shares symbols with live steps
    'generic_dead',            # default category
]

QUALITY_PENALTIES = {
    'verification_theater': 0.80,
    'circular_revisit': 0.60,
    'wasteful_exploration': 0.40,
    'failed_correction': 0.30,
    'abandoned_computation': 0.30,
    'generic_dead': 0.30,
    'redundant_verification': 0.20,
    'productive_dead_end': 0.05,
}


def classify_dead_step(
    step: ClassifiedStep,
    dag: ReasoningDAG,
    all_steps: list[ClassifiedStep],
    step_symbols: dict[int, set],
) -> str:
    """Classify WHY a dead step is dead — what kind of waste is it?

    Args:
        step: The dead step to classify.
        dag: The reasoning DAG (after backward reachability).
        all_steps: All steps in the trace.
        step_symbols: Pre-extracted symbols per step ID.

    Returns:
        One of the QUALITY_TYPES strings.
    """
    node = dag.nodes[step.id]
    live_ids = {nid for nid, n in dag.nodes.items() if n.is_live}
    dead_ids = {nid for nid, n in dag.nodes.items() if not n.is_live}

    # --- Verification Theater ---
    if step.step_type == 'verification':
        potential_targets = node.inputs
        if not potential_targets and step.id > 0:
            potential_targets = [step.id - 1]

        if potential_targets:
            targets_dead = all(t in dead_ids for t in potential_targets if t in dag.nodes)
            if targets_dead and len(potential_targets) > 0:
                return 'verification_theater'
            else:
                return 'redundant_verification'
        return 'redundant_verification'

    # --- Circular Revisit ---
    if step.step_type in ('computation', 'derivation'):
        my_symbols = step_symbols.get(step.id, set())
        if my_symbols:
            for lid in live_ids:
                live_syms = step_symbols.get(lid, set())
                if live_syms:
                    overlap = my_symbols & live_syms
                    union = my_symbols | live_syms
                    if union and len(overlap) / len(union) >= 0.6:
                        return 'circular_revisit'

    # --- Productive Dead End ---
    if step.step_type in ('exploration', 'derivation', 'computation'):
        my_symbols = step_symbols.get(step.id, set())
        if my_symbols:
            for other_step in all_steps:
                if other_step.id > step.id and other_step.id in live_ids:
                    other_syms = step_symbols.get(other_step.id, set())
                    overlap = my_symbols & other_syms
                    if len(overlap) >= 2:
                        return 'productive_dead_end'

    # --- Type-specific defaults ---
    if step.step_type == 'exploration':
        return 'wasteful_exploration'

    if step.step_type == 'correction':
        return 'failed_correction'

    if step.step_type == 'computation':
        return 'abandoned_computation'

    return 'generic_dead'


def compute_quality_reward(
    steps: list[ClassifiedStep],
    dag: ReasoningDAG,
    step_symbols: dict[int, set] | None = None,
) -> tuple[float, dict[str, int]]:
    """Compute StructPRM Level 1 quality-aware structural reward.

    Unlike raw DSR (Level 0) which penalizes all dead steps equally,
    quality-aware reward assigns graduated penalties based on the TYPE
    of structural waste. Productive dead ends get near-zero penalty.

    Args:
        steps: Classified steps from the trace.
        dag: DAG after backward reachability analysis.
        step_symbols: Pre-extracted symbols. If None, extracts them.

    Returns:
        Tuple of (reward, quality_counts).
        reward: Float in [0, 1]. 1.0 = perfectly clean trace.
        quality_counts: Counter of quality types for dead steps.
    """
    if not steps:
        return 1.0, {}

    if step_symbols is None:
        step_symbols = {s.id: _extract_symbols(s.content) for s in steps}

    total_penalty = 0.0
    quality_counts = Counter()

    for step in steps:
        node = dag.nodes[step.id]
        if node.is_live:
            continue

        quality = classify_dead_step(step, dag, steps, step_symbols)
        quality_counts[quality] += 1
        total_penalty += QUALITY_PENALTIES.get(quality, 0.3)

    reward = 1.0 - (total_penalty / len(steps))
    reward = max(0.0, min(1.0, reward))
    return reward, dict(quality_counts)


def full_quality_analysis(solution: str) -> dict:
    """Run complete quality-aware structural analysis on a reasoning trace.

    This is the Level 1 StructPRM scoring function. Suitable for:
    - Best-of-N selection
    - DPO pair construction
    - Online RL reward shaping

    Args:
        solution: Full reasoning trace text.

    Returns:
        Dict with DSR, quality reward, quality breakdown, and step details.
    """
    from .classifier import classify_trace
    from .dag_builder import build_dag
    from .reachability import backward_reachability, compute_dsr

    steps = classify_trace(solution)
    if not steps:
        return {
            'num_steps': 0,
            'dsr': 0.0,
            'quality_reward': 1.0,
            'quality_counts': {},
            'steps': [],
        }

    dag = build_dag(steps)
    dag = backward_reachability(dag)
    step_symbols = {s.id: _extract_symbols(s.content) for s in steps}

    dsr = compute_dsr(dag)
    quality_reward, quality_counts = compute_quality_reward(steps, dag, step_symbols)

    step_details = []
    for s in steps:
        node = dag.nodes[s.id]
        detail = {
            'id': s.id,
            'type': s.step_type,
            'is_live': node.is_live,
            'char_length': s.char_length,
        }
        if not node.is_live:
            detail['quality'] = classify_dead_step(s, dag, steps, step_symbols)
            detail['penalty'] = QUALITY_PENALTIES.get(detail['quality'], 0.3)
        step_details.append(detail)

    return {
        'num_steps': len(steps),
        'dsr': dsr,
        'quality_reward': quality_reward,
        'quality_counts': quality_counts,
        'num_live': dag.num_live,
        'num_dead': dag.num_dead,
        'steps': step_details,
    }
