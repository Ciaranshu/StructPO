"""
Backward Reachability Analysis

Given a dependency DAG, performs backward reachability from conclusion nodes
to determine which steps are "live" (contribute to the final answer) and
which are "dead" (structurally unreachable from the conclusion).

This is the core algorithm behind Dead Step Elimination (DSE) and the
structural preference signal used in StructPO.

Ported from DecoR poc/poc2_dse.py
"""

from .dag_builder import ReasoningDAG, DAGNode
from .classifier import ClassifiedStep


def backward_reachability(dag: ReasoningDAG) -> ReasoningDAG:
    """Mark nodes as live/dead via backward reachability from conclusions.
    
    Algorithm:
    1. Start from all conclusion nodes
    2. BFS backward through input edges
    3. All visited nodes are "live", rest are "dead"
    
    Args:
        dag: ReasoningDAG with nodes and edges already built.
        
    Returns:
        Same DAG with is_live and dead_reason fields updated.
    """
    if not dag.nodes:
        return dag
    
    # BFS from conclusion nodes
    live_ids = set()
    queue = list(dag.conclusion_ids)
    
    while queue:
        nid = queue.pop(0)
        if nid in live_ids:
            continue
        live_ids.add(nid)
        node = dag.nodes.get(nid)
        if node:
            for dep_id in node.inputs:
                if dep_id not in live_ids:
                    queue.append(dep_id)
    
    # Mark live/dead
    for nid, node in dag.nodes.items():
        if nid in live_ids:
            node.is_live = True
            node.dead_reason = ""
        else:
            node.is_live = False
            node.dead_reason = "unreachable_from_conclusion"
    
    return dag


def compute_dsr(dag: ReasoningDAG) -> float:
    """Compute Dead Step Ratio (DSR) after reachability analysis."""
    return dag.dead_step_ratio


def compute_typed_dsr(dag: ReasoningDAG, steps: list[ClassifiedStep]) -> dict:
    """Compute per-type dead step ratios.
    
    Returns dict like:
    {
        'verification': {'total': 5, 'dead': 2, 'ratio': 0.4},
        'computation': {'total': 8, 'dead': 1, 'ratio': 0.125},
        ...
    }
    """
    type_stats = {}
    for s in steps:
        t = s.step_type
        if t not in type_stats:
            type_stats[t] = {'total': 0, 'dead': 0}
        type_stats[t]['total'] += 1
        node = dag.nodes.get(s.id)
        if node and not node.is_live:
            type_stats[t]['dead'] += 1
    
    for t in type_stats:
        total = type_stats[t]['total']
        type_stats[t]['ratio'] = type_stats[t]['dead'] / total if total > 0 else 0.0
    
    return type_stats


def get_dead_steps(dag: ReasoningDAG, steps: list[ClassifiedStep]) -> list[ClassifiedStep]:
    """Return only the dead steps."""
    return [s for s in steps if not dag.nodes[s.id].is_live]


def get_live_steps(dag: ReasoningDAG, steps: list[ClassifiedStep]) -> list[ClassifiedStep]:
    """Return only the live steps."""
    return [s for s in steps if dag.nodes[s.id].is_live]


def full_structural_analysis(solution: str) -> dict:
    """Run the complete structural analysis pipeline on a reasoning trace.
    
    Pipeline: segment → classify → build DAG → reachability → metrics
    
    Args:
        solution: Full reasoning trace text.
        
    Returns:
        Dict with DSR, typed DSR, step details, and DAG info.
    """
    from .classifier import classify_trace
    from .dag_builder import build_dag
    
    # Step 1-2: Segment and classify
    steps = classify_trace(solution)
    
    if not steps:
        return {
            'num_steps': 0,
            'dsr': 0.0,
            'typed_dsr': {},
            'steps': [],
        }
    
    # Step 3: Build DAG
    dag = build_dag(steps)
    
    # Step 4: Backward reachability
    dag = backward_reachability(dag)
    
    # Step 5: Compute metrics
    dsr = compute_dsr(dag)
    typed_dsr = compute_typed_dsr(dag, steps)
    
    step_details = []
    for s in steps:
        node = dag.nodes[s.id]
        step_details.append({
            'id': s.id,
            'type': s.step_type,
            'is_live': node.is_live,
            'dead_reason': node.dead_reason,
            'char_length': s.char_length,
            'inputs': node.inputs,
            'outputs': node.outputs,
        })
    
    return {
        'num_steps': len(steps),
        'dsr': dsr,
        'num_live': dag.num_live,
        'num_dead': dag.num_dead,
        'typed_dsr': typed_dsr,
        'steps': step_details,
        'conclusion_ids': dag.conclusion_ids,
    }
