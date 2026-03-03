"""
Dependency DAG Builder for Reasoning Traces

Builds a lightweight directed acyclic graph (DAG) from classified reasoning steps.
Edges represent logical dependencies between steps.

Two layers of edges:
1. Sequential chain: each step depends on the previous (baseline)
2. Content-overlap edges: shared variable names, numbers, or expressions

This is the fast, API-free alternative to LLM-based R-IR parsing.
Runtime: <5ms per trace.
"""

import re
from dataclasses import dataclass, field
from .classifier import ClassifiedStep


@dataclass
class DAGNode:
    """A node in the dependency DAG."""
    step_id: int
    step_type: str
    inputs: list[int] = field(default_factory=list)   # steps this depends on
    outputs: list[int] = field(default_factory=list)  # steps depending on this
    is_live: bool = False
    dead_reason: str = ""


@dataclass
class ReasoningDAG:
    """A dependency DAG for a reasoning trace."""
    nodes: dict[int, DAGNode] = field(default_factory=dict)
    conclusion_ids: list[int] = field(default_factory=list)
    
    @property
    def num_nodes(self) -> int:
        return len(self.nodes)
    
    @property
    def num_live(self) -> int:
        return sum(1 for n in self.nodes.values() if n.is_live)
    
    @property
    def num_dead(self) -> int:
        return sum(1 for n in self.nodes.values() if not n.is_live)
    
    @property
    def dead_step_ratio(self) -> float:
        if not self.nodes:
            return 0.0
        return self.num_dead / self.num_nodes


# ============================================================
# Variable/Number Extraction for Content-Overlap Edges
# ============================================================

# Match LaTeX variable patterns like x, y, n, k, a_i, x_1, etc.
_VAR_PATTERN = re.compile(r'\\?([a-zA-Z])(?:_\{?(\d+|[a-zA-Z])\}?)?')

# Match numbers (integers and decimals)
_NUM_PATTERN = re.compile(r'\b(\d+(?:\.\d+)?)\b')

# Match LaTeX expressions that are likely to be shared across steps
_EXPR_PATTERN = re.compile(r'\\(?:frac|sqrt|binom)\{[^}]+\}\{[^}]+\}')


def _extract_symbols(text: str) -> set[str]:
    """Extract mathematical symbols (variables + significant numbers) from text."""
    symbols = set()
    
    # Extract variables
    for m in _VAR_PATTERN.finditer(text):
        var = m.group(1)
        sub = m.group(2)
        if sub:
            symbols.add(f"{var}_{sub}")
        else:
            symbols.add(var)
    
    # Extract significant numbers (skip 0, 1, 2 which are too common)
    for m in _NUM_PATTERN.finditer(text):
        num = m.group(1)
        try:
            val = float(num)
            if val > 2 and len(num) >= 2:  # Only "interesting" numbers
                symbols.add(num)
        except ValueError:
            pass
    
    # Extract LaTeX expressions
    for m in _EXPR_PATTERN.finditer(text):
        symbols.add(m.group(0))
    
    return symbols


def build_dag(steps: list[ClassifiedStep]) -> ReasoningDAG:
    """Build a dependency DAG from classified steps.
    
    Edge construction:
    1. Sequential: step[i] depends on step[i-1] (baseline chain)
    2. Content-overlap: step[j] depends on step[i] if they share
       significant symbols and i < j
    
    Args:
        steps: List of ClassifiedStep from the classifier.
        
    Returns:
        ReasoningDAG with nodes and edges.
    """
    dag = ReasoningDAG()
    
    if not steps:
        return dag
    
    # Create nodes
    for s in steps:
        dag.nodes[s.id] = DAGNode(
            step_id=s.id,
            step_type=s.step_type,
        )
    
    # Identify conclusions
    dag.conclusion_ids = [s.id for s in steps if s.step_type == 'conclusion']
    if not dag.conclusion_ids:
        # If no explicit conclusion, use last step
        dag.conclusion_ids = [steps[-1].id]
    
    # Extract symbols per step
    step_symbols = {s.id: _extract_symbols(s.content) for s in steps}
    
    # Build edges
    for i, step in enumerate(steps):
        node = dag.nodes[step.id]
        
        # Layer 1: Sequential dependency (each step depends on previous)
        if i > 0:
            prev_id = steps[i - 1].id
            if prev_id not in node.inputs:
                node.inputs.append(prev_id)
                dag.nodes[prev_id].outputs.append(step.id)
        
        # Layer 2: Content-overlap edges (skip sequential neighbors already connected)
        current_symbols = step_symbols[step.id]
        if current_symbols:
            for j in range(max(0, i - 10), i):  # Look back up to 10 steps
                prev_step = steps[j]
                if prev_step.id in node.inputs:
                    continue  # Already connected
                
                prev_symbols = step_symbols[prev_step.id]
                overlap = current_symbols & prev_symbols
                
                # Require meaningful overlap (>= 2 shared symbols)
                if len(overlap) >= 2:
                    node.inputs.append(prev_step.id)
                    dag.nodes[prev_step.id].outputs.append(step.id)
    
    return dag


def dag_to_dict(dag: ReasoningDAG) -> dict:
    """Serialize DAG to a JSON-compatible dict."""
    return {
        'num_nodes': dag.num_nodes,
        'conclusion_ids': dag.conclusion_ids,
        'nodes': {
            str(nid): {
                'step_id': n.step_id,
                'step_type': n.step_type,
                'inputs': n.inputs,
                'outputs': n.outputs,
                'is_live': n.is_live,
                'dead_reason': n.dead_reason,
            }
            for nid, n in dag.nodes.items()
        }
    }
