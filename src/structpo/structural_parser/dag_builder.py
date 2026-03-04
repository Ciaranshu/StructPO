"""
Dependency DAG Builder for Reasoning Traces

Builds a lightweight directed acyclic graph (DAG) from classified reasoning steps.
Edges represent logical dependencies between steps. Combined with backward
reachability, this identifies which steps are structurally connected to the
conclusion (live/productive) vs disconnected (dead/wasteful).

Three layers of edges:
1. Adjacent derivation chain: consecutive derivation/computation steps
2. Content-overlap edges: shared variable names, numbers, or expressions
3. Conclusion fallback: last-resort connection for orphan conclusions

Steps that introduce new topics without sharing symbols with later steps
become structurally dead — distinguishing productive exploration (connected
to the reasoning graph) from wasteful exploration (disconnected tangents).

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

# Match LaTeX commands with arguments: \frac{...}{...}, \sqrt{...}, \binom{...}{...}
_LATEX_EXPR_PATTERN = re.compile(r'\\(?:frac|sqrt|binom)\{[^}]+\}(?:\{[^}]+\})?')

# Match subscripted variables: x_1, a_i, x_{12}, a_{ij}
_SUBSCRIPT_PATTERN = re.compile(r'([a-zA-Z])_\{?([a-zA-Z0-9]+)\}?')

# Match LaTeX Greek or named symbols: \alpha, \beta, \pi, \theta, etc.
_LATEX_SYMBOL_PATTERN = re.compile(r'\\(alpha|beta|gamma|delta|epsilon|zeta|eta|theta|iota|kappa|lambda|mu|nu|xi|pi|rho|sigma|tau|phi|chi|psi|omega|infty|cdot|times|pm)')

# Match isolated math variables: single letters surrounded by math context
# Requires the letter to be near math operators, =, ^, (, ), digits, or LaTeX
_MATH_VAR_PATTERN = re.compile(
    r'(?:^|[\s=+\-*/^({,<>])([a-zA-Z])(?=[\s=+\-*/^)},.<>:\d]|$|\^|_)'
)

# Match numbers (integers and decimals), at least 2 digits or a decimal
_NUM_PATTERN = re.compile(r'(?<![a-zA-Z])(\d{2,}(?:\.\d+)?|\d+\.\d+)(?![a-zA-Z])')

# Common English single letters to exclude from math variable detection
_ENGLISH_WORDS = {'a', 'i', 'I', 'A'}


def _extract_symbols(text: str) -> set[str]:
    """Extract mathematical symbols from text.
    
    Carefully distinguishes math variables from English text:
    - Subscripted variables (x_1, a_i) are always included
    - LaTeX expressions (\frac, \sqrt, \binom) are included as-is
    - LaTeX Greek symbols (\alpha, \pi) are included
    - Isolated single letters are included ONLY if they appear in 
      clear math context (near =, +, -, *, ^, etc.)
    - Numbers >= 10 (or with decimals) are included as anchors
    - Common English words ('a', 'I') are excluded from single-letter matches
    """
    symbols = set()
    
    # Layer 1: Subscripted variables (highest confidence)
    for m in _SUBSCRIPT_PATTERN.finditer(text):
        symbols.add(f"{m.group(1)}_{m.group(2)}")
    
    # Layer 2: LaTeX expressions (exact string match)
    for m in _LATEX_EXPR_PATTERN.finditer(text):
        symbols.add(m.group(0))
    
    # Layer 3: LaTeX named symbols
    for m in _LATEX_SYMBOL_PATTERN.finditer(text):
        symbols.add(f"\\{m.group(1)}")
    
    # Layer 4: Math-context single letters
    for m in _MATH_VAR_PATTERN.finditer(text):
        var = m.group(1)
        if var not in _ENGLISH_WORDS:
            symbols.add(var)
    
    # Layer 5: Significant numbers (>= 10 or decimal, to avoid matching
    # trivial numbers like 1, 2, 3 that appear in step numbering)
    for m in _NUM_PATTERN.finditer(text):
        symbols.add(m.group(1))
    
    return symbols


def build_dag(steps: list[ClassifiedStep]) -> ReasoningDAG:
    """Build a dependency DAG from classified steps.
    
    Edge construction uses content-based dependency detection, NOT an
    unconditional sequential chain. This is critical: a sequential chain
    makes everything reachable (DSR=0 always), defeating the purpose of
    structural analysis.
    
    Edge types:
    1. Content-overlap: step[j] depends on step[i] if they share
       significant symbols (variables, numbers, expressions) and i < j.
    2. Conclusion fallback: if the conclusion has no content-based inputs,
       connect it to the immediately preceding step (last-resort chain).
    3. Adjacent derivation chain: consecutive derivation/computation steps
       are chained (they form a logical derivation flow).
    
    Steps that introduce NEW topics (exploration, tangents) without sharing
    symbols with later steps become structurally dead — exactly the behavior
    we need for DSR-based preference pairs.
    
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
        dag.conclusion_ids = [steps[-1].id]
    
    # Extract symbols per step
    step_symbols = {s.id: _extract_symbols(s.content) for s in steps}
    
    # Types that form natural derivation chains when adjacent
    _CHAIN_TYPES = {'derivation', 'computation', 'conclusion'}
    
    # Build edges
    for i, step in enumerate(steps):
        node = dag.nodes[step.id]
        
        # Layer 1: Adjacent derivation chain — consecutive derivation/computation
        # steps are likely a logical flow and should be chained.
        # Exploration, verification, and correction steps break the chain.
        if i > 0:
            prev_step = steps[i - 1]
            if step.step_type in _CHAIN_TYPES and prev_step.step_type in _CHAIN_TYPES:
                if prev_step.id not in node.inputs:
                    node.inputs.append(prev_step.id)
                    dag.nodes[prev_step.id].outputs.append(step.id)
        
        # Layer 2: Content-overlap edges (look back up to 10 steps)
        current_symbols = step_symbols[step.id]
        if current_symbols:
            for j in range(max(0, i - 10), i):
                prev_step = steps[j]
                if prev_step.id in node.inputs:
                    continue
                
                prev_symbols = step_symbols[prev_step.id]
                overlap = current_symbols & prev_symbols
                
                # Require at least 1 shared math symbol. This is safe now
                # because _extract_symbols only returns genuine math symbols,
                # not English letters.
                if len(overlap) >= 1:
                    node.inputs.append(prev_step.id)
                    dag.nodes[prev_step.id].outputs.append(step.id)
        
        # Layer 3: Conclusion fallback — if a conclusion step has no inputs
        # from content overlap or derivation chain, connect to previous step
        if step.step_type == 'conclusion' and not node.inputs and i > 0:
            prev_id = steps[i - 1].id
            node.inputs.append(prev_id)
            dag.nodes[prev_id].outputs.append(step.id)
    
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
