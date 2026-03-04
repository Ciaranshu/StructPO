"""R-IR data models using Pydantic.

Ported from DecoR poc/schemas.py
"""

from __future__ import annotations
from typing import Literal
from pydantic import BaseModel


class Step(BaseModel):
    """A single reasoning step in the R-IR."""
    id: int
    content: str
    type: Literal["assumption", "derivation", "computation", "verification", "conclusion"]
    inputs: list[int] = []       # IDs of steps this depends on
    outputs: list[int] = []      # IDs of steps depending on this
    deterministic: bool = False  # Whether this step is deterministically verifiable
    original_tokens: int = 0     # Token count of original content


class ReasoningIR(BaseModel):
    """The full Reasoning Intermediate Representation for one problem."""
    problem_id: str
    problem: str
    ground_truth: str
    original_answer: str
    original_reasoning: str
    original_token_count: int
    steps: list[Step]
    metadata: dict = {}
