"""
Fast Regex-Based Step Type Classifier

Classifies reasoning paragraphs into step types without API calls.
Achieves ~85%+ agreement with LLM-based classification.
Runtime: <10ms per trace (suitable for RL loop integration).

Step types:
  - verification: checking/confirming/validating previous work
  - computation: arithmetic/algebraic operations
  - exploration: trying alternative approaches
  - correction: self-correction after detecting errors
  - conclusion: contains final boxed answer
  - derivation: default (logical reasoning steps)

Ported from DecoR experiments/automated_typed_dse.py
"""

import re
from dataclasses import dataclass, field


# ============================================================
# Pattern Definitions
# ============================================================

VERIFICATION_PATTERNS = [
    r'let me (check|verify|double.?check|confirm|validate|make sure)',
    r'(checking|verifying|double.?checking|confirming)',
    r'let\'s (check|verify|double.?check|confirm|make sure)',
    r'to (check|verify|confirm|validate) (this|that|our|the)',
    r'we can (check|verify|confirm)',
    r'substitut(e|ing) back',
    r'plug(ging)? (it |this )?back',
    r'sanity check',
    r'indeed[,.]',
    r'this (is correct|checks out|confirms|agrees|matches|is consistent)',
    r'which (confirms|verifies|checks out|matches|agrees)',
    r'as expected',
    r'consistent with',
]

COMPUTATION_PATTERNS = [
    r'\d+\s*[+\-*/×÷^]\s*\d+\s*=\s*\d+',
    r'\\frac\{.*?\}\{.*?\}',
    r'\\sqrt\{.*?\}',
    r'= \d+',
    r'\\cdot',
    r'\\times',
    r'\\binom\{',
    r'\\sum',
    r'\\prod',
]

EXPLORATION_PATTERNS = [
    r'let me (try|consider|think about|explore|approach)',
    r'another (way|approach|method|strategy)',
    r'what if',
    r'alternatively',
    r'on the other hand',
    r'perhaps',
    r'maybe (we|I) (should|could|can)',
    r'i wonder',
    r'one approach',
    r'suppose',
]

CORRECTION_PATTERNS = [
    r'\bwait\b',
    r'\bactually\b[,.]',
    r'\bhold on\b',
    r'let me re(think|consider|do|calculate|start|examine)',
    r'that\'s (not right|wrong|incorrect)',
    r'i made (a |an )?(mistake|error)',
    r'\bcorrection\b',
    r'going back',
    r'this (is wrong|doesn\'t work|can\'t be right)',
    r'no[,.]? (that|this|wait)',
]

# Pre-compile all patterns for performance
_COMPILED_VERIFICATION = [re.compile(p, re.IGNORECASE) for p in VERIFICATION_PATTERNS]
_COMPILED_COMPUTATION = [re.compile(p) for p in COMPUTATION_PATTERNS]
_COMPILED_EXPLORATION = [re.compile(p, re.IGNORECASE) for p in EXPLORATION_PATTERNS]
_COMPILED_CORRECTION = [re.compile(p, re.IGNORECASE) for p in CORRECTION_PATTERNS]


STEP_TYPES = ['verification', 'computation', 'exploration', 'correction', 'conclusion', 'derivation']


@dataclass
class ClassifiedStep:
    """A classified reasoning step."""
    id: int
    step_type: str
    content: str
    char_length: int
    pattern_scores: dict = field(default_factory=dict)


def classify_paragraph(text: str) -> str:
    """Classify a paragraph into a step type using regex patterns.
    
    Returns one of: 'verification', 'computation', 'exploration', 
                    'correction', 'conclusion', 'derivation'
    """
    text_lower = text.lower()
    
    # Conclusion: contains boxed answer
    if r'\boxed{' in text or r'\boxed ' in text:
        return 'conclusion'
    
    # Score each type
    verif_score = sum(1 for p in _COMPILED_VERIFICATION if p.search(text_lower))
    expl_score = sum(1 for p in _COMPILED_EXPLORATION if p.search(text_lower))
    corr_score = sum(1 for p in _COMPILED_CORRECTION if p.search(text_lower))
    comp_score = sum(1 for p in _COMPILED_COMPUTATION if p.search(text))  # case-sensitive for math
    
    scores = {
        'verification': verif_score * 2,      # Weight verification higher
        'exploration': expl_score * 1.5,
        'correction': corr_score * 1.5,
        'computation': comp_score,
    }
    
    max_type = max(scores, key=scores.get)
    if scores[max_type] >= 2:
        return max_type
    
    return 'derivation'


def segment_trace(solution: str) -> list[str]:
    """Segment a solution text into paragraphs (reasoning steps).
    
    Splits on double newlines. Filters empty paragraphs.
    """
    paragraphs = re.split(r'\n\n+', solution)
    return [p.strip() for p in paragraphs if p.strip()]


def classify_trace(solution: str) -> list[ClassifiedStep]:
    """Segment and classify all steps in a reasoning trace.
    
    Args:
        solution: Full reasoning trace text.
        
    Returns:
        List of ClassifiedStep objects.
    """
    paragraphs = segment_trace(solution)
    steps = []
    for i, para in enumerate(paragraphs):
        step_type = classify_paragraph(para)
        steps.append(ClassifiedStep(
            id=i,
            step_type=step_type,
            content=para,
            char_length=len(para),
        ))
    return steps


def get_type_distribution(steps: list[ClassifiedStep]) -> dict[str, int]:
    """Get the count of each step type."""
    dist = {t: 0 for t in STEP_TYPES}
    for s in steps:
        dist[s.step_type] = dist.get(s.step_type, 0) + 1
    return dist


def compute_verification_density(steps: list[ClassifiedStep]) -> float:
    """Fraction of steps that are verification."""
    if not steps:
        return 0.0
    return sum(1 for s in steps if s.step_type == 'verification') / len(steps)


if __name__ == '__main__':
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Classify reasoning trace step types')
    parser.add_argument('--input', type=str, help='Path to text file with reasoning trace')
    parser.add_argument('--text', type=str, help='Direct text input')
    args = parser.parse_args()
    
    if args.input:
        with open(args.input) as f:
            text = f.read()
    elif args.text:
        text = args.text
    else:
        print("Provide --input <file> or --text <string>")
        sys.exit(1)
    
    steps = classify_trace(text)
    for s in steps:
        print(f"[{s.id}] {s.step_type:15s} ({s.char_length:5d} chars) | {s.content[:80]}...")
    
    dist = get_type_distribution(steps)
    print(f"\nType distribution: {dist}")
    print(f"Verification density: {compute_verification_density(steps):.1%}")
