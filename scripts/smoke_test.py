#!/usr/bin/env python3
"""
StructPO Smoke Test

Validates that all core components work correctly:
  1. Step classifier (regex-based)
  2. DAG builder (content-overlap edges)
  3. Backward reachability (live/dead detection)
  4. Structural annotator
  5. Preference pair builder
  6. Dataset loading (LIMO format)

Run: python scripts/smoke_test.py
No GPU needed. Should complete in < 30 seconds.
"""

import json
import sys
import time
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from structpo.structural_parser.classifier import (
    classify_paragraph, classify_trace, segment_trace,
    compute_verification_density, get_type_distribution,
)
from structpo.structural_parser.dag_builder import build_dag, _extract_symbols
from structpo.structural_parser.reachability import (
    backward_reachability, compute_dsr, compute_typed_dsr,
    full_structural_analysis, get_dead_steps, get_live_steps,
)
from structpo.preference_builder.annotator import annotate_trace
from structpo.preference_builder.pair_builder import (
    build_efficiency_pairs, build_productive_exploration_pairs,
    build_direction_pairs, build_all_pairs, AnnotatedTrace,
)


PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  {detail}")


def test_classifier():
    print("\n=== 1. Step Classifier ===")

    tests = [
        ("Let me verify this by substituting back.", "verification"),
        ("2 + 3 = 5, and 5 * 4 = 20", "computation"),
        ("What if we try a different approach? Alternatively...", "exploration"),
        ("Wait, actually that is wrong. Let me reconsider.", "correction"),
        (r"Therefore the answer is \boxed{42}.", "conclusion"),
        ("By the triangle inequality, we have a + b > c.", "derivation"),
    ]

    for text, expected in tests:
        got = classify_paragraph(text)
        check(f"classify '{expected}'", got == expected, f"got={got}")

    # Segmentation
    trace = "Step 1.\n\nStep 2.\n\nStep 3."
    segs = segment_trace(trace)
    check("segment_trace splits on double newline", len(segs) == 3, f"got {len(segs)}")

    # Full trace classification
    steps = classify_trace("Let me compute.\n\n2+3=5.\n\nThe answer is \\boxed{5}.")
    check("classify_trace returns list", len(steps) == 3)
    check("last step is conclusion", steps[-1].step_type == "conclusion")


def test_symbol_extraction():
    print("\n=== 2. Symbol Extraction ===")

    # Math variables should be extracted
    syms = _extract_symbols("x^2 + 2x + 1 = 49")
    check("extracts 'x' from equation", "x" in syms, f"got {syms}")
    check("extracts '49' from equation", "49" in syms, f"got {syms}")

    # English text should NOT produce single-letter noise
    syms_en = _extract_symbols("Hmm, what if I try completing the square differently?")
    english_noise = syms_en & {"H", "w", "t", "c", "s", "d"}
    check("no English letter noise", len(english_noise) == 0, f"got {syms_en}")

    # Subscripted variables
    syms_sub = _extract_symbols("Consider a_1, x_{12}, and n_k")
    check("extracts subscripted vars", "a_1" in syms_sub or "x_12" in syms_sub, f"got {syms_sub}")


def test_dag_builder():
    print("\n=== 3. DAG Builder ===")

    steps = classify_trace(
        "We need to find x such that x^2 = 49.\n\n"
        "So x = 7 or x = -7.\n\n"
        "Let me think about an unrelated tangent with no math.\n\n"
        "The answer is \\boxed{7}."
    )
    dag = build_dag(steps)
    check("DAG has correct node count", dag.num_nodes == len(steps))
    check("conclusion detected", len(dag.conclusion_ids) >= 1)

    # The tangent (step 2) should NOT be connected to the conclusion
    # since it has no shared symbols
    step2 = dag.nodes[2]
    check("tangent step has few/no outputs", len(step2.outputs) <= 1,
          f"outputs={step2.outputs}")


def test_reachability():
    print("\n=== 4. Backward Reachability ===")

    # Trace with clear dead exploration steps (must be classified as
    # exploration/verification, not derivation, to break the derivation chain)
    r = full_structural_analysis(
        "We need to find x where x^2 = 49.\n\n"
        "x = 7 or x = -7.\n\n"
        "What if we try a completely different method using polar coordinates?\n\n"  # dead exploration
        "Let me consider another approach with generating functions.\n\n"            # dead exploration
        "The answer is \\boxed{7}."
    )
    check("DSR > 0 for trace with dead exploration", r["dsr"] > 0, f"dsr={r['dsr']:.0%}")
    check("has dead steps", r["num_dead"] > 0, f"dead={r['num_dead']}")

    # All-productive trace
    r2 = full_structural_analysis(
        "We compute 15 * 23.\n\n"
        "15 * 23 = 15 * 20 + 15 * 3.\n\n"
        "15 * 20 = 300 and 15 * 3 = 45.\n\n"
        "300 + 45 = 345.\n\n"
        "The answer is \\boxed{345}."
    )
    check("productive trace has low DSR", r2["dsr"] <= 0.2, f"dsr={r2['dsr']:.0%}")


def test_annotator():
    print("\n=== 5. Structural Annotator ===")

    ann = annotate_trace(
        problem_id="test_1",
        trace_id=0,
        solution="We solve x^2 = 49.\n\nx = 7.\n\nThe answer is \\boxed{7}.",
        answer="7",
        is_correct=True,
    )
    check("annotator returns AnnotatedTrace", ann.problem_id == "test_1")
    check("dsr is float", isinstance(ann.dsr, float))
    check("num_steps > 0", ann.num_steps > 0, f"num_steps={ann.num_steps}")
    check("trace_length > 0", ann.trace_length > 0)


def test_pair_builder():
    print("\n=== 6. Preference Pair Builder ===")

    # Create synthetic annotated traces
    traces = [
        AnnotatedTrace(problem_id="p1", trace_id=0, solution="clean A", answer="a",
                        is_correct=True, num_steps=5, dsr=0.05,
                        verification_density=0.1, live_verification_rate=0.9, trace_length=100),
        AnnotatedTrace(problem_id="p1", trace_id=1, solution="wasteful B", answer="a",
                        is_correct=True, num_steps=10, dsr=0.50,
                        verification_density=0.3, live_verification_rate=0.2, trace_length=200),
        AnnotatedTrace(problem_id="p1", trace_id=2, solution="wrong C", answer="b",
                        is_correct=False, num_steps=8, dsr=0.45,
                        verification_density=0.2, live_verification_rate=0.1, trace_length=150),
    ]

    traces_by_problem = {"p1": traces}

    eff = build_efficiency_pairs(traces_by_problem)
    check("efficiency pairs built", len(eff) >= 1, f"count={len(eff)}")
    if eff:
        check("chosen has lower DSR", eff[0].chosen_dsr < eff[0].rejected_dsr)

    prod = build_productive_exploration_pairs(traces_by_problem)
    check("productive exploration pairs built", len(prod) >= 1, f"count={len(prod)}")

    direction = build_direction_pairs(traces_by_problem)
    check("direction pairs built", len(direction) >= 1, f"count={len(direction)}")
    if direction:
        check("chosen is correct", direction[0].chosen_correct is True)
        check("rejected is incorrect", direction[0].rejected_correct is False)

    all_pairs = build_all_pairs(traces)
    check("build_all_pairs works", len(all_pairs) >= 1, f"count={len(all_pairs)}")


def test_dataset_loading():
    print("\n=== 7. Dataset Loading ===")

    dse_path = Path("data/limo_cleaned/limo_dse.json")
    orig_path = Path("data/limo_cleaned/limo_original.json")

    check("limo_dse.json exists", dse_path.exists())
    check("limo_original.json exists", orig_path.exists())

    if dse_path.exists():
        data = json.loads(dse_path.read_text())
        check("DSE dataset has 817 samples", len(data) == 817, f"got {len(data)}")
        check("sample has conversations", "conversations" in data[0])
        check("conversation has human+gpt", len(data[0]["conversations"]) == 2)

    # Dataset info for LLaMA-Factory
    info_path = Path("data/limo_cleaned/dataset_info.json")
    check("dataset_info.json exists", info_path.exists())
    if info_path.exists():
        info = json.loads(info_path.read_text())
        check("dataset_info has limo_dse", "limo_dse" in info)
        check("file_name points to correct file",
              info["limo_dse"]["file_name"] == "limo_dse.json",
              f"got {info['limo_dse']['file_name']}")


def test_performance():
    print("\n=== 8. Performance (on real data) ===")

    dse_path = Path("data/limo_cleaned/limo_dse.json")
    if not dse_path.exists():
        check("SKIP: dataset not found", False, "data/limo_cleaned/limo_dse.json missing")
        return

    data = json.loads(dse_path.read_text())
    n = min(100, len(data))

    t0 = time.time()
    dsrs = []
    for sample in data[:n]:
        convs = sample.get("conversations", [])
        if len(convs) < 2:
            continue
        r = full_structural_analysis(convs[1]["value"])
        dsrs.append(r["dsr"])
    elapsed = time.time() - t0

    avg_ms = elapsed / len(dsrs) * 1000
    avg_dsr = sum(dsrs) / len(dsrs)
    nonzero = sum(1 for d in dsrs if d > 0)

    check(f"analyzed {len(dsrs)} traces in {elapsed:.1f}s", True)
    check(f"avg {avg_ms:.1f}ms per trace (target: <20ms)", avg_ms < 20, f"got {avg_ms:.1f}ms")
    check(f"avg DSR = {avg_dsr:.1%}", True)
    check(f"{nonzero}/{len(dsrs)} traces have DSR > 0", nonzero > len(dsrs) * 0.5,
          f"only {nonzero}")


def main():
    print("=" * 60)
    print("StructPO Smoke Test")
    print("=" * 60)

    test_classifier()
    test_symbol_extraction()
    test_dag_builder()
    test_reachability()
    test_annotator()
    test_pair_builder()
    test_dataset_loading()
    test_performance()

    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed")
    print("=" * 60)

    if FAIL > 0:
        sys.exit(1)
    print("\nAll smoke tests passed! Ready for BriCS deployment.")


if __name__ == "__main__":
    main()
