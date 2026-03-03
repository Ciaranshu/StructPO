"""Build 5 cleaned LIMO datasets from R-IR + DSE results.

Reads parsed R-IR and DSE results, then produces 5 training datasets:
1. LIMO-Original: original solutions (baseline)
2. LIMO-DSE: all dead steps removed (structural analysis)
3. LIMO-NoVerif: all verification steps removed (structural, ablation)
4. LIMO-LiveVerif: only dead verification steps removed (structural, precision)
5. LIMO-KeywordClean: verification keyword spans removed (heuristic baseline)

Output: LLaMA-Factory sharegpt format JSON files.

Usage:
    python -m experiments.build_limo_datasets
    python -m experiments.build_limo_datasets --min-parsed 700  # require at least N parsed
"""

import argparse
import json
import re
from pathlib import Path

from poc.config import DATA_DIR

LIMO_DIR = DATA_DIR / "limo"
LIMO_RAW = LIMO_DIR / "limo_raw.json"
LIMO_RIR_DIR = LIMO_DIR / "rir"
LIMO_DSE_DIR = LIMO_DIR / "dse"
CLEANED_DIR = LIMO_DIR / "cleaned"
CLEANED_DIR.mkdir(parents=True, exist_ok=True)

# System prompt for math reasoning (same as LIMO uses)
SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


def load_all_data():
    """Load raw LIMO data, R-IR parses, and DSE results."""
    raw_data = json.loads(LIMO_RAW.read_text())
    raw_by_id = {d["id"]: d for d in raw_data}

    parsed = {}
    for f in sorted(LIMO_RIR_DIR.glob("*.json")):
        rir = json.loads(f.read_text())
        limo_id = rir["metadata"].get("limo_id", int(f.stem.split("_")[1]))
        parsed[limo_id] = rir

    dse_results = {}
    for f in sorted(LIMO_DSE_DIR.glob("*.json")):
        dse = json.loads(f.read_text())
        limo_id = dse.get("limo_id", int(f.stem.split("_")[1]))
        dse_results[limo_id] = dse

    return raw_by_id, parsed, dse_results


def remove_steps_by_paragraphs(
    paragraph_texts: dict[str, str],
    steps: list[dict],
    step_ids_to_remove: set[int],
) -> str:
    """Remove step content from solution using paragraph-level mappings.

    Each step has a 'paragraphs' field listing which paragraph IDs it owns.
    We reconstruct the solution keeping only paragraphs belonging to
    steps NOT in step_ids_to_remove.

    Args:
        paragraph_texts: {paragraph_id_str: text} from R-IR
        steps: list of step dicts with 'paragraphs' field
        step_ids_to_remove: set of step IDs to remove

    Returns:
        Cleaned solution text with dead-step paragraphs removed.
    """
    if not step_ids_to_remove:
        # Return all paragraphs in order
        all_pids = sorted(int(k) for k in paragraph_texts)
        return "\n\n".join(paragraph_texts[str(p)] for p in all_pids).strip()

    # Collect paragraph IDs to remove
    remove_pids = set()
    for s in steps:
        if s["id"] in step_ids_to_remove:
            remove_pids.update(s.get("paragraphs", []))

    # SAFETY: never remove paragraphs containing \boxed{} (final answer)
    for pid in list(remove_pids):
        text = paragraph_texts.get(str(pid), "")
        if "boxed" in text:
            remove_pids.discard(pid)

    # Reconstruct solution from kept paragraphs (in order)
    all_pids = sorted(int(k) for k in paragraph_texts)
    kept = [paragraph_texts[str(p)] for p in all_pids if p not in remove_pids]

    result = "\n\n".join(kept)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


# Verification keywords for heuristic baseline (LIMO-KeywordClean)
VERIF_KEYWORDS = [
    "let me verify", "let me check", "let me double-check", "let me double check",
    "let's verify", "let's check", "let's double-check", "let's double check",
    "to verify", "to check", "to confirm",
    "i can verify", "i can check", "i'll verify", "i'll check",
    "we can verify", "we can check",
    "double-checking", "double checking",
    "verification:", "checking:",
    "let me make sure", "let's make sure",
    "sanity check",
]


def keyword_clean_solution(solution: str) -> str:
    """Remove verification spans using keyword matching (no structural analysis).

    For each verification keyword found, remove the sentence/paragraph containing it.
    This is the heuristic baseline to compare against DSE.
    """
    lines = solution.split("\n")
    cleaned_lines = []
    skip_until_blank = False

    for line in lines:
        line_lower = line.lower().strip()

        # Check if this line starts a verification span
        if any(kw in line_lower for kw in VERIF_KEYWORDS):
            skip_until_blank = True
            continue

        # If we're skipping and hit a blank line or new topic, stop skipping
        if skip_until_blank:
            if line.strip() == "":
                skip_until_blank = False
                cleaned_lines.append(line)
            # Also stop if line starts with a new step indicator
            elif any(line.strip().startswith(p) for p in [
                "Step", "Now", "Next", "First", "Second", "Third",
                "Then", "Finally", "Therefore", "Thus", "Hence",
                "So ", "Since", "Given", "We ", "The ", "From",
                "**Step", "**Now", "**Next",
            ]):
                skip_until_blank = False
                cleaned_lines.append(line)
            continue

        cleaned_lines.append(line)

    result = "\n".join(cleaned_lines)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def build_sharegpt_entry(question: str, solution: str) -> dict:
    """Build a single entry in LLaMA-Factory sharegpt format."""
    return {
        "conversations": [
            {"from": "human", "value": question},
            {"from": "gpt", "value": solution},
        ],
        "system": SYSTEM_PROMPT,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-parsed", type=int, default=100,
                        help="Minimum number of parsed samples required")
    args = parser.parse_args()

    raw_by_id, parsed, dse_results = load_all_data()

    # Only use samples that have both R-IR and DSE
    valid_ids = sorted(set(parsed.keys()) & set(dse_results.keys()))
    print(f"Raw samples: {len(raw_by_id)}")
    print(f"Parsed R-IR: {len(parsed)}")
    print(f"DSE results: {len(dse_results)}")
    print(f"Valid (both): {len(valid_ids)}")

    if len(valid_ids) < args.min_parsed:
        print(f"\nERROR: Only {len(valid_ids)} valid samples, need {args.min_parsed}.")
        print("Wait for parsing to complete, then re-run.")
        return

    # Build 5 datasets
    ds_original = []
    ds_dse = []
    ds_noverif = []
    ds_liveverif = []
    ds_keywordclean = []

    stats = {
        "total": 0, "dse_removed_steps": 0, "dse_removed_chars": 0,
        "noverif_removed_steps": 0, "liveverif_removed_steps": 0,
        "keywordclean_removed_chars": 0,
        "total_steps": 0, "total_chars": 0,
    }

    for limo_id in valid_ids:
        raw = raw_by_id[limo_id]
        rir = parsed[limo_id]
        dse = dse_results[limo_id]

        question = raw["question"]
        solution = raw["solution"]
        steps = rir["steps"]
        para_texts = rir.get("paragraph_texts", {})
        dead_ids = set(dse["dead_ids"])

        stats["total"] += 1
        stats["total_steps"] += len(steps)
        stats["total_chars"] += len(solution)

        # 1. LIMO-Original: unchanged
        ds_original.append(build_sharegpt_entry(question, solution))

        # 2. LIMO-DSE: remove all dead steps (using paragraph mappings)
        dead_step_ids = {s["id"] for s in steps if s["id"] in dead_ids}
        dse_solution = remove_steps_by_paragraphs(para_texts, steps, dead_step_ids)
        ds_dse.append(build_sharegpt_entry(question, dse_solution))
        stats["dse_removed_steps"] += len(dead_step_ids)
        stats["dse_removed_chars"] += len(solution) - len(dse_solution)

        # 3. LIMO-NoVerif: remove ALL verification steps
        verif_ids = {s["id"] for s in steps if s["type"] == "verification"}
        noverif_solution = remove_steps_by_paragraphs(para_texts, steps, verif_ids)
        ds_noverif.append(build_sharegpt_entry(question, noverif_solution))
        stats["noverif_removed_steps"] += len(verif_ids)

        # 4. LIMO-LiveVerif: remove only DEAD verification steps
        dead_verif_ids = {s["id"] for s in steps
                         if s["type"] == "verification" and s["id"] in dead_ids}
        liveverif_solution = remove_steps_by_paragraphs(para_texts, steps, dead_verif_ids)
        ds_liveverif.append(build_sharegpt_entry(question, liveverif_solution))
        stats["liveverif_removed_steps"] += len(dead_verif_ids)

        # 5. LIMO-KeywordClean: keyword-based verification removal (no structural analysis)
        keywordclean_solution = keyword_clean_solution(solution)
        ds_keywordclean.append(build_sharegpt_entry(question, keywordclean_solution))
        stats["keywordclean_removed_chars"] += len(solution) - len(keywordclean_solution)

    # Save datasets
    datasets = {
        "limo_original": ds_original,
        "limo_dse": ds_dse,
        "limo_noverif": ds_noverif,
        "limo_liveverif": ds_liveverif,
        "limo_keywordclean": ds_keywordclean,
    }

    for name, ds in datasets.items():
        out_path = CLEANED_DIR / f"{name}.json"
        out_path.write_text(json.dumps(ds, indent=2, ensure_ascii=False))
        # Compute avg solution length
        avg_len = sum(len(e["conversations"][1]["value"]) for e in ds) / len(ds) if ds else 0
        print(f"\n{name}: {len(ds)} samples, avg solution length: {avg_len:.0f} chars")
        print(f"  Saved to {out_path}")

    # Print summary stats
    n = stats["total"]
    print(f"\n{'='*60}")
    print(f"Dataset Building Summary ({n} samples)")
    print(f"  Total steps: {stats['total_steps']} ({stats['total_steps']/n:.1f}/sample)")
    print(f"  DSE removed: {stats['dse_removed_steps']} steps, {stats['dse_removed_chars']} chars")
    print(f"  NoVerif removed: {stats['noverif_removed_steps']} steps")
    print(f"  LiveVerif removed: {stats['liveverif_removed_steps']} steps")
    print(f"  KeywordClean removed: {stats['keywordclean_removed_chars']} chars")
    print(f"  Avg char reduction (DSE): {stats['dse_removed_chars']/stats['total_chars']*100:.1f}%")
    print(f"  Avg char reduction (Keyword): {stats['keywordclean_removed_chars']/stats['total_chars']*100:.1f}%")

    # Save stats
    stats_path = LIMO_DIR / "dataset_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"\nStats saved to {stats_path}")

    # Also create LLaMA-Factory dataset_info.json
    dataset_info = {}
    for name in datasets:
        dataset_info[name] = {
            "file_name": f"cleaned/{name}.json",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "system": "system"},
        }
    info_path = LIMO_DIR / "dataset_info.json"
    info_path.write_text(json.dumps(dataset_info, indent=2))
    print(f"LLaMA-Factory dataset_info saved to {info_path}")


if __name__ == "__main__":
    main()
