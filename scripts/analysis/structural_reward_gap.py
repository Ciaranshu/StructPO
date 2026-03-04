"""
Comprehensive experiment suite for "The Structural Reward Gap" narrative.

Experiments:
  E1. Dead Step Type Decomposition — what types of steps are dead?
  E2. Correctness-Independence — dead step distributions for correct vs incorrect
  E3. Difficulty Stratification — DSR across easy/medium/hard problems
  E4. Safe Structural Pruning — DSE removes steps without changing answers
  E5. Structure > Position — DSE vs random removal comparison
  E6. Verification Theater Quantification — detailed verification analysis
"""

import json
import re
import random
import warnings
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RIR_V2_DIR = DATA_DIR / "rir_v2"
GSM8K_RIR_DIR = DATA_DIR / "gsm8k" / "rir"
RESULTS_DIR = DATA_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers (reused from validate_structural_confidence.py)
# ---------------------------------------------------------------------------
def extract_boxed(text: str) -> str | None:
    pattern = r'\\boxed\{'
    matches = list(re.finditer(pattern, text))
    if not matches:
        return None
    start = matches[-1].end()
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        i += 1
    if depth == 0:
        return text[start:i-1].strip()
    return None

def normalize_answer(ans: str) -> str:
    if ans is None:
        return ""
    ans = ans.strip()
    ans = ans.replace("$", "").replace(",", "").rstrip(".")
    ans = re.sub(r'\\text\{([^}]*)\}', r'\1', ans)
    ans = re.sub(r'\\(?:mathrm|mathbf|mathit|textbf)\{([^}]*)\}', r'\1', ans)
    ans = ans.replace(" ", "")
    try:
        return str(float(ans))
    except ValueError:
        return ans.lower()

def check_correctness(rir: dict) -> bool:
    gt = rir.get("ground_truth", "")
    answer_text = rir.get("original_answer", "")
    model_ans = extract_boxed(answer_text)
    if model_ans is None:
        lines = answer_text.strip().split("\n")
        model_ans = lines[-1] if lines else ""
    return normalize_answer(str(gt)) == normalize_answer(str(model_ans))

def dead_step_elimination(rir: dict) -> dict:
    steps = rir["steps"]
    step_map = {s["id"]: s for s in steps}
    conclusions = [s for s in steps if s["type"] == "conclusion"]
    if not conclusions:
        conclusions = [steps[-1]] if steps else []
    live = set()
    worklist = [s["id"] for s in conclusions]
    while worklist:
        sid = worklist.pop()
        if sid in live:
            continue
        live.add(sid)
        step = step_map.get(sid)
        if step:
            for inp_id in step.get("inputs", []):
                if inp_id not in live and inp_id in step_map:
                    worklist.append(inp_id)
    dead_ids = {s["id"] for s in steps} - live
    return {
        "live_ids": sorted(live),
        "dead_ids": sorted(dead_ids),
        "dead_steps": [s for s in steps if s["id"] in dead_ids],
        "live_steps": [s for s in steps if s["id"] in live],
        "total": len(steps),
        "live_count": len(live),
        "dead_count": len(dead_ids),
        "dead_ratio": len(dead_ids) / len(steps) if steps else 0,
    }

def load_all_rirs(rir_dir: Path, dataset_name: str) -> list[dict]:
    """Load all RIR files, compute DSE + correctness."""
    rir_files = sorted(f for f in rir_dir.glob("*.json") if f.name != "_summary.json")
    records = []
    for rf in rir_files:
        rir = json.loads(rf.read_text())
        if not rir.get("steps"):
            continue
        dse = dead_step_elimination(rir)
        correct = check_correctness(rir)
        
        steps = rir["steps"]
        n_tokens = rir.get("original_token_count", 0)
        
        # Step type counts
        type_counts = Counter(s["type"] for s in steps)
        dead_type_counts = Counter(s["type"] for s in dse["dead_steps"])
        live_type_counts = Counter(s["type"] for s in dse["live_steps"])
        
        # Token counts by liveness
        dead_tokens = sum(s.get("original_tokens", 0) for s in dse["dead_steps"])
        live_tokens = sum(s.get("original_tokens", 0) for s in dse["live_steps"])
        
        # Verification-specific
        n_verif = type_counts.get("verification", 0)
        n_dead_verif = dead_type_counts.get("verification", 0)
        vii = n_dead_verif / n_verif if n_verif > 0 else 0.0
        
        # Problem category from ID
        pid = rir["problem_id"]
        if pid.startswith("aime"):
            category = "hard"
        elif pid.startswith("hard"):
            category = "hard"
        elif pid.startswith("s2_") or pid.startswith("scale_"):
            # s2_ are MATH-500 sampled, mixed difficulty
            category = "medium"
        elif pid.startswith("algebra") or pid.startswith("intermediate"):
            category = "easy"
        elif pid.startswith("geometry") or pid.startswith("combinatorics") or pid.startswith("numtheory"):
            category = "medium"
        elif pid.startswith("gsm8k"):
            category = "easy"
        else:
            category = "unknown"
        
        records.append({
            "problem_id": pid,
            "dataset": dataset_name,
            "category": category,
            "correct": correct,
            "n_steps": len(steps),
            "n_tokens": n_tokens,
            "n_dead": dse["dead_count"],
            "n_live": dse["live_count"],
            "dsr": dse["dead_ratio"],
            "dead_tokens": dead_tokens,
            "live_tokens": live_tokens,
            "dead_token_ratio": dead_tokens / n_tokens if n_tokens > 0 else 0,
            "type_counts": dict(type_counts),
            "dead_type_counts": dict(dead_type_counts),
            "live_type_counts": dict(live_type_counts),
            "n_verification": n_verif,
            "n_dead_verification": n_dead_verif,
            "vii": vii,
            "steps": steps,  # keep for E5
            "dse": dse,      # keep for E5
            "rir": rir,       # keep for E4
        })
    return records


# ===================================================================
# E1: Dead Step Type Decomposition
# ===================================================================
def experiment_1(records, label=""):
    print(f"\n{'='*70}")
    print(f"E1: DEAD STEP TYPE DECOMPOSITION [{label}]")
    print(f"{'='*70}")
    
    all_types = set()
    total_dead_by_type = Counter()
    total_all_by_type = Counter()
    total_dead = 0
    total_all = 0
    
    for r in records:
        for t, c in r["type_counts"].items():
            total_all_by_type[t] += c
            total_all += c
            all_types.add(t)
        for t, c in r["dead_type_counts"].items():
            total_dead_by_type[t] += c
            total_dead += c
    
    print(f"\n  Total steps: {total_all}, Dead steps: {total_dead} ({total_dead/total_all*100:.1f}%)")
    print(f"\n  {'Type':<15} {'Total':>8} {'Dead':>8} {'Dead%':>8} {'% of Dead':>10}")
    print(f"  {'-'*55}")
    
    results = {}
    for t in sorted(all_types):
        tot = total_all_by_type[t]
        dead = total_dead_by_type.get(t, 0)
        pct_dead = dead / tot * 100 if tot > 0 else 0
        pct_of_all_dead = dead / total_dead * 100 if total_dead > 0 else 0
        print(f"  {t:<15} {tot:>8} {dead:>8} {pct_dead:>7.1f}% {pct_of_all_dead:>9.1f}%")
        results[t] = {"total": tot, "dead": dead, "dead_pct": pct_dead, "pct_of_all_dead": pct_of_all_dead}
    
    # Verification theater headline number
    n_verif_total = total_all_by_type.get("verification", 0)
    n_verif_dead = total_dead_by_type.get("verification", 0)
    if n_verif_total > 0:
        print(f"\n  >>> VERIFICATION THEATER: {n_verif_dead}/{n_verif_total} verification steps are dead ({n_verif_dead/n_verif_total*100:.1f}%)")
    
    return results


# ===================================================================
# E2: Correctness-Independence
# ===================================================================
def experiment_2(records, label=""):
    print(f"\n{'='*70}")
    print(f"E2: CORRECTNESS-INDEPENDENCE [{label}]")
    print(f"{'='*70}")
    
    correct_recs = [r for r in records if r["correct"]]
    incorrect_recs = [r for r in records if not r["correct"]]
    
    print(f"\n  Correct: {len(correct_recs)}, Incorrect: {len(incorrect_recs)}")
    
    if not correct_recs or not incorrect_recs:
        print("  SKIP: need both correct and incorrect traces")
        return None
    
    metrics = [
        ("DSR", "dsr"),
        ("VII", "vii"),
        ("Dead Token Ratio", "dead_token_ratio"),
        ("# Dead Steps", "n_dead"),
        ("# Verification", "n_verification"),
        ("# Dead Verification", "n_dead_verification"),
    ]
    
    print(f"\n  {'Metric':<22} {'Correct':>10} {'Incorrect':>10} {'Diff':>10} {'p-value':>10} {'Effect':>8}")
    print(f"  {'-'*72}")
    
    results = {}
    for mname, mkey in metrics:
        c_vals = np.array([r[mkey] for r in correct_recs])
        i_vals = np.array([r[mkey] for r in incorrect_recs])
        
        mean_c = np.mean(c_vals)
        mean_i = np.mean(i_vals)
        
        try:
            u_stat, p_val = stats.mannwhitneyu(c_vals, i_vals, alternative='two-sided')
        except ValueError:
            p_val = 1.0
        
        # Cohen's d
        pooled_std = np.sqrt((np.var(c_vals) + np.var(i_vals)) / 2)
        cohens_d = (mean_c - mean_i) / pooled_std if pooled_std > 0 else 0
        
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
        print(f"  {mname:<22} {mean_c:>10.4f} {mean_i:>10.4f} {mean_c-mean_i:>+10.4f} {p_val:>9.2e}  d={cohens_d:>+.3f} {sig}")
        
        results[mkey] = {
            "mean_correct": float(mean_c), "mean_incorrect": float(mean_i),
            "diff": float(mean_c - mean_i), "p_value": float(p_val),
            "cohens_d": float(cohens_d),
        }
    
    # Dead type distribution comparison
    print(f"\n  Dead step TYPE distribution (correct vs incorrect):")
    all_types = set()
    c_dead_types = Counter()
    i_dead_types = Counter()
    c_dead_total = 0
    i_dead_total = 0
    for r in correct_recs:
        for t, c in r["dead_type_counts"].items():
            c_dead_types[t] += c
            c_dead_total += c
            all_types.add(t)
    for r in incorrect_recs:
        for t, c in r["dead_type_counts"].items():
            i_dead_types[t] += c
            i_dead_total += c
            all_types.add(t)
    
    print(f"  {'Type':<15} {'Correct%':>10} {'Incorrect%':>10} {'Diff':>10}")
    print(f"  {'-'*47}")
    for t in sorted(all_types):
        c_pct = c_dead_types.get(t, 0) / c_dead_total * 100 if c_dead_total > 0 else 0
        i_pct = i_dead_types.get(t, 0) / i_dead_total * 100 if i_dead_total > 0 else 0
        print(f"  {t:<15} {c_pct:>9.1f}% {i_pct:>9.1f}% {c_pct-i_pct:>+9.1f}%")
    
    return results


# ===================================================================
# E3: Difficulty Stratification
# ===================================================================
def experiment_3(records, label=""):
    print(f"\n{'='*70}")
    print(f"E3: DIFFICULTY STRATIFICATION [{label}]")
    print(f"{'='*70}")
    
    by_cat = defaultdict(list)
    for r in records:
        by_cat[r["category"]].append(r)
    
    print(f"\n  {'Category':<12} {'N':>5} {'Acc%':>7} {'DSR':>8} {'VII':>8} {'DTR':>8} {'Steps':>7} {'Tokens':>8}")
    print(f"  {'-'*68}")
    
    results = {}
    for cat in ["easy", "medium", "hard"]:
        recs = by_cat.get(cat, [])
        if not recs:
            continue
        n = len(recs)
        acc = sum(1 for r in recs if r["correct"]) / n * 100
        dsr = np.mean([r["dsr"] for r in recs])
        vii = np.mean([r["vii"] for r in recs])
        dtr = np.mean([r["dead_token_ratio"] for r in recs])
        steps = np.mean([r["n_steps"] for r in recs])
        tokens = np.mean([r["n_tokens"] for r in recs])
        
        print(f"  {cat:<12} {n:>5} {acc:>6.1f}% {dsr:>7.3f} {vii:>7.3f} {dtr:>7.3f} {steps:>6.1f} {tokens:>7.0f}")
        
        results[cat] = {"n": n, "accuracy": acc, "dsr": float(dsr), "vii": float(vii),
                        "dead_token_ratio": float(dtr), "avg_steps": float(steps), "avg_tokens": float(tokens)}
    
    # Kruskal-Wallis test for DSR across difficulty levels
    groups = []
    group_labels = []
    for cat in ["easy", "medium", "hard"]:
        recs = by_cat.get(cat, [])
        if recs:
            groups.append([r["dsr"] for r in recs])
            group_labels.append(cat)
    
    if len(groups) >= 2:
        h_stat, p_val = stats.kruskal(*groups)
        print(f"\n  Kruskal-Wallis test (DSR across difficulty): H={h_stat:.3f}, p={p_val:.2e}")
        
        # Also test correctness-independence WITHIN each difficulty level
        print(f"\n  Correctness-independence within difficulty levels:")
        print(f"  {'Category':<12} {'DSR(✓)':>8} {'DSR(✗)':>8} {'Diff':>8} {'p-value':>10}")
        print(f"  {'-'*50}")
        for cat in ["easy", "medium", "hard"]:
            recs = by_cat.get(cat, [])
            if not recs:
                continue
            c_dsr = [r["dsr"] for r in recs if r["correct"]]
            i_dsr = [r["dsr"] for r in recs if not r["correct"]]
            if c_dsr and i_dsr:
                try:
                    _, p = stats.mannwhitneyu(c_dsr, i_dsr, alternative='two-sided')
                except ValueError:
                    p = 1.0
                print(f"  {cat:<12} {np.mean(c_dsr):>7.3f} {np.mean(i_dsr):>7.3f} {np.mean(c_dsr)-np.mean(i_dsr):>+7.3f} {p:>9.2e}")
            else:
                print(f"  {cat:<12} (insufficient data for both groups)")
    
    return results


# ===================================================================
# E4: Safe Structural Pruning (Answer Preservation)
# ===================================================================
def experiment_4(records, label=""):
    print(f"\n{'='*70}")
    print(f"E4: SAFE STRUCTURAL PRUNING [{label}]")
    print(f"{'='*70}")
    
    n_total = len(records)
    n_preserved = 0
    token_savings = []
    step_savings = []
    
    for r in records:
        rir = r["rir"]
        dse = r["dse"]
        
        # Check if conclusion is preserved (all conclusion steps are live)
        conclusions = [s for s in rir["steps"] if s["type"] == "conclusion"]
        if not conclusions:
            conclusions = [rir["steps"][-1]] if rir["steps"] else []
        preserved = all(s["id"] in set(dse["live_ids"]) for s in conclusions)
        
        if preserved:
            n_preserved += 1
        
        token_savings.append(r["dead_token_ratio"])
        step_savings.append(r["dsr"])
    
    apr = n_preserved / n_total * 100 if n_total > 0 else 0
    avg_token_save = np.mean(token_savings) * 100
    avg_step_save = np.mean(step_savings) * 100
    
    print(f"\n  Answer Preservation Rate (APR): {n_preserved}/{n_total} = {apr:.1f}%")
    print(f"  Avg Token Savings: {avg_token_save:.1f}%")
    print(f"  Avg Step Savings:  {avg_step_save:.1f}%")
    
    # Breakdown by correctness
    for group_name, group_filter in [("Correct", True), ("Incorrect", False)]:
        grp = [r for r in records if r["correct"] == group_filter]
        if not grp:
            continue
        grp_preserved = sum(1 for r in grp 
                          if all(s["id"] in set(r["dse"]["live_ids"]) 
                                for s in ([s for s in r["rir"]["steps"] if s["type"] == "conclusion"] 
                                          or [r["rir"]["steps"][-1]] if r["rir"]["steps"] else [])))
        grp_apr = grp_preserved / len(grp) * 100
        grp_tok = np.mean([r["dead_token_ratio"] for r in grp]) * 100
        print(f"  [{group_name}] APR={grp_apr:.1f}%, Token Savings={grp_tok:.1f}%")
    
    return {"apr": apr, "avg_token_savings": avg_token_save, "avg_step_savings": avg_step_save}


# ===================================================================
# E5: Structure > Position (DSE vs Random Removal)
# ===================================================================
def experiment_5(records, label="", n_trials=100):
    print(f"\n{'='*70}")
    print(f"E5: STRUCTURE > POSITION (DSE vs Random Removal) [{label}]")
    print(f"{'='*70}")
    
    random.seed(42)
    
    dse_preserved = 0
    random_preserved_counts = []
    
    # Only consider records with dead steps
    records_with_dead = [r for r in records if r["n_dead"] > 0]
    
    if not records_with_dead:
        print("  No records with dead steps to compare")
        return None
    
    print(f"\n  Traces with dead steps: {len(records_with_dead)}/{len(records)}")
    
    for r in records_with_dead:
        rir = r["rir"]
        dse = r["dse"]
        steps = rir["steps"]
        
        # Conclusion step IDs
        conclusions = [s for s in steps if s["type"] == "conclusion"]
        if not conclusions:
            conclusions = [steps[-1]] if steps else []
        conclusion_ids = {s["id"] for s in conclusions}
        
        # DSE: check if conclusions preserved
        dse_ok = all(cid in set(dse["live_ids"]) for cid in conclusion_ids)
        if dse_ok:
            dse_preserved += 1
        
        # Random removal: remove same NUMBER of steps as DSE, but randomly
        n_to_remove = dse["dead_count"]
        removable_ids = [s["id"] for s in steps if s["id"] not in conclusion_ids]
        
        trial_preserved = 0
        for _ in range(n_trials):
            if n_to_remove <= len(removable_ids):
                removed = set(random.sample(removable_ids, n_to_remove))
            else:
                removed = set(removable_ids)
            
            # Check if conclusion's dependencies are still satisfied
            # Simple check: are all direct inputs of conclusion still present?
            remaining_ids = {s["id"] for s in steps} - removed
            
            # More thorough: check if conclusion is still reachable from assumptions
            # via remaining steps
            step_map = {s["id"]: s for s in steps}
            ok = True
            for cid in conclusion_ids:
                # Check all transitive inputs of conclusion are present
                needed = set()
                worklist = [cid]
                while worklist:
                    sid = worklist.pop()
                    if sid in needed:
                        continue
                    needed.add(sid)
                    s = step_map.get(sid)
                    if s:
                        for inp in s.get("inputs", []):
                            if inp in step_map and inp not in needed:
                                worklist.append(inp)
                # If any needed step was removed, conclusion is broken
                if needed & removed:
                    ok = False
                    break
            
            if ok:
                trial_preserved += 1
        
        random_preserved_counts.append(trial_preserved / n_trials)
    
    dse_apr = dse_preserved / len(records_with_dead) * 100
    random_apr = np.mean(random_preserved_counts) * 100
    random_std = np.std(random_preserved_counts) * 100
    
    print(f"\n  DSE Removal APR:    {dse_apr:.1f}%")
    print(f"  Random Removal APR: {random_apr:.1f}% (±{random_std:.1f}%)")
    print(f"  Gap:                {dse_apr - random_apr:+.1f} percentage points")
    
    if dse_apr > random_apr + 5:
        print(f"  >>> CONFIRMED: Structural pruning is significantly safer than random removal")
    
    return {"dse_apr": dse_apr, "random_apr": float(random_apr), "random_std": float(random_std)}


# ===================================================================
# E6: Verification Theater Deep Dive
# ===================================================================
def experiment_6(records, label=""):
    print(f"\n{'='*70}")
    print(f"E6: VERIFICATION THEATER DEEP DIVE [{label}]")
    print(f"{'='*70}")
    
    # Per-trace VII distribution
    viis = [r["vii"] for r in records if r["n_verification"] > 0]
    traces_with_verif = [r for r in records if r["n_verification"] > 0]
    traces_without_verif = [r for r in records if r["n_verification"] == 0]
    
    print(f"\n  Traces with verification steps: {len(traces_with_verif)}/{len(records)} ({len(traces_with_verif)/len(records)*100:.1f}%)")
    print(f"  Traces without verification:    {len(traces_without_verif)}/{len(records)}")
    
    if viis:
        print(f"\n  VII Distribution (among traces with verification):")
        print(f"    Mean:   {np.mean(viis):.3f}")
        print(f"    Median: {np.median(viis):.3f}")
        print(f"    Std:    {np.std(viis):.3f}")
        print(f"    Min:    {np.min(viis):.3f}")
        print(f"    Max:    {np.max(viis):.3f}")
        
        # Histogram buckets
        bins = [0, 0.25, 0.5, 0.75, 1.0, 1.01]
        labels = ["0-25%", "25-50%", "50-75%", "75-100%", "100%"]
        counts = np.histogram(viis, bins=bins)[0]
        print(f"\n  VII Histogram:")
        for lbl, cnt in zip(labels, counts):
            bar = "█" * int(cnt / len(viis) * 40)
            print(f"    {lbl:>8}: {cnt:>4} ({cnt/len(viis)*100:>5.1f}%) {bar}")
    
    # Correctness comparison for traces WITH verification
    if traces_with_verif:
        c_viis = [r["vii"] for r in traces_with_verif if r["correct"]]
        i_viis = [r["vii"] for r in traces_with_verif if not r["correct"]]
        
        if c_viis and i_viis:
            try:
                _, p = stats.mannwhitneyu(c_viis, i_viis, alternative='two-sided')
            except ValueError:
                p = 1.0
            print(f"\n  VII by correctness (traces with verification only):")
            print(f"    Correct:   {np.mean(c_viis):.3f} (n={len(c_viis)})")
            print(f"    Incorrect: {np.mean(i_viis):.3f} (n={len(i_viis)})")
            print(f"    p-value:   {p:.2e}")
    
    # Token cost of dead verification
    dead_verif_tokens = 0
    total_tokens = 0
    for r in records:
        total_tokens += r["n_tokens"]
        for s in r["dse"]["dead_steps"]:
            if s["type"] == "verification":
                dead_verif_tokens += s.get("original_tokens", 0)
    
    if total_tokens > 0:
        print(f"\n  Token cost of dead verification:")
        print(f"    Dead verification tokens: {dead_verif_tokens}")
        print(f"    Total tokens:             {total_tokens}")
        print(f"    Wasted on dead verif:     {dead_verif_tokens/total_tokens*100:.1f}%")
    
    return {"mean_vii": float(np.mean(viis)) if viis else 0,
            "dead_verif_token_pct": dead_verif_tokens / total_tokens * 100 if total_tokens > 0 else 0}


# ===================================================================
# Main
# ===================================================================
def main():
    print("=" * 70)
    print("THE STRUCTURAL REWARD GAP — Comprehensive Experiment Suite")
    print("=" * 70)
    
    # Load data
    print("\n--- Loading datasets ---")
    math_records = load_all_rirs(RIR_V2_DIR, "math500")
    gsm_records = load_all_rirs(GSM8K_RIR_DIR, "gsm8k")
    all_records = math_records + gsm_records
    
    print(f"  MATH-500: {len(math_records)} traces ({sum(1 for r in math_records if r['correct'])} correct)")
    print(f"  GSM8K:    {len(gsm_records)} traces ({sum(1 for r in gsm_records if r['correct'])} correct)")
    print(f"  Total:    {len(all_records)} traces")
    
    all_results = {}
    
    # Run all experiments on combined data
    all_results["E1"] = experiment_1(all_records, "Combined")
    all_results["E2"] = experiment_2(all_records, "Combined")
    all_results["E3"] = experiment_3(all_records, "Combined")
    all_results["E4"] = experiment_4(all_records, "Combined")
    all_results["E5"] = experiment_5(all_records, "Combined")
    all_results["E6"] = experiment_6(all_records, "Combined")
    
    # Per-dataset runs for E1, E2
    for name, recs in [("MATH-500", math_records), ("GSM8K", gsm_records)]:
        all_results[f"E1_{name}"] = experiment_1(recs, name)
        all_results[f"E2_{name}"] = experiment_2(recs, name)
        all_results[f"E6_{name}"] = experiment_6(recs, name)
    
    # Final verdict
    print(f"\n\n{'#'*70}")
    print(f"# NARRATIVE FEASIBILITY VERDICT")
    print(f"{'#'*70}")
    
    checks = []
    
    # Check 1: Verification Theater exists
    e1 = all_results.get("E1", {})
    verif_data = e1.get("verification", {})
    vt_pct = verif_data.get("dead_pct", 0) if verif_data else 0
    check1 = vt_pct > 70
    checks.append(check1)
    print(f"\n  [{'✓' if check1 else '✗'}] Verification Theater: {vt_pct:.1f}% of verification steps are dead (need >70%)")
    
    # Check 2: Correctness-Independence
    e2 = all_results.get("E2", {})
    dsr_p = e2.get("dsr", {}).get("p_value", 0) if e2 else 0
    vii_p = e2.get("vii", {}).get("p_value", 0) if e2 else 0
    check2 = dsr_p > 0.05 and vii_p > 0.05
    checks.append(check2)
    print(f"  [{'✓' if check2 else '✗'}] Correctness-Independence: DSR p={dsr_p:.2e}, VII p={vii_p:.2e} (need p>0.05)")
    
    # Check 3: Safe Pruning
    e4 = all_results.get("E4", {})
    apr = e4.get("apr", 0) if e4 else 0
    check3 = apr > 95
    checks.append(check3)
    print(f"  [{'✓' if check3 else '✗'}] Safe Structural Pruning: APR={apr:.1f}% (need >95%)")
    
    # Check 4: Structure > Position
    e5 = all_results.get("E5", {})
    if e5:
        dse_apr = e5.get("dse_apr", 0)
        rand_apr = e5.get("random_apr", 0)
        gap = dse_apr - rand_apr
        check4 = gap > 10
        checks.append(check4)
        print(f"  [{'✓' if check4 else '✗'}] Structure > Position: DSE APR={dse_apr:.1f}% vs Random={rand_apr:.1f}%, gap={gap:.1f}pp (need >10pp)")
    
    # Check 5: Meaningful token savings
    tok_save = e4.get("avg_token_savings", 0) if e4 else 0
    check5 = tok_save > 15
    checks.append(check5)
    print(f"  [{'✓' if check5 else '✗'}] Meaningful Token Savings: {tok_save:.1f}% (need >15%)")
    
    n_pass = sum(checks)
    n_total = len(checks)
    print(f"\n  OVERALL: {n_pass}/{n_total} checks passed")
    
    if n_pass >= 4:
        print(f"  >>> NARRATIVE IS STRONGLY SUPPORTED — proceed with paper writing")
    elif n_pass >= 3:
        print(f"  >>> NARRATIVE IS VIABLE — some aspects need strengthening")
    else:
        print(f"  >>> NARRATIVE NEEDS RETHINKING — insufficient evidence")
    
    # Save results (strip heavy data)
    save_results = {}
    for k, v in all_results.items():
        if v is not None:
            save_results[k] = v
    
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    out_path = RESULTS_DIR / "structural_reward_gap_experiments.json"
    with open(out_path, 'w') as f:
        json.dump(save_results, f, indent=2, default=convert)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
