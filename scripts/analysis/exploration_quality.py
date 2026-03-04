"""
Exploration Quality Analysis

Core question: What determines whether an exploration/verification step
is cognitively useful (even if structurally dead)?

We analyze this by comparing DSE vs Original model outputs on problems
where they DISAGREE (one correct, other wrong). This reveals when
removing "dead" steps actually removes useful cognitive scaffolding.

Key concepts:
- Structural Dead: not reachable from conclusion in dependency DAG
- Cognitively Useful: removal changes the model's answer (empirically observed)
- True Waste: removal doesn't change (or improves) the answer

Analyses:
1. Disagreement analysis: What's different about problems where Orig wins vs DSE wins?
2. Verification utility: When do verifications actually change the reasoning trajectory?
3. Exploration depth: Is there a "right amount" of exploration per difficulty?
4. Cognitive scaffolding theory: Formalize when dead steps serve implicit functions
"""

import json
import re
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path


RESULTS_DIR = Path("training/eval/results")
LIMO_DIR = Path("data/limo/cleaned")

VERIF_PATTERNS = [
    (r'let me (check|verify|double.?check|confirm|validate)', 'explicit_verify'),
    (r'(checking|verifying|double.?checking)', 'active_verify'),
    (r'substitut(e|ing) back', 'substitution_check'),
    (r'plug(ging)? (it )?back', 'plug_back'),
    (r'this (is correct|checks out|seems right|makes sense)', 'confirmation'),
    (r'sanity check', 'sanity_check'),
]

CORRECTION_PATTERNS = [
    (r'\bwait\b', 'wait'),
    (r'\bactually\b', 'actually'),
    (r'\bno,?\s', 'negation'),
    (r'\bhold on\b', 'hold_on'),
    (r'let me re(think|consider|do|calculate)', 'rethink'),
    (r'that\'s (not right|wrong|incorrect)', 'error_recognition'),
    (r'i made (a |an )?(mistake|error)', 'mistake_admission'),
    (r'\bcorrection\b', 'explicit_correction'),
    (r'going back', 'going_back'),
]

EXPLORATION_PATTERNS = [
    (r'let me (try|consider|think about|explore)', 'try_alternative'),
    (r'another (way|approach|method)', 'alternative_approach'),
    (r'what if', 'what_if'),
    (r'alternatively', 'alternatively'),
    (r'on the other hand', 'other_hand'),
    (r'let\'s (see|look at|examine)', 'examine'),
    (r'perhaps', 'perhaps'),
    (r'maybe', 'maybe'),
    (r'i wonder', 'wonder'),
]


def extract_thinking(output: str) -> str:
    """Extract thinking content from <think> tags."""
    match = re.search(r'<think>(.*?)</think>', output, re.DOTALL)
    return match.group(1) if match else output


def count_patterns(text: str, patterns: list) -> dict:
    """Count occurrences of each pattern category."""
    counts = {}
    for regex, name in patterns:
        counts[name] = len(re.findall(regex, text, re.IGNORECASE))
    return counts


def segment_reasoning(thinking: str) -> list[dict]:
    """Segment reasoning into phases based on transition markers.
    
    Returns list of segments with type and content.
    """
    segments = []
    # Split on double newlines or explicit markers
    paragraphs = re.split(r'\n\n+', thinking)
    
    for para in paragraphs:
        if not para.strip():
            continue
        para_lower = para.lower()
        
        # Classify paragraph
        if any(re.search(p, para_lower) for p, _ in VERIF_PATTERNS):
            seg_type = "verification"
        elif any(re.search(p, para_lower) for p, _ in CORRECTION_PATTERNS):
            seg_type = "correction"
        elif any(re.search(p, para_lower) for p, _ in EXPLORATION_PATTERNS):
            seg_type = "exploration"
        elif re.search(r'\\boxed\{', para):
            seg_type = "conclusion"
        else:
            seg_type = "derivation"
        
        segments.append({
            "type": seg_type,
            "length": len(para),
            "content_preview": para[:100],
        })
    
    return segments


def load_paired_results(bench):
    """Load Original and DSE results for a benchmark."""
    orig = json.loads((RESULTS_DIR / "decor-qwen3-4b-original" / f"{bench}_results.json").read_text())
    dse = json.loads((RESULTS_DIR / "decor-qwen3-4b-dse" / f"{bench}_results.json").read_text())
    return orig, dse


# ============================================================
# Analysis 1: Disagreement Deep Dive
# ============================================================

def analysis_disagreement():
    """Deep analysis of problems where DSE and Original disagree."""
    print("=" * 90)
    print("ANALYSIS 1: Disagreement Deep Dive — When Does Removing Dead Steps Hurt?")
    print("=" * 90)

    for bench in ["math500", "gpqa"]:
        orig, dse = load_paired_results(bench)
        
        dse_wins = []  # DSE correct, Orig wrong
        orig_wins = []  # Orig correct, DSE wrong
        
        for i, (o, d) in enumerate(zip(orig, dse)):
            oc, dc = o.get("correct", False), d.get("correct", False)
            if dc and not oc:
                dse_wins.append(i)
            elif oc and not dc:
                orig_wins.append(i)
        
        print(f"\n{'─' * 80}")
        print(f"{bench.upper()}: DSE wins {len(dse_wins)}, Orig wins {len(orig_wins)}")
        print(f"{'─' * 80}")
        
        # Compare reasoning structure in disagreement cases
        def analyze_group(indices, label):
            verif_totals = Counter()
            correction_totals = Counter()
            exploration_totals = Counter()
            orig_lengths = []
            dse_lengths = []
            seg_type_counts_orig = Counter()
            seg_type_counts_dse = Counter()
            
            # Detailed: count reasoning phases
            orig_phase_counts = []
            dse_phase_counts = []
            
            for i in indices:
                o_think = extract_thinking(orig[i].get("output", ""))
                d_think = extract_thinking(dse[i].get("output", ""))
                
                orig_lengths.append(len(o_think))
                dse_lengths.append(len(d_think))
                
                # Pattern counts
                for k, v in count_patterns(o_think, VERIF_PATTERNS).items():
                    verif_totals[f"orig_{k}"] += v
                for k, v in count_patterns(d_think, VERIF_PATTERNS).items():
                    verif_totals[f"dse_{k}"] += v
                    
                for k, v in count_patterns(o_think, CORRECTION_PATTERNS).items():
                    correction_totals[f"orig_{k}"] += v
                for k, v in count_patterns(d_think, CORRECTION_PATTERNS).items():
                    correction_totals[f"dse_{k}"] += v
                
                for k, v in count_patterns(o_think, EXPLORATION_PATTERNS).items():
                    exploration_totals[f"orig_{k}"] += v
                for k, v in count_patterns(d_think, EXPLORATION_PATTERNS).items():
                    exploration_totals[f"dse_{k}"] += v
                
                # Segment analysis
                o_segs = segment_reasoning(o_think)
                d_segs = segment_reasoning(d_think)
                for s in o_segs:
                    seg_type_counts_orig[s["type"]] += 1
                for s in d_segs:
                    seg_type_counts_dse[s["type"]] += 1
                
                orig_phase_counts.append(len(o_segs))
                dse_phase_counts.append(len(d_segs))
            
            n = len(indices)
            if n == 0:
                return
            
            print(f"\n  {label} (n={n}):")
            print(f"    Avg orig length: {np.mean(orig_lengths):,.0f} chars")
            print(f"    Avg DSE length:  {np.mean(dse_lengths):,.0f} chars")
            print(f"    Avg orig phases: {np.mean(orig_phase_counts):.1f}")
            print(f"    Avg DSE phases:  {np.mean(dse_phase_counts):.1f}")
            
            # Verification density
            orig_verif = sum(v for k, v in verif_totals.items() if k.startswith("orig_"))
            dse_verif = sum(v for k, v in verif_totals.items() if k.startswith("dse_"))
            orig_corr = sum(v for k, v in correction_totals.items() if k.startswith("orig_"))
            dse_corr = sum(v for k, v in correction_totals.items() if k.startswith("dse_"))
            orig_expl = sum(v for k, v in exploration_totals.items() if k.startswith("orig_"))
            dse_expl = sum(v for k, v in exploration_totals.items() if k.startswith("dse_"))
            
            print(f"    Verif instances:  orig={orig_verif/n:.1f}/problem, dse={dse_verif/n:.1f}/problem")
            print(f"    Corrections:      orig={orig_corr/n:.1f}/problem, dse={dse_corr/n:.1f}/problem")
            print(f"    Explorations:     orig={orig_expl/n:.1f}/problem, dse={dse_expl/n:.1f}/problem")
            
            print(f"    Phase distribution (Original):")
            total_o = sum(seg_type_counts_orig.values())
            for t, c in seg_type_counts_orig.most_common():
                print(f"      {t:<15}: {c:>5} ({c/total_o*100:.1f}%)")
            print(f"    Phase distribution (DSE):")
            total_d = sum(seg_type_counts_dse.values())
            for t, c in seg_type_counts_dse.most_common():
                print(f"      {t:<15}: {c:>5} ({c/total_d*100:.1f}%)")
        
        analyze_group(orig_wins, "Original Wins (DSE hurt by removing steps)")
        analyze_group(dse_wins, "DSE Wins (Original wasted on useless steps)")
        
        # For MATH-500: show level breakdown of disagreements
        if bench == "math500":
            print(f"\n  Level breakdown of disagreements:")
            for level in ["1", "2", "3", "4", "5"]:
                ow = [i for i in orig_wins if str(orig[i].get("level")) == level]
                dw = [i for i in dse_wins if str(orig[i].get("level")) == level]
                n_level = sum(1 for o in orig if str(o.get("level")) == level)
                if n_level:
                    print(f"    Level {level} (n={n_level}): DSE wins {len(dw)}, Orig wins {len(ow)}, net={len(dw)-len(ow):+d}")


# ============================================================
# Analysis 2: Verification Utility — When does verification change outcomes?
# ============================================================

def analysis_verification_utility():
    """Analyze when verification steps actually change reasoning outcomes."""
    print("\n" + "=" * 90)
    print("ANALYSIS 2: Verification Utility — Function of Dead Verification Steps")
    print("=" * 90)
    
    for bench in ["math500", "gpqa"]:
        orig, dse = load_paired_results(bench)
        
        # For each problem, measure verification density and correctness
        results = []
        for i, (o, d) in enumerate(zip(orig, dse)):
            o_think = extract_thinking(o.get("output", ""))
            d_think = extract_thinking(d.get("output", ""))
            
            o_verif = sum(count_patterns(o_think, VERIF_PATTERNS).values())
            d_verif = sum(count_patterns(d_think, VERIF_PATTERNS).values())
            o_corr = sum(count_patterns(o_think, CORRECTION_PATTERNS).values())
            d_corr = sum(count_patterns(d_think, CORRECTION_PATTERNS).values())
            
            # Key: did Original's extra verification lead to a correction?
            # If Orig has more verif AND more corrections AND gets it right
            # while DSE doesn't → that verification was USEFUL
            verif_led_to_correction = (
                o_verif > d_verif and 
                o_corr > d_corr and 
                o.get("correct") and 
                not d.get("correct")
            )
            
            results.append({
                "idx": i,
                "orig_correct": o.get("correct", False),
                "dse_correct": d.get("correct", False),
                "orig_verif": o_verif,
                "dse_verif": d_verif,
                "orig_corr": o_corr,
                "dse_corr": d_corr,
                "orig_len": len(o_think),
                "dse_len": len(d_think),
                "verif_led_to_correction": verif_led_to_correction,
                "level": str(o.get("level", "?")),
            })
        
        print(f"\n  {bench.upper()}:")
        
        # How often does more verification → more correction?
        more_verif = [r for r in results if r["orig_verif"] > r["dse_verif"]]
        more_verif_more_corr = [r for r in more_verif if r["orig_corr"] > r["dse_corr"]]
        verif_useful = [r for r in results if r["verif_led_to_correction"]]
        
        n = len(results)
        print(f"    Problems where Orig has more verification: {len(more_verif)}/{n} ({len(more_verif)/n*100:.1f}%)")
        print(f"    Of those, where more verif → more correction: {len(more_verif_more_corr)}/{len(more_verif)} ({len(more_verif_more_corr)/max(len(more_verif),1)*100:.1f}%)")
        print(f"    Cases where verif led to correction AND changed outcome: {len(verif_useful)}/{n} ({len(verif_useful)/n*100:.1f}%)")
        
        # Verification-outcome matrix
        # High verif vs Low verif × Correct vs Wrong
        median_verif_orig = np.median([r["orig_verif"] for r in results])
        
        print(f"\n    Verification-Outcome Matrix (Original, median verif={median_verif_orig:.0f}):")
        for label, condition in [
            ("High Verif", lambda r: r["orig_verif"] > median_verif_orig),
            ("Low Verif", lambda r: r["orig_verif"] <= median_verif_orig),
        ]:
            subset = [r for r in results if condition(r)]
            n_sub = len(subset)
            acc = sum(1 for r in subset if r["orig_correct"]) / max(n_sub, 1)
            print(f"      {label} (n={n_sub}): accuracy={acc*100:.1f}%")
        
        # Same for DSE
        median_verif_dse = np.median([r["dse_verif"] for r in results])
        print(f"\n    Verification-Outcome Matrix (DSE, median verif={median_verif_dse:.0f}):")
        for label, condition in [
            ("High Verif", lambda r: r["dse_verif"] > median_verif_dse),
            ("Low Verif", lambda r: r["dse_verif"] <= median_verif_dse),
        ]:
            subset = [r for r in results if condition(r)]
            n_sub = len(subset)
            acc = sum(1 for r in subset if r["dse_correct"]) / max(n_sub, 1)
            print(f"      {label} (n={n_sub}): accuracy={acc*100:.1f}%")


# ============================================================
# Analysis 3: Exploration Depth — Right amount per difficulty
# ============================================================

def analysis_exploration_depth():
    """Find optimal exploration depth as function of problem difficulty."""
    print("\n" + "=" * 90)
    print("ANALYSIS 3: Optimal Exploration Depth vs Difficulty")
    print("=" * 90)
    
    orig, dse = load_paired_results("math500")
    
    # For each problem, compute exploration richness
    for model_name, data in [("Original", orig), ("DSE", dse)]:
        print(f"\n  {model_name}:")
        
        by_level = defaultdict(list)
        for r in data:
            think = extract_thinking(r.get("output", ""))
            segs = segment_reasoning(think)
            
            n_verif = sum(1 for s in segs if s["type"] == "verification")
            n_correction = sum(1 for s in segs if s["type"] == "correction")
            n_exploration = sum(1 for s in segs if s["type"] == "exploration")
            n_derivation = sum(1 for s in segs if s["type"] == "derivation")
            total_segs = len(segs)
            
            explore_ratio = (n_verif + n_correction + n_exploration) / max(total_segs, 1)
            
            level = str(r.get("level", "?"))
            by_level[level].append({
                "correct": r.get("correct", False),
                "n_segs": total_segs,
                "n_verif": n_verif,
                "n_correction": n_correction,
                "n_exploration": n_exploration,
                "n_derivation": n_derivation,
                "explore_ratio": explore_ratio,
                "length": len(think),
            })
        
        print(f"  {'Level':<6} {'N':>4} {'Acc':>6} {'Segs':>5} {'Verif':>6} {'Corr':>5} {'Expl':>5} {'Deriv':>6} {'ExplRatio':>10}")
        for level in sorted(by_level.keys()):
            items = by_level[level]
            n = len(items)
            acc = sum(1 for x in items if x["correct"]) / n
            print(f"  {level:<6} {n:>4} {acc*100:>5.1f}% "
                  f"{np.mean([x['n_segs'] for x in items]):>5.1f} "
                  f"{np.mean([x['n_verif'] for x in items]):>6.1f} "
                  f"{np.mean([x['n_correction'] for x in items]):>5.1f} "
                  f"{np.mean([x['n_exploration'] for x in items]):>5.1f} "
                  f"{np.mean([x['n_derivation'] for x in items]):>6.1f} "
                  f"{np.mean([x['explore_ratio'] for x in items]):>9.1%}")
        
        # Within each level: do correct answers have different exploration ratios?
        print(f"\n  Exploration ratio: Correct vs Incorrect by level")
        print(f"  {'Level':<6} {'Correct ExplR':>14} {'Wrong ExplR':>12} {'Δ':>8} {'Interpretation'}")
        for level in sorted(by_level.keys()):
            items = by_level[level]
            correct = [x for x in items if x["correct"]]
            wrong = [x for x in items if not x["correct"]]
            if correct and wrong:
                c_ratio = np.mean([x["explore_ratio"] for x in correct])
                w_ratio = np.mean([x["explore_ratio"] for x in wrong])
                delta = c_ratio - w_ratio
                interp = "more expl helps" if delta > 0.02 else "less expl helps" if delta < -0.02 else "neutral"
                print(f"  {level:<6} {c_ratio:>13.1%} {w_ratio:>11.1%} {delta:>+7.1%} {interp}")


# ============================================================
# Analysis 4: Cognitive Scaffolding Theory
# ============================================================

def analysis_cognitive_scaffolding():
    """Formalize when structurally dead steps serve implicit cognitive functions."""
    print("\n" + "=" * 90)
    print("ANALYSIS 4: Cognitive Scaffolding — Implicit Functions of Dead Steps")
    print("=" * 90)
    
    print("""
  THEORETICAL FRAMEWORK: Three Functions of "Dead" Steps
  
  F1: CONFIRMATION SCAFFOLDING
      - Step verifies current state is correct
      - Not referenced downstream (structurally dead)
      - But prevents the model from going down wrong paths
      - Analog: "looking both ways before crossing" — action has no output
        but changes the agent's confidence state
      
  F2: EXPLORATION SCAFFOLDING  
      - Step explores an alternative that is ultimately abandoned
      - Not referenced (dead), but helped the model ELIMINATE wrong paths
      - Analog: "looking at a map" — the map isn't in the final route,
        but informed the choice
      
  F3: CALIBRATION SCAFFOLDING
      - Step re-derives something already known, increasing confidence
      - Redundant (dead) but calibrates the model's uncertainty
      - Most common in hard problems where the model needs reassurance
      - Analog: "counting your money twice" — same result, but reduces error rate
  
  KEY INSIGHT: 
  These three functions are INVISIBLE to structural analysis (backward reachability)
  because they operate on the model's IMPLICIT STATE, not on the explicit token chain.
  
  DSE removes all three. This is correct for training efficiency (the model should
  learn to reason WITHOUT needing scaffolding). But on HARD problems, the model
  at INFERENCE time may need this scaffolding because it hasn't fully internalized
  the reasoning pattern yet.
  
  This explains the difficulty-dependent effect:
  - Easy problems: Model has internalized the pattern → scaffolding is waste → DSE helps
  - Hard problems: Model hasn't internalized → scaffolding compensates → DSE hurts
""")
    
    # Empirical test: measure scaffolding density by difficulty
    orig, dse = load_paired_results("math500")
    
    print("  EMPIRICAL TEST: Scaffolding density where it matters")
    print()
    
    # Focus on Orig-wins cases (where DSE removal hurt)
    orig_wins_by_level = defaultdict(list)
    dse_wins_by_level = defaultdict(list)
    
    for i, (o, d) in enumerate(zip(orig, dse)):
        level = str(o.get("level", "?"))
        o_think = extract_thinking(o.get("output", ""))
        d_think = extract_thinking(d.get("output", ""))
        
        o_segs = segment_reasoning(o_think)
        d_segs = segment_reasoning(d_think)
        
        o_scaffold = sum(1 for s in o_segs if s["type"] in ("verification", "correction", "exploration"))
        d_scaffold = sum(1 for s in d_segs if s["type"] in ("verification", "correction", "exploration"))
        
        record = {
            "orig_scaffold": o_scaffold,
            "dse_scaffold": d_scaffold,
            "scaffold_diff": o_scaffold - d_scaffold,
            "orig_len": len(o_think),
            "dse_len": len(d_think),
        }
        
        if o.get("correct") and not d.get("correct"):
            orig_wins_by_level[level].append(record)
        elif d.get("correct") and not o.get("correct"):
            dse_wins_by_level[level].append(record)
    
    print(f"  {'Level':<6} {'Orig-wins':>10} {'DSE-wins':>10} {'Orig-wins scaffold_diff':>25} {'DSE-wins scaffold_diff':>25}")
    for level in ["1", "2", "3", "4", "5"]:
        ow = orig_wins_by_level.get(level, [])
        dw = dse_wins_by_level.get(level, [])
        ow_diff = np.mean([r["scaffold_diff"] for r in ow]) if ow else 0
        dw_diff = np.mean([r["scaffold_diff"] for r in dw]) if dw else 0
        print(f"  {level:<6} {len(ow):>10} {len(dw):>10} {ow_diff:>+24.1f} {dw_diff:>+24.1f}")
    
    print(f"""
  INTERPRETATION:
  - If Orig-wins have higher scaffold_diff → Original's extra scaffolding was what saved it
  - If DSE-wins have lower scaffold_diff → Original's extra scaffolding was pure waste
  
  This directly measures WHEN scaffolding is cognitively useful vs wasteful.
""")

    # Summary: what predicts when scaffolding is useful?
    print("  PREDICTIVE FEATURES for scaffolding utility:")
    print()
    
    all_orig_wins = []
    all_dse_wins = []
    for level in orig_wins_by_level:
        all_orig_wins.extend(orig_wins_by_level[level])
    for level in dse_wins_by_level:
        all_dse_wins.extend(dse_wins_by_level[level])
    
    if all_orig_wins and all_dse_wins:
        print(f"  Where scaffolding was USEFUL (Orig wins, n={len(all_orig_wins)}):")
        print(f"    Avg scaffold diff (Orig - DSE): {np.mean([r['scaffold_diff'] for r in all_orig_wins]):+.1f}")
        print(f"    Avg orig length: {np.mean([r['orig_len'] for r in all_orig_wins]):,.0f}")
        print(f"    Avg DSE length:  {np.mean([r['dse_len'] for r in all_orig_wins]):,.0f}")
        
        print(f"\n  Where scaffolding was WASTEFUL (DSE wins, n={len(all_dse_wins)}):")
        print(f"    Avg scaffold diff (Orig - DSE): {np.mean([r['scaffold_diff'] for r in all_dse_wins]):+.1f}")
        print(f"    Avg orig length: {np.mean([r['orig_len'] for r in all_dse_wins]):,.0f}")
        print(f"    Avg DSE length:  {np.mean([r['dse_len'] for r in all_dse_wins]):,.0f}")


# ============================================================
# Analysis 5: Implications for Training
# ============================================================

def analysis_training_implications():
    """What does this mean for how we should build training data?"""
    print("\n" + "=" * 90)
    print("ANALYSIS 5: Implications for Training Data Construction")
    print("=" * 90)
    
    orig_data = json.load(open(LIMO_DIR / "limo_original.json"))
    dse_data = json.load(open(LIMO_DIR / "limo_dse.json"))
    
    n = len(orig_data)
    orig_lens = [len(d['conversations'][1]['value']) for d in orig_data]
    dse_lens = [len(d['conversations'][1]['value']) for d in dse_data]
    reductions = [(o - d) / o * 100 if o > 0 else 0 for o, d in zip(orig_lens, dse_lens)]
    
    # Classify samples by DSE impact
    no_change = sum(1 for r in reductions if r < 5)
    light_dse = sum(1 for r in reductions if 5 <= r < 20)
    medium_dse = sum(1 for r in reductions if 20 <= r < 40)
    heavy_dse = sum(1 for r in reductions if r >= 40)
    
    print(f"\n  DSE Impact Distribution on Training Data ({n} samples):")
    print(f"    No change (<5% reduction):   {no_change:>4} ({no_change/n*100:.1f}%) — already clean")
    print(f"    Light DSE (5-20% reduction):  {light_dse:>4} ({light_dse/n*100:.1f}%) — minor cleanup")
    print(f"    Medium DSE (20-40% reduction): {medium_dse:>4} ({medium_dse/n*100:.1f}%) — significant")
    print(f"    Heavy DSE (40%+ reduction):   {heavy_dse:>4} ({heavy_dse/n*100:.1f}%) — aggressive pruning")
    
    print(f"""
  TRAINING STRATEGY RECOMMENDATIONS:
  
  Strategy 1: CONSERVATIVE DSE (safest, minimal risk)
    - Only apply DSE to samples with reduction < 20% (light DSE)
    - Heavy DSE samples → keep original (might remove useful scaffolding)
    - Expected: preserve hard-problem scaffolding, still clean easy problems
    - Samples: {no_change + light_dse} DSE + {medium_dse + heavy_dse} Original
    
  Strategy 2: TYPED DSE (most principled)
    - Remove ONLY these dead step types: redundant computation, repeated derivation
    - KEEP these even if dead: verification, exploration, correction
    - Requires step-type classification (from R-IR parsing)
    - This preserves all three scaffolding functions
    - Best suited for a system with DeepSeek API access
    
  Strategy 3: DIFFICULTY-AWARE DSE (requires difficulty labels)
    - Easy problems (short solutions): full DSE
    - Hard problems (long solutions): no DSE or light DSE
    - Threshold: solution length percentile P75
    - Simple but requires reliable difficulty estimation
    
  Strategy 4: CALIBRATED DSE (most sophisticated, future work)
    - For each training sample, estimate P(correct | DSE) vs P(correct | original)
    - Apply DSE only when estimated P(correct | DSE) ≥ P(correct | original)  
    - Requires a calibration model or held-out validation
    - This is essentially "learning when to prune"
    
  THE FUNDAMENTAL INSIGHT:
    Current DSE treats ALL dead steps as waste. But some dead steps serve as
    COGNITIVE SCAFFOLDING that helps the model navigate uncertain territory.
    The optimal approach is to remove TRUE waste while preserving scaffolding.
    
    The three scaffolding functions (confirmation, exploration, calibration)
    are all MORE important for hard problems. This explains why DSE hurts
    on Level 5 and GPQA but helps on Level 1-4.
    
    For the COLM paper, this analysis provides a clear explanation for the
    GPQA trade-off and points to Typed DSE / Calibrated DSE as future work.
""")

    # Quantify potential
    print(f"  Potential of Strategy 1 (Conservative DSE):")
    conservative_lens = [
        dse_lens[i] if reductions[i] < 20 else orig_lens[i]
        for i in range(n)
    ]
    avg_conservative = np.mean(conservative_lens)
    avg_orig = np.mean(orig_lens)
    avg_dse = np.mean(dse_lens)
    print(f"    Avg length: Original={avg_orig:.0f}, Full-DSE={avg_dse:.0f}, Conservative={avg_conservative:.0f}")
    print(f"    Reduction vs Original: Full-DSE={100*(1-avg_dse/avg_orig):.1f}%, Conservative={100*(1-avg_conservative/avg_orig):.1f}%")
    

def main():
    analysis_disagreement()
    analysis_verification_utility()
    analysis_exploration_depth()
    analysis_cognitive_scaffolding()
    analysis_training_implications()
    
    print("\n" + "=" * 90)
    print("COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()
