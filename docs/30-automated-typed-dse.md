# Automated Typed DSE: Making Dead Step Elimination Smarter

**Date**: 2026-03-02
**Purpose**: Design analysis for automating the decision of *what* to eliminate, moving beyond binary dead/live classification.

---

## 1. Problem Statement

Current DSE is binary: a step is either **live** (reachable from conclusion) or **dead** (unreachable). All dead steps are removed. But our exploration quality analysis (doc/29) showed:

- **MATH**: removing dead steps helps (+3.2% accuracy, −18.5% length)
- **GPQA**: removing dead steps hurts (−6.6% accuracy)
- **MATH L5**: marginal hurt (−4.5%)

The issue: some structurally dead steps serve **cognitive scaffolding** functions that are invisible to dependency analysis. We need an algorithm that can distinguish **true waste** from **useful scaffolding**.

---

## 2. Three Approaches (from Algorithm Modification)

### Approach A: Weighted Reachability (Modify the DSE Algorithm Itself)

**Core idea**: Instead of binary reachability (live/dead), compute a **continuous liveness score** for each step.

Current DSE:
```
live(s) = 1 if reachable from conclusion, else 0
```

Weighted DSE:
```
liveness(s) = Σ_c∈conclusions  decay^(dist(s,c)) × importance(c)
```

Where `dist(s,c)` = shortest path length from step s to conclusion c in the dependency DAG, and `decay` ∈ (0,1) is a hyperparameter.

**Effect**: Steps that are "almost live" (one missing edge away from the conclusion) get high liveness scores. Steps that are deeply disconnected get low scores. We then threshold: remove only steps with liveness < τ.

**Why this could work**: A verification step that checks an intermediate result is typically 1-2 edges away from the derivation chain. If the parser missed one edge, it becomes "dead" even though it's actually close to the live chain. Weighted reachability is robust to missing edges.

**Limitation**: Doesn't distinguish step *types* — a redundant computation and a useful verification both get the same score if they're equidistant from conclusions.

---

### Approach B: Type-Conditional DSE (Different Rules per Step Type)

**Core idea**: Apply different elimination policies based on step type.

```python
def typed_dse(step, is_structurally_dead):
    if not is_structurally_dead:
        return KEEP  # Live steps always kept
    
    match step.type:
        case "computation":
            return REMOVE  # Dead computation = pure waste
        case "derivation":
            return REMOVE  # Dead derivation = redundant logic
        case "verification":
            return KEEP    # Dead verification = potential scaffolding
        case "exploration":  
            return KEEP    # Dead exploration = potential search value
        case "assumption":
            return KEEP    # Dead assumption = context
        case "conclusion":
            return KEEP    # Never remove conclusions
```

**Why this could work**: Our data shows that verification has domain-dependent value (waste on math, useful on GPQA). By keeping dead verification, we preserve the deliberative reasoning that GPQA needs.

**Key advantage**: This is the simplest change — zero new models, zero API calls, just a 5-line policy change in the existing DSE code.

**Limitation**: Requires accurate step type classification (currently from DeepSeek API parser). Also, the policy is static — it doesn't learn from data.

---

### Approach C: Learned Elimination (Train a Classifier)

**Core idea**: Train a lightweight classifier to predict whether removing a dead step will change the answer.

**Architecture**:
```
Input: (step_content, step_type, step_context, problem_type) 
→ Binary classifier 
→ P(removal_changes_answer)
```

If P(removal_changes_answer) > threshold → KEEP, else REMOVE.

**Training data**: We already HAVE the data to train this:
- LIMO-Original and LIMO-DSE trained models
- Per-problem correctness for both
- We can identify exactly which problems DSE hurt (Original correct, DSE wrong)
- For those problems, the dead steps that were removed are the "useful scaffolding"

**Limitation**: Requires a training loop and may overfit to LIMO distribution.

---

## 3. The Key Insight: We Don't Need New Models

Looking at this more carefully, I realize there's a much simpler and more elegant approach that modifies the DSE algorithm itself:

### Approach D: **Counterfactual Reachability** (Best Approach)

**The fundamental problem with current DSE**: It uses a SINGLE dependency DAG parsed by an LLM. This DAG has systematic biases:
- The parser prompt says "only add inputs that are DIRECTLY USED" (poc1_parse_rir.py line 47)
- This means *implicit* dependencies (cognitive scaffolding) are systematically excluded
- The parser is designed to produce a MINIMAL DAG, which maximizes dead steps

**The fix**: Instead of one DAG, consider what the DAG WOULD look like under different plausible dependency interpretations.

```
Step is DEAD iff it is dead under ALL plausible dependency interpretations
Step is LIVE iff it is live under ANY plausible dependency interpretation
```

This is analogous to **must-analysis** in compiler theory: a step is dead only if it MUST be dead, not if it MIGHT be dead.

**Implementation without API calls**:

For each dead step s, check if adding ANY single edge from s to any live step would make s live. If yes, s is "fragile dead" — it could be live under a slightly different parse. Keep fragile-dead steps; remove only "robust dead" steps.

```python
def counterfactual_dse(rir, base_dse_result):
    """Only remove steps that are dead under all plausible edge additions."""
    live_ids = set(base_dse_result["live_ids"])
    dead_ids = set(base_dse_result["dead_ids"])
    robust_dead = set()
    fragile_dead = set()
    
    for dead_id in dead_ids:
        dead_step = step_map[dead_id]
        could_be_live = False
        
        # Try adding ONE edge from this step to any live step
        for live_id in live_ids:
            # Would adding dead_step → live_step make dead_step reachable?
            # Yes, if live_step is reachable from conclusion (it is, by definition)
            # So we just need: does this edge make semantic sense?
            if _plausible_edge(dead_step, step_map[live_id]):
                could_be_live = True
                break
        
        if could_be_live:
            fragile_dead.add(dead_id)  # KEEP — might be live
        else:
            robust_dead.add(dead_id)   # REMOVE — robustly dead
    
    return robust_dead, fragile_dead
```

The key function is `_plausible_edge`: does it make sense for step A to feed into step B?

**Heuristic plausibility rules** (no API needed):
1. **Type compatibility**: verification → derivation is plausible; computation → assumption is not
2. **Content overlap**: if step A mentions concepts/variables that appear in step B, an edge is plausible
3. **Temporal proximity**: adjacent steps are more likely to have missed edges
4. **File/topic overlap** (for agents): if step A and B touch the same file/topic

---

## 4. Synthesis: Recommended Algorithm

Combining the best ideas:

### **DSE v2: Robust Dead Step Elimination**

```
Phase 1: Standard backward reachability → live_ids, dead_ids

Phase 2: For each dead step:
  a) Check TYPE: if verification or exploration → mark as "fragile" candidate
  b) Check PROXIMITY: if adjacent to a live step → mark as "fragile" candidate  
  c) Check CONTENT OVERLAP: if shares entities/variables with any live step → "fragile"
  d) If none of (a,b,c) → "robust dead" → REMOVE

Phase 3: Apply domain-specific policy:
  - Math-only training: remove robust_dead + fragile_dead (aggressive, current behavior)
  - Cross-domain training: remove only robust_dead (conservative)
  - Adaptive: remove robust_dead always; for fragile_dead, keep if solution is long (>P75)
```

**This requires ZERO new models, ZERO API calls, and is fully automatic.**

The only change is in `poc2_dse.py`'s `dead_step_elimination` function — add a second pass that reclassifies some dead steps as "fragile dead" based on type + proximity + content overlap.

---

## 5. What This Means for the LIMO Training Pipeline

### Immediate (can do now, no retraining):

1. **Modify `build_limo_datasets.py`** to produce a 6th variant: **LIMO-TypedDSE**
   - Remove only `computation` and `derivation` dead steps
   - Keep `verification` dead steps
   - Uses existing R-IR data (already has step types)

2. **Modify `build_limo_datasets.py`** to produce a 7th variant: **LIMO-RobustDSE**
   - Implement the counterfactual reachability check
   - Remove only robustly dead steps
   - Uses existing R-IR data

### Requires retraining (4B, ~8 hours each):
- Train on LIMO-TypedDSE → evaluate on MATH-500 + GPQA
- Train on LIMO-RobustDSE → evaluate on MATH-500 + GPQA
- If GPQA recovers while MATH stays → we've solved the trade-off

### For the paper:
- Frame DSE v1 (current) as the "aggressive" policy
- Frame DSE v2 (typed/robust) as the "conservative" policy
- Show that the trade-off is controllable
- Future work: learn the optimal policy from data (Approach C)

---

## 6. Comparison with Existing Approaches

| Approach | Needs API? | Needs Retraining? | Needs New Data? | Principled? |
|:---------|:----------:|:-----------------:|:---------------:|:-----------:|
| Difficulty-Adaptive (doc/29) | No | Yes | No | Weak (length proxy) |
| **Type-Conditional DSE** | No | Yes | No | **Strong (type-based)** |
| **Counterfactual Reachability** | No | Yes | No | **Very strong (must-analysis)** |
| Learned Classifier | No | Yes | Partially | Strong (data-driven) |
| Full DeepSeek API re-parse | Yes | Yes | No | Medium (parser-dependent) |

**Recommendation**: Type-Conditional DSE is the quickest win (5-line code change + rebuild dataset + retrain). Counterfactual Reachability is the most principled and publishable.

---

## 7. Connection to Compiler Theory

This progression maps cleanly to compiler optimization theory:

| Compiler Concept | DSE v1 (Current) | DSE v2 (Proposed) |
|:-----------------|:-----------------|:------------------|
| Analysis type | May-analysis (might be dead) | Must-analysis (must be dead) |
| Conservatism | Aggressive (remove everything unreachable) | Conservative (remove only robustly unreachable) |
| Analog | `-O3` (aggressive, may break edge cases) | `-O1` (safe, preserves all plausible semantics) |
| Edge model | Single parse (one DAG) | Plausible parse set (DAG family) |

The "plausible parse set" idea connects to **abstract interpretation** in compilers: instead of analyzing one concrete execution, analyze a set of possible executions and only optimize when ALL executions agree.

This is a genuinely novel contribution if we can formalize it: **Abstract Interpretation for Reasoning Chain Optimization**.

---

## 8. Implementation Priority

1. **Type-Conditional DSE** — build dataset + train (1 day + 8h GPU)
2. **Quantify the effect** — if GPQA recovers, we have the paper story
3. **Counterfactual Reachability** — implement + build dataset + train (2 days + 8h GPU)
4. **Formalize** — connect to abstract interpretation for paper framing
5. **Learned Classifier** — future work (post-COLM)
