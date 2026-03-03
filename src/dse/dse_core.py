"""
Dead Step Elimination (DSE) Core Algorithm

Backward reachability from conclusion steps on R-IR dependency graphs.
Identifies and removes structurally dead steps.

Ported from DecoR poc/poc2_dse.py
"""

import json
from pathlib import Path


def dead_step_elimination(rir: dict) -> dict:
    """Run DSE on a single R-IR trace.
    
    Args:
        rir: Dict with 'steps' list, each step having 'id', 'type', 'inputs'.
        
    Returns:
        Dict with live_ids, dead_ids, dead_steps, live_steps, counts, ratios.
    """
    steps = rir["steps"]
    step_map = {s["id"]: s for s in steps}
    
    # Find conclusion nodes
    conclusions = [s for s in steps if s["type"] == "conclusion"]
    if not conclusions:
        conclusions = [steps[-1]] if steps else []
    
    # Backward reachability BFS
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


def run_dse_on_directory(rir_dir: str | Path) -> dict:
    """Run DSE on all R-IR JSON files in a directory.
    
    Args:
        rir_dir: Path to directory containing R-IR JSON files.
        
    Returns:
        Summary dict with per-trace results and aggregate stats.
    """
    rir_dir = Path(rir_dir)
    rir_files = sorted(f for f in rir_dir.glob("*.json") if f.name != "_summary.json")
    
    if not rir_files:
        print(f"No R-IR files found in {rir_dir}")
        return {}
    
    results = []
    for rf in rir_files:
        rir = json.loads(rf.read_text())
        pid = rir.get("problem_id", rf.stem)
        dse = dead_step_elimination(rir)
        
        # Check conclusion preservation
        concs = [s for s in rir["steps"] if s["type"] == "conclusion"]
        preserved = all(s["id"] in set(dse["live_ids"]) for s in concs) if concs else True
        
        # Dead type breakdown
        dt = {}
        for s in dse["dead_steps"]:
            dt[s["type"]] = dt.get(s["type"], 0) + 1
        
        # Token savings
        dead_tok = sum(s.get("original_tokens", 0) for s in dse["dead_steps"])
        total_tok = rir.get("original_token_count", 0)
        
        results.append({
            "id": pid,
            "total": dse["total"],
            "dead": dse["dead_count"],
            "dead_ratio": dse["dead_ratio"],
            "preserved": preserved,
            "token_savings": dead_tok / total_tok if total_tok else 0,
            "dead_types": dt,
        })
    
    n = len(results)
    avg_dr = sum(r["dead_ratio"] for r in results) / n if n else 0
    apr = sum(1 for r in results if r["preserved"]) / n if n else 0
    
    return {
        "avg_dead_ratio": avg_dr,
        "answer_preservation_rate": apr,
        "num_traces": n,
        "per_trace": results,
    }
