"""
Aggregates the per-run stats produced by compute_stats.score_phase into per-category summaries.
"""
import json
import statistics
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import EXPERIMENT


def aggregate_phase(llm_dir: str, category: str, phase: str = "baseline"):
    """
    Reads every run{r}/stats.json under .HEC-experiment/{llm_dir}/{category}/{phase}/ and, per task,
    collects incoherence and error across runs plus their means.
    Writes the summary to {phase}/aggregate.json and returns it.

    llm_dir/category/phase: which phase directory to summarise
    returns:                list of per-task aggregate dicts
    """
    phase_dir = EXPERIMENT / llm_dir / category / phase
    run_dirs = sorted((d for d in phase_dir.iterdir() if d.is_dir() and d.name.startswith("run")),
                      key=lambda d: int(d.name[len("run"):]))

    per_run = {}
    for d in run_dirs:
        stats_path = d / "stats.json"
        if not stats_path.exists():
            continue
        with open(stats_path, "r", encoding="utf-8") as f:
            rows = json.load(f)
        per_run[int(d.name[len("run"):])] = {r["key"]: r for r in rows}

    runs = sorted(per_run)
    if not runs:
        print(f"{llm_dir}/{category}/{phase}: no run stats found (run score_phase first)")
        return []

    # Align across runs by candidate key (the file name), which is stable from run to run. For the
    # baseline that is one key per task; for the refined phase it is one key per (task, question,
    # branch) condition, so conditions of the same task do not collide.
    keys = sorted({k for run_rows in per_run.values() for k in run_rows})
    aggregated = []
    for key in keys:
        inc = [per_run[r].get(key, {}).get("incoherence") for r in runs]
        err = [per_run[r].get(key, {}).get("error") for r in runs]
        inc_valid = [v for v in inc if v is not None]
        err_valid = [v for v in err if v is not None]
        sample = next(per_run[r][key] for r in runs if key in per_run[r])
        aggregated.append({
            "key":              key,
            "task_id":          sample.get("task_id"),
            "name":             sample.get("name"),
            "variant":          sample.get("variant"),
            "runs":             runs,
            "incoherence_list": inc,
            "mean_incoherence": statistics.mean(inc_valid) if inc_valid else None,
            "error_list":       err,
            "mean_error":       statistics.mean(err_valid) if err_valid else None,
        })

    out_path = phase_dir / "aggregate.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=2)
    print(f"{llm_dir}/{category}/{phase}: aggregated {len(aggregated)} task(s) over runs {runs} -> {out_path}")
    return aggregated


if __name__ == "__main__":

    # --- Step 4b: aggregate per-run stats into {phase}/aggregate.json ---
    # aggregate_phase("LLM1", "1a")                    # baseline
    # aggregate_phase("LLM1", "1a", phase="refined")   # round 5
    pass
