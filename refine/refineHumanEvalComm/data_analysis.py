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
        per_run[int(d.name[len("run"):])] = {r["task_id"]: r for r in rows}

    runs = sorted(per_run)
    if not runs:
        print(f"{llm_dir}/{category}/{phase}: no run stats found (run score_phase first)")
        return []

    task_ids = sorted({tid for run_rows in per_run.values() for tid in run_rows})
    aggregated = []
    for tid in task_ids:
        inc = [per_run[r].get(tid, {}).get("incoherence") for r in runs]
        err = [per_run[r].get(tid, {}).get("error") for r in runs]
        inc_valid = [v for v in inc if v is not None]
        err_valid = [v for v in err if v is not None]
        sample = next(per_run[r][tid] for r in runs if tid in per_run[r])
        aggregated.append({
            "task_id":          tid,
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
    #aggregate_phase("LLM1", "1a")
    pass
