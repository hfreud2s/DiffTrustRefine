"""
This file is used for computing the incoherence and error of the baseline candidates as well as the refined candidates.
"""
import json
import re
import statistics
import sys
from pathlib import Path
import time
import cloudpickle

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
sys.path.append(str(PROJECT_ROOT))
MBPP_JSON = PROJECT_ROOT / "MBPP" / ".data" / "sanitized-mbpp.json"

import difftrust
from MBPP.instance import Instance


def load_instances_by_id(task_ids: set[int]):
    """
    Loads Instance objects for the given task IDs from the sanitized MBPP JSON file.
    Tasks that fail to construct an Instance (e.g. due to exec errors) are skipped
    with a warning.

    task_ids: set of integer task IDs to load
    returns:  dict mapping task_id -> Instance for all successfully loaded tasks
    """
    with open(MBPP_JSON, "r", encoding="utf-8") as f:
        all_tasks = json.load(f)
    result = {}
    for task in all_tasks:
        tid = task["task_id"]
        if tid in task_ids:
            try:
                result[tid] = Instance(task)
            except Exception as e:
                print(f"  Could not build Instance for task_id={tid}: {e}")
    return result

def extract_code(raw: str):
    """
    Strips markdown code fences from a raw model response to get clean Python code.
    Removes a leading ```python fence and a trailing ``` fence if present.

    raw:     raw string from the model response
    returns: the code string with fences removed
    """
    code = raw.strip()
    code = re.sub(r"^```python\n?", "", code)
    code = re.sub(r"\n?```$", "", code)
    return code

def compile_candidates(raw_candidates: list[str], 
                       spec):
    """
    Compiles a list of raw model-generated code strings into Function objects.
    Each string is cleaned with extract_code before compilation.
    force_compile() is called so that compilation errors are surfaced immediately
    rather than lazily at evaluation time.

    raw_candidates: list of raw code strings from the model
    spec:           the Specification the candidates implement
    returns:        list of compiled Function objects
    """
    functions = [difftrust.function.Function(spec, extract_code(c)) for c in raw_candidates]
    return [f.force_compile() for f in functions]

def compute_baseline_stats(candidate_path:  Path, 
                           output_path:     Path,
                           nb_candidates:   int = 1000,
                           timeout:         float = 60.0):
    """
    Computes incoherence and error for all baseline candidate files in candidate_path.
    Results are written incrementally to output_path after each task so progress is
    not lost if the run is interrupted. Already-completed tasks are skipped on resume.

    Relies on the following globals defined in __main__:
      compute_disagreement, compute_error_fn

    candidate_path: directory of per-task cloudpickle files (e.g. baseline/runN/program_candidates/)
    output_path:    path to write the stats JSON (e.g. baseline/runN/stats_runN.json)
    """
    task_files = sorted(candidate_path.iterdir())
    task_ids   = {int(f.name.split("_")[1]) for f in task_files}
    instances  = load_instances_by_id(task_ids)
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        already_done = {r["task_id"] for r in results}
        print(f"Loaded {len(already_done)} existing task(s) from {output_path}, skipping them.")
    else:
        results = []
        already_done = set()
    for task_file in task_files:
        task_id = int(task_file.name.split("_")[1])
        if task_id in already_done:
            print(f"Skipping task {task_id} because it was already computed")
            continue
        inst = instances.get(task_id)
        if inst is None:
            print(f"No instance found for task_id={task_id}, skipping.")
            continue
        with open(task_file, "rb") as f:
            raw_candidates = cloudpickle.load(f)
        print(f"\n=== Task {task_id} ({inst.name}): {len(raw_candidates)} candidates ===")
        candidate_list = compile_candidates(raw_candidates, inst.spec)
        task_result = {
            "task_id": task_id,
            "name": inst.name,
            "nb_candidates": len(candidate_list),
        }
        try:
            start_time = time.time()
            if len(candidate_list) <= 1:
                dis = 0.0
            else:
                dis = difftrust.checking.timeout_call(
                    func=compute_disagreement,
                    args=(candidate_list, inst.filtered_generator, nb_candidates),
                    kwargs={},
                    timeout=timeout,
                )
            task_result["incoherence"] = dis
            print(f"  incoherence={dis:.4f} (computed in {time.time()-start_time:.2f}s)", end="")
        except Exception as e:
            task_result["incoherence"] = None
            task_result["incoherence_error"] = f"{type(e).__name__}: {e}"
            print(f"  incoherence=ERROR ({e})", end="")
        try:
            start_time = time.time()
            err = difftrust.checking.timeout_call(
                func=compute_error_fn,
                args=(candidate_list, inst.ground_truth, inst.filtered_generator, nb_candidates),
                kwargs={},
                timeout=timeout,
            )
            task_result["error"] = err
            print(f", error={err:.4f} (computed in {time.time()-start_time:.2f}s)")
        except Exception as e:
            task_result["error"] = None
            task_result["error_error"] = f"{type(e).__name__}: {e}"
            print(f", error=ERROR ({e})")
        results.append(task_result)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

def compute_complete_stats(candidate_path:  Path, 
                           output_path:     Path, 
                           candidate_no:    int = 1000,
                           timeout:         float = 60.0):
    """
    Computes incoherence (and error for oracle variants) for all description variant
    candidate files under candidate_path for a single task. Results are written to
    a stats.json inside candidate_path and also merged into the shared output_path.
    Already-completed files are skipped on resume.

    Error is only computed for files whose name ends with 'oracle', since only the
    oracle description has a well-defined ground truth to compare against.

    Relies on the following globals defined in __main__:
      nb_sample, timeout, compute_disagreement, compute_error_fn

    candidate_path: directory of candidate files for one task (e.g. refined/candidates/mbpp_X/)
    output_path:    shared output JSON that accumulates results across all tasks
    candidate_no:   integer task ID for the task being processed
    """
    inst = load_instances_by_id({candidate_no}).get(candidate_no)
    if inst is None:
        print(f"No instance found for task_id={candidate_no}, skipping.")
        return
    stats_path = candidate_path / "stats.json"
    if stats_path.exists():
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
        print(f"  Loaded {len(stats)} existing file result(s) from {stats_path}, skipping them.")
    else:
        stats = {}
    for file_path in sorted(candidate_path.iterdir()):
        if file_path.is_dir() or file_path.name == "stats.json":
            continue
        filename = file_path.name
        if filename in stats:
            print(f"  Skipping {filename} (already computed).")
            continue
        print(f"\n  Processing {filename}...")
        with open(file_path, "rb") as f:
            raw_candidates = cloudpickle.load(f)
        candidate_list = compile_candidates(raw_candidates, inst.spec)
        entry = {"nb_candidates": len(candidate_list)}
        try:
            start_time = time.time()
            if len(candidate_list) <= 1:
                dis = 0.0
            else:
                dis = difftrust.checking.timeout_call(
                    func=compute_disagreement,
                    args=(candidate_list, inst.filtered_generator, candidate_no),
                    kwargs={},
                    timeout=timeout,
                )
            entry["incoherence"] = dis
            print(f"    incoherence={dis:.4f} (computed in {time.time()-start_time:.2f}s)", end="")
        except Exception as e:
            entry["incoherence"] = None
            entry["incoherence_error"] = f"{type(e).__name__}: {e}"
            print(f"    incoherence=ERROR ({e})", end="")
        if filename.endswith("oracle"):
            try:
                start_time = time.time()
                err = difftrust.checking.timeout_call(
                    func=compute_error_fn,
                    args=(candidate_list, inst.ground_truth, inst.filtered_generator, candidate_no),
                    kwargs={},
                    timeout=timeout,
                )
                entry["error"] = err
                print(f", error={err:.4f} (computed in {time.time()-start_time:.2f}s)")
            except Exception as e:
                entry["error"] = None
                entry["error_error"] = f"{type(e).__name__}: {e}"
                print(f", error=ERROR ({e})")
        else:
            print()
        stats[filename] = entry
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"\n  Saved per-file stats to {stats_path}")
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            output_data = json.load(f)
    else:
        output_data = {}
    output_data[candidate_no] = stats
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    print(f"  Updated output stats at {output_path}")


if __name__ == "__main__":

    computation_type      = "pointwise"  
    
    if computation_type  == "pointwise":
        compute_disagreement = difftrust.disagreement.pointwise_incoherence
        compute_error_fn     = difftrust.disagreement.pointwise_error
    else:
        compute_disagreement = difftrust.disagreement.functional_incoherence
        compute_error_fn     = difftrust.disagreement.functional_error

    # --- Compute Baseline Stats ---

    # candidate_path  = Path() # e.g. THIS_DIR / ".MBPP-example-mini" / "baseline" / "candidates"
    # output_path     = Path() # e.g. THIS_DIR / ".MBPP-example-mini" / "baseline" / "baseline_stats.json"
    # compute_baseline_stats(candidate_path,output_path)


    # --- Compute Refined and Oracle Stats ---

    # candidate_path  = Path() # e.g. THIS_DIR / ".MBPP-example-mini" / "refined" / "candidates"
    # output_path     = Path() # e.g. THIS_DIR / ".MBPP-example-mini" / "refined" / "refined_stats.json"
    # for task_dir in candidate_path.iterdir():
    #     if task_dir.is_dir():
    #         candidate_no = int(task_dir.name.split("_")[1])   # "mbpp_1" → 1
    #         print(f"\n=== Task {candidate_no} ===")
    #         compute_complete_stats(task_dir, output_path, candidate_no)