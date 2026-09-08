"""
Computes incoherence and error for HumanEvalComm candidate files.
"""
import json
import re
from pathlib import Path

import cloudpickle

import time

import pathlib
import sys

THIS_DIR   = Path(__file__).resolve().parent          # .../refine/refineHumanEvalComm
REPO_ROOT  = THIS_DIR.parent.parent                   # .../DiffTrustRefine
for entry in (REPO_ROOT, THIS_DIR):
    if entry.as_posix() not in sys.path:
        sys.path.insert(0, entry.as_posix())


import difftrust


EXPERIMENT   = THIS_DIR / ".HEC-experiment"
DATA_DIR     = REPO_ROOT / "HumanEvalComm" / ".data"   # dataset pickles live in the benchmark folder
SOURCE_JSON  = DATA_DIR / "HumanEvalComm.json"
DATASET_NAME = "dataset-50"

KEY_RE = re.compile(r"^humanevalcomm_(\d+)(?:-(\w+))?$")


def parse_candidate_key(key: str):
    """
    Splits a candidate file name into (task_id, variant).

    "humanevalcomm_23"           -> (23, None)        the unmanipulated condition
    "humanevalcomm_23-prompt1a"  -> (23, "prompt1a")  a manipulated condition

    key:     the candidate file's name, i.e. the first two underscore-separated fields
             of the custom_id, as produced by group_candidates_by_task_id
    returns: (task_id, variant), or None if the name does not match the scheme
    """
    match = KEY_RE.match(key)
    if match is None:
        return None
    task_id, variant = match.groups()
    return int(task_id), variant


def load_instances_by_id(task_ids: set = None, dataset_name: str = DATASET_NAME):
    """
    Loads the checked Instance objects from .data/{dataset_name}.pkl, keyed by the numeric task id
    they carry as inst.task_id (the N in "HumanEval/N").

    Note that inst.name is the entry point, not the task id, and is not usable as a key: six of
    them occur twice.

    task_ids: optional set of ids to keep; None loads all of them
    returns:  dict mapping task_id -> Instance
    """
    with open(DATA_DIR / f"{dataset_name}.pkl", "rb") as f:
        instances = cloudpickle.load(f)

    return {inst.task_id: inst for inst in instances
            if task_ids is None or inst.task_id in task_ids}


def extract_code(raw: str):
    """
    Extracts the Python code from a raw model response.
    raw:     raw string from the model response
    returns: the code inside the first fenced block, else the de-fenced whole string
    """
    match = re.search(r"```[ \t]*[A-Za-z0-9_+-]*[ \t]*\r?\n(.*?)```", raw, re.DOTALL)
    if match:
        return match.group(1).strip()
    code = raw.strip()
    code = re.sub(r"^```[A-Za-z0-9_+-]*\n?", "", code)
    code = re.sub(r"\n?```$", "", code)
    return code.strip()


def compile_candidates(raw_candidates: list, spec):
    """
    Compiles raw model-generated code strings into Function objects.
    force_compile turns a compilation failure into a CompilationErrorFunction rather than
    raising, so one bad candidate does not abort the whole task.

    raw_candidates: list of raw code strings from the model
    spec:           the Specification the candidates implement
    returns:        list of compiled callables
    """
    functions = [difftrust.function.Function(spec, extract_code(c)) for c in raw_candidates]
    return [f.force_compile() for f in functions]


def compute_stats(candidate_path: Path,
                  output_path:    Path,
                  nb_samples:     int = 1000,
                  timeout:        float = 60.0,
                  dataset_name:   str = DATASET_NAME):
    """
    Computes incoherence and error for every candidate file in candidate_path.

    Results are written after each file so an interrupted
    run can resume, and files already present in output_path are skipped.

    candidate_path: directory of per-condition cloudpickle files
    output_path:    path to write the stats JSON
    nb_samples:     number of input samples per metric
    timeout:        seconds allowed per metric before it is recorded as None
    dataset_name:   stem of the .pkl file to read instances from
    """
    compute_incoherence = difftrust.disagreement.pointwise_incoherence
    compute_error       = difftrust.disagreement.pointwise_error

    parsed = []
    for f in sorted(candidate_path.iterdir()):
        key = parse_candidate_key(f.name)
        if key is None:
            print(f"Skipping {f.name}: not a HumanEvalComm candidate file")
            continue
        parsed.append((f, *key))

    instances = load_instances_by_id({tid for _, tid, _ in parsed}, dataset_name)

    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        already_done = {r["key"] for r in results}
        print(f"Loaded {len(already_done)} existing result(s) from {output_path}, skipping them.")
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results, already_done = [], set()

    for candidate_file, task_id, variant in parsed:
        if candidate_file.name in already_done:
            print(f"Skipping {candidate_file.name} because it was already computed")
            continue
        inst = instances.get(task_id)
        if inst is None:
            print(f"No instance found for task_id={task_id}, skipping {candidate_file.name}")
            continue
        spec = inst.spec

        with open(candidate_file, "rb") as f:
            raw_candidates = cloudpickle.load(f)
        condition = variant or "baseline"
        print(f"\n=== Task {task_id} ({inst.name}) [{condition}]: {len(raw_candidates)} candidates ===")
        candidate_list = compile_candidates(raw_candidates, spec)

        task_result = {
            "key":           candidate_file.name,
            "task_id":       task_id,
            "variant":       variant,
            "name":          inst.name,
            "nb_candidates": len(candidate_list),
        }
        try:
            start_time = time.time()
            if len(candidate_list) <= 1:
                dis = 0.0
            else:
                dis = difftrust.checking.timeout_call(
                    func=compute_incoherence,
                    args=(candidate_list, inst.filtered_generator, nb_samples),
                    kwargs={},
                    timeout=timeout,
                )
            task_result["incoherence"] = dis
            print(f"  incoherence={dis:.4f} (computed in {time.time()-start_time:.2f}s)", end="")
        except Exception as e:
            task_result["incoherence"] = None
            task_result["incoherence_error"] = f"{type(e).__name__}: {e}"
            print(f"  incoherence=ERROR ({type(e).__name__})", end="")
        try:
            start_time = time.time()
            err = difftrust.checking.timeout_call(
                func=compute_error,
                args=(candidate_list, inst.ground_truth, inst.filtered_generator, nb_samples),
                kwargs={},
                timeout=timeout,
            )
            task_result["error"] = err
            print(f", error={err:.4f} (computed in {time.time()-start_time:.2f}s)")
        except Exception as e:
            task_result["error"] = None
            task_result["error_error"] = f"{type(e).__name__}: {e}"
            print(f", error=ERROR ({type(e).__name__})")

        results.append(task_result)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")
    return results


def score_phase(llm_dir: str, category: str, phase: str = "baseline",
                runs: list = None, nb_samples: int = 1000, timeout: float = 60.0,
                dataset_name: str = DATASET_NAME):
    """
    Scores the candidate files of one (LLM, category, phase) run by run, writing a stats.json into
    each run folder. It walks
        .HEC-experiment/{llm_dir}/{category}/{phase}/run{r}/
    and, for each run present, calls compute_stats to score every candidate file there against the
    Specification named by that file's variant suffix. compute_stats is resumable, so re-running
    picks up where an interrupted run left off.

    llm_dir/category/phase: locate the phase directory
    runs:         which run indices to score (default: every run folder found)
    nb_samples:   input samples per metric
    timeout:      seconds per metric before it is recorded as None
    dataset_name: dataset pickle to read instances from
    returns:      dict run_index -> stats.json path
    """
    phase_dir = EXPERIMENT / llm_dir / category / phase
    run_dirs = sorted((d for d in phase_dir.iterdir() if d.is_dir() and d.name.startswith("run")),
                      key=lambda d: int(d.name[len("run"):]))
    if runs is not None:
        wanted = set(runs)
        run_dirs = [d for d in run_dirs if int(d.name[len("run"):]) in wanted]
    written = {}
    for d in run_dirs:
        print(f"\n########## scoring {llm_dir}/{category}/{phase}/{d.name} ##########")
        compute_stats(d, d / "stats.json", nb_samples=nb_samples, timeout=timeout,
                      dataset_name=dataset_name)
        written[int(d.name[len("run"):])] = d / "stats.json"
    return written


if __name__ == "__main__":

    # --- Step 4a: score each run of a category (writes run{r}/stats.json) ---
    # score_phase("LLM1", "1a")               # all runs of the baseline phase
    # score_phase("LLM1", "1a", runs=[0])     # just run0

    # --- Or a single run folder directly ---
    # run_dir = EXPERIMENT / "LLM1" / "1a" / "baseline" / "run0"
    # compute_stats(run_dir, run_dir / "stats.json")
    pass
