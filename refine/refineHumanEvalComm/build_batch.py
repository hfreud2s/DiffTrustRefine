"""
Builds the batch requests for the HumanEvalComm refinement experiment.

Step 1: the baseline phase. `build_baseline_batch()` samples program candidates
directly from each category's task description, with no clarifying-question refinement. It writes
one OpenRouter batch input file per (LLM, category). Submitting the files and post-processing the
results into the run folders happen in later steps.
"""
import json
import pathlib
import sys

THIS_DIR  = pathlib.Path(__file__).resolve().parent          # .../refine/refineHumanEvalComm
REPO_ROOT = THIS_DIR.parent.parent                           # .../DiffTrustRefine
for entry in (REPO_ROOT, THIS_DIR):
    if entry.as_posix() not in sys.path:
        sys.path.insert(0, entry.as_posix())

import cloudpickle

DATA_DIR   = REPO_ROOT / "HumanEvalComm" / ".data"
EXPERIMENT = THIS_DIR / ".HEC-experiment"

# Category folder name -> the Instance variant it maps to.
# None is the unmanipulated ("original") description; the rest are keys of
# Instance.manipulated_specs (HumanEvalComm prefixes its variant keys with "prompt").
CATEGORIES = {
    "original": None,
    "1a":   "prompt1a",
    "1c":   "prompt1c",
    "1p":   "prompt1p",
    "2ac":  "prompt2ac",
    "2ap":  "prompt2ap",
    "2cp":  "prompt2cp",
    "3acp": "prompt3acp",
}


def load_instances(dataset_name: str):
    """
    Loads the checked Instance objects from the benchmark's .data/{dataset_name}.pkl.
    The pickle is a plain list of Instances (not a Dataset).

    dataset_name: stem of the .pkl file, e.g. "dataset-50"
    returns:      list of Instance
    """
    with open(DATA_DIR / f"{dataset_name}.pkl", "rb") as f:
        return cloudpickle.load(f)


def spec_for(inst, variant):
    """
    Returns the Specification the candidates for this category are generated from, or None if the
    instance has no such variant. HumanEvalComm does not define every manipulation for every task,
    so a None here means "skip this task for this category".

    inst:    an Instance
    variant: None for the "original" category, else a key of Instance.manipulated_specs
    """
    if variant is None:
        return inst.spec
    return inst.manipulated_specs.get(variant)


def build_prompt(spec):
    """
    Builds the OpenRouter chat messages for one code-generation request. The prompt asks the model
    to implement the specification and return the full code in a single fenced Python block.

    spec:    a Specification (has .name, .signature, and a readable str form)
    returns: list of message dicts (system + user)
    """
    system_msg = {"role": "system", "content": "You are world class python programmer"}
    user_content = (
        f"Please write a function with the following specification:\n\n"
        f"{spec}\n\n"
        f"Please write the full code in one go like this :\n\n"
        f"```python\n"
        f"# imports ...\n"
        f"# Auxiliary functions and classes ...\n"
        f"def {spec.name}{spec.signature}:\n"
        f"    # {spec.name} code ...\n"
        f"# End of python code\n"
        f"```\n\n"
        f"Please ensure the function adheres strictly to the provided specifications.\n"
    )
    return [system_msg, {"role": "user", "content": user_content}]


def build_baseline_batch(llm_dir:        str,
                         model:          str,
                         dataset_name:   str = "dataset-50",
                         categories:     list = None,
                         num_candidates: int = 10,
                         num_runs:       int = 10,
                         temperature:    float = None):
    """
    Writes one OpenRouter baseline batch file per category for a single LLM.

    For every task that has a Specification in the category, and for every (run, sample) pair, one
    request is emitted. The output file for a category is
        .HEC-experiment/{llm_dir}/{category}/baseline_batch_request.jsonl
    with one JSON object per line in OpenRouter's batch shape: {"custom_id": ..., "body": ...}.
    custom_id is "run{r}__humanevalcomm_{task_id}__sample{i}".

    llm_dir:        experiment subfolder for this LLM, e.g. "LLM1"
    model:          the OpenRouter model id to request (placeholder until real ids are set)
    dataset_name:   which dataset pickle to read (default "dataset-50")
    categories:     which category folders to build (default: all eight)
    num_candidates: samples per (task, run)
    num_runs:       number of runs
    temperature:    optional sampling temperature added to the request body
    """
    categories = categories or list(CATEGORIES)
    instances  = load_instances(dataset_name)

    for category in categories:
        variant  = CATEGORIES[category]
        out_dir  = EXPERIMENT / llm_dir / category
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "baseline_batch_request.jsonl"

        written = 0
        skipped = 0
        with open(out_path, "w", encoding="utf-8") as batch_file:
            for inst in instances:
                spec = spec_for(inst, variant)
                if spec is None:
                    skipped += 1
                    continue
                body = {"model": model, "messages": build_prompt(spec)}
                if temperature is not None:
                    body["temperature"] = temperature
                for run in range(num_runs):
                    for i in range(num_candidates):
                        request = {
                            "custom_id": f"run{run}__humanevalcomm_{inst.task_id}__sample{i}",
                            "body":      body,
                        }
                        batch_file.write(json.dumps(request) + "\n")
                        written += 1
        print(f"{llm_dir}/{category}: {written} requests -> {out_path} "
              f"({skipped} task(s) had no '{category}' spec)")


if __name__ == "__main__":

    # Placeholder model ids. Replace with the real OpenRouter model strings for LLM1..LLM4.
    MODELS = {
        "LLM1": "openrouter/model-1",
        "LLM2": "openrouter/model-2",
        "LLM3": "openrouter/model-3",
        "LLM4": "openrouter/model-4",
    }

    dataset_name   = "dataset-50"
    num_candidates = 10
    num_runs       = 10
    temperature    = None

    # --- Baseline ---
    for llm_dir, model in MODELS.items():
        build_baseline_batch(
            llm_dir,
            model,
            dataset_name,
            num_candidates=num_candidates,
            num_runs=num_runs,
            temperature=temperature,
        )
