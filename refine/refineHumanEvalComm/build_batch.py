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

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import CATEGORIES, CODER_MODELS, EXPERIMENT, DATASET_NAME, load_instances, spec_for, provider_obj


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
                         dataset_name:   str = DATASET_NAME,
                         categories:     list = None,
                         num_candidates: int = 10,
                         num_runs:       int = 10,
                         temperature:    float = None,
                         provider=None):
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
            for inst in instances.values():
                spec = spec_for(inst, variant)
                if spec is None:
                    skipped += 1
                    continue
                body = {"model": model, "messages": build_prompt(spec)}
                if temperature is not None:
                    body["temperature"] = temperature
                if provider is not None:
                    body["provider"] = provider_obj(provider)
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



# Which refined descriptions become program candidates. Per question we generate from the oracle's
# true answer and from both of the coder's YES/NO branches, so scoring later yields both the
# true-refinement effect (oracle) and the expected-incoherence-reduction signal (desc1/desc2).
ORACLE_VARIANTS = [("oracle_description", "oracle")]                 # oracle's true answer
CODER_VARIANTS  = [("description1", "desc1"), ("description2", "desc2")]  # coder YES/NO branches
DESC_VARIANTS   = ORACLE_VARIANTS + CODER_VARIANTS                        # all three (combined default)


def effective_oracle_description(q):
    """The oracle description to sample programs from: the auditor's (#2) tightened rewrite when a
    leak was flagged and a rewrite was supplied, otherwise the oracle's original true description."""
    if q.get("oracle_leak") and q.get("oracle_rewrite"):
        return q["oracle_rewrite"]
    return q.get("oracle_description")


def build_refined_prompt(description: str, spec):
    """
    Like build_prompt, but the task text is a refined description string (not a Specification). The
    function name and signature for the code skeleton still come from spec (the entry point is stable
    across a task's variants and is what scoring uses).
    """
    system_msg = {"role": "system", "content": "You are world class python programmer"}
    user_content = (
        f"Please write a function with the following specification:\n\n"
        f"{description}\n\n"
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


def build_refined_candidates_batch(llm_dir:        str,
                                   category:       str,
                                   model:          str = None,
                                   dataset_name:   str = DATASET_NAME,
                                   num_candidates: int = 10,
                                   num_runs:       int = 10,
                                   temperature:    float = None,
                                   provider=None,
                                   variants=None,
                                   request_name:   str = "refined_candidates_batch_request.jsonl"):
    """
    Writes the batch that generates refined program candidates for one (LLM, category). For every
    task in {category}/refined/questions_and_descriptions.json, every question, and every filled
    description branch (oracle / desc1 / desc2), one request is emitted per (run, sample). Output:
        .HEC-experiment/{llm_dir}/{category}/refined/refined_candidates_batch_request.jsonl
    custom_id "run{r}__humanevalcomm_{task_id}__{qkey}__{branch}__sample{i}". Post-process with
    batch_processing.postprocess_refined_candidates, then score with compute_stats.score_phase(
    phase="refined") and data_analysis.aggregate_phase(phase="refined").

    model:          coder slug (defaults to this LLM's roster model - the same coder as the baseline)
    num_candidates: samples per (branch, run); num_runs: runs, mirroring the baseline
    """
    refined_dir = EXPERIMENT / llm_dir / category / "refined"
    with open(refined_dir / "questions_and_descriptions.json", "r", encoding="utf-8") as f:
        entries = json.load(f)
    instances = load_instances(dataset_name, {e["task_id"] for e in entries})
    if model is None:
        model = CODER_MODELS[llm_dir]
    if variants is None:
        variants = DESC_VARIANTS

    out_path = refined_dir / request_name
    written = skipped = rewrites_used = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for e in entries:
            inst = instances.get(e["task_id"])
            if inst is None:
                continue
            spec = inst.spec
            for qkey, q in e["questions"].items():
                for field, label in variants:
                    if field == "oracle_description":
                        description = effective_oracle_description(q)
                        if q.get("oracle_leak") and q.get("oracle_rewrite"):
                            rewrites_used += 1
                    else:
                        description = q.get(field)
                    if not description:
                        skipped += 1
                        continue
                    body = {"model": model, "messages": build_refined_prompt(description, spec)}
                    if temperature is not None:
                        body["temperature"] = temperature
                    if provider is not None:
                        body["provider"] = provider_obj(provider)
                    for run in range(num_runs):
                        for i in range(num_candidates):
                            f.write(json.dumps({
                                "custom_id": f"run{run}__humanevalcomm_{e['task_id']}__{qkey}__{label}__sample{i}",
                                "body":      body,
                            }) + "\n")
                            written += 1
    note = f", {rewrites_used} oracle rewrite(s) used" if rewrites_used else ""
    print(f"{llm_dir}/{category}: {written} refined candidate request(s) -> {out_path} "
          f"({skipped} empty branch(es) skipped{note})")
    return out_path


def build_refined_coder_candidates_batch(llm_dir, category, **kwargs):
    """Refined candidates for the coder YES/NO branches only (desc1/desc2). Separate stage so these
    can be sampled while the oracle-description decision is still open. Output:
    {category}/refined/refined_coder_candidates_batch_request.jsonl; post-process with
    postprocess_refined_candidates(result_name="refined_coder_candidates_batch_result.jsonl")."""
    return build_refined_candidates_batch(
        llm_dir, category, variants=CODER_VARIANTS,
        request_name="refined_coder_candidates_batch_request.jsonl", **kwargs)


def build_refined_oracle_candidates_batch(llm_dir, category, **kwargs):
    """Refined candidates for the oracle's true description only. Output:
    {category}/refined/refined_oracle_candidates_batch_request.jsonl; post-process with
    postprocess_refined_candidates(result_name="refined_oracle_candidates_batch_result.jsonl").
    Both stages write disjoint per-(run,task,qkey,branch) files into the same run dirs, so scoring
    reads whichever branches are present."""
    return build_refined_candidates_batch(
        llm_dir, category, variants=ORACLE_VARIANTS,
        request_name="refined_oracle_candidates_batch_request.jsonl", **kwargs)


if __name__ == "__main__":


    num_candidates = 10
    num_runs       = 10
    temperature    = None

    # --- Baseline ---
    for llm_dir, model in CODER_MODELS.items():
        build_baseline_batch(
            llm_dir,
            model,
            DATASET_NAME,
            num_candidates=num_candidates,
            num_runs=num_runs,
            temperature=temperature,
        )

    # --- Round 5: refined candidates (oracle + both coder branches, per question) ---
    # build_refined_candidates_batch("LLM1", "1a", num_candidates=num_candidates, num_runs=num_runs)
