"""
Shared constants and helpers for the HumanEvalComm refinement pipeline.

Centralises what build_batch / batch_processing / compute_stats / data_analysis / refine_descriptions
all need: the experiment paths, the dataset location, the category -> variant map, and instance
loading. Importing this also puts the repo root on sys.path so `difftrust` and the pickled
Specification objects resolve.
"""
import pathlib
import sys

THIS_DIR  = pathlib.Path(__file__).resolve().parent          # .../refine/refineHumanEvalComm
REPO_ROOT = THIS_DIR.parent.parent                           # .../DiffTrustRefine
for entry in (REPO_ROOT, THIS_DIR):
    if entry.as_posix() not in sys.path:
        sys.path.insert(0, entry.as_posix())

import cloudpickle

EXPERIMENT   = THIS_DIR / ".HEC-mini"                  # per-(LLM, category) experiment data
DATA_DIR     = REPO_ROOT / "HumanEvalComm" / ".data"         # dataset pickles + source JSON (benchmark folder)
SOURCE_JSON  = DATA_DIR / "HumanEvalComm.json"
DATASET_NAME = "dataset-10"

BASE = "openrouter" # choose "litellm" or "openrouter"

API_BASE_OR = "https://openrouter.ai/api/beta/batches"
API_BASE_LL = "https://litellm.ai.cispa.de/"
API_BASE = API_BASE_LL if BASE == "litellm" else API_BASE_OR

# Category folder name -> the Instance variant it maps to.
# None is the unmanipulated ("original") description; the rest are keys of Instance.manipulated_specs
# (HumanEvalComm prefixes its variant keys with "prompt").
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


# --- Model list: one place for every role ---
# Coders (four vendors), the oracle (simulates the user, answers from the ground truth), and
# the auditor (independent judge, a different vendor from every coder and from the oracle).
CODER_MODELS = {
    "LLM1": "openai/gpt-5.6-luna:batch",
    "LLM2": "anthropic/claude-sonnet-5:batch",
    "LLM3": "deepseek/deepseek-v4-pro-0813:batch",
    "LLM4": "qwen/qwen3.5-9b:batch",
}
ORACLE_MODEL  = "anthropic/claude-opus-4.8:batch"
AUDITOR_MODEL = "google/gemini-3.8-flash:batch"


def blank_question(text: str, source: str = "model"):
    """A fresh question entry for questions_and_descriptions.json.
    source is "model" for a coder-generated question or "auditor" for one appended by the auditor."""
    return {
        "question":           text,
        "source":             source,
        "description1":       None,   # coder LLM, YES branch
        "description2":       None,   # coder LLM, NO branch
        "oracle_description": None,   # oracle LLM, true answer
        "oracle_leak":        None,   # auditor #2
        "oracle_rewrite":     None,   # auditor #2
    }


def load_instances(dataset_name: str = DATASET_NAME, task_ids: set = None):
    """
    Loads the checked Instance objects from the benchmark's .data/{dataset_name}.pkl, keyed by
    inst.task_id (the N in "HumanEval/N"). inst.name is the entry point, not unique across tasks,
    so it is not a usable key.

    dataset_name: stem of the .pkl file (default "dataset-50")
    task_ids:     optional set of ids to keep; None loads all
    returns:      dict mapping task_id -> Instance
    """
    with open(DATA_DIR / f"{dataset_name}.pkl", "rb") as f:
        instances = cloudpickle.load(f)
    return {inst.task_id: inst for inst in instances
            if task_ids is None or inst.task_id in task_ids}


def spec_for(inst, variant):
    """
    The Specification for a category: the baseline spec for the 'original' category (variant None),
    else the manipulated variant's spec, or None if the task has no such variant (skip it).
    """
    if variant is None:
        return inst.spec
    return inst.manipulated_specs.get(variant)
