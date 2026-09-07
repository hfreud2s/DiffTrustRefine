"""
This file is used to prepare the jsonl-files needed for batch processing.
Use build_baseline_batch() to generate a file of batch requests for the baseline program candidates.
Use build_refinement_batch() to generate batch requests to get candidates for the refined and oracle descriptions.
"""
import json
import re
import inspect
import importlib.util
from pathlib import Path

THIS_DIR     = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent

_spec = importlib.util.spec_from_file_location(
    "specification", PROJECT_ROOT / "difftrust" / "core" / "specification.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
Specification = _mod.Specification


def load_mbpp_tasks(mbpp_file_path: Path, 
                    task_ids: list = None):
    """
    Loads all MBPP tasks from a JSON file.
    If task_ids is provided, only tasks whose task_id appears in the list are returned.

    mbpp_file_path: path to the sanitized-mbpp.json file
    task_ids:       optional list of integer task IDs to filter by
    """
    with open(mbpp_file_path, "r", encoding="utf-8") as f:
        all_tasks = json.load(f)
    if not task_ids:
        return all_tasks
    return [t for t in all_tasks if t["task_id"] in task_ids]

def extract_name_and_signature(task: dict):
    """
    Extracts the target function's name and signature from an MBPP task.

    task: a single MBPP task dict with keys 'code' and 'test_list'
    returns: (function_name, signature_string)
    """
    code      = task["code"]
    tests_str = "\n".join(task["test_list"])
    func_names = re.findall(r"def\s+(\w+)\s*\(.*\)\s*:", code)
    name = next((fn for fn in func_names if fn in tests_str), None)
    if name is None:
        raise ValueError(f"Could not extract function name for task {task['task_id']}")
    namespace = {}
    exec(code, namespace)
    signature = str(inspect.signature(namespace[name]))
    return name, signature

def make_spec_from_task(task: dict):
    """
    Builds a Specification directly from an MBPP task.

    task: a single MBPP task dict
    returns: Specification with name, signature, and description
    """
    name, signature = extract_name_and_signature(task)
    description = task["prompt"].replace("Write a python function to ", "")
    description = description.replace("Write a function to ", "")
    return Specification(name=name, signature=signature, description=description)

def build_prompt(spec: Specification):
    """
    Builds the OpenAI chat messages list for a code generation request.
    The prompt instructs the model to implement the given specification
    and return the full code in a single fenced Python block.

    spec: a Specification object with name, signature, and description
    returns: list of message dicts (system + user)
    """
    system_msg = {"role": "system", "content": "You are world class python programmer"}
    user_content = (
        f"Can you write a function with the following specification:\n\n"
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

def build_baseline_batch(mbpp_file_path:  Path,
                         batch_file_path: Path,
                         model:           str,
                         num_candidates:  int = 10,
                         temperature:     float = None,
                         task_ids:        list=None):
    """
    Writes an OpenAI batch .jsonl file for the baseline experiment.
    Each task produces num_candidates identical requests (using the 
    description from the task's original prompt).
    Custom IDs follow the format: mbpp_{task_id}_sample_{i}

    Optional: Provide a list of MBPP task ids to use only a subset
    of tasks for smaller experiments.

    mbpp_file_path:  path to sanitized-mbpp.json
    batch_file_path: output path for the .jsonl batch file
    model:           OpenAI model name
    num_candidates:  number of samples to request per task
    temperature:     optional sampling temperature
    task_ids:        list of task IDs to include
    """
    tasks   = load_mbpp_tasks(mbpp_file_path, None)
    written = 0
    with open(batch_file_path, "w", encoding="utf-8") as batch_file:
        for task in tasks:
            if task["task_id"] not in task_ids:
                continue
            try:
                spec = make_spec_from_task(task)
            except ValueError as e:
                print(f"Skipping task {task['task_id']}: {e}")
                continue
            messages = build_prompt(spec)
            body     = {"model": model, "messages": messages}
            if temperature is not None:
                body["temperature"] = temperature
            for sample_idx in range(num_candidates):
                request = {
                    "custom_id": f"mbpp_{task['task_id']}_sample_{sample_idx}",
                    "method":    "POST",
                    "url":       "/v1/chat/completions",
                    "body":      body,
                }
                batch_file.write(json.dumps(request) + "\n")
                written += 1
    print(f"Wrote {written} requests to {batch_file_path}")

def load_questions(file_path: Path):
    """
    Loads the questions JSON file and returns a dict keyed by task_id.
    Each value is the 'questions' sub-dict for that task
    (e.g. {"q1": {"question": ..., "description1": ..., ...}, ...}).

    file_path: path to questions.json
    returns: dict mapping task_id -> questions dict
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return {entry["task_id"]: entry["questions"] for entry in json.load(f)}

def build_refinement_batch(mbpp_file_path: Path, 
                           questions_path: Path, 
                           batch_file_path: Path,
                           model: str, 
                           num_samples: int):
    """
    Writes an OpenAI batch .jsonl file for the refinement experiment.
    For each task, requests are generated for all description variants
    (number of questions x 3 description types: oracle, desc1, desc2),
    each with num_samples samples.
    Custom IDs follow the format: mbpp_{task_id}_{q_key}_{label}_sample_{i}

    mbpp_file_path:  path to sanitized-mbpp.json
    questions_path:  path to questions.json (must already contain description fields)
    batch_file_path: output path for the .jsonl batch file
    model:           OpenAI model name
    num_samples:     number of samples to request per variant
    """
    tasks     = load_mbpp_tasks(mbpp_file_path)
    questions = load_questions(questions_path)
    written   = 0
    with open(batch_file_path, "w", encoding="utf-8") as batch_file:
        for task in tasks:
            task_id = task["task_id"]
            if task_id not in questions:
                print(f"No questions for task {task_id}, skipping")
                continue
            try:
                name, signature = extract_name_and_signature(task)
            except ValueError as e:
                print(f"Skipping task {task_id}: {e}")
                continue
            task_questions = questions[task_id]
            for q_key, desc_field, label in DESC_VARIANTS:
                description = task_questions[q_key][desc_field]
                spec        = Specification(name=name, signature=signature, description=description)
                messages    = build_prompt(spec)
                body        = {"model": model, "messages": messages}
                for sample_idx in range(num_samples):
                    request = {
                        "custom_id": f"mbpp_{task_id}_{q_key}_{label}_sample_{sample_idx}",
                        "method":    "POST",
                        "url":       "/v1/chat/completions",
                        "body":      body,
                    }
                    batch_file.write(json.dumps(request) + "\n")
                    written += 1
    print(f"Wrote {written} requests to {batch_file_path}")


"""
DESC_VARIANTS has to be adjusted if the number of questions or description changes.
"""
DESC_VARIANTS = [
    ("q1", "oracle_description", "oracle"),
    ("q1", "description1",       "desc1"),
    ("q1", "description2",       "desc2"),
    ("q2", "oracle_description", "oracle"),
    ("q2", "description1",       "desc1"),
    ("q2", "description2",       "desc2"),
    ("q3", "oracle_description", "oracle"),
    ("q3", "description1",       "desc1"),
    ("q3", "description2",       "desc2"),
]

if __name__ == "__main__":

    mbpp_file_path  = PROJECT_ROOT / "MBPP" / ".data" / "sanitized-mbpp.json"
    model           = "gpt-5-2025-08-07"
    num_candidates  = 10
    temperature     = None

    task_ids = [106, 117, 407, 415, 421, 429, 438, 572, 614, 780]   # optional, only necessary for small experiments

    # --- Baseline ---
    # batch_file_path = Path() # e.g. THIS_DIR / ".MBPP-example-mini" / "baseline" / "batch_request.jsonl"
    # build_baseline_batch(
    #     mbpp_file_path,
    #     batch_file_path,
    #     model,
    #     num_candidates,
    #     temperature,
    #     task_ids
    # )

    # --- Refinement ---
    # batch_file_path     = Path() # e.g. THIS_DIR / ".MBPP-example-mini" / "refined" / "batch_request.jsonl"
    # question_file_path  = Path() # e.g. THIS_DIR / ".MBPP-example-mini" / "refined" / "questions_and_descriptions.json"
    # build_refinement_batch(
    #     mbpp_file_path,
    #     question_file_path,
    #     batch_file_path,
    #     model,
    #     num_candidates
    # )