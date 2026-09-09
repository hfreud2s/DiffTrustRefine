"""
Refinement phase for HumanEvalComm: clarifying questions and refined descriptions.

The refinement chain is: questions -> refined (yes/no) descriptions -> oracle (true) description ->
candidates from each description -> score. Each round consumes the previous round's results.

Step 5: question generation. Each coder LLM asks its own clarifying questions (per default)
about a category's (possibly ambiguous) description. build_questions_batch() writes the batch request file.
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import CATEGORIES, EXPERIMENT, DATASET_NAME, load_instances, spec_for


def needs_refinement(llm_dir: str, category: str, phase: str = "baseline"):
    """
    The task ids of a (LLM, category) that need clarifying questions: those whose baseline mean
    incoherence is above zero. Reads the cross-run summary {phase}/aggregate.json produced by
    data_analysis.aggregate_phase.

    returns: set of task ids with mean_incoherence > 0
    """
    aggregate_path = EXPERIMENT / llm_dir / category / phase / "aggregate.json"
    with open(aggregate_path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    return {row["task_id"] for row in rows
            if row.get("mean_incoherence") is not None and row["mean_incoherence"] > 0}


def infer_model(llm_dir: str, category: str):
    """Reads the model originally used for the category, from its baseline request file."""
    req_path = EXPERIMENT / llm_dir / category / "baseline_batch_request.jsonl"
    with open(req_path, "r", encoding="utf-8") as f:
        first = json.loads(f.readline())
    return first["body"]["model"]


def generate_binary_questions(description: str, num_questions: int = 3):
    """
    Builds the prompt that asks the model to identify genuinely underspecified behaviours in a task
    description as yes/no questions. Ported from the MBPP pipeline; takes the description of the
    category being refined (the ambiguous one for a manipulated category).
    """
    prompt = (
        f"You are analyzing a Python programming task to identify genuine implementation ambiguities.\n\n"
        f"Task specification:\n{description}\n\n"
        f"Identify {num_questions} binary question(s) about this specification where:\n"
        f"- Each question targets a specific behavior that is genuinely underspecified\n"
        f"- Each question can be answered with either YES or NO\n"
        f"- A YES answer and a NO answer would lead to observably different code (different outputs on at least one input)\n"
        f"- The question cannot be answered just by reading the specification carefully\n"
        f"- The question is about WHAT the function should return or do, not HOW to implement it\n\n"
        f"For example: Consider the specification \"Sort a given list of integers.\"\n"
        f"Good question: 'Should the list be sorted in increasing order?'\n\n"
        f"Bad question: 'How should the function handle edge cases?' (too vague, not binary)\n\n"
        f"Output format:\n"
        f"question 1: ...\nquestion 2: ...\nquestion 3: ...\n\n"
        f"Output only the questions, nothing else."
    )
    return prompt


def build_questions_batch(llm_dir: str, category: str, model: str = None,
                          num_questions: int = 3, dataset_name: str = DATASET_NAME):
    """
    Writes the OpenRouter batch request that asks the LLM to generate clarifying questions for the
    tasks of (llm_dir, category) whose baseline incoherence is above zero.

    Output: .HEC-experiment/{llm_dir}/{category}/refined/questions_batch_request.jsonl
    One item per task, custom_id "questions__humanevalcomm_{task_id}", body is a chat request 
    whose prompt embeds that category's description.

    llm_dir/category: which (LLM, category) to build for
    model:            Question-generation model (defaults to the one the baseline batch used)
    num_questions:    binary questions to ask per task (default 3)
    returns:          the path written, or None if nothing needs refinement
    """
    variant = CATEGORIES[category]
    need    = needs_refinement(llm_dir, category)
    if not need:
        print(f"{llm_dir}/{category}: no task has incoherence > 0, nothing to refine")
        return None
    instances = load_instances(dataset_name)
    if model is None:
        model = infer_model(llm_dir, category)

    out_dir  = EXPERIMENT / llm_dir / category / "refined"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "questions_batch_request.jsonl"

    written = skipped = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for task_id in sorted(need):
            inst = instances.get(task_id)
            spec = spec_for(inst, variant) if inst is not None else None
            if spec is None:
                skipped += 1
                continue
            content = generate_binary_questions(spec.description, num_questions)
            request = {
                "custom_id": f"questions__humanevalcomm_{task_id}",
                "body":      {"model": model, "messages": [{"role": "user", "content": content}]},
            }
            f.write(json.dumps(request) + "\n")
            written += 1
    print(f"{llm_dir}/{category}: {written} question request(s) -> {out_path} "
          f"({skipped} skipped; {len(need)} tasks need refinement)")
    return out_path


if __name__ == "__main__":

    # --- Round 1: build the clarifying-question batch, then submit it with batch_processing ---
    # build_questions_batch("LLM1", "1a", num_questions=3)

    # from batch_processing import create_batch, save_batch_meta, wait_for_batch, save_results
    # req = EXPERIMENT / "LLM1" / "1a" / "refined" / "questions_batch_request.jsonl"
    # batch = create_batch(req) 
    # save_batch_meta(batch, req)

    # --- Later: wait for a batch and save its results ---
    # batch_id = json.load(open(EXPERIMENT / "LLM1" / "1a" / "refined" / "questions_batch_meta.json"))["batch_id"]
    # batch    = wait_for_batch(batch_id, poll=60)
    # save_results(batch, EXPERIMENT / "LLM1" / "1a" / "refined" / "questions_batch_result.jsonl")
    pass
