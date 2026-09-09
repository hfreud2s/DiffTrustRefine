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
from common import CATEGORIES, EXPERIMENT, DATASET_NAME, ORACLE_MODEL, load_instances, spec_for
from batch_processing import create_batch, save_batch_meta, wait_for_batch, save_results


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


def generate_binary_descriptions(description: str, question: str):
    """
    Builds the prompt asking the coder to write two refined specifications, one for a YES answer and
    one for a NO answer to the question, each standalone and behaviour-focused and differing only in
    what the answer resolves. Ported from the MBPP pipeline. It uses the category's (possibly
    ambiguous) description, since this is the coder's own what-if scoring.
    """
    return (
        f"Given this Python programming task:\n"
        f"{description}\n\n"
        f"And this yes/no question about it:\n"
        f"{question}\n\n"
        f"Write exactly two refined specifications, one for YES and one for NO, describing what the "
        f"function should do in each case.\n"
        f"Each answer must:\n"
        f"- Describe the expected RETURN VALUE or BEHAVIOR of the function, not implementation steps\n"
        f"- Contain no additional information besides the one that answers the question\n"
        f"- Be genuinely different from the other answer in observable output\n"
        f"- Contain no code or examples\n\n"
        f"Output format:\n"
        f"description 1: ...\ndescription 2: ...\n\n"
        f"One example:\n"
        f"Suppose the original task description was \"Sort a list of integers.\"\n"
        f"And the question would be \"Should the list be sorted in ascending order?\"\n"
        f"The correct output would be:\n\n"
        f"description 1: Sort a list of integers in ascending order.\n"
        f"description 2: Sort a list of integers in descending order.\n\n"
        f"Bad output would be something like "
        f"\"The function should return the list of elements sorted in descending order (in-place)\" "
        f"since this is not a standalone task description and adds extra information (the in-place sorting)."
    )


def build_descriptions_batch(llm_dir: str, category: str, model: str = None,
                             dataset_name: str = DATASET_NAME):
    """
    Writes the OpenRouter batch asking the coder LLM for the YES/NO refined descriptions of every
    question in {category}/refined/questions_and_descriptions.json, including any q_auditor question
    (it is scored like the rest). One request per (task, question); custom_id
    "descriptions__humanevalcomm_{task_id}__{qkey}". Output:
    {category}/refined/descriptions_batch_request.jsonl. Model defaults to the coder's own baseline
    slug. Post-process with batch_processing.postprocess_descriptions.
    """
    variant     = CATEGORIES[category]
    refined_dir = EXPERIMENT / llm_dir / category / "refined"
    with open(refined_dir / "questions_and_descriptions.json", "r", encoding="utf-8") as f:
        entries = json.load(f)
    instances = load_instances(dataset_name, {e["task_id"] for e in entries})
    if model is None:
        model = infer_model(llm_dir, category)

    out_path = refined_dir / "descriptions_batch_request.jsonl"
    written = skipped = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for e in entries:
            inst = instances.get(e["task_id"])
            spec = spec_for(inst, variant) if inst is not None else None
            if spec is None:
                skipped += 1
                continue
            for qkey, q in e["questions"].items():
                content = generate_binary_descriptions(spec.description, q["question"])
                f.write(json.dumps({
                    "custom_id": f"descriptions__humanevalcomm_{e['task_id']}__{qkey}",
                    "body": {"model": model, "messages": [{"role": "user", "content": content}]},
                }) + "\n")
                written += 1
    print(f"{llm_dir}/{category}: {written} description request(s) -> {out_path} "
          f"({skipped} task(s) skipped)")
    return out_path



def generate_oracle_prompt(description: str, question: str, code: str, test: str):
    """
    Builds the prompt that asks the oracle to write the ground-truth ("true") refined
    specification: the one refinement the ambiguous description should have had, resolved by
    consulting the reference solution and its tests rather than guessing. The oracle answers the
    question from what the canonical code actually does. This is the evaluation signal against which
    the coder's own YES/NO descriptions are later measured.

    description: the category's (ambiguous) task description, as the coder saw it
    question:    the clarifying question whose answer the description must embed
    code:        the reference solution (inst.code) that resolves the ambiguity
    test:        the reference test cases (inst.test), for extra disambiguation
    returns:     the prompt string
    """
    return (
        f"Given this Python programming task:\n"
        f"{description}\n\n"
        f"And this yes/no question about it:\n"
        f"{question}\n\n"
        f"To answer the question you can use the ground truth for reference:\n"
        f"{code}\n\n"
        f"You can also analyse the following test cases if necessary:\n"
        f"{test}\n\n"
        f"Your task is to write a refined specification for the programming task. "
        f"The description must:\n"
        f"- Contain the answer to the question\n"
        f"- Describe the expected RETURN VALUE or BEHAVIOR of the function, not implementation steps\n"
        f"- Contain no additional information besides the one that answers the question\n"
        f"- Contain no code or examples\n\n"
        f"Output format:\n"
        f"description: ...\n\n"
        f"One example:\n"
        f"Suppose the original task description was \"Sort a list of integers.\"\n"
        f"And the question would be \"Should the list be sorted in ascending order?\"\n"
        f"Suppose the ground truth sorts ascending. Then the correct output would be:\n\n"
        f"description: Sort a list of integers in ascending order.\n\n"
        f"Bad output would be something like: "
        f"\"Based on the given ground truth and test cases, the function should return the list "
        f"sorted in ascending order (in-place).\" "
        f"It is bad because it mentions the ground truth and test cases and adds extra information "
        f"(the in-place sorting)."
    )


def build_oracle_batch(llm_dir: str, category: str, model: str = ORACLE_MODEL,
                       dataset_name: str = DATASET_NAME):
    """
    Writes the OpenRouter batch asking the fixed oracle for the ground-truth description of every
    question in {category}/refined/questions_and_descriptions.json, including any q_auditor question.
    One request per (task, question); custom_id "oracle__humanevalcomm_{task_id}__{qkey}". Output:
    {category}/refined/oracle_batch_request.jsonl. Post-process with
    batch_processing.postprocess_oracle, which fills oracle_description.

    Unlike the coder rounds, the model is the fixed ORACLE_MODEL (the same simulated user across all
    LLMs and categories), not the coder's own slug.
    """
    variant     = CATEGORIES[category]
    refined_dir = EXPERIMENT / llm_dir / category / "refined"
    with open(refined_dir / "questions_and_descriptions.json", "r", encoding="utf-8") as f:
        entries = json.load(f)
    instances = load_instances(dataset_name, {e["task_id"] for e in entries})

    out_path = refined_dir / "oracle_batch_request.jsonl"
    written = skipped = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for e in entries:
            inst = instances.get(e["task_id"])
            spec = spec_for(inst, variant) if inst is not None else None
            if spec is None:
                skipped += 1
                continue
            for qkey, q in e["questions"].items():
                content = generate_oracle_prompt(spec.description, q["question"], inst.code, inst.test)
                f.write(json.dumps({
                    "custom_id": f"oracle__humanevalcomm_{e['task_id']}__{qkey}",
                    "body": {"model": model, "messages": [{"role": "user", "content": content}]},
                }) + "\n")
                written += 1
    print(f"{llm_dir}/{category}: {written} oracle request(s) -> {out_path} "
          f"({skipped} task(s) skipped)")
    return out_path


if __name__ == "__main__":

    # --- Round 1: build the clarifying-question batch, then submit it with batch_processing ---
    # build_questions_batch("LLM1", "1a", num_questions=3)

    # req = EXPERIMENT / "LLM1" / "1a" / "refined" / "questions_batch_request.jsonl"
    # batch = create_batch(req) 
    # save_batch_meta(batch, req)

    # --- Later: wait for a batch and save its results ---
    # batch_id = json.load(open(EXPERIMENT / "LLM1" / "1a" / "refined" / "questions_batch_meta.json"))["batch_id"]
    # batch    = wait_for_batch(batch_id, poll=60)
    # save_results(batch, EXPERIMENT / "LLM1" / "1a" / "refined" / "questions_batch_result.jsonl")

    # --- Round 3: build the YES/NO refined descriptions, then submit and retrieve ---
    # build_descriptions_batch("LLM1", "1a")
    # req = EXPERIMENT / "LLM1" / "1a" / "refined" / "descriptions_batch_request.jsonl"
    # batch = create_batch(req)
    # save_batch_meta(batch, req)
    # batch_id = json.load(open(EXPERIMENT / "LLM1" / "1a" / "refined" / "descriptions_batch_meta.json"))["batch_id"]
    # batch    = wait_for_batch(batch_id, poll=60)
    # save_results(batch, EXPERIMENT / "LLM1" / "1a" / "refined" / "descriptions_batch_result.jsonl")

    # --- Round 4: build the oracle's ground-truth description, then submit and retrieve ---
    # build_oracle_batch("LLM1", "1a")   # uses the fixed ORACLE_MODEL
    # req = EXPERIMENT / "LLM1" / "1a" / "refined" / "oracle_batch_request.jsonl"
    # batch = create_batch(req)
    # save_batch_meta(batch, req)
    # batch_id = json.load(open(EXPERIMENT / "LLM1" / "1a" / "refined" / "oracle_batch_meta.json"))["batch_id"]
    # batch    = wait_for_batch(batch_id, poll=60)
    # save_results(batch, EXPERIMENT / "LLM1" / "1a" / "refined" / "oracle_batch_result.jsonl")
    pass
