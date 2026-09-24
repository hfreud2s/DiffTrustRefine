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
from common import CATEGORIES, EXPERIMENT, DATASET_NAME, ORACLE_MODEL, load_instances, spec_for, provider_obj
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
                          num_questions: int = 3, dataset_name: str = DATASET_NAME,
                          provider=None, temperature: float = 0.0):
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
            body = {"model": model, "messages": [{"role": "user", "content": content}]}
            if temperature is not None:
                body["temperature"] = temperature
            if provider is not None:
                body["provider"] = provider_obj(provider)
            f.write(json.dumps({
                "custom_id": f"questions__humanevalcomm_{task_id}",
                "body":      body,
            }) + "\n")
            written += 1
    print(f"{llm_dir}/{category}: {written} question request(s) -> {out_path} "
          f"({skipped} skipped; {len(need)} tasks need refinement)")
    return out_path


def generate_binary_descriptions(description: str, question: str):
    """
    Builds the prompt asking the coder to write two refined specifications, one for a YES answer and
    one for a NO answer to the question. Each is self-contained but describes the SAME task at the
    SAME level of (under)specification as the (possibly ambiguous) category description, changing
    only what this one question resolves. The two differ from each other in nothing but that answer,
    so no other ambiguity of the task is resolved or invented. Ported from the MBPP pipeline.
    """
    return (
        f"Given this Python programming task:\n"
        f"{description}\n\n"
        f"And this yes/no question about it:\n"
        f"{question}\n\n"
        f"Write exactly two refined task specifications: one assuming the answer to the question is "
        f"YES, one assuming NO. Each must be SELF-CONTAINED (a reader who has not seen the original "
        f"task or the question can read it on its own), but describe the SAME task at the SAME level "
        f"of detail as the original, changing only what this question resolves. Self-contained does "
        f"not mean fully specified: if the original task is vague about something, your specification "
        f"stays exactly as vague about it.\n"
        f"Rules:\n"
        f"- Use ONLY information stated in the original task, plus the answer to this one question. Do "
        f"not invent, assume, or resolve anything the original leaves open: not input constraints "
        f"(e.g. lengths, non-emptiness, types), not edge cases, not other ambiguities. If the "
        f"original does not state it, your specification must not state it either.\n"
        f"- Treat any input/output examples in the task as illustrations only. Do NOT turn an "
        f"incidental property of an example into a stated rule (e.g. if an example happens to use "
        f"equal-length inputs, do not assume the inputs are always equal length).\n"
        f"- The two specifications must be identical to each other except for the part this question's "
        f"answer changes.\n"
        f"- Describe the expected RETURN VALUE or BEHAVIOR, not implementation steps. No code or examples.\n\n"
        f"Output format:\n"
        f"description 1: ...\ndescription 2: ...\n\n"
        f"Example 1:\n"
        f"Task \"Sort a list of integers.\", question \"Should the list be sorted in ascending order?\"\n"
        f"description 1: Sort a list of integers in ascending order.\n"
        f"description 2: Sort a list of integers in descending order.\n\n"
        f"Example 2:\n"
        f"Task \"Given a list of numbers, return the ones that are above the average.\", "
        f"question \"Should numbers equal to the average be included?\"\n"
        f"description 1: Given a list of numbers, return the numbers that are greater than or equal to "
        f"the average of the list.\n"
        f"description 2: Given a list of numbers, return the numbers that are strictly greater than the "
        f"average of the list.\n"
        f"Here it would be wrong to add that the result is sorted, or that the list is non-empty since the "
        f"original states neither."
    )


def build_descriptions_batch(llm_dir: str, category: str, model: str = None,
                             dataset_name: str = DATASET_NAME, provider=None,
                             temperature: float = 0.0):
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
                body = {"model": model, "messages": [{"role": "user", "content": content}]}
                if temperature is not None:
                    body["temperature"] = temperature
                if provider is not None:
                    body["provider"] = provider_obj(provider)
                f.write(json.dumps({
                    "custom_id": f"descriptions__humanevalcomm_{e['task_id']}__{qkey}",
                    "body": body,
                }) + "\n")
                written += 1
    print(f"{llm_dir}/{category}: {written} description request(s) -> {out_path} "
          f"({skipped} task(s) skipped)")
    return out_path



def generate_oracle_prompt(description: str, question: str, code: str, test: str):
    """
    Builds the prompt that asks the oracle for the ground-truth ("true") refined specification: the
    task with THIS one question resolved the way the reference solution actually behaves, and nothing
    else changed. The reference code and tests are used ONLY to decide this question's answer; every
    other under-specification is left as open as the original, so the oracle branch stays comparable
    to the coder's YES/NO branches and isolates this question's effect. The oracle must not reveal the
    reference solution, the tests, or specific values (checked later by auditor #2).

    description: the category's (ambiguous) task description, as the coder saw it
    question:    the clarifying question whose answer the specification must embed
    code:        the reference solution (inst.code), used only to determine this question's answer
    test:        the reference test cases (inst.test), for the same purpose only
    """
    return (
        f"Given this Python programming task (it is under-specified in several ways):\n"
        f"{description}\n\n"
        f"And this yes/no question about it:\n"
        f"{question}\n\n"
        f"Here is the reference solution. Use it ONLY to determine the correct answer to this one "
        f"question:\n{code}\n\n"
        f"And its test cases, for the same purpose only:\n{test}\n\n"
        f"Write a single refined specification: the task with THIS question resolved the way the "
        f"reference solution actually behaves, and nothing else changed. It must:\n"
        f"- Be self-contained (readable on its own) but describe the SAME task at the SAME level of "
        f"detail as the original, resolving ONLY this question. Self-contained does NOT mean fully "
        f"specified: leave every other under-specification exactly as open as the original (and treat "
        f"any examples in the task as illustrations, not constraints).\n"
        f"- Use the reference solution and tests ONLY to decide this question's answer. Do not "
        f"resolve, assume, or state anything else the original leaves open: not other ambiguities, "
        f"edge cases, input constraints, or types.\n"
        f"- Not mention or reveal the reference solution, the tests, or any specific example/test "
        f"values. Describe the expected RETURN VALUE or BEHAVIOR, not implementation steps. No code "
        f"or examples.\n\n"
        f"Output format:\n"
        f"description: ...\n\n"
        f"Example: original task \"Sort a list of integers.\", question \"Should the list be sorted "
        f"in ascending order?\", and suppose the reference solution sorts ascending.\n"
        f"description: Sort a list of integers in ascending order.\n\n"
        f"Bad output: \"Based on the reference solution and test cases, sort the list in ascending "
        f"order in place.\" It is bad because it mentions the reference/tests and adds information the "
        f"question did not ask about (in-place)."
    )


def build_oracle_batch(llm_dir: str, category: str, model: str = ORACLE_MODEL,
                       dataset_name: str = DATASET_NAME, provider=None,
                       temperature: float = 0.0):
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
                body = {"model": model, "messages": [{"role": "user", "content": content}]}
                if temperature is not None:
                    body["temperature"] = temperature
                if provider is not None:
                    body["provider"] = provider_obj(provider)
                f.write(json.dumps({
                    "custom_id": f"oracle__humanevalcomm_{e['task_id']}__{qkey}",
                    "body": body,
                }) + "\n")
                written += 1
    print(f"{llm_dir}/{category}: {written} oracle request(s) -> {out_path} "
          f"({skipped} task(s) skipped)")
    return out_path


if __name__ == "__main__":

    # --- Which tasks need refinement (baseline mean incoherence > 0), the set every round below acts on ---
    # print(sorted(needs_refinement("LLM1", "1a")))

    # --- Round 1: build the clarifying-question batch, then submit it with batch_processing ---
    # build_questions_batch("LLM1", "1a", num_questions=3)

    # req = EXPERIMENT / "LLM1" / "1a" / "refined" / "questions_batch_request.jsonl"
    # batch = create_batch(req) 
    # save_batch_meta(batch, req)

    # --- Later: wait for a batch and save its results ---
    # batch_id = json.load(open(EXPERIMENT / "LLM1" / "1a" / "refined" / "questions_batch_meta.json"))["batch_id"]
    # batch    = wait_for_batch(batch_id, poll=60)
    # save_results(batch, EXPERIMENT / "LLM1" / "1a" / "refined" / "questions_batch_result.jsonl")
    # then in batch_processing.py: postprocess_questions("LLM1", "1a")
    # (optional) Round 2 = auditor #1 on the question set: see audit.py

    # --- Round 3: build the YES/NO refined descriptions, then submit and retrieve ---
    # build_descriptions_batch("LLM1", "1a")
    # req = EXPERIMENT / "LLM1" / "1a" / "refined" / "descriptions_batch_request.jsonl"
    # batch = create_batch(req)
    # save_batch_meta(batch, req)
    # batch_id = json.load(open(EXPERIMENT / "LLM1" / "1a" / "refined" / "descriptions_batch_meta.json"))["batch_id"]
    # batch    = wait_for_batch(batch_id, poll=60)
    # save_results(batch, EXPERIMENT / "LLM1" / "1a" / "refined" / "descriptions_batch_result.jsonl")
    # then in batch_processing.py: postprocess_descriptions("LLM1", "1a")

    # --- Round 4: build the oracle's ground-truth description, then submit and retrieve ---
    # build_oracle_batch("LLM1", "1a")   # uses the fixed ORACLE_MODEL
    # req = EXPERIMENT / "LLM1" / "1a" / "refined" / "oracle_batch_request.jsonl"
    # batch = create_batch(req)
    # save_batch_meta(batch, req)
    # batch_id = json.load(open(EXPERIMENT / "LLM1" / "1a" / "refined" / "oracle_batch_meta.json"))["batch_id"]
    # batch    = wait_for_batch(batch_id, poll=60)
    # save_results(batch, EXPERIMENT / "LLM1" / "1a" / "refined" / "oracle_batch_result.jsonl")
    # then in batch_processing.py: postprocess_oracle("LLM1", "1a")

    # --- Round 5 (refined candidates) is built in build_batch.py: build_refined_candidates_batch("LLM1", "1a"),
    #     then batch_processing.postprocess_refined_candidates, compute_stats.score_phase(phase="refined"),
    #     data_analysis.aggregate_phase(phase="refined") ---
    pass
