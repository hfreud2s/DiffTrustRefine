"""
LLM auditor for the HumanEvalComm refinement pipeline.

A single fixed auditor LLM, kept separate from the coder LLMs and from the oracle (to avoid
self-evaluation bias), does two audits:

  #1 question-set audit: for a task with an injected ambiguity, make sure a question
     targeting that ambiguity is present. The auditor compares the original (clear) description with
     the manipulated one; if none of the generated questions covers the introduced ambiguity, it
     writes the missing question, which is appended under the key "q_auditor" with source="auditor"
     and the task's `true_question_missing` flag is set. This is evaluation-only: in a real
     deployment you could not know which question is the "right" one, so it would not join the pool.

  #2 oracle-answer audit: flag oracle answers that leak information beyond what answers the
     question and suggest a tightened rewrite for a human to review.

Auditing runs through the OpenRouter Batch API like every other step: build a request file, submit
it with batch_processing.create_batch, then post-process the results back into
questions_and_descriptions.json.
"""
import json
import re
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from common import EXPERIMENT, CATEGORIES, DATASET_NAME, AUDITOR_MODEL, load_instances, spec_for, provider_obj


# ---------------------------------------------------------------------------
# Auditor #1: is the injected-ambiguity question in the set?
# ---------------------------------------------------------------------------

def generate_audit_question_prompt(original: str, manipulated: str, questions: list):
    """
    Prompt asking the auditor to decide whether any candidate question targets the decision the
    manipulation left open, and, if not, to write the single binary question that a coder seeing only
    the ambiguous spec would ask. The original spec is used only to locate the decision; the auditor
    is told to judge coverage by meaning (not wording) and to phrase any missing question
    answer-neutrally, so it does not leak the ground-truth resolution.
    """
    listed = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(questions)) or "(none)"
    return (
        "You are auditing the clarifying questions a coding model asked about a Python task.\n\n"
        "A clear ORIGINAL specification was deliberately manipulated into an AMBIGUOUS one by "
        "removing, obscuring, or contradicting a detail. Your job is to check whether the model's "
        "questions already include one that surfaces the decision this manipulation left open.\n\n"
        f"ORIGINAL (clear) specification:\n{original}\n\n"
        f"MANIPULATED (ambiguous) specification:\n{manipulated}\n\n"
        f"The model saw ONLY the ambiguous specification and asked:\n{listed}\n\n"
        "Step 1. Compare the two specifications to identify the single decision the manipulation "
        "left underspecified. Use the original only to locate that decision, not to grade the "
        "questions.\n"
        "Step 2. Judge coverage by meaning, not wording. A listed question covers the decision if "
        "answering it would force the same underspecified choice, even when it is phrased more "
        "generally or differently than you would. Do not require it to match the ground-truth "
        "answer or your exact phrasing.\n"
        "Step 3. If none covers it, write the single binary yes/no question that a competent "
        "programmer who had seen ONLY the ambiguous specification would ask to surface that "
        "decision. It must:\n"
        "  - be answerable YES or NO, and about WHAT the function should do, not how\n"
        "  - stay at the same abstraction level as the listed questions\n"
        "  - be answer-neutral: do NOT encode or presuppose the correct behavior, and do NOT "
        "mention specifics that only the ORIGINAL specification reveals. Someone who has not seen "
        "the original should find both YES and NO plausible.\n"
        "For example, for a sorting task prefer \"Should the list be sorted in ascending order?\" "
        "over \"Should the list be sorted ascending before reversing the values between 1 and 9?\", "
        "which leaks the intended answer.\n\n"
        "Output exactly three lines, nothing else:\n"
        "covered: YES or NO\n"
        "covering: the number of the covering question, or NONE\n"
        "question: the missing question, or NONE\n"
    )


def build_audit_questions_batch(llm_dir: str, category: str, model: str = AUDITOR_MODEL,
                                dataset_name: str = DATASET_NAME, provider=None,
                                temperature: float = 0.0):
    """
    Writes the OpenRouter batch that asks the auditor to check each task's question set against the
    injected ambiguity. Reads {category}/refined/questions_and_descriptions.json and, per task,
    embeds the original and manipulated descriptions plus the coder's questions.

    Output: {category}/refined/audit_questions_batch_request.jsonl, custom_id
    "audit_questions__humanevalcomm_{task_id}". `model` is required (the fixed auditor model);
    submit it with batch_processing.create_batch.
    """
    variant     = CATEGORIES[category]
    refined_dir = EXPERIMENT / llm_dir / category / "refined"
    with open(refined_dir / "questions_and_descriptions.json", "r", encoding="utf-8") as f:
        entries = json.load(f)
    instances = load_instances(dataset_name, {e["task_id"] for e in entries})

    out_path = refined_dir / "audit_questions_batch_request.jsonl"
    written = skipped = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for e in entries:
            inst = instances.get(e["task_id"])
            variant_spec = spec_for(inst, variant) if inst is not None else None
            if variant_spec is None:
                skipped += 1
                continue
            # audit only the coder's own questions, not one a previous audit already appended
            questions = [q["question"] for q in e["questions"].values() if q.get("source") != "auditor"]
            content = generate_audit_question_prompt(inst.spec.description, variant_spec.description, questions)
            body = {"model": model, "messages": [{"role": "user", "content": content}]}
            if temperature is not None:
                body["temperature"] = temperature
            if provider is not None:
                body["provider"] = provider_obj(provider)
            f.write(json.dumps({
                "custom_id": f"audit_questions__humanevalcomm_{e['task_id']}",
                "body":      body,
            }) + "\n")
            written += 1
    print(f"{llm_dir}/{category}: {written} audit request(s) -> {out_path} ({skipped} skipped)")
    return out_path


_COVERED_RE = re.compile(r"covered\s*:\s*(YES|NO)", re.IGNORECASE)
_MISSING_RE = re.compile(r"question\s*:\s*(.+)", re.IGNORECASE | re.DOTALL)


def parse_audit(text: str):
    """Parses the auditor's reply into (covered: bool, missing_question: str | None)."""
    m = _COVERED_RE.search(text)
    covered = bool(m) and m.group(1).upper() == "YES"
    missing = None
    if not covered:
        mq = _MISSING_RE.search(text)
        if mq:
            first = mq.group(1).strip().splitlines()[0].strip()
            if first and first.upper() != "NONE":
                missing = first
    return covered, missing


def postprocess_audit_questions(llm_dir: str, category: str,
                                result_name: str = "audit_questions_batch_result.jsonl"):
    """
    Applies the auditor's verdicts to {category}/refined/questions_and_descriptions.json.

    If the auditor found the injected ambiguity already covered, nothing changes. Otherwise the
    task's `true_question_missing` is set True and the auditor's question is appended under the key
    "q_auditor" with source="auditor" (scored later for incoherence reduction, but flagged as not
    part of the practical question pool).
    """
    from common import blank_question
    from batch_processing import extract_text   # reuse the batch-result text extractor
    refined_dir = EXPERIMENT / llm_dir / category / "refined"
    qd_path = refined_dir / "questions_and_descriptions.json"
    entries = json.load(open(qd_path, encoding="utf-8"))
    by_id = {e["task_id"]: e for e in entries}

    updated = added = skipped = 0
    with open(refined_dir / result_name, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            text = extract_text(entry)
            if text is None:
                skipped += 1
                continue
            task_id = int(entry["custom_id"].split("__")[1].split("_")[1])
            e = by_id.get(task_id)
            if e is None:
                continue
            covered, missing = parse_audit(text)
            e["true_question_missing"] = not covered
            if not covered and missing and "q_auditor" not in e["questions"]:
                e["questions"]["q_auditor"] = blank_question(missing, source="auditor")
                added += 1
            updated += 1

    with open(qd_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)
    print(f"{llm_dir}/{category}: audited {updated} task(s), added {added} auditor question(s), "
          f"{skipped} failed response(s)")
    return qd_path


# ---------------------------------------------------------------------------
# Auditor #2: does the oracle answer leak information beyond the question?
# ---------------------------------------------------------------------------

def generate_audit_oracle_prompt(manipulated: str, question: str, oracle_description: str):
    """
    Prompt asking the auditor whether an oracle-written refined description leaks information beyond
    what answers its one question. The auditor sees ONLY the ambiguous task and the question (never
    the ground truth): anything the description settles that is neither already in the ambiguous task
    nor a direct answer to this question is a leak. It flags the leak and suggests a tightened
    rewrite for a human to review; it never edits anything itself.
    """
    return (
        "You are auditing a refined task specification that a reference ('oracle') wrote to answer "
        "ONE clarifying question about an under-specified Python task.\n\n"
        f"The ambiguous task:\n{manipulated}\n\n"
        f"The one question this description is meant to answer:\n{question}\n\n"
        f"The oracle's refined description:\n{oracle_description}\n\n"
        "The description is allowed to contain exactly two things: (1) what the ambiguous task "
        "already stated, and (2) the answer to THIS question. Anything else is LEAKAGE: resolving a "
        "DIFFERENT open point of the task, adding a constraint the task did not state (input "
        "lengths, non-emptiness, types, ordering, edge cases), or referring to a reference solution, "
        "tests, or specific example values. Judge by meaning; restating or paraphrasing the task is "
        "not leakage.\n\n"
        "If it leaks, say what leaked (the specific extra content) and give a rewrite that keeps the "
        "task and this question's answer but removes everything else, at the same level of detail as "
        "the ambiguous task.\n\n"
        "Output exactly three lines, nothing else:\n"
        "leak: YES or NO\n"
        "leaked: the specific leaked information, or NONE\n"
        "rewrite: the tightened description, or NONE\n"
    )


def build_audit_oracle_batch(llm_dir: str, category: str, model: str = AUDITOR_MODEL,
                             dataset_name: str = DATASET_NAME, provider=None,
                             temperature: float = 0.0):
    """
    Writes the batch that asks the auditor to leak-check every filled oracle_description in
    {category}/refined/questions_and_descriptions.json (one request per (task, question) that has an
    oracle_description, including q_auditor). custom_id
    "audit_oracle__humanevalcomm_{task_id}__{qkey}"; output
    {category}/refined/audit_oracle_batch_request.jsonl. Submit with batch_processing.create_batch;
    post-process with postprocess_audit_oracle.
    """
    variant     = CATEGORIES[category]
    refined_dir = EXPERIMENT / llm_dir / category / "refined"
    with open(refined_dir / "questions_and_descriptions.json", "r", encoding="utf-8") as f:
        entries = json.load(f)
    instances = load_instances(dataset_name, {e["task_id"] for e in entries})

    out_path = refined_dir / "audit_oracle_batch_request.jsonl"
    written = skipped = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for e in entries:
            inst = instances.get(e["task_id"])
            spec = spec_for(inst, variant) if inst is not None else None
            if spec is None:
                skipped += 1
                continue
            for qkey, q in e["questions"].items():
                oracle = q.get("oracle_description")
                if not oracle:
                    continue
                content = generate_audit_oracle_prompt(spec.description, q["question"], oracle)
                body = {"model": model, "messages": [{"role": "user", "content": content}]}
                if temperature is not None:
                    body["temperature"] = temperature
                if provider is not None:
                    body["provider"] = provider_obj(provider)
                f.write(json.dumps({
                    "custom_id": f"audit_oracle__humanevalcomm_{e['task_id']}__{qkey}",
                    "body":      body,
                }) + "\n")
                written += 1
    print(f"{llm_dir}/{category}: {written} oracle-audit request(s) -> {out_path} ({skipped} task(s) skipped)")
    return out_path


_LEAK_RE    = re.compile(r"leak\s*:\s*(YES|NO)", re.IGNORECASE)
_LEAKED_RE  = re.compile(r"leaked\s*:\s*(.+?)(?:\n\s*rewrite\s*:|$)", re.IGNORECASE | re.DOTALL)
_REWRITE_RE = re.compile(r"rewrite\s*:\s*(.+)", re.IGNORECASE | re.DOTALL)


def parse_oracle_audit(text: str):
    """Parses the auditor's reply into (leak: bool, leaked: str | None, rewrite: str | None)."""
    m = _LEAK_RE.search(text)
    leak = bool(m) and m.group(1).upper() == "YES"

    def grab(rx):
        g = rx.search(text)
        if not g:
            return None
        val = g.group(1).strip()
        return None if not val or val.upper() == "NONE" else val

    return leak, grab(_LEAKED_RE), grab(_REWRITE_RE)


def postprocess_audit_oracle(llm_dir: str, category: str,
                             result_name: str = "audit_oracle_batch_result.jsonl"):
    """
    Applies the oracle-audit verdicts to {category}/refined/questions_and_descriptions.json: for each
    (task, question) it fills oracle_leak (the leaked content, or None if clean) and oracle_rewrite
    (the auditor's tightened description, or None). Nothing is auto-applied; a human reviews the
    flags and decides whether to swap in a rewrite.
    """
    from batch_processing import extract_text
    refined_dir = EXPERIMENT / llm_dir / category / "refined"
    qd_path = refined_dir / "questions_and_descriptions.json"
    entries = json.load(open(qd_path, encoding="utf-8"))
    by_id = {e["task_id"]: e for e in entries}

    flagged = clean = skipped = 0
    with open(refined_dir / result_name, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            text  = extract_text(entry)
            parts = entry.get("custom_id", "").split("__")   # audit_oracle, humanevalcomm_{id}, {qkey}
            if text is None or len(parts) != 3:
                skipped += 1
                continue
            task_id, qkey = int(parts[1].split("_")[1]), parts[2]
            e = by_id.get(task_id)
            if e is None or qkey not in e["questions"]:
                skipped += 1
                continue
            leak, leaked, rewrite = parse_oracle_audit(text)
            q = e["questions"][qkey]
            q["oracle_leak"]    = leaked if leak else None
            q["oracle_rewrite"] = rewrite if leak else None
            flagged += leak
            clean   += (not leak)

    with open(qd_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)
    print(f"{llm_dir}/{category}: oracle audit -> {flagged} flagged, {clean} clean, {skipped} skipped")
    return qd_path


if __name__ == "__main__":

    # --- Auditor #1: build the audit batch, submit, then apply the verdicts ---
    # build_audit_questions_batch("LLM1", "1a", AUDITOR_MODEL)
    # from batch_processing import create_batch, save_batch_meta, wait_for_batch, save_results
    # req = EXPERIMENT / "LLM1" / "1a" / "refined" / "audit_questions_batch_request.jsonl"
    # batch = create_batch(req) 
    # save_batch_meta(batch, req)

    # --- Later: wait for a batch and save its results ---
    # batch_id = json.load(open(EXPERIMENT / "LLM1" / "1a" / "refined" / "audit_questions_batch_meta.json"))["batch_id"]
    # batch    = wait_for_batch(batch_id, poll=60)
    # save_results(batch, EXPERIMENT / "LLM1" / "1a" / "refined" / "audit_questions_batch_result.jsonl")

    # postprocess_audit_questions("LLM1", "1a")

    # --- Auditor #2 (round 4.5): leak-check the oracle descriptions, then apply the flags ---
    # build_audit_oracle_batch("LLM1", "1a")
    # req = EXPERIMENT / "LLM1" / "1a" / "refined" / "audit_oracle_batch_request.jsonl"
    # batch = create_batch(req)
    # save_batch_meta(batch, req)
    # batch_id = json.load(open(EXPERIMENT / "LLM1" / "1a" / "refined" / "audit_oracle_batch_meta.json"))["batch_id"]
    # batch    = wait_for_batch(batch_id, poll=60)
    # save_results(batch, EXPERIMENT / "LLM1" / "1a" / "refined" / "audit_oracle_batch_result.jsonl")
    # postprocess_audit_oracle("LLM1", "1a")
    pass
