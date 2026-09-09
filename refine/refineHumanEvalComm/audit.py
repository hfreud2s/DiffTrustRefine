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
from common import EXPERIMENT, CATEGORIES, DATASET_NAME, AUDITOR_MODEL, load_instances, spec_for


# ---------------------------------------------------------------------------
# Auditor #1: is the injected-ambiguity question in the set?
# ---------------------------------------------------------------------------

def generate_audit_question_prompt(original: str, manipulated: str, questions: list):
    """
    Prompt asking the auditor to decide whether any candidate question targets the ambiguity that
    the manipulation introduced (the difference between the original and manipulated descriptions),
    and, if not, to write the single binary question that does.
    """
    listed = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(questions)) or "(none)"
    return (
        "You are auditing clarifying questions for a Python programming task.\n\n"
        "A clear ORIGINAL specification was deliberately manipulated into an AMBIGUOUS one by "
        "removing, obscuring, or contradicting some detail. Check whether the generated questions "
        "already contain one that targets that specific introduced ambiguity.\n\n"
        f"ORIGINAL (clear) specification:\n{original}\n\n"
        f"MANIPULATED (ambiguous) specification:\n{manipulated}\n\n"
        f"Generated candidate questions:\n{listed}\n\n"
        "First work out what the manipulation made ambiguous (the difference between the two "
        "specifications). Then decide whether any listed question, if answered, would resolve that "
        "specific ambiguity. If one does, report which. If none does, write the single binary "
        "yes/no question that targets the introduced ambiguity: it must be answerable with YES or "
        "NO, be about WHAT the function should do (not how), and match the style of the listed "
        "questions.\n\n"
        "Output exactly three lines, nothing else:\n"
        "covered: YES or NO\n"
        "covering: the number of the covering question, or NONE\n"
        "question: the missing question, or NONE\n"
    )


def build_audit_questions_batch(llm_dir: str, category: str, model: str = AUDITOR_MODEL,
                                dataset_name: str = DATASET_NAME):
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
            f.write(json.dumps({
                "custom_id": f"audit_questions__humanevalcomm_{e['task_id']}",
                "body":      {"model": model, "messages": [{"role": "user", "content": content}]},
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
    pass
