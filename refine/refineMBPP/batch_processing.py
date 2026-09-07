"""
This file contains all necessary function to handle batch requests:
- submit batch requests 
- re-submit requests that weren't completed
- merge files with completed requests
- postprocess batch results
"""
import json
import re
import cloudpickle
from pathlib import Path
from openai import OpenAI

THIS_DIR = Path(__file__).resolve().parent

def create_batch(client: OpenAI, 
                 batch_file_path: Path):
    """
    Uploads a .jsonl batch file to the OpenAI Files API then creates an OpenAI batch job from this file.

    client:     an authenticated OpenAI client instance
    batch_file_path: path to the .jsonl file to upload
    """
    with open(batch_file_path, "rb") as f:
        batch_file = client.files.create(file=f, purpose="batch")
    
    return client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )

def recreate_batch(original_path: Path, 
                   error_path: Path, 
                   output_path: Path = None):
    """
    Builds a new batch .jsonl file containing only the requests that failed
    in a previous batch run. Failed requests are identified by their custom_id
    in the error file. If no output_path is given, the repair file is written
    next to the original with '_repair' appended to the stem.

    original_path: path to the original batch .jsonl file
    error_path:    path to the error output .jsonl file from the failed batch
    output_path:   optional explicit output path for the repair file
    returns:       path to the written repair file
    """
    with open(error_path, "r", encoding="utf-8") as f:
        failed_ids = {json.loads(line)["custom_id"] for line in f if line.strip()}
    if output_path is None:
        output_path = original_path.with_stem(original_path.stem + "_repair")
    written = 0
    with open(original_path, "r", encoding="utf-8") as src, \
         open(output_path, "w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            request = json.loads(line)
            if request["custom_id"] in failed_ids:
                dst.write(json.dumps(request) + "\n")
                written += 1
    print(f"Wrote {written} repair requests to {output_path}")
    return output_path

def merge_batches(path1: Path, 
                  path2: Path, 
                  output_path: Path = None):
    """
    Merges two batch result .jsonl files into a single deduplicated file.
    Intended for combining a first batch result with a repair batch result.
    If no output_path is given, the merged file is written next to path1
    with '_merged' appended to the stem.

    path1:       path to the first batch result .jsonl file
    path2:       path to the second (repair) batch result .jsonl file
    output_path: optional explicit output path for the merged file
    returns:     path to the written merged file
    """
    entries: dict[str, dict] = {}
    with open(path1, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            entries[entry["custom_id"]] = entry
    overlaps = 0
    with open(path2, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry["custom_id"] in entries:
                overlaps += 1
            entries[entry["custom_id"]] = entry
    if overlaps:
        print(f"Warning: {overlaps} overlapping custom_id(s) found; path2 entries take precedence.")
    if output_path is None:
        output_path = path1.with_stem(path1.stem + "_merged")
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in entries.values():
            f.write(json.dumps(entry) + "\n")
    print(f"Wrote {len(entries)} entries to {output_path}")
    return output_path

def jsonl_to_json(jsonl_file_path: Path):
    """
    Converts a .jsonl file (one JSON object per line) to a formatted .json file
    (a JSON array with 4-space indentation). Deletes the original .jsonl file.
    Returns the path of the written .json file.
    """
    json_file_path = jsonl_file_path.with_suffix(".json")
    with open(jsonl_file_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    with open(json_file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    jsonl_file_path.unlink()
    return json_file_path

def extract_text(entry: dict):
    """
    Extracts the model's text response from a batch result entry,
    handling both Anthropic and OpenAI batch output formats transparently.
    Returns None if the entry did not succeed.

    Anthropic entries succeed when result.type == 'succeeded'.
    OpenAI entries succeed when response.status_code == 200.

    entry: a single result entry from a batch output file
    returns: the response text string, or None for failed entries
    """
    if "result" in entry:
        if entry["result"]["type"] != "succeeded":
            return None
        return entry["result"]["message"]["content"][0]["text"]
    if entry["response"]["status_code"] != 200:
        return None
    return entry["response"]["body"]["choices"][0]["message"]["content"]

def group_candidates_by_task_id(json_batch_file_path: Path):
    """
    Groups model-generated code candidates from a batch result file by task ID.
    Task ID is extracted as the first two underscore-separated parts of the custom_id
    (e.g. 'mbpp_2_sample_0' -> 'mbpp_2'). Failed entries are skipped.

    json_batch_file_path: path to a batch result .json file
    returns: dict mapping task_id_string -> list of code strings
    """
    with open(json_batch_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    grouped = dict()
    for item in data:
        text = extract_text(item)
        if text is None:
            continue
        task_id = "_".join(item["custom_id"].split("_")[:2])  # "mbpp_2"
        grouped.setdefault(task_id, []).append(text)
    return grouped

def postprocess_baseline_candidates(batch_result_path: Path,
                                    candidate_dir: Path):
    """
    Converts baseline batch results into per-task cloudpickle files.
    Each task gets a file named after its task_id string (e.g. 'mbpp_2')
    containing a list of all generated code candidates.

    batch_result_path:  path to the raw batch output .jsonl file
    candidate_dir:      directory where per-task candidate files are written
    """
    json_batch_file_path = jsonl_to_json(batch_result_path)
    grouped = group_candidates_by_task_id(json_batch_file_path)
    for task_id, task_candidates in grouped.items():
        task_path = Path.joinpath(candidate_dir, f"{task_id}")
        task_path.parent.mkdir(parents=True, exist_ok=True)
        with open(task_path, "wb") as task_file:
            cloudpickle.dump(task_candidates, task_file)

def get_questions_by_task_id(json_question_file_path: Path, 
                             output_path: Path):
    """
    Parses a batch result file containing generated questions and writes a structured
    questions.json. Failed entries are skipped. Questions are extracted via regex
    from the model's response text.

    json_question_file_path: path to the batch result .json file
    output_path:             path to write the structured questions.json
    """
    with open(json_question_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    tasks = []
    for entry in data:
        text = extract_text(entry)
        if text is None:
            continue
        task_id   = int(entry["custom_id"].split("_")[-1])
        questions = {f"q{i+1}": {"question": q} for i, q in enumerate(re.findall(r"question \d+: (.+)", text))}
        tasks.append({"task_id": task_id, "questions": questions})
    tasks.sort(key=lambda t: t["task_id"])
    with open(output_path, "w", encoding="utf-8") as out:
        json.dump(tasks, out, indent=4)

def postprocess_questions(jsonl_question_file_path: Path,
                          question_path: Path):
    """
    Converts a question generation batch output .jsonl into a structured questions.json.
    Combines jsonl_to_json and get_questions_by_task_id.

    jsonl_question_file_path: path to the raw batch output .jsonl file
    question_path:            path to write the final questions.json
    """
    json_question_file_path = jsonl_to_json(jsonl_question_file_path)
    get_questions_by_task_id(json_question_file_path, question_path)

def postprocess_refined_descriptions(jsonl_descriptions_path: Path,
                                     QA_path: Path):
    """
    Parses a description refinement batch result and merges description1 and description2
    into the existing questions.json (QA_path). Failed entries are skipped.
    The model response is expected to contain 'description 1: ...' and 'description 2: ...'

    jsonl_descriptions_path: path to the raw batch output .jsonl file
    QA_path:                 path to questions.json, which is read and updated in place
    """
    json_descriptions_path = jsonl_to_json(jsonl_descriptions_path)
    with open(json_descriptions_path, "r", encoding="utf-8") as f:
        descriptions_data = json.load(f)
    desc_lookup = {}
    for entry in descriptions_data:
        text = extract_text(entry)
        if text is None:
            continue
        parts   = entry["custom_id"].split("_")
        task_id = int(parts[2])
        q_key   = parts[3]
        match   = re.search(r"description 1:\s*(.+?)\n\n?description 2:\s*(.+)", text, re.DOTALL)
        if match:
            desc_lookup.setdefault(task_id, {})[q_key] = {
                "description1": match.group(1).strip(),
                "description2": match.group(2).strip(),
            }
    with open(QA_path, "r", encoding="utf-8") as f:
        qa_data = json.load(f)
    for task in qa_data:
        task_id = task["task_id"]
        for q_key, q_entry in task["questions"].items():
            descs = desc_lookup.get(task_id, {}).get(q_key)
            if descs:
                q_entry["description1"] = descs["description1"]
                q_entry["description2"] = descs["description2"]
    with open(QA_path, "w", encoding="utf-8") as f:
        json.dump(qa_data, f, indent=4)

def postprocess_oracle_descriptions(jsonl_oracle_descriptions_path: Path,
                                    QA_path: Path):
    """
    Parses an oracle description batch result and merges oracle_description
    into the existing questions.json (QA_path). Failed entries are skipped.
    The model response is expected to contain 'description: ...'

    jsonl_oracle_descriptions_path: path to the raw batch output .jsonl file
    QA_path:                        path to questions.json, which is read and updated in place
    """
    json_oracle_descriptions_path = jsonl_to_json(jsonl_oracle_descriptions_path)
    with open(json_oracle_descriptions_path, "r", encoding="utf-8") as f:
        descriptions_data = json.load(f)
    desc_lookup = {}
    for entry in descriptions_data:
        text = extract_text(entry)
        if text is None:
            continue
        parts   = entry["custom_id"].split("_")
        task_id = int(parts[2])
        q_key   = parts[3]
        match   = re.search(r"description:\s*(.+)", text, re.DOTALL)
        if match:
            desc_lookup.setdefault(task_id, {})[q_key] = {
                "oracle_description": match.group(1).strip()
            }
    with open(QA_path, "r", encoding="utf-8") as f:
        qa_data = json.load(f)
    for task in qa_data:
        task_id = task["task_id"]
        for q_key, q_entry in task["questions"].items():
            descs = desc_lookup.get(task_id, {}).get(q_key)
            if descs:
                q_entry["oracle_description"] = descs["oracle_description"]
    with open(QA_path, "w", encoding="utf-8") as f:
        json.dump(qa_data, f, indent=4)

def postprocess_refined_candidates(jsonl_refinement_path: Path,
                                   target_path: Path):
    """
    Converts QA refinement batch results into per-task/per-description cloudpickle files.
    Results are grouped by (task_id, question_no, description_type) and written to:
    target_path/program_candidates/mbpp_{task_id}/{question_no}_{description_type}
    Failed entries are skipped.

    jsonl_refinement_path: path to the raw batch output .jsonl file
    target_path:           root directory for the output candidate files
    """
    json_refinement_path = jsonl_to_json(jsonl_refinement_path)
    with open(json_refinement_path, "r") as f:
        refinement_data = json.load(f)
    candidate_collection = dict()
    for entry in refinement_data:
        code = extract_text(entry)
        if code is None:
            continue
        custom_id        = entry["custom_id"]
        id_parts         = custom_id.split("_")
        task_id          = id_parts[1]
        question_no      = id_parts[2]
        description_type = id_parts[3]
        dict_id = f"mbpp_{task_id}_{question_no}_{description_type}"
        if not candidate_collection.get(dict_id):
            candidate_collection[dict_id] = []
        candidate_collection[dict_id].append(code)
    for id, candidates in candidate_collection.items():
        id_parts         = id.split("_")
        task_id          = id_parts[1]
        question_no      = id_parts[2]
        description_type = id_parts[3]
        path_postfix = rf'mbpp_{task_id}/{question_no}_{description_type}'
        path = Path.joinpath(target_path, path_postfix)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as task_file:
            cloudpickle.dump(candidates, task_file)

if __name__ == "__main__":
    client = OpenAI()

    # --- Submit ---
    # batch_file_path     = Path() # e.g. THIS_DIR / ".MBPP-example-mini" / "baseline" / "batch_request.jsonl"
    # batch               = create_batch(client, batch_file_path)

    # --- Re-submit if results are incomplete ---
    # batch_file_path     = Path()
    # failed_file_path    = Path()
    # retry_file_path     = recreate_batch(batch_file_path, failed_file_path)
    # batch               = create_batch(client, retry_file_path)

    # --- Merge after re-submit ---
    # incomplete_file_path    = Path()
    # new_results_path        = Path()
    # complete_results        = Path()
    # merge_batches(
    #     incomplete_file_path,
    #     new_results_path,
    #     complete_results
    # )

    # --- Postprocess baseline candidates ---
    # result_file_path    = Path() # e.g. THIS_DIR / ".MBPP-example-mini" / "baseline" / "batch_result.jsonl"
    # candidate_dir       = Path() # e.g. THIS_DIR / ".MBPP-example-mini" / "baseline" / "candidates"
    # postprocess_baseline_candidates(
    #     result_file_path,
    #     candidate_dir
    # )

    # --- Postprocess questions ---
    # refined_dir             = Path() # e.g. THIS_DIR / ".MBPP-example-mini" / "refined"
    # question_results_path   = Path() # e.g. refined_dir / "batch_results" / "batch_results_questions.jsonl"
    # question_file           = Path() # e.g. refined_dir / "questions_and_descriptions.json"
    # postprocess_questions(
    #     question_results_path,
    #     question_file
    # )

    # --- Postprocess refined descriptions ---
    # refined_dir           = Path() # e.g. THIS_DIR / ".MBPP-example-mini" / "refined"
    # descriptions_results  = Path() # e.g. refined_dir / "batch_results" / "batch_result_refined_descriptions.jsonl"
    # question_file         = Path() # e.g. refined_dir / "questions_and_descriptions.json"
    # postprocess_refined_descriptions(
    #     descriptions_results,
    #     question_file
    # )


    # --- Postprocess oracle descriptions ---
    # refined_dir     = Path() # e.g. THIS_DIR / THIS_DIR / ".MBPP-example-mini" / "refined"
    # oracle_results  = Path() # e.g. refined_dir / "batch_results" / "batch_result_oracle.jsonl"
    # question_file   = Path() # e.g. refined_dir / "questions_and_descriptions.json"
    # postprocess_oracle_descriptions(
    #     oracle_results,
    #     question_file
    # )
    
    # --- Postprocess refined candidates ---
    # candidate_batch_results = Path() # e.g. THIS_DIR / ".MBPP-example-mini" / "refined" / "batch_results" / "refined_candidates.jsonl"
    # candidate_dir           = Path() # e.g. THIS_DIR / ".MBPP-example-mini" / "refined" / "candidates"
    # postprocess_refined_candidates(
    #     candidate_batch_results,
    #     candidate_dir
    # )