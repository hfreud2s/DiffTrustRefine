"""
This file contains the most important parts of the experiment:
- generate questions based on task descriptions
- generate refined descriptions based on possible answers to the questions
- ask the oracle for the true refined description
"""
from pathlib import Path
import sys
import json
import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

THIS_DIR = Path(__file__).resolve().parent
DIFFTRUST_ROOT = THIS_DIR.parent
sys.path.append(str(DIFFTRUST_ROOT))
MBPP_JSON = DIFFTRUST_ROOT / "MBPP" / ".data" / "sanitized-mbpp.json"

from MBPP.instance import Instance
from difftrust.core.specification import Specification


def load_instances_by_id(task_ids: set[int]):
    with open(MBPP_JSON, "r", encoding="utf-8") as f:
        all_tasks = json.load(f)
    result = {}
    for task in all_tasks:
        tid = task["task_id"]
        if tid in task_ids:
            try:
                result[tid] = Instance(task)
            except Exception as e:
                print(f"  Could not build Instance for task_id={tid}: {e}")
    return result

def needs_refinement(baseline_stats_path: Path):
    """
    Reads the file at baseline_stats_path and returns the set of task ids of all taks that need refinement. Refinement is needed if incoherence > 0.
    """
    with open(baseline_stats_path) as f:
        data = json.load(f)
    need_refinement = set()
    for task in data:
        if "needs_refinement" in task.keys():
            if task.get("needs_refinement"): need_refinement.add(int(task["task_id"]))
        else:
            if task.get("incoherence") > 0: need_refinement.add(int(task["task_id"]))
    return need_refinement

def generate_binary_questions(spec: Specification, 
                              num_questions: int = 1):
    """
    Builds the prompt that asks the model to identify genuinely underspecified
    behaviors in a task specification as yes/no questions.
    The prompt enforces that each question must be binary, answerable only by
    reading the spec, and lead to observably different implementations.

    spec:          the Specification whose description is embedded in the prompt
    num_questions: how many questions to ask the model to generate
    returns:       the prompt string
    """
    prompt = (
        f"You are analyzing a Python programming task to identify genuine implementation ambiguities.\n\n"
        f"Task specification:\n{spec.description}\n\n"
        f"Identify {num_questions} binary question(s) about this specification where:\n"
        f"- Each question targets a specific behavior that is genuinely underspecified\n"
        f"- Each question can be answered with either YES or NO\n"
        f"- A YES answer and a NO answer would lead to observably different code (different outputs on at least one input)\n"
        f"- The question cannot be answered just by reading the specification carefully\n"
        f"- The question is about WHAT the function should return or do, not HOW to implement it\n\n"
        f"For example: Consider the specification \"Sort a given list of integers.\"\n"
        f"Good question: 'Should the list be sorted in increasing order?'\n\n"
        f"Bad question: 'How should the function handle edge cases?' (too vague, not binary)\n\n"
        f"Bad question: 'Should the list be sorted in increasing or decreasing order?' (binary, but cannot be answered with YES or NO)\n\n"
        f"Output format:\n"
        f"question 1: ...\nquestion 2: ...\nquestion 3: ...\n\n"
        f"Output only the questions, nothing else."
    )
    return prompt

def generate_binary_descriptions(spec: Specification, 
                                 question: str):
    """
    Builds the prompt that asks the model to write two refined specifications —
    one for the YES answer and one for the NO answer to the given question.
    The prompt enforces that each description is standalone, behavior-focused,
    and contains no extra information beyond resolving the question.

    spec:     the Specification whose description is embedded in the prompt
    question: the yes/no question about the specification
    returns:  the prompt string
    """
    prompt = (
        f"Given this Python programming task:\n"
        f"{spec.description}\n\n"
        f"And this yes/no question about it:\n"
        f"{question}\n\n"
        f"Write exactly two refined specifications "
        f"— one for YES and one for NO — describing what the function should do in each case.\n"
        f"Each answer must:\n"
        f"- Describe the expected RETURN VALUE or BEHAVIOR of the function, not implementation steps\n"
        f"- Contain no additional information besides the one that answers the question\n"
        f"- Be genuinely different from the other answer in observable output\n"
        f"- Contain no code or examples\n\n"
        f"Output format:\n"
        f"description 1: ...\ndescription 2: ...\n\n"
        f"One example:\n"
        f"Suppose the originial task description was \"Sort a list of integers.\"\n"
        f"And the question would be \"Should the list be sorted in ascending order?\"\n"
        f"The correct output would be:\n\n"
        f"description 1: Sort a list of integers in ascending order.\n"
        f"description 2: Sort a list of integers in descending order.\n\n"
        f"Bad output would be something like "
        f"\"The function should return the list of elements sorted in descending order (in-place)\" "
        f"Since this is not a standalone task description and also adds additional information (the in-place sorting)."
    )
    return prompt

def generate_oracle_prompt_v2(inst: Instance,
                              question: str,
                              desc1: str,
                              desc2: str,
                              other_questions: list[str]) -> str:
    """
    Improved oracle prompt that prevents information leakage by:
    1. Anchoring the oracle to the binary choice (desc1 / desc2) so it cannot
       introduce information beyond what resolving the question requires.
    2. Explicitly listing the other questions for the same task so the model
       knows which topics are off-limits.
    3. Adding a single-new-fact test so the model self-checks its output.

    inst:            the Instance containing the spec, ground truth code, and test list
    question:        the question whose answer should be embedded in the refined description
    desc1:           the YES-branch description already generated for this question
    desc2:           the NO-branch description already generated for this question
    other_questions: questions asked about the same task that this oracle must NOT address
    returns:         the prompt string
    """
    other_q_block = (
        "\n".join(f"  - {q}" for q in other_questions)
        if other_questions else "  (none)"
    )
    prompt = (
        f"You are writing the oracle (ground-truth) refined specification for a Python programming task.\n\n"
        f"Original task:\n"
        f"{inst.spec.description}\n\n"
        f"A yes/no question has been asked about this task:\n"
        f"Question: {question}\n\n"
        f"The two possible answers to this question correspond to these two descriptions:\n"
        f"  description 1 (one possible answer): {desc1}\n"
        f"  description 2 (other possible answer): {desc2}\n\n"
        f"Use the ground truth implementation and test cases below to determine which answer is correct:\n"
        f"Ground truth:\n{inst.code}\n\n"
        f"Test cases:\n{inst.test_list}\n\n"
        f"Other questions asked independently about this same task — do NOT address these in your answer:\n"
        f"{other_q_block}\n\n"
        f"Write the oracle description following these rules:\n"
        f"1. Choose one of the two descriptions above as your starting point (or write a minimal variation of it)\n"
        f"2. Answer ONLY the question — add nothing that is not needed to answer it\n"
        f"3. Do NOT reveal anything covered by the other questions listed above\n"
        f"4. Do NOT mention return types, parameter types, edge cases, or implementation details unless the question specifically asks about them\n"
        f"5. Do NOT reference the ground truth, test cases, or how the function is implemented\n"
        f"6. Contain no code or examples\n\n"
        f"Self-check before writing: someone reading only the original task description and your oracle description should gain exactly ONE new piece of information — the answer to the question above, and nothing else.\n\n"
        f"Output format:\n"
        f"description: ...\n\n"
        f"Example:\n"
        f"Original task: \"Sort a list of integers.\"\n"
        f"Question: \"Should the list be sorted in ascending order?\"\n"
        f"description 1: \"Sort a list of integers in ascending order.\"\n"
        f"description 2: \"Sort a list of integers in descending order.\"\n"
        f"Other questions: \"Should the function sort in-place?\"\n"
        f"Correct oracle output:\n"
        f"description: Sort a list of integers in ascending order.\n\n"
        f"Bad oracle output:\n"
        f"description: Sort a list of integers in ascending order and return the sorted list.\n"
        f"(Bad because 'return the sorted list' addresses the in-place question, which is off-limits.)"
    )
    return prompt


def generate_oracle_prompt(inst: Instance,
                           question: str):
    """
    Builds the prompt that asks the model to write the ground-truth refined
    specification by consulting the reference code and test cases.
    The model is told to answer the question based on what the code actually does,
    not based on the original description.

    inst:     the Instance containing the spec, ground truth code, and test list
    question: the question whose answer should be embedded in the refined description
    returns:  the prompt string
    """
    prompt = (
        f"Given this Python programming task:\n"
        f"{inst.spec.description}\n\n"
        f"And this question about it:\n"
        f"{question}\n\n"
        f"To answer the question you can use the ground truth for reference:\n"
        f"{inst.code}\n\n"
        f"You can also analyse the following test cases if necessary:\n"
        f"{inst.test_list}\n\n"
        f"Your task is to write a refined specification for the programming task. "
        f"The description must:\n"
        f"- Contain the answer to the question\n"
        f"- Describe the expected RETURN VALUE or BEHAVIOR of the function, not implementation steps\n"
        f"- Contain no additional information besides the one that answers the question\n"
        f"- Contain no code or examples\n\n"
        f"Output format:\n"
        f"description: ...\n\n"
        f"An example:\n"
        f"Suppose the originial task description was \"Sort a list of integers.\"\n"
        f"And the question would be \"Should the list be sorted in ascending or descending order?\"\n"
        f"Suppose the answer would be \"ascending\" "
        f"Then the correct output would be:\n\n"
        f"description: Sort a list of integers in ascending order.\n\n"
        f"Bad output would be something like: "
        f"\"Based on the given ground truth and test cases, "
        f"the function should return the list of elements sorted in ascending order. "
        f"The list also should be sorted in-place.\"\n"
        f"The output is bad, since it doesn't adhere to the output format, "
        f"mentions the ground truth and test cases, and gives additional information (in-place sorting)."
    )
    return prompt

def generate_batch_request(custom_id:   str,
                           model:       str,
                           content:     str,
                           max_tokens:  int = 1024,
                           role:        str = "user"):
    """
    Wraps a prompt string into an Anthropic batch Request object.

    custom_id:  unique identifier for this request within the batch
    model:      Anthropic model name (e.g. 'claude-opus-4-6')
    content:    the prompt string
    max_tokens: maximum tokens in the model response (default 1024)
    role:       message role (default 'user')
    returns:    an Anthropic Request object ready for batch submission
    """
    return Request(
        custom_id=custom_id,
        params=MessageCreateParamsNonStreaming(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": role, "content": content}]
        )
    )

def send_batch(requests: list):
    """
    Sends a list of Anthropic batch requests to the Messages Batch API.
    Reads the API key from difftrust/config.json.
    Prints the batch object returned by the API (contains the batch ID for tracking).

    requests: list of Anthropic Request objects built by generate_batch_request
    """
    client = anthropic.Anthropic()
    message_batch = client.messages.batches.create(requests=requests)
    print(message_batch)

def generate_questions(question_file:       Path, 
                       needs_refinement:    set[int], 
                       refinement_llm:      str, 
                       num_questions:       int):
    """
    Sends an Anthropic batch to generate clarifying questions for all tasks in
    needs_refinement that don't already have questions in question_file.
    If question_file doesn't exist yet it is created as an empty JSON array.
    Tasks with existing questions are skipped (resumable).

    question_file:    path to questions.json (read and updated in place)
    needs_refinement: set of integer task IDs that need clarification
    refinement_llm:   Anthropic model name to use
    num_questions:    number of questions to generate per task
    """
    instances = load_instances_by_id(needs_refinement)
    try:
        with open(question_file) as f:
            results = json.load(f)
        print(f"Loaded {len(results)} existing entries from {question_file}")
    except FileNotFoundError:
        results = []
        with open(question_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Created new file at {question_file}")
    requests = []
    for task_id, inst in instances.items():
        task = next((r for r in results if r.get("task_id") == task_id), {})
        if task.get("questions") is not None:
            print(f"Skipping task {task_id} ({inst.name}) - questions already completed.")
            continue
        print(f"Generating questions for task {task_id} ({inst.name})...")
        question_prompt = generate_binary_questions(inst.spec, num_questions)
        request = generate_batch_request(f"questions_task_{task_id}", refinement_llm, question_prompt)
        requests.append(request)
    if len(requests) > 0:
        send_batch(requests)

def generate_refined_descriptions(question_file:        Path,
                                   needs_refinement:    set[int],
                                   refinement_llm:      str):
    """
    Sends an Anthropic batch to generate description1 and description2 for each
    question in question_file. Questions that already have description1 are skipped
    (resumable). Tasks with no questions are skipped silently.

    question_file:    path to questions.json (must already exist with questions populated)
    needs_refinement: set of integer task IDs to process
    refinement_llm:   Anthropic model name to use
    """
    try:
        with open(question_file) as f:
            tasks = json.load(f)
        print(f"Loaded {len(tasks)} existing entries from {question_file}")
    except FileNotFoundError:
        print(f"No file found at {question_file}")
        return
    instances = load_instances_by_id(needs_refinement)
    requests = []
    for task_id, inst in instances.items():
        task = next((r for r in tasks if r.get("task_id") == task_id), {})
        questions = task.get("questions")
        if not questions:
            continue
        for q_key in questions.keys():
            if questions[q_key].get("description1") is not None:
                print(f"Skipping task {task_id} question {q_key} because refined descriptions exist already.")
                continue
            refinement_prompt = generate_binary_descriptions(inst.spec, questions[q_key]["question"])
            request = generate_batch_request(f"descriptions_task_{task_id}_{q_key}", refinement_llm, refinement_prompt)
            requests.append(request)
    if len(requests) > 0:
        send_batch(requests)

def generate_oracle_description(question_file:      Path,
                                needs_refinement:   set[int],
                                refinement_llm:     str):
    """
    Sends an Anthropic batch to generate the oracle_description for each question
    in question_file, using the ground truth code and test cases as reference.
    Questions that already have oracle_description are skipped (resumable).
    Tasks with no questions are skipped silently.

    question_file:    path to questions.json (must already exist with questions populated)
    needs_refinement: set of integer task IDs to process
    refinement_llm:   Anthropic model name to use
    """
    try:
        with open(question_file) as f:
            tasks = json.load(f)
        print(f"Loaded {len(tasks)} existing entries from {question_file}")
    except FileNotFoundError:
        print(f"No file found at {question_file}")
        return
    instances = load_instances_by_id(needs_refinement)
    requests = []
    for task_id, inst in instances.items():
        task = next((r for r in tasks if r.get("task_id") == task_id), {})
        questions = task.get("questions")
        if not questions:
            continue
        for q_key in questions.keys():
            if questions[q_key].get("oracle_description") is not None:
                print(f"Skipping task {task_id} question {q_key} because oracle descriptions exist already.")
                continue
            refinement_prompt = generate_oracle_prompt(inst, questions[q_key]["question"])
            request = generate_batch_request(f"oracle_task_{task_id}_{q_key}", refinement_llm, refinement_prompt)
            requests.append(request)
    if len(requests) > 0:
        send_batch(requests)

if __name__ == "__main__":
    num_questions  = 3
    refinement_llm = "claude-opus-4-6"

    baseline_stats_path = Path() # e.g. THIS_DIR / ".MBPP-example-mini" / "baseline" / "baseline_stats.json"
    question_file_path  = Path() # e.g. THIS_DIR / ".MBPP-example-mini" / "refined" / "questions_and_descriptions.json"

    need_refinement = needs_refinement(baseline_stats_path)

    # --- Generate Questions, Refined Descriptions or True Answers ---
    # generate_questions(question_file_path, need_refinement, refinement_llm, num_questions)
    # generate_refined_descriptions(question_file_path, need_refinement, refinement_llm)
    # generate_oracle_description(question_file_path, need_refinement, refinement_llm)