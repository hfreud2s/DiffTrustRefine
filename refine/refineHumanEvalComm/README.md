# refineHumanEvalComm

Folder structure for the HumanEvalComm refinement experiment. All experiment data lives under `.HEC-experiment/`. The experiment is run on four LLMs (`LLM1`..`LLM4`, placeholder names). Each task in HumanEvalComm comes with an unmanipulated description plus several ambiguous variants, so every LLM holds one folder per category. Each category is evaluated in a `baseline` and a `refined` phase, and each phase is repeated over 10 runs (`run0`..`run9`). Only the `original` category does not have a refinement phase and serves only as a baseline.

```
refineHumanEvalComm/
└── .HEC-experiment/
    ├── LLM1/
    │   ├── original/                # unmanipulated task description
    │   │   └── baseline/
    │   │       ├── run0/
    │   │       │   └── .gitkeep
    │   │       ├── run1/
    │   │       ├── ...
    │   │       └── run9/
    │   ├── 1a/
    │   │   ├── baseline/
    │   │   │   ├── run0/
    │   │   │   │   └── .gitkeep
    │   │   │   ├── run1/
    │   │   │   ├── ...
    │   │   │   └── run9/
    │   │   └── refined/
    │   │       ├── run0/
    │   │       ├── run1/
    │   │       ├── ...
    │   │       └── run9/
    │   ├── 1c/                      # baseline/ and refined/, run0..run9 each (as above)
    │   ├── 1p/
    │   ├── 2ac/
    │   ├── 2ap/
    │   ├── 2cp/
    │   └── 3acp/
    ├── LLM2/                        # same 8 categories
    ├── LLM3/                        # same 8 categories
    └── LLM4/                        # same 8 categories
```

Categories: `original` (unmanipulated description) plus the 7 HumanEvalComm manipulation variants `1a`, `1c`, `1p`, `2ac`, `2ap`, `2cp`, `3acp`. The digit is how many manipulation kinds are combined; the letters are the kinds (`a` = ambiguity, `c` = inconsistency, `p` = incompleteness).

## Pipeline at a glance

Run the baseline phase for each LLM, then the refinement phase for the tasks that turn out ambiguous. Every batch goes through the OpenRouter Batch API. Shared paths, the `CATEGORIES` map and instance loading live in `common.py`.

Baseline phase (how self-consistent the model is on a description as written):

1. Build baseline candidate requests for all categories: `build_baseline_batch` (`build_batch.py`) writes one batch file per (LLM, category).
2. Submit and retrieve: `submit_llm` / `retry_llm`, then `wait_for_batch` + `save_results` (`batch_processing.py`).
3. Turn the results into per-run candidate files: `postprocess_baseline` (`batch_processing.py`).
4. Score incoherence and error per run, then average across runs: `score_phase` (`compute_stats.py`), then `aggregate_phase` (`data_analysis.py`).

Refinement phase (whether asking a clarifying question helps):

5. For the tasks with incoherence > 0, generate clarifying questions: `needs_refinement` + `build_questions_batch` (`refine_descriptions.py`), submit and retrieve as in step 2, then parse them into `questions_and_descriptions.json`: `postprocess_questions` (`batch_processing.py`).
6. Audit the question set (auditor #1): `build_audit_questions_batch` (`audit.py`), submit, then `postprocess_audit_questions`. It checks that a question targeting the injected ambiguity is present and if not it appends one as `q_auditor` and flags `true_question_missing`. Evaluation-only.
7. Generate the coder's YES/NO refined descriptions for every question: `build_descriptions_batch` (`refine_descriptions.py`) + `postprocess_descriptions` (`batch_processing.py`).
8. Generate the oracle's true description per question (round 4). (to build) Then audit those oracle answers for leakage (auditor #2), build candidates from each description, and score and aggregate exactly as steps 3-4 with `phase="refined"`.

Each step is detailed below.

## Step details

The experiment is built up one step at a time. Each step is a function that must be run before the next.

### Step 1: Baseline batch generation (`build_batch.py`)

`build_baseline_batch(llm_dir, model, dataset_name="dataset-50", categories=None, num_candidates=10, num_runs=10, temperature=None)` is called first, once per LLM. It samples program candidates straight from each category's task description, with no clarifying-question refinement (that is the later `refined` phase). The coder models are configured in `common.CODER_MODELS`.

For the given LLM it writes one OpenRouter batch input file per category:

```
.HEC-experiment/{llm_dir}/{category}/baseline_batch_request.jsonl
```

Each line is one OpenRouter batch item, `{"custom_id": ..., "body": {"model": ..., "messages": [...]}}`. The `custom_id` is `run{r}__humanevalcomm_{task_id}__sample{i}`.

What it does in detail:

- Instances are read from the benchmark folder `HumanEvalComm/.data/{dataset_name}.pkl`, so candidates are generated from the exact Specification that `compute_stats.py` later scores them against.
- A task is skipped for a category when it has no Specification for that variant. HumanEvalComm does not define every manipulation for every task, so the combined categories (`2ap`, `2cp`, `3acp`) cover far fewer tasks than `original`, `1a`, `1c`, `1p`, `2ac`.
- Category folder names map to dataset variants as `original` -> the unmanipulated spec, `1a` -> `prompt1a`, `1c` -> `prompt1c`, `1p` -> `prompt1p`, `2ac` -> `prompt2ac`, `2ap` -> `prompt2ap`, `2cp` -> `prompt2cp`, `3acp` -> `prompt3acp`.

### Step 2: Submit a batch and retrieve results (`batch_processing.py`)

Requires `OPENROUTER_API_KEY` in the environment. This uses OpenRouter's Batch API (`POST /api/beta/batches`), which takes the requests inline and returns the results inline in the status response once the batch is complete.

`submit_llm(llm_dir)` is the entry point for one LLM. It submits each category's `baseline_batch_request.jsonl` as its own batch (option B: one batch per category) and writes a per-batch meta file named after the request (e.g. `baseline_batch_meta.json`) next to each request file, recording the batch id and status. Internally it calls `create_batch(request_path)`, which reads the request file and posts `{endpoint, model, requests}` in that order (OpenRouter stream-parses the body, so `endpoint` and `model` must precede `requests`).

To retrieve, once a batch has run: `wait_for_batch(batch_id, poll=60)` polls `GET /api/beta/batches/{id}` until the batch is terminal (`completed`, `failed`, `expired`, `cancelled`), then `save_results(batch, out_path)` writes the inline results to `{category}/baseline_batch_result.jsonl`, one `{custom_id, response, error}` item per line. That file is the input for step 3 (post-processing into the run folders). The batch id is read back from that category's `baseline_batch_meta.json`.

Notes:

- One batch per category. The largest dataset-50 category (`original`, 5000 requests) is a ~5.6 MB request; OpenRouter documents no hard request cap, and the only fixed limit is the 24h completion window.
- OpenRouter deletes batch inputs and results 30 days after creation, so retrieve and post-process within that window.

OpenRouter caps how many requests may be submitted per minute (about 20000). Submission is therefore throttled: `submit_llm` and `retry_llm` count the requests sent in each 60s window and wait when the next batch would cross `max_requests_per_min` (default 20000), and they back off and retry on any 429. If a run is still cut short, `pending_categories(llm_dir)` lists the categories that have a request file but no `batch_meta.json`, and `retry_llm(llm_dir)` resubmits exactly those. `submit_llm` skips categories that already have a `batch_meta.json`, so re-running it resumes rather than double-submitting.

### Step 3: Post-process results into candidate files (`batch_processing.py`)

Once a category's results are retrieved (`baseline_batch_result.jsonl`), `postprocess_baseline(llm_dir, categories=None)` turns them into the per-run candidate files the stats step reads. For each category it groups responses by (run, task) and writes one cloudpickle file per (run, task):

```
.HEC-experiment/{llm_dir}/{category}/baseline/run{r}/humanevalcomm_{task_id}[-{variant}]
```

Each file holds that task's list of raw candidate strings for that run, ordered by sample index. The variant suffix (`-prompt1a`, `-prompt1c`, ...) records which condition the candidates belong to. Failed responses (non-200) are skipped and counted. Called with no `categories`, it processes every category that has a result file.

### Step 4: Score runs and aggregate (`compute_stats.py`, `data_analysis.py`)

Scoring is per run, aggregation is across runs.

`compute_stats.score_phase(llm_dir, category, phase="baseline", runs=None, nb_samples=1000, timeout=60)` scores the candidate files run by run. For each `{category}/{phase}/run{r}/` it computes pointwise incoherence and error for every candidate file and writes `run{r}/stats.json` (a list of `{key, task_id, variant, name, nb_candidates, incoherence, error}`). It reads the instances from the benchmark's `dataset-50.pkl`, is resumable (tasks already in a `stats.json` are skipped), and takes an optional `runs` list to score only some runs. Underneath, `compute_stats(candidate_path, output_path, ...)` scores a single folder.

`data_analysis.aggregate_phase(llm_dir, category, phase="baseline")` then reads every `run{r}/stats.json`, aligns the runs by task_id, and writes `{phase}/aggregate.json`: one row per task with the per-run `incoherence_list` and `error_list` plus `mean_incoherence` and `mean_error` (Nones dropped before the mean). 

## Refinement phase

The refined phase clarifies a category's description and regenerates candidates from the clarified description, to test whether asking a question reduces incoherence and error and to score the quality of each question. Each coder LLM asks its own questions per default, and everything is stored under `{LLM}/{category}/refined/`.

1. clarifying questions
2. refined (yes/no) descriptions, one pair per question
3. oracle (true) description per question, answered from the ground truth
4. candidates generated from each refined/oracle description, into `{category}/refined/run{r}/`
5. score and aggregate, exactly as the baseline phase (step 4)

### Step 5: Refinement round 1 — clarifying questions (`refine_descriptions.py`)

`needs_refinement(llm_dir, category)` reads `{category}/baseline/aggregate.json` and returns the tasks whose mean incoherence is above zero. Only those enter the refinement pipeline.

`build_questions_batch(llm_dir, category, model=None, num_questions=3)` writes `{category}/refined/questions_batch_request.jsonl`: one OpenRouter item per needing-refinement task, custom_id `questions__humanevalcomm_{task_id}`, whose prompt embeds that category's description and asks for `num_questions` binary yes/no questions. Submit it with `batch_processing.create_batch` / `save_batch_meta`, the same way as the baseline batch.

### Step 6: Question post-processing (`batch_processing.py`)

`postprocess_questions(llm_dir, category)` parses a retrieved question batch (`{category}/refined/questions_batch_result.jsonl`) into `{category}/refined/questions_and_descriptions.json`, the file every remaining refinement round reads and extends. One entry per task:

```json
{
  "task_id": 1,
  "true_question_missing": false,      // set by auditor #1
  "questions": {
    "q1": {
      "question": "...",
      "source": "model",               // "model" (coder) or "auditor" (added for evaluation only)
      "description1": null,            // coder LLM, YES branch (later round)
      "description2": null,            // coder LLM, NO branch (later round)
      "oracle_description": null,      // oracle LLM, true answer (later round)
      "oracle_leak": null,             // auditor #2
      "oracle_rewrite": null           // auditor #2
    }
  }
}
```

It only adds tasks not already present, so it is safe to re-run. The blank-question skeleton is `common.blank_question`.

### Step 7: Auditor #1 — question-set audit (`audit.py`)

A single fixed auditor LLM, separate from the coder LLMs and the oracle (to avoid self-evaluation bias), checks that each task's question set contains a question targeting the injected ambiguity. `build_audit_questions_batch(llm_dir, category, model)` writes `{category}/refined/audit_questions_batch_request.jsonl` (custom_id `audit_questions__humanevalcomm_{task_id}`). Each prompt gives the auditor the original (clear) and manipulated (ambiguous) descriptions plus the coder's questions, and asks for a three-line verdict (`covered` / `covering` / `question`). `model` is required (the auditor model). Submit it with `batch_processing.create_batch`.

`postprocess_audit_questions(llm_dir, category)` applies the verdicts back into `questions_and_descriptions.json`: it sets `true_question_missing` per task, and when the injected ambiguity was not covered it appends the auditor's question under the distinct key `q_auditor` with `source: "auditor"`. That question is scored later for incoherence reduction but is flagged so it stays out of the practical question pool (in a real deployment you could not know which question is the right one).

### Round 3: Coder YES/NO descriptions (`refine_descriptions.py`, `batch_processing.py`)

`build_descriptions_batch(llm_dir, category, model=None)` asks the coder LLM (its own model, inferred from the baseline request) to turn each question into a YES description and a NO description, one request per (task, question), including any `q_auditor` question so it is scored like the rest. custom_id `descriptions__humanevalcomm_{task_id}__{qkey}`; output `{category}/refined/descriptions_batch_request.jsonl`; submit with `create_batch`.

`postprocess_descriptions(llm_dir, category)` parses `description 1:` / `description 2:` from the results and fills `description1` / `description2` for each question in `questions_and_descriptions.json`. These two branches are what the coder scores for incoherence reduction (imagining each answer); the oracle's true answer is generated separately in the next round.

### Round 4: Oracle true description (`refine_descriptions.py`, `batch_processing.py`)

`build_oracle_batch(llm_dir, category, model=ORACLE_MODEL)` asks the fixed oracle for the ground-truth ("true") refined description of every question in `questions_and_descriptions.json`, including any `q_auditor` question. Unlike the coder rounds, the model is the single fixed `ORACLE_MODEL` (the same simulated user across all LLMs and categories). Each prompt gives the oracle the category's (ambiguous) description, the question, and the reference solution (`inst.code`) with its test cases (`inst.test`), and asks it to answer the question from what the code actually does, in the format `description: ...`. One request per (task, question); custom_id `oracle__humanevalcomm_{task_id}__{qkey}`; output `{category}/refined/oracle_batch_request.jsonl`; submit with `create_batch`.

`postprocess_oracle(llm_dir, category)` parses `description:` from the results and fills `oracle_description` for each question in `questions_and_descriptions.json`. This is the ground-truth branch the coder's YES/NO descriptions are measured against, and the text auditor #2 (still to build) then checks it for information leakage, filling `oracle_leak` / `oracle_rewrite`.
