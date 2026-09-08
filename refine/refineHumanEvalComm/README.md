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

## Pipeline

The experiment is built up one step at a time. Each step is a function that must be run before the next.

### Step 1: Baseline batch generation (`build_batch.py`)

`build_baseline_batch(llm_dir, model, dataset_name="dataset-50", categories=None, num_candidates=10, num_runs=10, temperature=None)` is called first, once per LLM. It samples program candidates straight from each category's task description, with no clarifying-question refinement (that is the later `refined` phase). Set the real OpenRouter model ids for `LLM1`..`LLM4` in the `__main__` block before running.

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

`submit_llm(llm_dir)` is the entry point for one LLM. It submits each category's `baseline_batch_request.jsonl` as its own batch (option B: one batch per category) and writes a `batch_meta.json` next to each request file recording the batch id and status. Internally it calls `create_batch(request_path)`, which reads the request file and posts `{endpoint, model, requests}` in that order (OpenRouter stream-parses the body, so `endpoint` and `model` must precede `requests`).

To retrieve, once a batch has run: `wait_for_batch(batch_id, poll=60)` polls `GET /api/beta/batches/{id}` until the batch is terminal (`completed`, `failed`, `expired`, `cancelled`), then `save_results(batch, out_path)` writes the inline results to `{category}/baseline_batch_result.jsonl`, one `{custom_id, response, error}` item per line. That file is the input for step 3 (post-processing into the run folders). The batch id is read back from `{category}/batch_meta.json`.

Notes:

- One batch per category. The largest dataset-50 category (`original`, 5000 requests) is a ~5.6 MB request; OpenRouter documents no hard request cap, and the only fixed limit is the 24h completion window.
- OpenRouter deletes batch inputs and results 30 days after creation, so retrieve and post-process within that window.

OpenRouter caps how many requests may be submitted per minute (about 20000). Submission is therefore throttled: `submit_llm` and `retry_llm` count the requests sent in each 60s window and wait when the next batch would cross `max_requests_per_min` (default 20000), and they back off and retry on any 429. If a run is still cut short, `pending_categories(llm_dir)` lists the categories that have a request file but no `batch_meta.json`, and `retry_llm(llm_dir)` resubmits exactly those. `submit_llm` skips categories that already have a `batch_meta.json`, so re-running it resumes rather than double-submitting.

### Step 3: Post-process results into candidate files (`batch_processing.py`)

Once a category's results are retrieved (`baseline_batch_result.jsonl`), `postprocess_baseline(llm_dir, categories=None)` turns them into the per-run candidate files the stats step reads. For each category it groups responses by (run, task) and writes one cloudpickle file per (run, task):

```
.HEC-experiment/{llm_dir}/{category}/baseline/run{r}/humanevalcomm_{task_id}[-{variant}]
```

Each file holds that task's list of raw candidate strings for that run, ordered by sample index. The variant suffix (`-prompt1a`, `-prompt1c`, ...) records which Specification the candidates were generated from, so the stats step can score them against the right spec. Failed responses (non-200) are skipped and counted. Called with no `categories`, it processes every category that has a result file.
