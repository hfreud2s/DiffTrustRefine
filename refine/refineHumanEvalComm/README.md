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

