# refineHumanEvalComm

Folder structure for the HumanEvalComm refinement experiment. All experiment data lives under `.HEC-experiment/`. The experiment is run on four LLMs (`LLM1`..`LLM4`, placeholder names). Each task in HumanEvalComm comes with an unmanipulated description plus several ambiguous variants, so every LLM holds one folder per category. Each category is evaluated in a `baseline` and a `refined` phase, and each phase is repeated over 10 runs (`run0`..`run9`). Each run folder carries a `.gitkeep` so the empty scaffold is preserved in git.

```
refineHumanEvalComm/
└── .HEC-experiment/
    ├── LLM1/
    │   ├── original/                # unmanipulated task description
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
    │   ├── 1a/                      # baseline/ and refined/, run0..run9 each (as above)
    │   ├── 1c/
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
