# Experiment Specifications

This directory groups batch specifications by purpose so pilot runs, benchmarks, and baselines are easy to locate and compare.

## Layout

- `pilots/`
  Single-case or very small runs used to validate a workflow before launching a larger benchmark.
- `single_study/`
  Focused study-grounded experiments centered on one clinical trial.
- `multi_study/`
  Curated benchmark specs that expand across multiple clinical trials, participant profiles, and question sets.
- `baselines/`
  Simpler comparison runs used for ablations and side-by-side evaluation.
- `examples/`
  Small illustrative examples that show the expected structure of a batch specification.

## Recommended Run Order

1. `pilots/nct03877237_single_case_pilot_v1.json`
   Use this first to confirm the live study-grounded path works end to end.
2. `pilots/nct03877237_comparison_pilot_full_agentic_v1.json`
   Use this with the matching `generic_rag` and `vanilla_llm` pilot specs for a small three-way comparison before any long rerun.
3. `pilots/nct03877237_gonogo_full_agentic_v1.json`
   Use this with the matching `generic_rag` and `vanilla_llm` go/no-go specs when you want the smallest possible preflight matrix: one study, one profile, one question set, all three workflow variants.
4. `pilots/nct03877237_full_agentic_followup_pilot_v1.json`
   Use this after offline evaluator or recovery changes when you want the smallest useful Full Agentic-only live follow-up: one study, both participant profiles, and all three question sets.
5. `single_study/nct03877237_study_grounded_batch_v1.json`
   Use this when you want a richer single-study batch around the reference case.
6. `multi_study/hf_interventional_multi_study_v1.json`
   Use this for the real multi-study benchmark after the pilot succeeds.
7. `baselines/hf_interventional_multi_study_generic_rag_v1.json`
   Use this as the retrieval-grounded baseline without the full agentic planning and revision loop.
8. `baselines/hf_interventional_multi_study_vanilla_llm_v1.json`
   Use this as the no-retrieval baseline when you want a cleaner ablation against the full system.

The comparison pilot set is:

- `pilots/nct03877237_comparison_pilot_full_agentic_v1.json`
- `pilots/nct03877237_comparison_pilot_generic_rag_v1.json`
- `pilots/nct03877237_comparison_pilot_vanilla_llm_v1.json`

Each pilot spec expands to the same 6 cases:

- 1 study: `nct03877237`
- 2 participant profiles
- 3 question sets

Run them with `--base-run-id <prepared_run_id>` so the checked-in specs stay reusable across rebuilt corpora.

The go/no-go preflight set is:

- `pilots/nct03877237_gonogo_full_agentic_v1.json`
- `pilots/nct03877237_gonogo_generic_rag_v1.json`
- `pilots/nct03877237_gonogo_vanilla_llm_v1.json`

Each go/no-go spec runs the same smallest comparison slice:

- 1 study: `nct03877237`
- 1 participant profile: `example_us_medium_literacy`
- 1 question set: `study_specific_basics`

Suggested command sequence:

```powershell
C:\Users\Ahmed\AppData\Local\Programs\Python\Python311\python.exe run_pipeline.py run-batch-experiment --spec-file configs/experiments/pilots/nct03877237_gonogo_full_agentic_v1.json --base-run-id <prepared_run_id>
C:\Users\Ahmed\AppData\Local\Programs\Python\Python311\python.exe run_pipeline.py run-batch-experiment --spec-file configs/experiments/pilots/nct03877237_gonogo_generic_rag_v1.json --base-run-id <prepared_run_id>
C:\Users\Ahmed\AppData\Local\Programs\Python\Python311\python.exe run_pipeline.py run-batch-experiment --spec-file configs/experiments/pilots/nct03877237_gonogo_vanilla_llm_v1.json --base-run-id <prepared_run_id>
C:\Users\Ahmed\AppData\Local\Programs\Python\Python311\python.exe run_pipeline.py compare-batch-results --batch-summary <full_agentic_batch_summary.json> --batch-summary <generic_rag_batch_summary.json> --batch-summary <vanilla_llm_batch_summary.json> --comparison-id nct03877237_gonogo_v1
```

Use this sequence before the long live reruns so the reporting tables, grounding diagnostics, and failure summaries can be checked on a minimal matrix.

The post-fix Full Agentic follow-up pilot is:

- `pilots/nct03877237_full_agentic_followup_pilot_v1.json`

It runs:

- 1 study: `nct03877237`
- 2 participant profiles
- 3 question sets

Use it after offline evaluator or recovery updates when you want a small live check focused on:

- study procedures coverage
- benefits coverage
- alternatives coverage
- QA citation behavior
- recovery-pass provenance

## Naming

- `pilot` means a small validation run before scaling.
- `single_study` means one clinical-trial-centered evaluation.
- `multi_study` means a benchmark driven by a cohort file or multiple study records.
- `baseline` means a deliberately simpler comparison condition.

## Reporting Labels

- `reporting_role` keeps benchmark outputs separate from pilot and setup runs.
- Recommended values:
  - `pilot` for one-off live sanity checks
  - `case_study` for single-study illustrative runs
  - `scientific_evaluation` for headline benchmark results
  - `baseline` for comparison conditions

## Workflow Variants

- `full_agentic`
  The orchestrated system with role-separated evidence, planning, sufficiency checks, and draft revision.
- `generic_rag`
  Retrieval-grounded generation without the role-separated planning/revision loop.
- `vanilla_llm`
  Generation without retrieval grounding, used as a simple baseline.

Each new experiment specification should fit into one of these categories so the evaluation sequence remains easy to follow.
