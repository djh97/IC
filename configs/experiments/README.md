# Experiment Specifications

This directory groups experiment specifications by scope and reporting role so example runs, benchmarks, and comparison conditions are easy to locate and reproduce.

## Layout

- `pilots/`
  Small validation runs for checking end-to-end behavior on a limited case set.
- `single_study/`
  Focused study-grounded experiments centered on one clinical trial.
- `multi_study/`
  Benchmark specs that expand across multiple clinical trials, participant profiles, and question sets.
- `baselines/`
  Simpler comparison conditions used for ablations and side-by-side evaluation.
- `examples/`
  Illustrative examples showing the expected structure of a batch specification.

## Included Specification Types

- `pilots/nct03877237_single_case_pilot_v1.json`
  Single-case study-grounded pilot centered on the reference trial.
- `pilots/nct03877237_comparison_pilot_full_agentic_v1.json`
- `pilots/nct03877237_comparison_pilot_generic_rag_v1.json`
- `pilots/nct03877237_comparison_pilot_vanilla_llm_v1.json`
  Small three-way comparison set with the same study, participant profiles, and question sets across all workflow variants.
- `pilots/nct03877237_gonogo_full_agentic_v1.json`
- `pilots/nct03877237_gonogo_generic_rag_v1.json`
- `pilots/nct03877237_gonogo_vanilla_llm_v1.json`
  Minimal one-case comparison slice for quick end-to-end verification of the reporting pipeline.
- `pilots/nct03877237_full_agentic_followup_pilot_v1.json`
  Focused Full Agentic-only pilot covering one study, both participant profiles, and all three question sets.
- `single_study/nct03877237_study_grounded_batch_v1.json`
  Expanded single-study evaluation centered on the reference trial.
- `multi_study/hf_interventional_multi_study_v1.json`
  Multi-study Full Agentic benchmark.
- `baselines/hf_interventional_multi_study_generic_rag_v1.json`
  Retrieval-grounded baseline without the role-separated planning and revision workflow.
- `baselines/hf_interventional_multi_study_vanilla_llm_v1.json`
  Prompt-only baseline without retrieval grounding.

## Example Commands

Run any checked-in specification from the repository root with a prepared corpus identifier:

```powershell
python run_pipeline.py run-batch-experiment --spec-file <spec_file> --base-run-id <prepared_run_id>
```

For example, a small three-way comparison can be run with:

```powershell
python run_pipeline.py run-batch-experiment --spec-file configs/experiments/pilots/nct03877237_comparison_pilot_full_agentic_v1.json --base-run-id <prepared_run_id>
python run_pipeline.py run-batch-experiment --spec-file configs/experiments/pilots/nct03877237_comparison_pilot_generic_rag_v1.json --base-run-id <prepared_run_id>
python run_pipeline.py run-batch-experiment --spec-file configs/experiments/pilots/nct03877237_comparison_pilot_vanilla_llm_v1.json --base-run-id <prepared_run_id>
python run_pipeline.py compare-batch-results --batch-summary <full_agentic_batch_summary.json> --batch-summary <generic_rag_batch_summary.json> --batch-summary <vanilla_llm_batch_summary.json> --comparison-id <comparison_id>
```

The checked-in specs are reusable across rebuilt corpora because the corpus identifier is supplied at runtime with `--base-run-id`.

## Naming

- `pilot` means a small validation run before scaling.
- `single_study` means one clinical-trial-centered evaluation.
- `multi_study` means a benchmark driven by a cohort file or multiple study records.
- `baseline` means a deliberately simpler comparison condition.

## Reporting Labels

- `reporting_role` keeps benchmark outputs separate from pilot and setup runs.
- Recommended values:
  - `pilot` for small validation runs
  - `case_study` for single-study illustrative runs
  - `scientific_evaluation` for headline benchmark results
  - `baseline` for comparison conditions

## Workflow Variants

- `full_agentic`
  The orchestrated system with role-separated evidence, planning, sufficiency checks, and draft revision.
- `generic_rag`
  Retrieval-grounded generation without the role-separated planning and revision loop.
- `vanilla_llm`
  Generation without retrieval grounding, used as a simple baseline.

Each new experiment specification should fit into one of these categories so evaluation outputs remain easy to interpret.
