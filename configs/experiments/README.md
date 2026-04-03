# Experiment Specs

This directory is organized by experiment purpose so live runs do not turn into a flat pile of unrelated specs.

## Layout

- `pilots/`
  Single-case or very small sanity-check runs to validate a path before spending endpoint time on a larger benchmark.
- `single_study/`
  Focused study-grounded experiments centered on one clinical trial.
- `multi_study/`
  Curated benchmark specs that expand across multiple clinical trials, participant profiles, and question sets.
- `baselines/`
  Simpler comparison runs such as regulatory-only setups.
- `examples/`
  Small illustrative examples that show how a batch spec is structured.

## Recommended Run Order

1. `pilots/nct03877237_single_case_pilot_v1.json`
   Use this first to confirm the live study-grounded path works end to end.
2. `single_study/nct03877237_study_grounded_batch_v1.json`
   Use this when you want a richer single-study batch around the reference case.
3. `multi_study/hf_interventional_multi_study_v1.json`
   Use this for the real multi-study benchmark after the pilot succeeds.

## Naming

- `pilot` means one-off live validation before scaling.
- `single_study` means one clinical-trial-centered evaluation.
- `multi_study` means a benchmark driven by a cohort file or multiple study records.
- `baseline` means a deliberately simpler comparison condition.

The goal is that every new experiment spec should have an obvious home and a clear role in the evaluation sequence.
