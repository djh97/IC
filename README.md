# Regulatory-Grounded Personalized Informed Consent

This repository contains a software pipeline for personalized informed consent generation and evaluation in clinical-trial settings. It combines public regulatory guidance, ClinicalTrials.gov study records, posted consent forms, retrieval, participant-aware generation, structured consent formalization, and batch evaluation.

## Repository Layout

- `informed_consent/`: core implementation package
- `configs/`: experiment specifications, patient profiles, question sets, retrieval benchmarks, and study cohorts
- `prompts/`: prompt templates used by the orchestration and generation pipeline
- `data/`: raw source material, processed-data placeholders, and evaluation inputs
- `docs/`: project documentation
- `scripts/`: helper scripts
- `tests/`: automated tests
- `run_pipeline.py`: command-line entry point

## Included Versus Generated Content

- Versioned in the repository:
  - implementation code, prompt templates, experiment specifications, tests, and example inputs
  - public-source manifests and registry snapshots under `data/raw/public/manifests/`
  - a small public-source example subset under `data/raw/public/` for reference and testing
- Generated locally:
  - prepared corpora, retrieval indices, run outputs, evaluation tables, and other experiment artifacts under `artifacts/`
- Downloaded or refreshed on demand:
  - public-source materials collected through the ingestion commands in the `Source Ingestion` section

The checked-in public-source files are intentionally minimal. The manifests and ingestion commands are used to rebuild the larger public-source bundle locally when needed.

## Requirements

- Python 3.11+
- Access to a Hugging Face Inference Endpoint

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file from `.env.example` and set the required Hugging Face values:

```env
HF_TOKEN=<set_me>
HF_MODEL_ID=Qwen/Qwen3-8B
HF_INFERENCE_ENDPOINT=<set_me>
HF_ENABLE_THINKING=false
```

Inspect the active configuration:

```bash
python run_pipeline.py show-config --pretty
```

## Common Workflow

1. Initialize a run

```bash
python run_pipeline.py init-run --purpose repository_setup
```

2. Prepare a corpus

```bash
python run_pipeline.py prepare-corpus --source-dir data/raw/examples
```

3. Build the hybrid retrieval index

```bash
python run_pipeline.py build-hybrid-index --run-id YOUR_RUN_ID
```

4. Run a batch experiment

```bash
python run_pipeline.py run-batch-experiment --spec-file configs/experiments/examples/example_batch.json --base-run-id YOUR_RUN_ID
```

5. Evaluate the saved run

```bash
python run_pipeline.py evaluate-run --run-id YOUR_RUN_ID
```

## Experiment Specifications

- Single-case pilot:

```bash
python run_pipeline.py run-batch-experiment --spec-file configs/experiments/pilots/nct03877237_single_case_pilot_v1.json --base-run-id YOUR_RUN_ID
```

- Single-study benchmark:

```bash
python run_pipeline.py run-batch-experiment --spec-file configs/experiments/single_study/nct03877237_study_grounded_batch_v1.json --base-run-id YOUR_RUN_ID
```

- Multi-study benchmark:

```bash
python run_pipeline.py run-batch-experiment --spec-file configs/experiments/multi_study/hf_interventional_multi_study_v1.json --base-run-id YOUR_RUN_ID
```

- Public-regulatory baseline:

```bash
python run_pipeline.py run-batch-experiment --spec-file configs/experiments/baselines/public_regulatory_batch_v1.json --base-run-id YOUR_RUN_ID
```

See [configs/experiments/README.md](configs/experiments/README.md) for the experiment layout and run order.

## Interactive Commands

Query a prepared corpus:

```bash
python run_pipeline.py query-corpus --run-id YOUR_RUN_ID --query "Can the participant withdraw from the study without penalty?"
```

Generate a personalized consent draft:

```bash
python run_pipeline.py draft-personalized-consent --run-id YOUR_RUN_ID --patient-profile-file configs/patient_profiles/example_us_low_literacy.json --template-file data/raw/examples/base_consent_template.txt
```

Answer a participant question:

```bash
python run_pipeline.py answer-consent-question --run-id YOUR_RUN_ID --question "Can I leave the study later without any penalty?" --patient-profile-file configs/patient_profiles/example_us_low_literacy.json
```

Generate a structured consent record:

```bash
python run_pipeline.py formalize-consent --run-id YOUR_RUN_ID --patient-profile-file configs/patient_profiles/example_us_low_literacy.json
```

Route a free-form request through the orchestrator:

```bash
python run_pipeline.py handle-user-request --run-id YOUR_RUN_ID --user-input "Can I leave the study later without penalty?" --patient-profile-file configs/patient_profiles/example_us_low_literacy.json
```

## Source Ingestion

Preview public sources listed in the registry:

```bash
python run_pipeline.py plan-public-sources
```

Download public sources:

```bash
python run_pipeline.py download-public-sources
```

Fetch ClinicalTrials.gov study records by NCT ID:

```bash
python run_pipeline.py fetch-clinicaltrials-studies --nct-id NCT03877237
```

Fetch study records by query:

```bash
python run_pipeline.py fetch-clinicaltrials-studies --query-term "heart failure" --max-studies 5
```

## Evaluation and Review

Evaluate retrieval quality:

```bash
python run_pipeline.py evaluate-retrieval-benchmark --run-id YOUR_RUN_ID --spec-file configs/retrieval_benchmarks/public_regulatory_checks_v1.json
```

Export a manual-review bundle:

```bash
python run_pipeline.py export-manual-review-bundle --run-id YOUR_RUN_ID
```

Export an evaluation reference pack:

```bash
python run_pipeline.py export-evaluation-reference-pack --run-id YOUR_RUN_ID
```

## Outputs

Generated outputs are written to `artifacts/`. This directory is excluded from version control and is intended for local run outputs, evaluation summaries, tables, and case artifacts.
