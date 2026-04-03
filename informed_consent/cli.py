from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import asdict
from pathlib import Path
import json

from .pipeline import ConsentPipeline


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Local tooling for the informed consent implementation.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser(
        "init-run",
        help="Initialize a reproducible run folder and log the current inputs.",
    )
    init_parser.add_argument("--purpose", default="prototype_bootstrap", help="Short description of the run purpose.")
    init_parser.add_argument("--source-dir", type=Path, help="Directory containing source documents to inventory.")
    init_parser.add_argument("--template-file", type=Path, help="Plain-text base consent template.")
    init_parser.add_argument("--patient-profile-file", type=Path, help="JSON file describing the patient profile.")
    init_parser.add_argument("--tags", nargs="*", default=[], help="Optional tags to attach to the run manifest.")
    init_parser.add_argument("--notes", default="", help="Optional free-text run notes.")

    corpus_parser = subparsers.add_parser(
        "prepare-corpus",
        help="Create a new run and prepare retrieval-ready source chunks from local documents.",
    )
    corpus_parser.add_argument("--source-dir", type=Path, required=True, help="Directory containing local source documents.")
    corpus_parser.add_argument("--purpose", default="corpus_preparation", help="Short description of the run purpose.")
    corpus_parser.add_argument("--template-file", type=Path, help="Optional plain-text base consent template.")
    corpus_parser.add_argument("--patient-profile-file", type=Path, help="Optional JSON file describing the patient profile.")
    corpus_parser.add_argument("--tags", nargs="*", default=[], help="Optional tags to attach to the run manifest.")
    corpus_parser.add_argument("--notes", default="", help="Optional free-text run notes.")

    hybrid_parser = subparsers.add_parser(
        "build-hybrid-index",
        help="Build a local dense embedding index to enable hybrid retrieval for a prepared run.",
    )
    hybrid_parser.add_argument("--run-id", required=True, help="Run identifier returned by prepare-corpus.")

    batch_parser = subparsers.add_parser(
        "run-batch-experiment",
        help="Create isolated case runs from a prepared corpus and execute a versioned batch spec.",
    )
    batch_parser.add_argument("--spec-file", type=Path, required=True, help="JSON batch specification file.")
    batch_parser.add_argument("--base-run-id", help="Optional prepared corpus run id to override the one in the spec file.")
    batch_parser.add_argument("--dry-run", action="store_true", help="Prepare isolated case runs and request bundles without live model calls.")

    compare_batches_parser = subparsers.add_parser(
        "compare-batch-results",
        help="Aggregate multiple saved batch summaries into one comparison table for baseline and ablation reporting.",
    )
    compare_batches_parser.add_argument(
        "--batch-summary",
        action="append",
        dest="batch_summaries",
        required=True,
        help="Path to a batch_summary.json file. Repeatable.",
    )
    compare_batches_parser.add_argument("--comparison-id", help="Optional label for the exported comparison files.")

    query_parser = subparsers.add_parser(
        "query-corpus",
        help="Run a retrieval query against a prepared run corpus.",
    )
    query_parser.add_argument("--run-id", required=True, help="Run identifier returned by prepare-corpus.")
    query_parser.add_argument("--query", required=True, help="Question or search string to run against the prepared chunks.")
    query_parser.add_argument("--top-k", type=int, help="Optional override for the number of hits to return.")
    query_parser.add_argument("--retrieval-mode", choices=["lexical", "dense", "hybrid"], help="Optional retrieval-mode override.")
    query_parser.add_argument("--source-group", action="append", dest="source_groups", default=[], help="Optional source-group filter. Repeatable.")
    query_parser.add_argument("--source-id", action="append", dest="source_ids", default=[], help="Optional source-id filter. Repeatable.")
    query_parser.add_argument("--filter-logic", choices=["intersection", "union"], help="How to combine source-group and source-id filters when both are provided.")

    draft_parser = subparsers.add_parser(
        "draft-personalized-consent",
        help="Prepare or generate a grounded personalized consent draft from a prepared corpus run.",
    )
    draft_parser.add_argument("--run-id", required=True, help="Run identifier returned by prepare-corpus.")
    draft_parser.add_argument("--patient-profile-file", type=Path, help="JSON file describing the participant profile.")
    draft_parser.add_argument("--template-file", type=Path, help="Optional plain-text base consent template.")
    draft_parser.add_argument("--generation-query", help="Optional retrieval query override.")
    draft_parser.add_argument("--top-k", type=int, help="Optional override for the number of retrieval hits.")
    draft_parser.add_argument("--retrieval-mode", choices=["lexical", "dense", "hybrid"], help="Optional retrieval-mode override.")
    draft_parser.add_argument("--source-group", action="append", dest="source_groups", default=[], help="Optional source-group filter. Repeatable.")
    draft_parser.add_argument("--source-id", action="append", dest="source_ids", default=[], help="Optional source-id filter. Repeatable.")
    draft_parser.add_argument("--filter-logic", choices=["intersection", "union"], help="How to combine source-group and source-id filters when both are provided.")
    draft_parser.add_argument("--workflow-variant", choices=["full_agentic", "generic_rag", "vanilla_llm"], help="Execution mode for draft generation.")
    draft_parser.add_argument("--dry-run", action="store_true", help="Prepare artifacts without making a live model call.")

    qa_parser = subparsers.add_parser(
        "answer-consent-question",
        help="Prepare or generate a grounded answer to a participant consent question.",
    )
    qa_parser.add_argument("--run-id", required=True, help="Run identifier containing the prepared corpus.")
    qa_parser.add_argument("--question", required=True, help="Participant question to answer.")
    qa_parser.add_argument("--patient-profile-file", type=Path, help="Optional JSON file describing the participant profile.")
    qa_parser.add_argument("--top-k", type=int, help="Optional override for the number of retrieval hits.")
    qa_parser.add_argument("--retrieval-mode", choices=["lexical", "dense", "hybrid"], help="Optional retrieval-mode override.")
    qa_parser.add_argument("--source-group", action="append", dest="source_groups", default=[], help="Optional source-group filter. Repeatable.")
    qa_parser.add_argument("--source-id", action="append", dest="source_ids", default=[], help="Optional source-id filter. Repeatable.")
    qa_parser.add_argument("--filter-logic", choices=["intersection", "union"], help="How to combine source-group and source-id filters when both are provided.")
    qa_parser.add_argument("--workflow-variant", choices=["full_agentic", "generic_rag", "vanilla_llm"], help="Execution mode for question answering.")
    qa_parser.add_argument("--dry-run", action="store_true", help="Prepare artifacts without making a live model call.")

    formalize_parser = subparsers.add_parser(
        "formalize-consent",
        help="Prepare or generate a structured consent record from a personalized consent draft.",
    )
    formalize_parser.add_argument("--run-id", required=True, help="Run identifier containing the personalized draft.")
    formalize_parser.add_argument("--patient-profile-file", type=Path, help="Optional JSON file describing the participant profile.")
    formalize_parser.add_argument("--draft-file", type=Path, help="Optional path to a personalized consent draft JSON file.")
    formalize_parser.add_argument("--dry-run", action="store_true", help="Prepare artifacts without making a live model call.")

    routed_parser = subparsers.add_parser(
        "handle-user-request",
        help="Let the orchestrator classify a free-form user request and route it to the appropriate agent path.",
    )
    routed_parser.add_argument("--run-id", required=True, help="Run identifier containing the prepared corpus and any saved artifacts.")
    routed_parser.add_argument("--user-input", required=True, help="Free-form user request to route.")
    routed_parser.add_argument("--patient-profile-file", type=Path, help="Optional JSON file describing the participant profile.")
    routed_parser.add_argument("--template-file", type=Path, help="Optional plain-text base consent template.")
    routed_parser.add_argument("--draft-file", type=Path, help="Optional personalized consent draft JSON file.")
    routed_parser.add_argument("--top-k", type=int, help="Optional override for the number of retrieval hits.")
    routed_parser.add_argument("--retrieval-mode", choices=["lexical", "dense", "hybrid"], help="Optional retrieval-mode override.")
    routed_parser.add_argument("--source-group", action="append", dest="source_groups", default=[], help="Optional source-group filter. Repeatable.")
    routed_parser.add_argument("--source-id", action="append", dest="source_ids", default=[], help="Optional source-id filter. Repeatable.")
    routed_parser.add_argument("--filter-logic", choices=["intersection", "union"], help="How to combine source-group and source-id filters when both are provided.")
    routed_parser.add_argument("--workflow-variant", choices=["full_agentic", "generic_rag", "vanilla_llm"], help="Execution mode for routed generation and question answering.")
    routed_parser.add_argument("--dry-run", action="store_true", help="Prepare artifacts without making a live model call.")

    evaluate_parser = subparsers.add_parser(
        "evaluate-run",
        help="Compute evaluation summaries and metric files from saved run artifacts.",
    )
    evaluate_parser.add_argument("--run-id", required=True, help="Run identifier to evaluate.")

    retrieval_benchmark_parser = subparsers.add_parser(
        "evaluate-retrieval-benchmark",
        help="Score retrieval quality against a versioned benchmark spec without requiring live model calls.",
    )
    retrieval_benchmark_parser.add_argument("--run-id", required=True, help="Prepared run identifier to benchmark.")
    retrieval_benchmark_parser.add_argument("--spec-file", type=Path, required=True, help="JSON retrieval benchmark spec file.")
    retrieval_benchmark_parser.add_argument("--mode", action="append", dest="modes", default=[], help="Optional retrieval mode override. Repeatable.")
    retrieval_benchmark_parser.add_argument("--top-k", type=int, help="Optional default top-k override.")
    retrieval_benchmark_parser.add_argument("--filter-logic", choices=["intersection", "union"], help="Optional default filter logic override for benchmark queries.")

    review_parser = subparsers.add_parser(
        "export-manual-review-bundle",
        help="Export a CSV bundle for small-scale manual review from saved run artifacts.",
    )
    review_parser.add_argument("--run-id", required=True, help="Run identifier to export for review.")

    reference_parser = subparsers.add_parser(
        "export-evaluation-reference-pack",
        help="Export a reusable evaluation reference pack combining study facts, regulatory checklist evidence, posted consent forms, and the manual-review rubric.",
    )
    reference_parser.add_argument("--run-id", required=True, help="Run identifier to export the reference pack for.")
    reference_parser.add_argument("--source-id", help="Optional study source id or NCT id to use as the primary study reference.")

    source_plan_parser = subparsers.add_parser(
        "plan-public-sources",
        help="Show which public registry sources would be downloaded.",
    )
    source_plan_parser.add_argument("--registry-file", type=Path, help="Optional source registry JSON file.")
    source_plan_parser.add_argument("--group-id", action="append", dest="group_ids", default=[], help="Optional source group filter. Repeatable.")
    source_plan_parser.add_argument("--source-id", action="append", dest="source_ids", default=[], help="Optional source id filter. Repeatable.")

    source_download_parser = subparsers.add_parser(
        "download-public-sources",
        help="Download selected public sources from the configured registry into the local raw data area.",
    )
    source_download_parser.add_argument("--registry-file", type=Path, help="Optional source registry JSON file.")
    source_download_parser.add_argument("--output-dir", type=Path, help="Optional target directory. Defaults to data/raw/public.")
    source_download_parser.add_argument("--group-id", action="append", dest="group_ids", default=[], help="Optional source group filter. Repeatable.")
    source_download_parser.add_argument("--source-id", action="append", dest="source_ids", default=[], help="Optional source id filter. Repeatable.")

    clinicaltrials_fetch_parser = subparsers.add_parser(
        "fetch-clinicaltrials-studies",
        help="Fetch real ClinicalTrials.gov study records into the local raw data area for study-specific grounding.",
    )
    clinicaltrials_fetch_parser.add_argument("--output-dir", type=Path, help="Optional target directory. Defaults to data/raw/public.")
    clinicaltrials_fetch_parser.add_argument("--nct-id", action="append", dest="nct_ids", default=[], help="ClinicalTrials.gov NCT identifier. Repeatable.")
    clinicaltrials_fetch_parser.add_argument("--query-term", help="Optional ClinicalTrials.gov query.term search string.")
    clinicaltrials_fetch_parser.add_argument("--max-studies", type=int, default=10, help="Maximum number of studies to fetch when using --query-term.")
    clinicaltrials_fetch_parser.add_argument("--page-size", type=int, default=10, help="Page size to request from ClinicalTrials.gov when using --query-term.")

    show_parser = subparsers.add_parser(
        "show-config",
        help="Print the default local configuration as JSON.",
    )
    show_parser.add_argument("--pretty", action="store_true", help="Pretty-print the configuration.")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    pipeline = ConsentPipeline()

    if args.command == "init-run":
        manifest = pipeline.bootstrap_run(
            purpose=args.purpose,
            source_dir=args.source_dir,
            template_path=args.template_file,
            patient_profile_path=args.patient_profile_file,
            tags=args.tags,
            notes=args.notes,
        )
        print(f"Run initialized: {manifest.run_id}")
        print(f"Manifest: {manifest.artifact_paths['run_dir']}")
        return

    if args.command == "prepare-corpus":
        manifest = pipeline.prepare_corpus(
            purpose=args.purpose,
            source_dir=args.source_dir,
            template_path=args.template_file,
            patient_profile_path=args.patient_profile_file,
            tags=args.tags,
            notes=args.notes,
        )
        print(f"Corpus prepared: {manifest.run_id}")
        print(f"Run directory: {manifest.artifact_paths['run_dir']}")
        return

    if args.command == "build-hybrid-index":
        payload = pipeline.build_hybrid_index(run_id=args.run_id)
        print(json.dumps(payload, indent=2))
        return

    if args.command == "run-batch-experiment":
        payload = pipeline.run_batch_experiment(
            spec_path=args.spec_file,
            dry_run=args.dry_run,
            base_run_id_override=args.base_run_id,
        )
        print(json.dumps(payload, indent=2))
        return

    if args.command == "compare-batch-results":
        payload = pipeline.compare_batch_results(
            [Path(item) for item in args.batch_summaries],
            comparison_id=args.comparison_id,
        )
        print(json.dumps(payload, indent=2))
        return

    if args.command == "query-corpus":
        hits = pipeline.query_prepared_corpus(
            run_id=args.run_id,
            query=args.query,
            top_k=args.top_k,
            retrieval_mode=args.retrieval_mode,
            source_group_filters=args.source_groups,
            source_id_filters=args.source_ids,
            filter_logic=args.filter_logic,
        )
        print(json.dumps(hits, indent=2))
        return

    if args.command == "draft-personalized-consent":
        payload = pipeline.draft_personalized_consent(
            run_id=args.run_id,
            patient_profile_path=args.patient_profile_file,
            template_path=args.template_file,
            generation_query=args.generation_query,
            top_k=args.top_k,
            retrieval_mode=args.retrieval_mode,
            source_group_filters=args.source_groups,
            source_id_filters=args.source_ids,
            filter_logic=args.filter_logic,
            workflow_variant=args.workflow_variant,
            dry_run=args.dry_run,
        )
        print(json.dumps(payload, indent=2))
        return

    if args.command == "answer-consent-question":
        payload = pipeline.answer_consent_question(
            run_id=args.run_id,
            question=args.question,
            patient_profile_path=args.patient_profile_file,
            top_k=args.top_k,
            retrieval_mode=args.retrieval_mode,
            source_group_filters=args.source_groups,
            source_id_filters=args.source_ids,
            filter_logic=args.filter_logic,
            workflow_variant=args.workflow_variant,
            dry_run=args.dry_run,
        )
        print(json.dumps(payload, indent=2))
        return

    if args.command == "formalize-consent":
        payload = pipeline.formalize_consent(
            run_id=args.run_id,
            patient_profile_path=args.patient_profile_file,
            draft_path=args.draft_file,
            dry_run=args.dry_run,
        )
        print(json.dumps(payload, indent=2))
        return

    if args.command == "handle-user-request":
        payload = pipeline.handle_user_request(
            run_id=args.run_id,
            user_input=args.user_input,
            patient_profile_path=args.patient_profile_file,
            template_path=args.template_file,
            draft_path=args.draft_file,
            top_k=args.top_k,
            retrieval_mode=args.retrieval_mode,
            source_group_filters=args.source_groups,
            source_id_filters=args.source_ids,
            filter_logic=args.filter_logic,
            workflow_variant=args.workflow_variant,
            dry_run=args.dry_run,
        )
        print(json.dumps(payload, indent=2))
        return

    if args.command == "evaluate-run":
        payload = pipeline.evaluate_run(run_id=args.run_id)
        print(json.dumps(payload, indent=2))
        return

    if args.command == "evaluate-retrieval-benchmark":
        payload = pipeline.evaluate_retrieval_benchmark(
            run_id=args.run_id,
            spec_path=args.spec_file,
            modes=args.modes,
            top_k=args.top_k,
            filter_logic=args.filter_logic,
        )
        print(json.dumps(payload, indent=2))
        return

    if args.command == "export-manual-review-bundle":
        payload = pipeline.export_manual_review_bundle(run_id=args.run_id)
        print(json.dumps(payload, indent=2))
        return

    if args.command == "export-evaluation-reference-pack":
        payload = pipeline.export_evaluation_reference_pack(
            run_id=args.run_id,
            source_id=args.source_id,
        )
        print(json.dumps(payload, indent=2))
        return

    if args.command == "plan-public-sources":
        payload = pipeline.plan_public_sources(
            registry_path=args.registry_file,
            group_ids=args.group_ids,
            source_ids=args.source_ids,
        )
        print(json.dumps(payload, indent=2))
        return

    if args.command == "download-public-sources":
        payload = pipeline.download_public_sources(
            registry_path=args.registry_file,
            output_dir=args.output_dir,
            group_ids=args.group_ids,
            source_ids=args.source_ids,
        )
        print(json.dumps(payload, indent=2))
        return

    if args.command == "fetch-clinicaltrials-studies":
        payload = pipeline.fetch_clinicaltrials_studies(
            output_dir=args.output_dir,
            nct_ids=args.nct_ids,
            query_term=args.query_term,
            max_studies=args.max_studies,
            page_size=args.page_size,
        )
        print(json.dumps(payload, indent=2))
        return

    if args.command == "show-config":
        payload = {
            "study_id": pipeline.config.study_id,
            "site_id": pipeline.config.site_id,
            "models": asdict(pipeline.config.models),
            "retrieval": asdict(pipeline.config.retrieval),
            "paths": {key: str(value) for key, value in asdict(pipeline.config.paths).items()},
        }
        if args.pretty:
            print(json.dumps(payload, indent=2))
        else:
            print(json.dumps(payload))
        return

if __name__ == "__main__":
    main()
