from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from informed_consent.pipeline import ConsentPipeline


FAILURE_FIELDS = [
    "failure_missing_selected_study_grounding",
    "failure_foreign_study_contamination",
    "failure_regulatory_only_grounding",
    "failure_unsupported_claim_risk",
    "failure_omitted_required_element",
    "failure_overconfident_answer",
    "failure_malformed_structured_output",
    "failure_grounding_gap_declared",
]

NUMERIC_AGGREGATE_FIELDS = [
    "qa_answered_count",
    "qa_answered_question_count",
    "qa_abstained_count",
    "qa_abstained_question_count",
    "qa_clarified_count",
    "qa_abstention_rate",
    "qa_uncertainty_flag_count",
    "qa_uncertainty_rate",
    "qa_unsupported_marker_count",
    "qa_unsupported_sentence_count",
    "qa_citationless_sentence_count",
    "qa_citationless_sentence_rate",
    "qa_selected_study_hit_count",
    "qa_foreign_study_hit_count",
    "qa_regulatory_hit_count",
    "qa_total_hit_count",
    "draft_study_specific_hit_count",
    "draft_selected_study_hit_count",
    "draft_foreign_study_hit_count",
    "draft_regulatory_hit_count",
    "draft_total_hit_count",
    "draft_study_specific_hit_ratio",
    "draft_required_element_coverage_ratio",
    "draft_missing_required_element_count",
    "draft_citation_marker_coverage_ratio",
    "draft_sentence_citation_coverage_ratio",
    "draft_citationless_sentence_count",
    "draft_citationless_sentence_rate",
    "draft_unsupported_marker_count",
    "draft_unsupported_sentence_count",
    "draft_flesch_kincaid_grade",
    "structured_required_field_presence_ratio",
    "qa_average_citation_marker_coverage_ratio",
    "qa_average_sentence_citation_coverage_ratio",
    "qa_average_flesch_kincaid_grade",
]


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def normalize_str(value: Any) -> str:
    return str(value or "").strip()


def compute_case_row(original_row: dict[str, Any], summary_payload: dict[str, Any]) -> dict[str, Any]:
    summary = summary_payload.get("summary", {})
    summary_metadata = summary.get("metadata", {})
    if not isinstance(summary_metadata, dict):
        summary_metadata = {}
    draft_summary = summary.get("draft", {})
    structured_summary = summary.get("structured_record", {})
    qa_summary = summary.get("qa_answers", {})
    failure_taxonomy = summary.get("failure_taxonomy", {})
    if not isinstance(failure_taxonomy, dict):
        failure_taxonomy = {}
    case_failure_flags = failure_taxonomy.get("case_failure_flags", {})
    if not isinstance(case_failure_flags, dict):
        case_failure_flags = {}
    draft_failure_flags = draft_summary.get("failure_flags", {})
    if not isinstance(draft_failure_flags, dict):
        draft_failure_flags = {}
    qa_failure_flags = qa_summary.get("failure_flags", {})
    if not isinstance(qa_failure_flags, dict):
        qa_failure_flags = {}
    structured_failure_flags = structured_summary.get("failure_flags", {})
    if not isinstance(structured_failure_flags, dict):
        structured_failure_flags = {}

    qa_per_question = qa_summary.get("per_question", []) or []
    grade_values = [
        item.get("flesch_kincaid_grade")
        for item in qa_per_question
        if isinstance(item.get("flesch_kincaid_grade"), (int, float))
    ]
    qa_avg_fkg = round(sum(grade_values) / len(grade_values), 4) if grade_values else None

    row = dict(original_row)
    row.update(
        {
            "status": "completed",
            "config_path": summary_metadata.get("config_path") or row.get("config_path"),
            "git_commit_hash": summary_metadata.get("git_commit_hash") or row.get("git_commit_hash"),
            "model_id": summary_metadata.get("model_id") or row.get("model_id"),
            "embedding_model_id": summary_metadata.get("embedding_model_id") or row.get("embedding_model_id"),
            "corpus_version": summary_metadata.get("corpus_version") or row.get("corpus_version"),
            "index_version": summary_metadata.get("index_version") or row.get("index_version"),
            "random_seed": summary_metadata.get("random_seed")
            if summary_metadata.get("random_seed") is not None
            else row.get("random_seed"),
            "retrieval_mode": summary_metadata.get("retrieval_mode") or row.get("retrieval_mode"),
            "retrieval_top_k": summary_metadata.get("retrieval_top_k") or row.get("retrieval_top_k"),
            "retrieval_filter_logic": summary_metadata.get("retrieval_filter_logic") or row.get("retrieval_filter_logic"),
            "retrieval_filter_logic_config": summary_metadata.get("retrieval_filter_logic_config")
            or row.get("retrieval_filter_logic_config"),
            "retrieval_filter_logic_effective": summary_metadata.get("retrieval_filter_logic")
            or row.get("retrieval_filter_logic_effective"),
            "draft_retrieval_filter_logic_effective": summary_metadata.get("draft_retrieval_filter_logic_effective"),
            "qa_retrieval_filter_logic_effective": "|".join(summary_metadata.get("qa_retrieval_filter_logic_effective", []) or []),
            "draft_retrieval_strategy_effective": summary_metadata.get("draft_retrieval_strategy_effective"),
            "qa_retrieval_strategy_effective": "|".join(summary_metadata.get("qa_retrieval_strategy_effective", []) or []),
            "draft_system_prompt_id": summary_metadata.get("draft_system_prompt_id"),
            "draft_user_prompt_id": summary_metadata.get("draft_user_prompt_id"),
            "formalization_system_prompt_id": summary_metadata.get("formalization_system_prompt_id"),
            "formalization_user_prompt_id": summary_metadata.get("formalization_user_prompt_id"),
            "qa_system_prompt_ids": "|".join(summary_metadata.get("qa_system_prompt_ids", []) or []),
            "qa_user_prompt_ids": "|".join(summary_metadata.get("qa_user_prompt_ids", []) or []),
            "question_count": qa_summary.get("question_count", 0),
            "qa_answered_count": qa_summary.get("answered_count"),
            "qa_answered_question_count": qa_summary.get("answered_question_count"),
            "qa_abstained_count": qa_summary.get("abstained_count"),
            "qa_abstained_question_count": qa_summary.get("abstained_question_count"),
            "qa_clarified_count": qa_summary.get("clarified_count"),
            "qa_abstention_rate": qa_summary.get("abstention_rate"),
            "qa_uncertainty_flag_count": qa_summary.get("uncertainty_flag_count"),
            "qa_uncertainty_rate": qa_summary.get("uncertainty_rate"),
            "qa_unsupported_marker_count": qa_summary.get("unsupported_marker_count"),
            "qa_unsupported_sentence_count": qa_summary.get("unsupported_sentence_count"),
            "qa_citationless_sentence_count": qa_summary.get("citationless_sentence_count"),
            "qa_citationless_sentence_rate": qa_summary.get("citationless_sentence_rate"),
            "qa_selected_study_hit_count": qa_summary.get("selected_study_hit_count"),
            "qa_selected_study_hit_present": qa_summary.get("selected_study_hit_present"),
            "qa_foreign_study_hit_count": qa_summary.get("foreign_study_hit_count"),
            "qa_foreign_study_hit_present": qa_summary.get("foreign_study_hit_present"),
            "qa_regulatory_hit_count": qa_summary.get("regulatory_hit_count"),
            "qa_total_hit_count": qa_summary.get("total_hit_count"),
            "qa_study_specific_grounding_met": qa_summary.get("study_specific_grounding_met"),
            "qa_study_specific_grounding_gap": qa_summary.get("study_specific_grounding_gap"),
            "qa_grounding_source_ids_used": "|".join(qa_summary.get("grounding_source_ids_used", []) or []),
            "qa_foreign_source_ids_detected": "|".join(qa_summary.get("foreign_source_ids_detected", []) or []),
            "draft_expected_study_specific_grounding": draft_summary.get("expected_study_specific_grounding"),
            "draft_study_specific_grounding_met": draft_summary.get("study_specific_grounding_met"),
            "draft_study_specific_grounding_gap": draft_summary.get("study_specific_grounding_gap"),
            "draft_selected_study_hit_count": draft_summary.get("selected_study_hit_count"),
            "draft_selected_study_hit_present": draft_summary.get("selected_study_hit_present"),
            "draft_foreign_study_hit_count": draft_summary.get("foreign_study_hit_count"),
            "draft_foreign_study_hit_present": draft_summary.get("foreign_study_hit_present"),
            "draft_study_specific_hit_count": draft_summary.get("study_specific_hit_count"),
            "draft_regulatory_hit_count": draft_summary.get("regulatory_hit_count"),
            "draft_total_hit_count": draft_summary.get("total_hit_count"),
            "draft_has_study_specific_evidence": draft_summary.get("has_study_specific_evidence"),
            "draft_study_specific_hit_ratio": draft_summary.get("study_specific_hit_ratio"),
            "draft_grounding_source_ids_used": "|".join(draft_summary.get("grounding_source_ids_used", []) or []),
            "draft_foreign_source_ids_detected": "|".join(draft_summary.get("foreign_source_ids_detected", []) or []),
            "draft_required_element_coverage_ratio": draft_summary.get("required_element_coverage_ratio"),
            "draft_missing_required_element_count": draft_summary.get("missing_required_element_count"),
            "draft_citation_marker_coverage_ratio": draft_summary.get("citation_marker_coverage_ratio"),
            "draft_sentence_citation_coverage_ratio": draft_summary.get("sentence_citation_coverage_ratio"),
            "draft_citationless_sentence_count": draft_summary.get("citationless_sentence_count"),
            "draft_citationless_sentence_rate": draft_summary.get("citationless_sentence_rate"),
            "draft_unsupported_marker_count": draft_summary.get("unsupported_marker_count"),
            "draft_unsupported_sentence_count": draft_summary.get("unsupported_sentence_count"),
            "draft_flesch_kincaid_grade": draft_summary.get("flesch_kincaid_grade"),
            "draft_grounding_gap_declared": draft_summary.get("grounding_gap_declared"),
            "draft_unsupported_claim_risk": draft_summary.get("unsupported_claim_risk"),
            "structured_schema_repair_applied": structured_summary.get("schema_repair_applied"),
            "structured_required_field_presence_ratio": structured_summary.get("required_field_presence_ratio"),
            "structured_malformed_output": structured_summary.get("malformed_structured_output"),
            "qa_average_citation_marker_coverage_ratio": qa_summary.get("average_citation_marker_coverage_ratio"),
            "qa_average_sentence_citation_coverage_ratio": qa_summary.get("average_sentence_citation_coverage_ratio"),
            "qa_average_flesch_kincaid_grade": qa_avg_fkg,
            "failure_missing_selected_study_grounding": case_failure_flags.get("missing_selected_study_grounding"),
            "failure_foreign_study_contamination": case_failure_flags.get("foreign_study_contamination"),
            "failure_regulatory_only_grounding": case_failure_flags.get("regulatory_only_grounding"),
            "failure_unsupported_claim_risk": case_failure_flags.get("unsupported_claim_risk"),
            "failure_omitted_required_element": case_failure_flags.get("omitted_required_element"),
            "failure_overconfident_answer": case_failure_flags.get("overconfident_answer"),
            "failure_malformed_structured_output": case_failure_flags.get("malformed_structured_output"),
            "failure_grounding_gap_declared": case_failure_flags.get("grounding_gap_declared"),
            "draft_failure_missing_selected_study_grounding": draft_failure_flags.get("missing_selected_study_grounding"),
            "draft_failure_foreign_study_contamination": draft_failure_flags.get("foreign_study_contamination"),
            "draft_failure_regulatory_only_grounding": draft_failure_flags.get("regulatory_only_grounding"),
            "draft_failure_unsupported_claim_risk": draft_failure_flags.get("unsupported_claim_risk"),
            "draft_failure_omitted_required_element": draft_failure_flags.get("omitted_required_element"),
            "qa_failure_missing_selected_study_grounding": qa_failure_flags.get("missing_selected_study_grounding"),
            "qa_failure_foreign_study_contamination": qa_failure_flags.get("foreign_study_contamination"),
            "qa_failure_regulatory_only_grounding": qa_failure_flags.get("regulatory_only_grounding"),
            "qa_failure_unsupported_claim_risk": qa_failure_flags.get("unsupported_claim_risk"),
            "qa_failure_overconfident_answer": qa_failure_flags.get("overconfident_answer"),
            "structured_failure_malformed_structured_output": structured_failure_flags.get("malformed_structured_output"),
        }
    )
    return row


def compute_aggregate_metrics(completed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate_metrics: dict[str, Any] = {}
    for key in NUMERIC_AGGREGATE_FIELDS:
        numeric_values = [float(row[key]) for row in completed_rows if isinstance(row.get(key), (int, float))]
        aggregate_metrics[f"average_{key}"] = round(sum(numeric_values) / len(numeric_values), 4) if numeric_values else None

    schema_repairs = [
        row.get("structured_schema_repair_applied")
        for row in completed_rows
        if isinstance(row.get("structured_schema_repair_applied"), bool)
    ]
    aggregate_metrics["structured_schema_repair_rate"] = (
        round(sum(1 for value in schema_repairs if value) / len(schema_repairs), 4) if schema_repairs else None
    )

    study_specific_flags = [
        row.get("draft_has_study_specific_evidence")
        for row in completed_rows
        if isinstance(row.get("draft_has_study_specific_evidence"), bool)
    ]
    aggregate_metrics["draft_study_specific_evidence_case_rate"] = (
        round(sum(1 for value in study_specific_flags if value) / len(study_specific_flags), 4)
        if study_specific_flags
        else None
    )

    expected_study_specific_flags = [
        row.get("draft_expected_study_specific_grounding")
        for row in completed_rows
        if isinstance(row.get("draft_expected_study_specific_grounding"), bool)
    ]
    aggregate_metrics["draft_expected_study_specific_grounding_case_rate"] = (
        round(sum(1 for value in expected_study_specific_flags if value) / len(expected_study_specific_flags), 4)
        if expected_study_specific_flags
        else None
    )

    expected_grounding_rows = [row for row in completed_rows if row.get("draft_expected_study_specific_grounding") is True]
    grounding_success_flags = [
        row.get("draft_study_specific_grounding_met")
        for row in expected_grounding_rows
        if isinstance(row.get("draft_study_specific_grounding_met"), bool)
    ]
    aggregate_metrics["draft_study_specific_grounding_success_rate"] = (
        round(sum(1 for value in grounding_success_flags if value) / len(grounding_success_flags), 4)
        if grounding_success_flags
        else None
    )
    grounding_gap_flags = [
        row.get("draft_study_specific_grounding_gap")
        for row in expected_grounding_rows
        if isinstance(row.get("draft_study_specific_grounding_gap"), bool)
    ]
    aggregate_metrics["draft_study_specific_grounding_gap_rate"] = (
        round(sum(1 for value in grounding_gap_flags if value) / len(grounding_gap_flags), 4)
        if grounding_gap_flags
        else None
    )

    for failure_field in FAILURE_FIELDS:
        flags = [row.get(failure_field) for row in completed_rows if isinstance(row.get(failure_field), bool)]
        aggregate_metrics[f"{failure_field}_rate"] = (
            round(sum(1 for value in flags if value) / len(flags), 4) if flags else None
        )
    return aggregate_metrics


def write_batch_failure_summary(
    pipeline: ConsentPipeline,
    *,
    batch_payload: dict[str, Any],
    metric_rows: list[dict[str, Any]],
    version_suffix: str,
) -> Path:
    completed_rows = [row for row in metric_rows if row.get("status") == "completed"]
    grouped_completed_rows: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in completed_rows:
        key = (
            normalize_str(row.get("workflow_variant")) or "unknown",
            normalize_str(row.get("question_set_label")) or "unknown",
        )
        grouped_completed_rows.setdefault(key, []).append(row)

    failure_summary_rows: list[dict[str, Any]] = []
    for (workflow_value, question_set_value), rows_for_group in sorted(grouped_completed_rows.items()):
        case_count = len(rows_for_group)
        for failure_field in FAILURE_FIELDS:
            flags = [bool(row.get(failure_field)) for row in rows_for_group if isinstance(row.get(failure_field), bool)]
            failure_count = sum(1 for flag in flags if flag)
            failure_summary_rows.append(
                {
                    "batch_run_id": batch_payload["batch_run_id"],
                    "batch_id": batch_payload["batch_id"],
                    "workflow_variant": workflow_value,
                    "question_set_label": question_set_value,
                    "failure_type": failure_field.removeprefix("failure_"),
                    "case_count": case_count,
                    "failure_count": failure_count,
                    "failure_rate": round(failure_count / max(case_count, 1), 4),
                }
            )
    return pipeline.artifacts.write_table_csv(
        f"{batch_payload['batch_run_id']}_batch_failure_summary_{version_suffix}.csv",
        failure_summary_rows,
    )


def reevaluate_batch_summaries(
    batch_summary_paths: list[Path],
    *,
    version_suffix: str,
    comparison_id: str,
) -> dict[str, Any]:
    pipeline = ConsentPipeline()
    refreshed_summary_paths: list[Path] = []

    for batch_summary_path in batch_summary_paths:
        batch_payload = load_json(batch_summary_path)
        case_metrics_csv = Path(str(batch_payload["case_metrics_csv"]))
        with case_metrics_csv.open("r", encoding="utf-8", newline="") as handle:
            original_rows = list(csv.DictReader(handle))
        row_by_case_run_id = {normalize_str(row.get("case_run_id")): row for row in original_rows}

        refreshed_rows: list[dict[str, Any]] = []
        for case in batch_payload.get("cases", []):
            case_run_id = normalize_str(case.get("case_run_id"))
            if not case_run_id:
                continue
            pipeline.evaluate_run(case_run_id)
            updated_summary = load_json(Path(str(case["evaluation_summary_path"])))
            original_row = row_by_case_run_id.get(case_run_id, {"case_run_id": case_run_id})
            refreshed_rows.append(compute_case_row(original_row, {"summary": updated_summary}))

        refreshed_case_metrics_csv = pipeline.artifacts.write_table_csv(
            f"{batch_payload['batch_run_id']}_batch_case_metrics_{version_suffix}.csv",
            refreshed_rows,
        )
        completed_rows = [row for row in refreshed_rows if row.get("status") == "completed"]
        batch_payload["case_metrics_csv"] = str(refreshed_case_metrics_csv)
        batch_payload["aggregate_metrics"] = compute_aggregate_metrics(completed_rows)
        batch_payload["failure_summary_csv"] = str(
            write_batch_failure_summary(
                pipeline,
                batch_payload=batch_payload,
                metric_rows=refreshed_rows,
                version_suffix=version_suffix,
            )
        )
        refreshed_summary_path = batch_summary_path.with_name(f"batch_summary_{version_suffix}.json")
        refreshed_summary_path.write_text(json.dumps(batch_payload, indent=2), encoding="utf-8")
        refreshed_summary_paths.append(refreshed_summary_path)

    comparison_payload = pipeline.compare_batch_results(refreshed_summary_paths, comparison_id=comparison_id)
    return {
        "refreshed_batch_summaries": [str(path) for path in refreshed_summary_paths],
        "comparison_payload": comparison_payload,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-evaluate saved batch case artifacts offline and rebuild versioned comparison tables."
    )
    parser.add_argument(
        "--batch-summary",
        type=Path,
        action="append",
        dest="batch_summaries",
        required=True,
        help="Batch summary JSON file to refresh. Repeat once per workflow batch.",
    )
    parser.add_argument(
        "--version-suffix",
        default="reevaluated_v2",
        help="Suffix appended to refreshed batch-summary and case-metrics outputs.",
    )
    parser.add_argument(
        "--comparison-id",
        default="batch_comparison_reevaluated_v2",
        help="Comparison identifier used for the regenerated comparison tables.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = reevaluate_batch_summaries(
        args.batch_summaries,
        version_suffix=args.version_suffix,
        comparison_id=args.comparison_id,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
