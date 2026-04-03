from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any
import json
import re

from .types import EvaluationRecord


WORD_PATTERN = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
CITATION_PATTERN = re.compile(r"\[\d+\]")

DRAFT_REQUIRED_ELEMENT_THRESHOLD = 0.7
DRAFT_SENTENCE_CITATION_THRESHOLD = 0.5
SUMMARY_SENTENCE_CITATION_THRESHOLD = 0.5
CRITICAL_PLANNED_ELEMENT_IDS = {"study_procedures", "benefits", "alternatives"}


REQUIRED_ELEMENT_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "voluntary_participation": [
        re.compile(r"\bvoluntary\b", re.IGNORECASE),
        re.compile(r"\byour choice\b", re.IGNORECASE),
        re.compile(r"\bchoice whether to take part\b", re.IGNORECASE),
        re.compile(r"\bchoose to join\b", re.IGNORECASE),
        re.compile(r"\bchoose whether to (join|take part)\b", re.IGNORECASE),
        re.compile(r"\btaking part is your choice\b", re.IGNORECASE),
        re.compile(r"\bjoining (the )?study is your choice\b", re.IGNORECASE),
    ],
    "study_procedures": [
        re.compile(r"\bstudy procedures?\b", re.IGNORECASE),
        re.compile(r"\bstudy steps\b", re.IGNORECASE),
        re.compile(r"\bstudy visits?\b", re.IGNORECASE),
    ],
    "risks": [
        re.compile(r"\brisks?\b", re.IGNORECASE),
    ],
    "benefits": [
        re.compile(r"\bpossible benefits?\b", re.IGNORECASE),
        re.compile(r"\bpotential benefits?\b", re.IGNORECASE),
        re.compile(r"\bbenefits? of (the )?(study|research)\b", re.IGNORECASE),
        re.compile(r"\bhow (the )?study may help\b", re.IGNORECASE),
        re.compile(r"\brisks? and benefits?\b", re.IGNORECASE),
        re.compile(r"\bbenefits? and risks?\b", re.IGNORECASE),
    ],
    "alternatives": [
        re.compile(r"\balternatives?\b", re.IGNORECASE),
        re.compile(r"\bother options\b", re.IGNORECASE),
    ],
    "questions": [
        re.compile(r"\bask questions?\b", re.IGNORECASE),
        re.compile(r"\bquestions? at any time\b", re.IGNORECASE),
        re.compile(r"\bcontact (the )?(study|research) team\b", re.IGNORECASE),
        re.compile(r"\bwho to contact\b", re.IGNORECASE),
    ],
    "withdrawal_rights": [
        re.compile(r"\bstop participating\b", re.IGNORECASE),
        re.compile(r"\bstop taking part\b", re.IGNORECASE),
        re.compile(r"\bwithdraw\b", re.IGNORECASE),
        re.compile(r"\bwithout (any )?penalty\b", re.IGNORECASE),
        re.compile(r"\bloss of benefits\b", re.IGNORECASE),
        re.compile(r"\blosing benefits\b", re.IGNORECASE),
        re.compile(r"\bwithout being punished\b", re.IGNORECASE),
    ],
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def deduplicate_qa_index_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest_by_question_id: dict[str, dict[str, Any]] = {}
    ordered_question_ids: list[str] = []
    for row in rows:
        question_id = str(row.get("question_id", "")).strip()
        if not question_id:
            continue
        if question_id not in latest_by_question_id:
            ordered_question_ids.append(question_id)
        current = latest_by_question_id.get(question_id)
        if current is None:
            latest_by_question_id[question_id] = row
            continue

        current_has_answer = bool(current.get("answer_path"))
        incoming_has_answer = bool(row.get("answer_path"))
        if incoming_has_answer and not current_has_answer:
            latest_by_question_id[question_id] = row
            continue
        if incoming_has_answer == current_has_answer:
            latest_by_question_id[question_id] = row

    return [latest_by_question_id[question_id] for question_id in ordered_question_ids if question_id in latest_by_question_id]


def tokenize_words(text: str) -> list[str]:
    return WORD_PATTERN.findall(text)


def has_nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def load_patient_profile_if_available(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "inputs" / "patient_profile.json"
    if not path.exists():
        return {}
    payload = load_json(path)
    return payload if isinstance(payload, dict) else {}


def target_grade_threshold(health_literacy: str, *, artifact_type: str) -> float:
    literacy = health_literacy.lower().strip()
    if artifact_type == "qa":
        if literacy == "low":
            return 8.0
        if literacy == "medium":
            return 10.0
        return 12.0
    if literacy == "low":
        return 6.5
    if literacy == "medium":
        return 9.0
    return 12.0


def split_sentences(text: str) -> list[str]:
    return [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", text) if segment.strip()]


def sentence_citation_metrics(text: str) -> dict[str, float]:
    sentences = split_sentences(text)
    if not sentences:
        return {
            "sentence_with_citation_count": 0.0,
            "sentence_without_citation_count": 0.0,
            "sentence_citation_coverage_ratio": 0.0,
        }

    sentence_with_citation_count = sum(1 for sentence in sentences if CITATION_PATTERN.search(sentence))
    sentence_without_citation_count = len(sentences) - sentence_with_citation_count
    return {
        "sentence_with_citation_count": float(sentence_with_citation_count),
        "sentence_without_citation_count": float(sentence_without_citation_count),
        "sentence_citation_coverage_ratio": round(sentence_with_citation_count / len(sentences), 4),
    }


def estimate_syllables(word: str) -> int:
    cleaned = re.sub(r"[^a-z]", "", word.lower())
    if not cleaned:
        return 1
    vowels = "aeiouy"
    count = 0
    previous_is_vowel = False
    for char in cleaned:
        is_vowel = char in vowels
        if is_vowel and not previous_is_vowel:
            count += 1
        previous_is_vowel = is_vowel
    if cleaned.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def readability_metrics(text: str) -> dict[str, float]:
    words = tokenize_words(text)
    sentences = split_sentences(text)
    if not words:
        return {
            "word_count": 0.0,
            "sentence_count": 0.0,
            "avg_words_per_sentence": 0.0,
            "flesch_reading_ease": 0.0,
            "flesch_kincaid_grade": 0.0,
        }

    sentence_count = max(len(sentences), 1)
    word_count = len(words)
    syllable_count = sum(estimate_syllables(word) for word in words)
    avg_words_per_sentence = word_count / sentence_count
    syllables_per_word = syllable_count / word_count
    flesch_reading_ease = 206.835 - 1.015 * avg_words_per_sentence - 84.6 * syllables_per_word
    flesch_kincaid_grade = 0.39 * avg_words_per_sentence + 11.8 * syllables_per_word - 15.59

    return {
        "word_count": float(word_count),
        "sentence_count": float(sentence_count),
        "avg_words_per_sentence": round(avg_words_per_sentence, 4),
        "flesch_reading_ease": round(flesch_reading_ease, 4),
        "flesch_kincaid_grade": round(flesch_kincaid_grade, 4),
    }


def extract_citation_markers(text: str) -> list[str]:
    return sorted(set(CITATION_PATTERN.findall(text)))


def evaluate_required_elements(text: str) -> dict[str, bool]:
    results: dict[str, bool] = {}
    for name, patterns in REQUIRED_ELEMENT_PATTERNS.items():
        results[name] = any(pattern.search(text) for pattern in patterns)
    return results


def summarize_personalized_draft(
    draft: dict[str, Any],
    *,
    available_markers: list[str],
    health_literacy: str,
) -> dict[str, Any]:
    text = str(draft.get("personalized_consent_text", "")).strip()
    summary_text = str(draft.get("key_information_summary", "")).strip()
    stored_markers = draft.get("citation_markers_used", [])
    if not isinstance(stored_markers, list):
        stored_markers = []
    stored_markers = [str(item) for item in stored_markers]
    summary_markers = draft.get("key_information_citation_markers_used", [])
    if not isinstance(summary_markers, list):
        summary_markers = []
    summary_markers = [str(item) for item in summary_markers]
    inline_markers = extract_citation_markers(text)
    summary_inline_markers = extract_citation_markers(summary_text)
    unsupported_markers = sorted(set(inline_markers) - set(available_markers))
    unsupported_summary_markers = sorted(set(summary_inline_markers) - set(available_markers))
    required_elements = evaluate_required_elements(text)
    readability = readability_metrics(text)
    summary_readability = readability_metrics(summary_text)
    citation_sentence_metrics = sentence_citation_metrics(text)
    summary_citation_sentence_metrics = sentence_citation_metrics(summary_text)
    draft_grade_threshold = target_grade_threshold(health_literacy, artifact_type="draft")
    summary_grade_threshold = target_grade_threshold(health_literacy, artifact_type="summary")
    repair_notes = draft.get("schema_repair_notes", [])
    if not isinstance(repair_notes, list):
        repair_notes = []
    revision_metadata = draft.get("revision_metadata", {})
    if not isinstance(revision_metadata, dict):
        revision_metadata = {}

    return {
        "artifact_present": True,
        "text_present": bool(text),
        "summary_present": bool(summary_text),
        "summary_identical_to_full_text": bool(text and summary_text and text == summary_text),
        "stored_citation_marker_count": len(stored_markers),
        "inline_citation_marker_count": len(inline_markers),
        "unsupported_inline_citation_marker_count": len(unsupported_markers),
        "summary_stored_citation_marker_count": len(summary_markers),
        "summary_inline_citation_marker_count": len(summary_inline_markers),
        "unsupported_summary_citation_marker_count": len(unsupported_summary_markers),
        "citation_marker_coverage_ratio": round(
            len(set(inline_markers)) / max(len(available_markers), 1),
            4,
        ) if available_markers else 0.0,
        "summary_citation_marker_coverage_ratio": round(
            len(set(summary_inline_markers)) / max(len(available_markers), 1),
            4,
        ) if available_markers else 0.0,
        "schema_repair_applied": bool(repair_notes),
        "required_element_coverage_ratio": round(
            sum(1 for value in required_elements.values() if value) / len(required_elements),
            4,
        ) if required_elements else 0.0,
        "target_health_literacy": health_literacy,
        "draft_grade_threshold": draft_grade_threshold,
        "draft_grade_target_met": readability["flesch_kincaid_grade"] <= draft_grade_threshold,
        "summary_grade_threshold": summary_grade_threshold,
        "summary_grade_target_met": summary_readability["flesch_kincaid_grade"] <= summary_grade_threshold,
        "revision_attempted": bool(revision_metadata.get("revision_attempted")),
        "revision_applied": bool(revision_metadata.get("revision_applied")),
        "revision_accepted": bool(revision_metadata.get("revision_accepted")),
        **required_elements,
        **readability,
        "summary_word_count": summary_readability["word_count"],
        "summary_sentence_count": summary_readability["sentence_count"],
        "summary_avg_words_per_sentence": summary_readability["avg_words_per_sentence"],
        "summary_flesch_reading_ease": summary_readability["flesch_reading_ease"],
        "summary_flesch_kincaid_grade": summary_readability["flesch_kincaid_grade"],
        "sentence_with_citation_count": citation_sentence_metrics["sentence_with_citation_count"],
        "sentence_without_citation_count": citation_sentence_metrics["sentence_without_citation_count"],
        "sentence_citation_coverage_ratio": citation_sentence_metrics["sentence_citation_coverage_ratio"],
        "summary_sentence_with_citation_count": summary_citation_sentence_metrics["sentence_with_citation_count"],
        "summary_sentence_without_citation_count": summary_citation_sentence_metrics["sentence_without_citation_count"],
        "summary_sentence_citation_coverage_ratio": summary_citation_sentence_metrics["sentence_citation_coverage_ratio"],
    }


def calculate_draft_quality_score(draft_summary: dict[str, Any]) -> float:
    required_element_coverage = float(draft_summary.get("required_element_coverage_ratio", 0.0))
    sentence_citation_coverage = float(draft_summary.get("sentence_citation_coverage_ratio", 0.0))
    summary_sentence_citation_coverage = float(draft_summary.get("summary_sentence_citation_coverage_ratio", 0.0))
    draft_grade_target_met = 1.0 if draft_summary.get("draft_grade_target_met") else 0.0
    summary_grade_target_met = 1.0 if draft_summary.get("summary_grade_target_met") else 0.0
    readability_penalty = 0.0
    if not draft_summary.get("draft_grade_target_met"):
        readability_penalty += 0.08
    if not draft_summary.get("summary_grade_target_met"):
        readability_penalty += 0.07

    score = (
        0.40 * required_element_coverage
        + 0.28 * sentence_citation_coverage
        + 0.12 * summary_sentence_citation_coverage
        + 0.10 * draft_grade_target_met
        + 0.10 * summary_grade_target_met
    )
    return round(max(score - readability_penalty, 0.0), 4)


def build_draft_revision_audit(
    draft_summary: dict[str, Any],
    *,
    draft_content_plan: dict[str, Any] | None = None,
    required_element_threshold: float = DRAFT_REQUIRED_ELEMENT_THRESHOLD,
    sentence_citation_threshold: float = DRAFT_SENTENCE_CITATION_THRESHOLD,
    summary_sentence_citation_threshold: float = SUMMARY_SENTENCE_CITATION_THRESHOLD,
) -> dict[str, Any]:
    missing_required_elements = [
        name
        for name in REQUIRED_ELEMENT_PATTERNS
        if not bool(draft_summary.get(name))
    ]
    issues: list[str] = []
    revision_targets: list[str] = []
    planned_required_elements: list[str] = []
    if isinstance(draft_content_plan, dict):
        plan_elements = draft_content_plan.get("elements")
        if isinstance(plan_elements, list):
            for item in plan_elements:
                if not isinstance(item, dict):
                    continue
                element_id = str(item.get("element_id", "")).strip()
                status = str(item.get("status", "")).strip()
                if element_id in CRITICAL_PLANNED_ELEMENT_IDS and status in {"supported", "partially_supported"}:
                    planned_required_elements.append(element_id)
    planned_required_elements = list(dict.fromkeys(planned_required_elements))
    missing_planned_required_elements = [
        element_id
        for element_id in planned_required_elements
        if not bool(draft_summary.get(element_id))
    ]

    if float(draft_summary.get("required_element_coverage_ratio", 0.0)) < required_element_threshold:
        issues.append("required_element_coverage_below_threshold")
        revision_targets.append(
            "Add the missing consent topics with short grounded statements that stay faithful to the source evidence."
        )
    if float(draft_summary.get("sentence_citation_coverage_ratio", 0.0)) < sentence_citation_threshold:
        issues.append("draft_sentence_citation_coverage_below_threshold")
        revision_targets.append(
            "Increase inline citations in the full draft, especially for study facts and participant-rights sentences."
        )
    if float(draft_summary.get("summary_sentence_citation_coverage_ratio", 0.0)) < summary_sentence_citation_threshold:
        issues.append("summary_sentence_citation_coverage_below_threshold")
        revision_targets.append(
            "Ensure the key-information summary contains short cited sentences rather than uncited general statements."
        )
    if not bool(draft_summary.get("draft_grade_target_met")):
        issues.append("draft_readability_target_not_met")
        revision_targets.append(
            "Use shorter sentences and simpler wording in the full draft while preserving grounded facts."
        )
    if not bool(draft_summary.get("summary_grade_target_met")):
        issues.append("summary_readability_target_not_met")
        revision_targets.append(
            "Simplify the key-information summary so it matches the participant literacy target."
        )
    if bool(draft_summary.get("summary_identical_to_full_text")):
        issues.append("summary_duplicates_full_draft")
        revision_targets.append(
            "Keep the summary concise and distinct from the fuller participant-facing draft."
        )
    if missing_planned_required_elements:
        issues.append("planned_required_elements_missing")
        revision_targets.append(
            "Do not omit planned study procedures, benefits, or alternatives when the content plan says they are supported."
        )

    return {
        "needs_revision": bool(issues),
        "issues": issues,
        "missing_required_elements": missing_required_elements,
        "planned_required_elements": planned_required_elements,
        "missing_planned_required_elements": missing_planned_required_elements,
        "revision_targets": revision_targets,
        "quality_score": calculate_draft_quality_score(draft_summary),
        "thresholds": {
            "required_element_coverage_ratio": required_element_threshold,
            "sentence_citation_coverage_ratio": sentence_citation_threshold,
            "summary_sentence_citation_coverage_ratio": summary_sentence_citation_threshold,
            "draft_grade_target_met": True,
            "summary_grade_target_met": True,
        },
        "metrics": draft_summary,
    }


def compare_draft_revision_candidates(
    initial_summary: dict[str, Any],
    revised_summary: dict[str, Any],
    *,
    initial_audit: dict[str, Any] | None = None,
    revised_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    initial_score = calculate_draft_quality_score(initial_summary)
    revised_score = calculate_draft_quality_score(revised_summary)
    improved_required_elements = float(revised_summary.get("required_element_coverage_ratio", 0.0)) - float(
        initial_summary.get("required_element_coverage_ratio", 0.0)
    )
    improved_sentence_citations = float(revised_summary.get("sentence_citation_coverage_ratio", 0.0)) - float(
        initial_summary.get("sentence_citation_coverage_ratio", 0.0)
    )
    improved_summary_citations = float(revised_summary.get("summary_sentence_citation_coverage_ratio", 0.0)) - float(
        initial_summary.get("summary_sentence_citation_coverage_ratio", 0.0)
    )
    readability_regressed = (
        float(revised_summary.get("flesch_kincaid_grade", 0.0)) - float(initial_summary.get("flesch_kincaid_grade", 0.0))
    )
    summary_readability_regressed = (
        float(revised_summary.get("summary_flesch_kincaid_grade", 0.0))
        - float(initial_summary.get("summary_flesch_kincaid_grade", 0.0))
    )

    accept_revision = revised_score > initial_score + 0.01 or (
        revised_summary.get("draft_grade_target_met")
        and revised_summary.get("summary_grade_target_met")
        and (
            improved_required_elements > 0.0
            or improved_sentence_citations > 0.05
            or improved_summary_citations > 0.05
        )
    )
    if not accept_revision and revised_score >= initial_score and improved_required_elements > 0.0:
        accept_revision = True
    if accept_revision and initial_summary.get("draft_grade_target_met") and not revised_summary.get("draft_grade_target_met"):
        accept_revision = False
    if accept_revision and initial_summary.get("summary_grade_target_met") and not revised_summary.get("summary_grade_target_met"):
        accept_revision = False
    if accept_revision and readability_regressed > 1.0 and improved_required_elements <= 0.0:
        accept_revision = False
    if accept_revision and summary_readability_regressed > 1.0 and improved_summary_citations <= 0.0:
        accept_revision = False
    if accept_revision and initial_audit and revised_audit:
        initial_missing_planned = len(initial_audit.get("missing_planned_required_elements", []))
        revised_missing_planned = len(revised_audit.get("missing_planned_required_elements", []))
        if revised_missing_planned > initial_missing_planned:
            accept_revision = False

    reasons: list[str] = []
    if accept_revision:
        reasons.append("revised_draft_quality_score_improved")
        if improved_required_elements > 0.0:
            reasons.append("required_element_coverage_improved")
        if improved_sentence_citations > 0.0:
            reasons.append("draft_sentence_citation_coverage_improved")
        if improved_summary_citations > 0.0:
            reasons.append("summary_sentence_citation_coverage_improved")
    else:
        reasons.append("initial_draft_retained")
        if revised_score < initial_score:
            reasons.append("revision_quality_score_regressed")
        elif revised_score == initial_score:
            reasons.append("revision_quality_score_did_not_change")
        else:
            reasons.append("revision_quality_gain_too_small")
        if initial_summary.get("draft_grade_target_met") and not revised_summary.get("draft_grade_target_met"):
            reasons.append("revision_lost_draft_readability_target")
        if initial_summary.get("summary_grade_target_met") and not revised_summary.get("summary_grade_target_met"):
            reasons.append("revision_lost_summary_readability_target")
        if readability_regressed > 1.0 and improved_required_elements <= 0.0:
            reasons.append("revision_draft_readability_regressed")
        if summary_readability_regressed > 1.0 and improved_summary_citations <= 0.0:
            reasons.append("revision_summary_readability_regressed")
        if initial_audit and revised_audit:
            initial_missing_planned = len(initial_audit.get("missing_planned_required_elements", []))
            revised_missing_planned = len(revised_audit.get("missing_planned_required_elements", []))
            if revised_missing_planned > initial_missing_planned:
                reasons.append("revision_lost_planned_required_elements")

    return {
        "accept_revision": accept_revision,
        "initial_quality_score": initial_score,
        "revised_quality_score": revised_score,
        "quality_score_delta": round(revised_score - initial_score, 4),
        "required_element_coverage_delta": round(improved_required_elements, 4),
        "draft_sentence_citation_coverage_delta": round(improved_sentence_citations, 4),
        "summary_sentence_citation_coverage_delta": round(improved_summary_citations, 4),
        "draft_readability_delta": round(readability_regressed, 4),
        "summary_readability_delta": round(summary_readability_regressed, 4),
        "reasons": reasons,
    }


def build_evaluation_records(
    run_id: str,
    *,
    metric_group: str,
    case_id: str,
    metrics: dict[str, Any],
) -> list[EvaluationRecord]:
    records: list[EvaluationRecord] = []
    for metric_name, metric_value in metrics.items():
        records.append(
            EvaluationRecord(
                run_id=run_id,
                case_id=case_id,
                metric_name=metric_name,
                metric_value=metric_value,
                metric_group=metric_group,
            )
        )
    return records


def evaluate_run_outputs(run_id: str, run_dir: Path) -> dict[str, Any]:
    outputs_dir = run_dir / "outputs"
    evaluation_records: list[EvaluationRecord] = []
    patient_profile = load_patient_profile_if_available(run_dir)
    health_literacy = str(patient_profile.get("health_literacy", "medium"))

    retrieval_hits = load_json(outputs_dir / "personalization_retrieval_hits.json") if (outputs_dir / "personalization_retrieval_hits.json").exists() else []
    available_markers = [f"[{idx}]" for idx in range(1, len(retrieval_hits) + 1)]

    summary: dict[str, Any] = {
        "run_id": run_id,
        "available_citation_markers": available_markers,
        "draft": {},
        "structured_record": {},
        "qa_answers": {},
    }

    draft_path = outputs_dir / "personalized_consent_draft.json"
    if draft_path.exists():
        draft = load_json(draft_path)
        draft_summary = summarize_personalized_draft(
            draft,
            available_markers=available_markers,
            health_literacy=health_literacy,
        )
        request_bundle_path = outputs_dir / "personalization_request_bundle.json"
        expected_study_specific_grounding = False
        has_study_specific_evidence = False
        if request_bundle_path.exists():
            request_bundle = load_json(request_bundle_path)
            source_group_filters = request_bundle.get("source_group_filters", [])
            if not isinstance(source_group_filters, list):
                source_group_filters = []
            source_group_filters = [str(item) for item in source_group_filters]
            source_id_filters = request_bundle.get("source_id_filters", [])
            if not isinstance(source_id_filters, list):
                source_id_filters = []
            source_id_filters = [str(item) for item in source_id_filters]
            draft_summary["workflow_variant"] = str(request_bundle.get("workflow_variant", "full_agentic")).strip() or "full_agentic"
            expected_study_specific_grounding = bool(source_id_filters) or ("trial_materials" in source_group_filters)
            draft_summary["expected_study_specific_grounding"] = expected_study_specific_grounding

        evidence_package_path = outputs_dir / "personalization_evidence_package.json"
        if evidence_package_path.exists():
            evidence_package = load_json(evidence_package_path)
            role_counts = evidence_package.get("role_counts", {})
            if not isinstance(role_counts, dict):
                role_counts = {}
            study_specific_hit_count = int(role_counts.get("study_specific", 0) or 0)
            regulatory_hit_count = int(role_counts.get("regulatory", 0) or 0)
            other_hit_count = int(role_counts.get("other", 0) or 0)
            total_role_hits = max(study_specific_hit_count + regulatory_hit_count + other_hit_count, 1)
            draft_summary["study_specific_hit_count"] = study_specific_hit_count
            draft_summary["regulatory_hit_count"] = regulatory_hit_count
            draft_summary["other_hit_count"] = other_hit_count
            has_study_specific_evidence = study_specific_hit_count > 0
            draft_summary["has_study_specific_evidence"] = has_study_specific_evidence
            draft_summary["study_specific_hit_ratio"] = round(study_specific_hit_count / total_role_hits, 4)
        draft_summary["expected_study_specific_grounding"] = expected_study_specific_grounding
        draft_summary["study_specific_grounding_met"] = (not expected_study_specific_grounding) or has_study_specific_evidence
        draft_summary["study_specific_grounding_gap"] = expected_study_specific_grounding and not has_study_specific_evidence
        summary["draft"] = draft_summary
        evaluation_records.extend(
            build_evaluation_records(
                run_id,
                metric_group="draft",
                case_id="personalized_consent_draft",
                metrics=draft_summary,
            )
        )
    else:
        summary["draft"] = {"artifact_present": False}

    structured_path = outputs_dir / "structured_consent_record.json"
    if structured_path.exists():
        structured = load_json(structured_path)
        required_fields = [
            "purposes",
            "data_types",
            "valid_until",
            "withdrawal_policy",
            "study_purpose_summary",
            "study_procedures_summary",
            "risks_summary",
            "benefits_summary",
            "alternatives_summary",
            "question_rights_summary",
            "voluntary_participation_statement",
            "withdrawal_rights_summary",
            "participant_rights",
            "consent_summary",
            "cited_markers",
            "metadata",
        ]
        present_fields = [field for field in required_fields if field in structured]
        missing_fields = [field for field in required_fields if field not in structured]
        metadata = structured.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        structured_summary = {
            "artifact_present": True,
            "required_field_presence_ratio": round(len(present_fields) / len(required_fields), 4),
            "missing_field_count": len(missing_fields),
            "purpose_count": len(structured.get("purposes", [])) if isinstance(structured.get("purposes"), list) else 0,
            "data_type_count": len(structured.get("data_types", [])) if isinstance(structured.get("data_types"), list) else 0,
            "participant_rights_count": len(structured.get("participant_rights", [])) if isinstance(structured.get("participant_rights"), list) else 0,
            "cited_marker_count": len(structured.get("cited_markers", [])) if isinstance(structured.get("cited_markers"), list) else 0,
            "withdrawal_policy_present": has_nonempty_text(structured.get("withdrawal_policy")),
            "study_purpose_summary_present": has_nonempty_text(structured.get("study_purpose_summary")),
            "study_procedures_summary_present": has_nonempty_text(structured.get("study_procedures_summary")),
            "risks_summary_present": has_nonempty_text(structured.get("risks_summary")),
            "benefits_summary_present": has_nonempty_text(structured.get("benefits_summary")),
            "alternatives_summary_present": has_nonempty_text(structured.get("alternatives_summary")),
            "question_rights_summary_present": has_nonempty_text(structured.get("question_rights_summary")),
            "voluntary_participation_statement_present": has_nonempty_text(structured.get("voluntary_participation_statement")),
            "withdrawal_rights_summary_present": has_nonempty_text(structured.get("withdrawal_rights_summary")),
            "consent_summary_present": has_nonempty_text(structured.get("consent_summary")),
            "schema_repair_applied": bool(metadata.get("schema_repair_notes")),
        }
        summary["structured_record"] = {
            **structured_summary,
            "missing_fields": missing_fields,
        }
        evaluation_records.extend(
            build_evaluation_records(
                run_id,
                metric_group="structured_record",
                case_id="structured_consent_record",
                metrics=structured_summary,
            )
        )
    else:
        summary["structured_record"] = {"artifact_present": False}

    qa_dir = outputs_dir / "qa"
    qa_index_path = qa_dir / "qa_index.jsonl"
    qa_index_rows = deduplicate_qa_index_rows(load_jsonl(qa_index_path))
    qa_answer_summaries: list[dict[str, Any]] = []
    if qa_index_rows:
        abstained_question_count = 0
        for row in qa_index_rows:
            answer_path_raw = row.get("answer_path")
            if not answer_path_raw:
                abstained_question_count += 1
                continue
            answer_path = Path(answer_path_raw)
            if not answer_path.exists():
                abstained_question_count += 1
                continue

            answer = load_json(answer_path)
            answer_text = str(answer.get("answer_text", "")).strip()
            inline_markers = extract_citation_markers(answer_text)
            stored_markers = answer.get("citation_markers_used", [])
            if not isinstance(stored_markers, list):
                stored_markers = []
            stored_markers = [str(item) for item in stored_markers]
            available_for_question = answer.get("available_citation_markers", [])
            if not isinstance(available_for_question, list):
                available_for_question = []
            available_for_question = [str(item) for item in available_for_question]
            unsupported_markers = answer.get("unsupported_citation_markers", [])
            if not isinstance(unsupported_markers, list):
                unsupported_markers = []
            unsupported_markers = [str(item) for item in unsupported_markers]
            repair_notes = answer.get("schema_repair_notes", [])
            if not isinstance(repair_notes, list):
                repair_notes = []

            answer_summary = {
                "question_id": row.get("question_id"),
                "question": row.get("question"),
                "artifact_present": True,
                "text_present": bool(answer_text),
                "retrieval_hit_count": len(load_json(Path(row["retrieval_hits_path"]))) if row.get("retrieval_hits_path") else 0,
                "stored_citation_marker_count": len(stored_markers),
                "inline_citation_marker_count": len(inline_markers),
                "unsupported_inline_citation_marker_count": len(unsupported_markers),
                "citation_marker_coverage_ratio": round(
                    len(set(inline_markers)) / max(len(available_for_question), 1),
                    4,
                ) if available_for_question else 0.0,
                "uncertainty_noted": bool(answer.get("uncertainty_noted")),
                "schema_repair_applied": bool(repair_notes),
                "target_health_literacy": health_literacy,
                **readability_metrics(answer_text),
                **sentence_citation_metrics(answer_text),
            }
            qa_grade_threshold = target_grade_threshold(health_literacy, artifact_type="qa")
            answer_summary["qa_grade_threshold"] = qa_grade_threshold
            answer_summary["qa_grade_target_met"] = answer_summary["flesch_kincaid_grade"] <= qa_grade_threshold
            qa_answer_summaries.append(answer_summary)
            evaluation_records.extend(
                build_evaluation_records(
                    run_id,
                    metric_group="qa_answer",
                    case_id=str(row.get("question_id")),
                    metrics={
                        key: value
                        for key, value in answer_summary.items()
                        if key not in {"question_id", "question"}
                    },
                )
            )

    if qa_answer_summaries:
        avg_flesch = sum(item["flesch_reading_ease"] for item in qa_answer_summaries) / len(qa_answer_summaries)
        avg_words = sum(item["word_count"] for item in qa_answer_summaries) / len(qa_answer_summaries)
        avg_citation_coverage = sum(item["citation_marker_coverage_ratio"] for item in qa_answer_summaries) / len(qa_answer_summaries)
        avg_sentence_citation_coverage = sum(item["sentence_citation_coverage_ratio"] for item in qa_answer_summaries) / len(qa_answer_summaries)
        qa_aggregate = {
            "artifact_present": True,
            "question_count": len(qa_index_rows),
            "answered_question_count": len(qa_answer_summaries),
            "abstained_question_count": abstained_question_count,
            "abstention_rate": round(abstained_question_count / max(len(qa_index_rows), 1), 4),
            "answers_with_uncertainty_count": sum(1 for item in qa_answer_summaries if item["uncertainty_noted"]),
            "uncertainty_rate": round(
                sum(1 for item in qa_answer_summaries if item["uncertainty_noted"]) / max(len(qa_answer_summaries), 1),
                4,
            ) if qa_answer_summaries else 0.0,
            "answers_with_schema_repair_count": sum(1 for item in qa_answer_summaries if item["schema_repair_applied"]),
            "answers_meeting_grade_target_count": sum(1 for item in qa_answer_summaries if item["qa_grade_target_met"]),
            "average_flesch_reading_ease": round(avg_flesch, 4),
            "average_word_count": round(avg_words, 4),
            "average_citation_marker_coverage_ratio": round(avg_citation_coverage, 4),
            "average_sentence_citation_coverage_ratio": round(avg_sentence_citation_coverage, 4),
            "per_question": qa_answer_summaries,
        }
        summary["qa_answers"] = qa_aggregate
        evaluation_records.extend(
            build_evaluation_records(
                run_id,
                metric_group="qa_answers_aggregate",
                case_id="qa_answers",
                metrics={
                    "artifact_present": True,
                    "question_count": qa_aggregate["question_count"],
                    "answered_question_count": qa_aggregate["answered_question_count"],
                    "abstained_question_count": qa_aggregate["abstained_question_count"],
                    "abstention_rate": qa_aggregate["abstention_rate"],
                    "answers_with_uncertainty_count": qa_aggregate["answers_with_uncertainty_count"],
                    "uncertainty_rate": qa_aggregate["uncertainty_rate"],
                    "answers_with_schema_repair_count": qa_aggregate["answers_with_schema_repair_count"],
                    "answers_meeting_grade_target_count": qa_aggregate["answers_meeting_grade_target_count"],
                    "average_flesch_reading_ease": qa_aggregate["average_flesch_reading_ease"],
                    "average_word_count": qa_aggregate["average_word_count"],
                    "average_citation_marker_coverage_ratio": qa_aggregate["average_citation_marker_coverage_ratio"],
                    "average_sentence_citation_coverage_ratio": qa_aggregate["average_sentence_citation_coverage_ratio"],
                },
            )
        )
    else:
        summary["qa_answers"] = {
            "artifact_present": bool(qa_index_rows),
            "question_count": len(qa_index_rows),
            "answered_question_count": 0,
            "abstained_question_count": len(qa_index_rows),
            "abstention_rate": round(len(qa_index_rows) / max(len(qa_index_rows), 1), 4) if qa_index_rows else 0.0,
            "uncertainty_rate": 0.0,
        }

    qualitative_bundle = {
        "run_id": run_id,
        "retrieval_hits": retrieval_hits,
        "personalized_consent_draft": load_json(draft_path) if draft_path.exists() else None,
        "structured_consent_record": load_json(structured_path) if structured_path.exists() else None,
        "qa_answers": [
            {
                "question_id": row.get("question_id"),
                "question": row.get("question"),
                "answer": load_json(Path(row["answer_path"])) if row.get("answer_path") and Path(row["answer_path"]).exists() else None,
                "retrieval_hits": load_json(Path(row["retrieval_hits_path"])) if row.get("retrieval_hits_path") and Path(row["retrieval_hits_path"]).exists() else [],
            }
            for row in qa_index_rows
        ],
    }

    return {
        "summary": summary,
        "records": [asdict(record) for record in evaluation_records],
        "qualitative_bundle": qualitative_bundle,
    }
