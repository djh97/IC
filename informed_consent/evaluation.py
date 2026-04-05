from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any
import json
import re

from .types import EvaluationRecord


WORD_PATTERN = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
CITATION_PATTERN = re.compile(r"\[\d+\]")
UNCERTAINTY_PATTERN = re.compile(
    r"\b(insufficient|uncertain|not enough|not specified|not provided|not available|cannot tell|can't tell|do not know|unknown|not supported)\b",
    re.IGNORECASE,
)
ASSERTIVE_QA_PATTERN = re.compile(
    r"\b(?:this study|the study|participants?|you)\s+"
    r"(?:tests?|includes?|involves?|requires?|will|would|must|can|may need to|may be asked to|have to|need to)\b",
    re.IGNORECASE,
)

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
        re.compile(r"\bparticipation is voluntary\b", re.IGNORECASE),
        re.compile(r"\byou do not have to (join|take part|participate)\b", re.IGNORECASE),
        re.compile(r"\byou can choose not to (join|take part|participate)\b", re.IGNORECASE),
        re.compile(r"\bit is up to you whether to (join|take part|participate)\b", re.IGNORECASE),
    ],
    "study_procedures": [
        re.compile(r"\bstudy procedures?\b", re.IGNORECASE),
        re.compile(r"\bstudy steps\b", re.IGNORECASE),
        re.compile(r"\bstudy visits?\b", re.IGNORECASE),
        re.compile(r"\bstudy visits? include\b", re.IGNORECASE),
        re.compile(r"\bthe (study|research) (involves|includes)\b", re.IGNORECASE),
        re.compile(r"\bduring the study\b", re.IGNORECASE),
        re.compile(r"\byou will (have|attend|receive|take|complete|provide|undergo|be asked to|need to)\b", re.IGNORECASE),
        re.compile(r"\byou will (walk|do|visit|return|come in|meet|use|wear)\b", re.IGNORECASE),
        re.compile(r"\bparticipants? will (have|attend|receive|take|complete|provide|undergo)\b", re.IGNORECASE),
        re.compile(
            r"\b(?:do|complete|take|undergo)\s+(?:a|an|the)?\s*"
            r"(?:6-?minute walk test|walk test|questionnaires?|blood (?:sample|samples|test)|"
            r"assessment|assessments|visits?|monitoring|follow-?up)\b",
            re.IGNORECASE,
        ),
        re.compile(r"\bregular (check-?ups|visits|monitoring|assessments)\b", re.IGNORECASE),
    ],
    "risks": [
        re.compile(r"\brisks?\b", re.IGNORECASE),
    ],
    "benefits": [
        re.compile(r"\bpossible benefits?\b", re.IGNORECASE),
        re.compile(r"\bpotential benefits?\b", re.IGNORECASE),
        re.compile(r"\bbenefits? of (the )?(study|research)\b", re.IGNORECASE),
        re.compile(r"\bhow (the )?study may help\b", re.IGNORECASE),
        re.compile(r"\bthere (may be|is) no direct (medical )?benefit\b", re.IGNORECASE),
        re.compile(r"\bthere (?:may be|is) no direct (medical )?benefit to you\b", re.IGNORECASE),
        re.compile(r"\bno direct (medical )?benefit\b", re.IGNORECASE),
        re.compile(r"\bno direct (medical )?benefit to you\b", re.IGNORECASE),
        re.compile(r"\bno guarantee of direct benefit\b", re.IGNORECASE),
        re.compile(r"\bdirect benefit is not guaranteed\b", re.IGNORECASE),
        re.compile(r"\bthere is no guarantee (?:that you will )?benefit directly\b", re.IGNORECASE),
        re.compile(r"\byou may not benefit directly\b", re.IGNORECASE),
        re.compile(r"\b(?:it|this study|the study) may not directly benefit you\b", re.IGNORECASE),
        re.compile(r"\b(?:it|this study|the study) might not directly benefit you\b", re.IGNORECASE),
        re.compile(r"\bthis (study|research) may help (future patients|others)\b", re.IGNORECASE),
        re.compile(r"\bmay help (future patients|others)\b", re.IGNORECASE),
        re.compile(r"\bpossible benefit\b", re.IGNORECASE),
    ],
    "alternatives": [
        re.compile(r"\balternatives?\b", re.IGNORECASE),
        re.compile(r"\bother options\b", re.IGNORECASE),
        re.compile(r"\bother treatment options\b", re.IGNORECASE),
        re.compile(r"\bstandard (treatment|care)\b", re.IGNORECASE),
        re.compile(r"\bother choices\b", re.IGNORECASE),
        re.compile(r"\balternatives? to (joining|taking part|participating)\b", re.IGNORECASE),
        re.compile(r"\binstead of (joining|taking part in|participating in) (this )?(study|research)\b", re.IGNORECASE),
        re.compile(r"\bif you (choose|prefer) not to join\b", re.IGNORECASE),
        re.compile(r"\bother options? (are|may be) available\b", re.IGNORECASE),
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
        re.compile(r"\byou (may|can) leave the study at any time\b", re.IGNORECASE),
        re.compile(r"\byou (may|can) stop participating at any time\b", re.IGNORECASE),
        re.compile(r"\byou (may|can) withdraw at any time\b", re.IGNORECASE),
        re.compile(r"\bleave the study at any time\b", re.IGNORECASE),
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


def sentence_support_diagnostics(
    text: str,
    *,
    unsupported_markers: list[str] | None = None,
    missing_required_evidence: bool = False,
) -> dict[str, float]:
    sentences = split_sentences(text)
    if not sentences:
        return {
            "sentence_count": 0.0,
            "citationless_sentence_count": 0.0,
            "citationless_sentence_rate": 0.0,
            "unsupported_sentence_count": 0.0,
        }

    unsupported_marker_set = set(unsupported_markers or [])
    citationless_sentence_count = 0
    unsupported_sentence_count = 0
    for sentence in sentences:
        sentence_markers = set(extract_citation_markers(sentence))
        sentence_has_unsupported_marker = bool(sentence_markers & unsupported_marker_set)
        if not sentence_markers:
            citationless_sentence_count += 1
        if sentence_has_unsupported_marker:
            unsupported_sentence_count += 1
            continue
        if not sentence_markers and missing_required_evidence and not UNCERTAINTY_PATTERN.search(sentence):
            unsupported_sentence_count += 1

    return {
        "sentence_count": float(len(sentences)),
        "citationless_sentence_count": float(citationless_sentence_count),
        "citationless_sentence_rate": round(citationless_sentence_count / len(sentences), 4),
        "unsupported_sentence_count": float(unsupported_sentence_count),
    }


def normalize_source_id_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    normalized: list[str] = []
    for item in value:
        source_id = str(item or "").strip().lower()
        if source_id:
            normalized.append(source_id)
    return list(dict.fromkeys(normalized))


def normalize_prompt_identifiers(bundle: dict[str, Any]) -> dict[str, Any]:
    prompt_identifiers = bundle.get("prompt_identifiers", {})
    if not isinstance(prompt_identifiers, dict):
        prompt_identifiers = {}
    system_prompt_id = str(
        prompt_identifiers.get("system_prompt_id")
        or bundle.get("system_prompt_id")
        or Path(str(bundle.get("system_prompt_path", "")).strip()).name
    ).strip()
    user_prompt_id = str(
        prompt_identifiers.get("user_prompt_id")
        or bundle.get("user_prompt_id")
        or Path(str(bundle.get("user_prompt_path", "")).strip()).name
    ).strip()
    return {
        "system_prompt_id": system_prompt_id or None,
        "user_prompt_id": user_prompt_id or None,
        "system_prompt_path": str(
            prompt_identifiers.get("system_prompt_path") or bundle.get("system_prompt_path") or ""
        ).strip()
        or None,
        "user_prompt_path": str(
            prompt_identifiers.get("user_prompt_path") or bundle.get("user_prompt_path") or ""
        ).strip()
        or None,
    }


def merge_prompt_identifier_sets(items: list[dict[str, Any]]) -> dict[str, list[str]]:
    merged = {
        "system_prompt_ids": [],
        "user_prompt_ids": [],
        "system_prompt_paths": [],
        "user_prompt_paths": [],
    }
    for item in items:
        if not isinstance(item, dict):
            continue
        for source_key, target_key in (
            ("system_prompt_id", "system_prompt_ids"),
            ("user_prompt_id", "user_prompt_ids"),
            ("system_prompt_path", "system_prompt_paths"),
            ("user_prompt_path", "user_prompt_paths"),
        ):
            value = str(item.get(source_key) or "").strip()
            if value and value not in merged[target_key]:
                merged[target_key].append(value)
    return merged


def merge_unique_string_values(values: list[Any]) -> list[str]:
    merged: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in merged:
            merged.append(text)
    return merged


def explicit_grounding_gap_declared(text: str, *, limitations: list[str] | None = None) -> bool:
    if isinstance(limitations, list) and any(str(item).strip() for item in limitations):
        return True
    return bool(UNCERTAINTY_PATTERN.search(text))


def should_flag_overconfident_answer(
    *,
    answer_text: str,
    uncertainty_noted: bool,
    unsupported_claim_risk: bool,
    study_specific_grounding_gap: bool,
    grounding_gap_declared: bool,
    unsupported_marker_count: int,
    unsupported_sentence_count: int,
) -> bool:
    if uncertainty_noted or not unsupported_claim_risk:
        return False
    if unsupported_marker_count > 0:
        return True
    if study_specific_grounding_gap and not grounding_gap_declared:
        return True
    return unsupported_sentence_count > 0 and bool(ASSERTIVE_QA_PATTERN.search(answer_text))


def compute_grounding_diagnostics(
    retrieval_hits: list[dict[str, Any]],
    *,
    selected_source_ids: list[str] | None = None,
    expected_study_specific_grounding: bool = False,
) -> dict[str, Any]:
    selected_ids = set(normalize_source_id_list(selected_source_ids or []))
    grounding_source_ids_used: list[str] = []
    foreign_source_ids_detected: list[str] = []
    selected_study_hit_count = 0
    foreign_study_hit_count = 0
    regulatory_hit_count = 0
    other_hit_count = 0

    for hit in retrieval_hits:
        if not isinstance(hit, dict):
            continue
        source_id = str(hit.get("source_id") or "").strip().lower()
        metadata = hit.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        source_group = str(metadata.get("source_group") or metadata.get("group_id") or "").strip().lower()
        if source_id and source_id not in grounding_source_ids_used:
            grounding_source_ids_used.append(source_id)
        if source_group == "regulatory_guidance":
            regulatory_hit_count += 1
            continue
        if source_group == "trial_materials":
            if selected_ids and source_id in selected_ids:
                selected_study_hit_count += 1
            elif selected_ids and source_id:
                foreign_study_hit_count += 1
                if source_id not in foreign_source_ids_detected:
                    foreign_source_ids_detected.append(source_id)
            else:
                other_hit_count += 1
            continue
        other_hit_count += 1

    total_hit_count = len(retrieval_hits)
    selected_study_hit_present = selected_study_hit_count > 0
    foreign_study_hit_present = foreign_study_hit_count > 0
    regulatory_hit_present = regulatory_hit_count > 0
    study_specific_grounding_met = (not expected_study_specific_grounding) or selected_study_hit_present
    study_specific_grounding_gap = expected_study_specific_grounding and not selected_study_hit_present
    regulatory_only_grounding = regulatory_hit_present and not selected_study_hit_present and not foreign_study_hit_present

    return {
        "selected_study_source_ids": sorted(selected_ids),
        "selected_study_hit_count": selected_study_hit_count,
        "selected_study_hit_present": selected_study_hit_present,
        "foreign_study_hit_count": foreign_study_hit_count,
        "foreign_study_hit_present": foreign_study_hit_present,
        "regulatory_hit_count": regulatory_hit_count,
        "regulatory_hit_present": regulatory_hit_present,
        "other_hit_count": other_hit_count,
        "total_hit_count": total_hit_count,
        "study_specific_hit_count": selected_study_hit_count + foreign_study_hit_count,
        "study_specific_hit_present": (selected_study_hit_count + foreign_study_hit_count) > 0,
        "study_specific_grounding_met": study_specific_grounding_met,
        "study_specific_grounding_gap": study_specific_grounding_gap,
        "regulatory_only_grounding": regulatory_only_grounding,
        "grounding_source_ids_used": grounding_source_ids_used,
        "foreign_source_ids_detected": foreign_source_ids_detected,
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
    missing_required_elements = [name for name, present in required_elements.items() if not present]
    readability = readability_metrics(text)
    summary_readability = readability_metrics(summary_text)
    citation_sentence_metrics = sentence_citation_metrics(text)
    summary_citation_sentence_metrics = sentence_citation_metrics(summary_text)
    support_diagnostics = sentence_support_diagnostics(text, unsupported_markers=unsupported_markers)
    summary_support_diagnostics = sentence_support_diagnostics(summary_text, unsupported_markers=unsupported_summary_markers)
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
        "unsupported_marker_count": len(unsupported_markers),
        "unsupported_inline_citation_marker_count": len(unsupported_markers),
        "summary_stored_citation_marker_count": len(summary_markers),
        "summary_inline_citation_marker_count": len(summary_inline_markers),
        "summary_unsupported_marker_count": len(unsupported_summary_markers),
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
        "missing_required_element_count": len(missing_required_elements),
        "missing_required_elements": missing_required_elements,
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
        "citationless_sentence_count": support_diagnostics["citationless_sentence_count"],
        "citationless_sentence_rate": support_diagnostics["citationless_sentence_rate"],
        "unsupported_sentence_count": support_diagnostics["unsupported_sentence_count"],
        "summary_sentence_with_citation_count": summary_citation_sentence_metrics["sentence_with_citation_count"],
        "summary_sentence_without_citation_count": summary_citation_sentence_metrics["sentence_without_citation_count"],
        "summary_sentence_citation_coverage_ratio": summary_citation_sentence_metrics["sentence_citation_coverage_ratio"],
        "summary_citationless_sentence_count": summary_support_diagnostics["citationless_sentence_count"],
        "summary_citationless_sentence_rate": summary_support_diagnostics["citationless_sentence_rate"],
        "summary_unsupported_sentence_count": summary_support_diagnostics["unsupported_sentence_count"],
        "grounding_gap_declared": explicit_grounding_gap_declared(
            f"{summary_text}\n{text}",
            limitations=draft.get("grounding_limitations") if isinstance(draft.get("grounding_limitations"), list) else [],
        ),
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
        if "study_procedures" in missing_planned_required_elements:
            revision_targets.append(
                "Add one explicit participant-action sentence describing what the person would do in the study when the study evidence supports it."
            )
        if "benefits" in missing_planned_required_elements:
            revision_targets.append(
                "Add one explicit benefits sentence when supported, or say clearly that direct benefit is not guaranteed; do not treat 'the team will explain risks and benefits' as enough."
            )
        if "alternatives" in missing_planned_required_elements:
            revision_targets.append(
                "Add one explicit sentence stating that other treatment options or other choices besides joining may be available when the regulatory evidence supports it."
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
    initial_missing_planned = len(initial_audit.get("missing_planned_required_elements", [])) if initial_audit else 0
    revised_missing_planned = len(revised_audit.get("missing_planned_required_elements", [])) if revised_audit else 0
    planned_completeness_improved = initial_missing_planned > revised_missing_planned
    completeness_improved = improved_required_elements > 0.0 or planned_completeness_improved

    accept_revision = revised_score > initial_score + 0.01 or (
        revised_summary.get("draft_grade_target_met")
        and revised_summary.get("summary_grade_target_met")
        and (
            improved_required_elements > 0.0
            or improved_sentence_citations > 0.05
            or improved_summary_citations > 0.05
        )
    )
    if not accept_revision and revised_score >= initial_score and completeness_improved:
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
        if revised_missing_planned > initial_missing_planned:
            accept_revision = False
        if initial_missing_planned > 0 and not completeness_improved:
            accept_revision = False

    reasons: list[str] = []
    if accept_revision:
        reasons.append("revised_draft_quality_score_improved")
        if improved_required_elements > 0.0:
            reasons.append("required_element_coverage_improved")
        if planned_completeness_improved:
            reasons.append("planned_required_elements_recovered")
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
            if revised_missing_planned > initial_missing_planned:
                reasons.append("revision_lost_planned_required_elements")
            elif initial_missing_planned > 0 and not completeness_improved:
                reasons.append("revision_did_not_improve_planned_completeness")

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
    metadata: dict[str, Any] | None = None,
) -> list[EvaluationRecord]:
    records: list[EvaluationRecord] = []
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, (dict, list, tuple, set)):
            continue
        records.append(
            EvaluationRecord(
                run_id=run_id,
                case_id=case_id,
                metric_name=metric_name,
                metric_value=metric_value,
                metric_group=metric_group,
                metadata=metadata or {},
            )
        )
    return records


def evaluate_run_outputs(run_id: str, run_dir: Path) -> dict[str, Any]:
    outputs_dir = run_dir / "outputs"
    evaluation_records: list[EvaluationRecord] = []
    patient_profile = load_patient_profile_if_available(run_dir)
    health_literacy = str(patient_profile.get("health_literacy", "medium"))
    manifest_payload = load_json(run_dir / "manifest.json") if (run_dir / "manifest.json").exists() else {}
    if not isinstance(manifest_payload, dict):
        manifest_payload = {}
    runtime_metadata = manifest_payload.get("runtime_metadata", {})
    if not isinstance(runtime_metadata, dict):
        runtime_metadata = {}
    context_metadata = manifest_payload.get("context_metadata", {})
    if not isinstance(context_metadata, dict):
        context_metadata = {}

    retrieval_hits = load_json(outputs_dir / "personalization_retrieval_hits.json") if (outputs_dir / "personalization_retrieval_hits.json").exists() else []
    if not isinstance(retrieval_hits, list):
        retrieval_hits = []
    available_markers = [f"[{idx}]" for idx in range(1, len(retrieval_hits) + 1)]
    draft_request_bundle = load_json(outputs_dir / "personalization_request_bundle.json") if (outputs_dir / "personalization_request_bundle.json").exists() else {}
    if not isinstance(draft_request_bundle, dict):
        draft_request_bundle = {}
    formalization_request_bundle = load_json(outputs_dir / "formalization_request_bundle.json") if (outputs_dir / "formalization_request_bundle.json").exists() else {}
    if not isinstance(formalization_request_bundle, dict):
        formalization_request_bundle = {}
    qa_dir = outputs_dir / "qa"
    qa_index_path = qa_dir / "qa_index.jsonl"
    qa_index_rows = deduplicate_qa_index_rows(load_jsonl(qa_index_path))
    qa_request_bundles: list[dict[str, Any]] = []
    for row in qa_index_rows:
        request_bundle_path = row.get("request_bundle_path")
        if not request_bundle_path:
            continue
        path = Path(str(request_bundle_path))
        if not path.exists():
            continue
        payload = load_json(path)
        if isinstance(payload, dict):
            qa_request_bundles.append(payload)
    qa_prompt_identifiers = merge_prompt_identifier_sets(
        [normalize_prompt_identifiers(payload) for payload in qa_request_bundles]
    )
    draft_prompt_identifiers = normalize_prompt_identifiers(draft_request_bundle) if draft_request_bundle else {}
    formalization_prompt_identifiers = normalize_prompt_identifiers(formalization_request_bundle) if formalization_request_bundle else {}
    qa_effective_filter_logics = merge_unique_string_values(
        [payload.get("filter_logic_used") for payload in qa_request_bundles]
    )
    qa_effective_strategies = merge_unique_string_values(
        [
            payload.get("retrieval_strategy_used")
            or ("no_retrieval" if str(payload.get("retrieval_mode_used") or "").strip().lower() == "none" else "single_pass")
            for payload in qa_request_bundles
        ]
    )
    draft_effective_filter_logic = str(draft_request_bundle.get("filter_logic_used") or "").strip() or None
    draft_effective_strategy = str(
        draft_request_bundle.get("retrieval_strategy_used")
        or ("no_retrieval" if str(draft_request_bundle.get("retrieval_mode_used") or "").strip().lower() == "none" else "single_pass")
    ).strip() if draft_request_bundle else None
    draft_effective_strategy = draft_effective_strategy or None
    summary_metadata = {
        "run_id": run_id,
        "study_id": str(manifest_payload.get("study_id") or "").strip() or None,
        "site_id": str(manifest_payload.get("site_id") or "").strip() or None,
        "study_source_id": str(context_metadata.get("study_source_id") or "").strip() or None,
        "workflow_variant": str(
            draft_request_bundle.get("workflow_variant")
            or context_metadata.get("workflow_variant")
            or ""
        ).strip() or None,
        "patient_profile_label": str(context_metadata.get("patient_profile_label") or "").strip() or None,
        "question_set_label": str(context_metadata.get("question_set_label") or "").strip() or None,
        "model_id": str(
            draft_request_bundle.get("model_id")
            or formalization_request_bundle.get("model_id")
            or runtime_metadata.get("model_id")
            or ""
        ).strip() or None,
        "embedding_model_id": str(
            draft_request_bundle.get("embedding_model_id")
            or formalization_request_bundle.get("embedding_model_id")
            or runtime_metadata.get("embedding_model_id")
            or ""
        ).strip() or None,
        "retrieval_mode": str(
            draft_request_bundle.get("retrieval_mode_used")
            or context_metadata.get("retrieval_mode")
            or runtime_metadata.get("retrieval_default_mode")
            or ""
        ).strip() or None,
        "retrieval_top_k": draft_request_bundle.get("top_k") or context_metadata.get("retrieval_top_k") or runtime_metadata.get("retrieval_default_top_k"),
        "retrieval_filter_logic_config": str(context_metadata.get("retrieval_filter_logic") or "").strip() or None,
        "retrieval_filter_logic": str(
            draft_request_bundle.get("filter_logic_used")
            or context_metadata.get("retrieval_filter_logic")
            or ""
        ).strip() or None,
        "draft_retrieval_filter_logic_effective": draft_effective_filter_logic,
        "qa_retrieval_filter_logic_effective": qa_effective_filter_logics,
        "draft_retrieval_strategy_effective": draft_effective_strategy,
        "qa_retrieval_strategy_effective": qa_effective_strategies,
        "config_path": str(context_metadata.get("config_path") or runtime_metadata.get("config_path") or "").strip() or None,
        "git_commit_hash": str(runtime_metadata.get("git_commit_hash") or "").strip() or None,
        "corpus_version": str(context_metadata.get("base_run_id") or runtime_metadata.get("corpus_version") or "").strip() or None,
        "index_version": str(runtime_metadata.get("index_version") or "").strip() or None,
        "base_run_id": str(context_metadata.get("base_run_id") or "").strip() or None,
        "batch_run_id": str(context_metadata.get("batch_run_id") or "").strip() or None,
        "batch_id": str(context_metadata.get("batch_id") or "").strip() or None,
        "reporting_role": str(context_metadata.get("reporting_role") or "").strip() or None,
        "random_seed": runtime_metadata.get("random_seed"),
        "draft_system_prompt_id": draft_prompt_identifiers.get("system_prompt_id"),
        "draft_user_prompt_id": draft_prompt_identifiers.get("user_prompt_id"),
        "formalization_system_prompt_id": formalization_prompt_identifiers.get("system_prompt_id"),
        "formalization_user_prompt_id": formalization_prompt_identifiers.get("user_prompt_id"),
        "qa_system_prompt_ids": qa_prompt_identifiers.get("system_prompt_ids", []),
        "qa_user_prompt_ids": qa_prompt_identifiers.get("user_prompt_ids", []),
        "prompt_identifiers": {
            "draft": draft_prompt_identifiers,
            "formalization": formalization_prompt_identifiers,
            "qa": qa_prompt_identifiers,
        },
    }

    summary: dict[str, Any] = {
        "run_id": run_id,
        "available_citation_markers": available_markers,
        "metadata": summary_metadata,
        "draft": {},
        "structured_record": {},
        "qa_answers": {},
        "failure_taxonomy": {},
    }

    draft_path = outputs_dir / "personalized_consent_draft.json"
    if draft_path.exists():
        draft = load_json(draft_path)
        draft_summary = summarize_personalized_draft(
            draft,
            available_markers=available_markers,
            health_literacy=health_literacy,
        )
        source_group_filters = draft_request_bundle.get("source_group_filters", [])
        if not isinstance(source_group_filters, list):
            source_group_filters = []
        source_group_filters = [str(item).strip().lower() for item in source_group_filters if str(item).strip()]
        source_id_filters = normalize_source_id_list(draft_request_bundle.get("source_id_filters", []))
        expected_study_specific_grounding = bool(source_id_filters) or ("trial_materials" in source_group_filters)
        draft_grounding = compute_grounding_diagnostics(
            retrieval_hits,
            selected_source_ids=source_id_filters,
            expected_study_specific_grounding=expected_study_specific_grounding,
        )
        draft_support_diagnostics = sentence_support_diagnostics(
            str(draft.get("personalized_consent_text", "")).strip(),
            unsupported_markers=[
                marker
                for marker in extract_citation_markers(str(draft.get("personalized_consent_text", "")).strip())
                if marker not in set(available_markers)
            ],
            missing_required_evidence=draft_grounding["study_specific_grounding_gap"] and not draft_summary.get("grounding_gap_declared"),
        )
        draft_summary.update(draft_grounding)
        draft_summary["workflow_variant"] = summary_metadata.get("workflow_variant") or "full_agentic"
        draft_summary["expected_study_specific_grounding"] = expected_study_specific_grounding
        draft_summary["has_study_specific_evidence"] = draft_grounding["selected_study_hit_present"]
        draft_summary["study_specific_hit_ratio"] = round(
            draft_grounding["study_specific_hit_count"] / max(draft_grounding["total_hit_count"], 1),
            4,
        ) if draft_grounding["total_hit_count"] else 0.0
        draft_summary["grounding_source_ids_used_count"] = len(draft_grounding["grounding_source_ids_used"])
        draft_summary["citationless_sentence_count"] = draft_support_diagnostics["citationless_sentence_count"]
        draft_summary["citationless_sentence_rate"] = draft_support_diagnostics["citationless_sentence_rate"]
        draft_summary["unsupported_sentence_count"] = draft_support_diagnostics["unsupported_sentence_count"]
        draft_summary["unsupported_claim_risk"] = bool(
            draft_summary.get("unsupported_marker_count", 0)
            or draft_summary.get("unsupported_sentence_count", 0)
            or (draft_grounding["study_specific_grounding_gap"] and not draft_summary.get("grounding_gap_declared"))
        )
        draft_summary["failure_flags"] = {
            "missing_selected_study_grounding": bool(draft_grounding["study_specific_grounding_gap"]),
            "foreign_study_contamination": bool(draft_grounding["foreign_study_hit_present"]),
            "regulatory_only_grounding": bool(draft_grounding["regulatory_only_grounding"]),
            "unsupported_claim_risk": bool(draft_summary["unsupported_claim_risk"]),
            "omitted_required_element": bool(draft_summary.get("missing_required_element_count", 0)),
            "overconfident_answer": False,
            "malformed_structured_output": False,
            "grounding_gap_declared": bool(draft_summary.get("grounding_gap_declared")),
        }
        summary["draft"] = draft_summary
        evaluation_records.extend(
            build_evaluation_records(
                run_id,
                metric_group="draft",
                case_id="personalized_consent_draft",
                metrics=draft_summary,
                metadata=summary_metadata,
            )
        )
    else:
        summary["draft"] = {
            "artifact_present": False,
            "failure_flags": {
                "missing_selected_study_grounding": False,
                "foreign_study_contamination": False,
                "regulatory_only_grounding": False,
                "unsupported_claim_risk": False,
                "omitted_required_element": False,
                "overconfident_answer": False,
                "malformed_structured_output": False,
                "grounding_gap_declared": False,
            },
        }

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
        structured_summary["malformed_structured_output"] = bool(
            structured_summary["schema_repair_applied"] or structured_summary["missing_field_count"] > 0
        )
        structured_summary["failure_flags"] = {
            "missing_selected_study_grounding": False,
            "foreign_study_contamination": False,
            "regulatory_only_grounding": False,
            "unsupported_claim_risk": False,
            "omitted_required_element": False,
            "overconfident_answer": False,
            "malformed_structured_output": bool(structured_summary["malformed_structured_output"]),
            "grounding_gap_declared": False,
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
                metadata=summary_metadata,
            )
        )
    else:
        summary["structured_record"] = {
            "artifact_present": False,
            "failure_flags": {
                "missing_selected_study_grounding": False,
                "foreign_study_contamination": False,
                "regulatory_only_grounding": False,
                "unsupported_claim_risk": False,
                "omitted_required_element": False,
                "overconfident_answer": False,
                "malformed_structured_output": False,
                "grounding_gap_declared": False,
            },
        }

    qa_answer_summaries: list[dict[str, Any]] = []
    if qa_index_rows:
        abstained_question_count = 0
        clarified_question_count = 0
        for row in qa_index_rows:
            question_id = str(row.get("question_id", "")).strip()
            question = str(row.get("question", "")).strip()
            qa_request_bundle: dict[str, Any] = {}
            request_path_raw = row.get("request_bundle_path")
            if request_path_raw:
                request_path = Path(str(request_path_raw))
                if request_path.exists():
                    payload = load_json(request_path)
                    if isinstance(payload, dict):
                        qa_request_bundle = payload
            qa_source_group_filters = qa_request_bundle.get("source_group_filters", [])
            if not isinstance(qa_source_group_filters, list):
                qa_source_group_filters = []
            qa_source_group_filters = [str(item).strip().lower() for item in qa_source_group_filters if str(item).strip()]
            qa_source_id_filters = normalize_source_id_list(qa_request_bundle.get("source_id_filters", []))
            qa_expected_study_specific_grounding = bool(qa_source_id_filters) or ("trial_materials" in qa_source_group_filters)
            question_retrieval_hits: list[dict[str, Any]] = []
            retrieval_hits_path_raw = row.get("retrieval_hits_path")
            if retrieval_hits_path_raw:
                retrieval_hits_path = Path(str(retrieval_hits_path_raw))
                if retrieval_hits_path.exists():
                    payload = load_json(retrieval_hits_path)
                    if isinstance(payload, list):
                        question_retrieval_hits = payload
            qa_grounding = compute_grounding_diagnostics(
                question_retrieval_hits,
                selected_source_ids=qa_source_id_filters,
                expected_study_specific_grounding=qa_expected_study_specific_grounding,
            )
            answer_path_raw = row.get("answer_path")
            clarification_path_raw = row.get("clarification_path")
            clarification_present = False
            if clarification_path_raw:
                clarification_path = Path(str(clarification_path_raw))
                clarification_present = clarification_path.exists()
            if not answer_path_raw:
                abstained_question_count += 1
                if clarification_present:
                    clarified_question_count += 1
                qa_answer_summaries.append(
                    {
                        "question_id": question_id,
                        "question": question,
                        "artifact_present": clarification_present,
                        "text_present": False,
                        "status": "clarified" if clarification_present else "abstained",
                        "answered": False,
                        "abstained": True,
                        "clarified": clarification_present,
                        "retrieval_hit_count": len(question_retrieval_hits),
                        "stored_citation_marker_count": 0,
                        "inline_citation_marker_count": 0,
                        "unsupported_marker_count": 0,
                        "unsupported_inline_citation_marker_count": 0,
                        "citation_marker_coverage_ratio": 0.0,
                        "uncertainty_noted": clarification_present,
                        "schema_repair_applied": False,
                        "target_health_literacy": health_literacy,
                        "word_count": 0.0,
                        "sentence_count": 0.0,
                        "avg_words_per_sentence": 0.0,
                        "flesch_reading_ease": 0.0,
                        "flesch_kincaid_grade": 0.0,
                        "sentence_with_citation_count": 0.0,
                        "sentence_without_citation_count": 0.0,
                        "sentence_citation_coverage_ratio": 0.0,
                        "citationless_sentence_count": 0.0,
                        "citationless_sentence_rate": 0.0,
                        "unsupported_sentence_count": 0.0,
                        "qa_grade_threshold": target_grade_threshold(health_literacy, artifact_type="qa"),
                        "qa_grade_target_met": False,
                        "expected_study_specific_grounding": qa_expected_study_specific_grounding,
                        **qa_grounding,
                        "grounding_gap_declared": clarification_present,
                        "unsupported_claim_risk": False,
                        "failure_flags": {
                            "missing_selected_study_grounding": bool(qa_grounding["study_specific_grounding_gap"]),
                            "foreign_study_contamination": bool(qa_grounding["foreign_study_hit_present"]),
                            "regulatory_only_grounding": bool(qa_grounding["regulatory_only_grounding"]),
                            "unsupported_claim_risk": False,
                            "omitted_required_element": False,
                            "overconfident_answer": False,
                            "malformed_structured_output": False,
                            "grounding_gap_declared": clarification_present,
                        },
                    }
                )
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
            grounding_gap_declared = explicit_grounding_gap_declared(
                answer_text,
                limitations=answer.get("grounding_limitations") if isinstance(answer.get("grounding_limitations"), list) else [],
            )
            support_diagnostics = sentence_support_diagnostics(
                answer_text,
                unsupported_markers=unsupported_markers,
                missing_required_evidence=qa_grounding["study_specific_grounding_gap"] and not grounding_gap_declared,
            )

            answer_summary = {
                "question_id": question_id,
                "question": question,
                "artifact_present": True,
                "text_present": bool(answer_text),
                "status": "answered",
                "answered": True,
                "abstained": False,
                "clarified": False,
                "retrieval_hit_count": len(question_retrieval_hits),
                "stored_citation_marker_count": len(stored_markers),
                "inline_citation_marker_count": len(inline_markers),
                "unsupported_marker_count": len(unsupported_markers),
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
                "citationless_sentence_count": support_diagnostics["citationless_sentence_count"],
                "citationless_sentence_rate": support_diagnostics["citationless_sentence_rate"],
                "unsupported_sentence_count": support_diagnostics["unsupported_sentence_count"],
                "expected_study_specific_grounding": qa_expected_study_specific_grounding,
                **qa_grounding,
                "grounding_gap_declared": grounding_gap_declared,
            }
            qa_grade_threshold = target_grade_threshold(health_literacy, artifact_type="qa")
            answer_summary["qa_grade_threshold"] = qa_grade_threshold
            answer_summary["qa_grade_target_met"] = answer_summary["flesch_kincaid_grade"] <= qa_grade_threshold
            answer_summary["unsupported_claim_risk"] = bool(
                answer_summary["unsupported_marker_count"]
                or answer_summary["unsupported_sentence_count"]
                or (qa_grounding["study_specific_grounding_gap"] and not grounding_gap_declared)
            )
            answer_summary["failure_flags"] = {
                "missing_selected_study_grounding": bool(qa_grounding["study_specific_grounding_gap"]),
                "foreign_study_contamination": bool(qa_grounding["foreign_study_hit_present"]),
                "regulatory_only_grounding": bool(qa_grounding["regulatory_only_grounding"]),
                "unsupported_claim_risk": bool(answer_summary["unsupported_claim_risk"]),
                "omitted_required_element": False,
                "overconfident_answer": should_flag_overconfident_answer(
                    answer_text=answer_text,
                    uncertainty_noted=bool(answer_summary["uncertainty_noted"]),
                    unsupported_claim_risk=bool(answer_summary["unsupported_claim_risk"]),
                    study_specific_grounding_gap=bool(qa_grounding["study_specific_grounding_gap"]),
                    grounding_gap_declared=bool(grounding_gap_declared),
                    unsupported_marker_count=int(answer_summary["unsupported_marker_count"]),
                    unsupported_sentence_count=int(answer_summary["unsupported_sentence_count"]),
                ),
                "malformed_structured_output": False,
                "grounding_gap_declared": bool(grounding_gap_declared or answer_summary["uncertainty_noted"]),
            }
            qa_answer_summaries.append(answer_summary)
            evaluation_records.extend(
                build_evaluation_records(
                    run_id,
                    metric_group="qa_answer",
                    case_id=question_id,
                    metrics={
                        key: value
                        for key, value in answer_summary.items()
                        if key not in {"question_id", "question", "failure_flags"}
                    },
                    metadata=summary_metadata,
                )
            )

    if qa_answer_summaries:
        answered_items = [item for item in qa_answer_summaries if item.get("answered")]
        avg_flesch = sum(item["flesch_reading_ease"] for item in answered_items) / len(answered_items) if answered_items else 0.0
        avg_words = sum(item["word_count"] for item in answered_items) / len(answered_items) if answered_items else 0.0
        avg_citation_coverage = sum(item["citation_marker_coverage_ratio"] for item in answered_items) / len(answered_items) if answered_items else 0.0
        avg_sentence_citation_coverage = sum(item["sentence_citation_coverage_ratio"] for item in answered_items) / len(answered_items) if answered_items else 0.0
        avg_fkg = sum(item["flesch_kincaid_grade"] for item in answered_items) / len(answered_items) if answered_items else 0.0
        total_answer_sentence_count = sum(float(item.get("sentence_count", 0.0)) for item in answered_items)
        grounding_source_ids_used = sorted(
            {
                source_id
                for item in qa_answer_summaries
                for source_id in item.get("grounding_source_ids_used", [])
            }
        )
        foreign_source_ids_detected = sorted(
            {
                source_id
                for item in qa_answer_summaries
                for source_id in item.get("foreign_source_ids_detected", [])
            }
        )
        qa_failure_flags = {
            "missing_selected_study_grounding": any(item.get("failure_flags", {}).get("missing_selected_study_grounding") for item in qa_answer_summaries),
            "foreign_study_contamination": any(item.get("failure_flags", {}).get("foreign_study_contamination") for item in qa_answer_summaries),
            "regulatory_only_grounding": any(item.get("failure_flags", {}).get("regulatory_only_grounding") for item in qa_answer_summaries),
            "unsupported_claim_risk": any(item.get("failure_flags", {}).get("unsupported_claim_risk") for item in qa_answer_summaries),
            "omitted_required_element": False,
            "overconfident_answer": any(item.get("failure_flags", {}).get("overconfident_answer") for item in qa_answer_summaries),
            "malformed_structured_output": False,
            "grounding_gap_declared": any(item.get("failure_flags", {}).get("grounding_gap_declared") for item in qa_answer_summaries),
        }
        qa_aggregate = {
            "artifact_present": True,
            "question_count": len(qa_index_rows),
            "answered_question_count": len(answered_items),
            "answered_count": len(answered_items),
            "abstained_question_count": abstained_question_count,
            "abstained_count": abstained_question_count,
            "clarified_count": clarified_question_count,
            "abstention_rate": round(abstained_question_count / max(len(qa_index_rows), 1), 4),
            "answers_with_uncertainty_count": sum(1 for item in answered_items if item["uncertainty_noted"]),
            "uncertainty_flag_count": sum(1 for item in answered_items if item["uncertainty_noted"]),
            "uncertainty_rate": round(
                sum(1 for item in answered_items if item["uncertainty_noted"]) / max(len(answered_items), 1),
                4,
            ) if answered_items else 0.0,
            "answers_with_schema_repair_count": sum(1 for item in answered_items if item["schema_repair_applied"]),
            "answers_meeting_grade_target_count": sum(1 for item in answered_items if item["qa_grade_target_met"]),
            "average_flesch_reading_ease": round(avg_flesch, 4),
            "average_flesch_kincaid_grade": round(avg_fkg, 4),
            "average_word_count": round(avg_words, 4),
            "average_citation_marker_coverage_ratio": round(avg_citation_coverage, 4),
            "average_sentence_citation_coverage_ratio": round(avg_sentence_citation_coverage, 4),
            "unsupported_marker_count": sum(int(item.get("unsupported_marker_count", 0)) for item in answered_items),
            "unsupported_sentence_count": sum(int(item.get("unsupported_sentence_count", 0)) for item in answered_items),
            "citationless_sentence_count": sum(int(item.get("citationless_sentence_count", 0)) for item in answered_items),
            "citationless_sentence_rate": round(
                sum(float(item.get("citationless_sentence_count", 0.0)) for item in answered_items) / max(total_answer_sentence_count, 1),
                4,
            ) if answered_items else 0.0,
            "selected_study_hit_count": sum(int(item.get("selected_study_hit_count", 0)) for item in qa_answer_summaries),
            "selected_study_hit_present": any(bool(item.get("selected_study_hit_present")) for item in qa_answer_summaries),
            "foreign_study_hit_count": sum(int(item.get("foreign_study_hit_count", 0)) for item in qa_answer_summaries),
            "foreign_study_hit_present": any(bool(item.get("foreign_study_hit_present")) for item in qa_answer_summaries),
            "regulatory_hit_count": sum(int(item.get("regulatory_hit_count", 0)) for item in qa_answer_summaries),
            "total_hit_count": sum(int(item.get("total_hit_count", 0)) for item in qa_answer_summaries),
            "study_specific_grounding_met": not any(
                item.get("expected_study_specific_grounding") and not item.get("selected_study_hit_present")
                for item in qa_answer_summaries
            ),
            "study_specific_grounding_gap": any(
                item.get("expected_study_specific_grounding") and not item.get("selected_study_hit_present")
                for item in qa_answer_summaries
            ),
            "grounding_source_ids_used": grounding_source_ids_used,
            "foreign_source_ids_detected": foreign_source_ids_detected,
            "failure_flags": qa_failure_flags,
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
                    "answered_count": qa_aggregate["answered_count"],
                    "abstained_question_count": qa_aggregate["abstained_question_count"],
                    "abstained_count": qa_aggregate["abstained_count"],
                    "clarified_count": qa_aggregate["clarified_count"],
                    "abstention_rate": qa_aggregate["abstention_rate"],
                    "answers_with_uncertainty_count": qa_aggregate["answers_with_uncertainty_count"],
                    "uncertainty_flag_count": qa_aggregate["uncertainty_flag_count"],
                    "uncertainty_rate": qa_aggregate["uncertainty_rate"],
                    "answers_with_schema_repair_count": qa_aggregate["answers_with_schema_repair_count"],
                    "answers_meeting_grade_target_count": qa_aggregate["answers_meeting_grade_target_count"],
                    "average_flesch_reading_ease": qa_aggregate["average_flesch_reading_ease"],
                    "average_flesch_kincaid_grade": qa_aggregate["average_flesch_kincaid_grade"],
                    "average_word_count": qa_aggregate["average_word_count"],
                    "average_citation_marker_coverage_ratio": qa_aggregate["average_citation_marker_coverage_ratio"],
                    "average_sentence_citation_coverage_ratio": qa_aggregate["average_sentence_citation_coverage_ratio"],
                    "unsupported_marker_count": qa_aggregate["unsupported_marker_count"],
                    "unsupported_sentence_count": qa_aggregate["unsupported_sentence_count"],
                    "citationless_sentence_count": qa_aggregate["citationless_sentence_count"],
                    "citationless_sentence_rate": qa_aggregate["citationless_sentence_rate"],
                    "selected_study_hit_count": qa_aggregate["selected_study_hit_count"],
                    "selected_study_hit_present": qa_aggregate["selected_study_hit_present"],
                    "foreign_study_hit_count": qa_aggregate["foreign_study_hit_count"],
                    "foreign_study_hit_present": qa_aggregate["foreign_study_hit_present"],
                    "regulatory_hit_count": qa_aggregate["regulatory_hit_count"],
                    "total_hit_count": qa_aggregate["total_hit_count"],
                    "study_specific_grounding_met": qa_aggregate["study_specific_grounding_met"],
                    "study_specific_grounding_gap": qa_aggregate["study_specific_grounding_gap"],
                },
                metadata=summary_metadata,
            )
        )
    else:
        summary["qa_answers"] = {
            "artifact_present": bool(qa_index_rows),
            "question_count": len(qa_index_rows),
            "answered_question_count": 0,
            "answered_count": 0,
            "abstained_question_count": len(qa_index_rows),
            "abstained_count": len(qa_index_rows),
            "clarified_count": 0,
            "abstention_rate": round(len(qa_index_rows) / max(len(qa_index_rows), 1), 4) if qa_index_rows else 0.0,
            "uncertainty_flag_count": 0,
            "uncertainty_rate": 0.0,
            "unsupported_marker_count": 0,
            "unsupported_sentence_count": 0,
            "citationless_sentence_count": 0,
            "citationless_sentence_rate": 0.0,
            "selected_study_hit_count": 0,
            "selected_study_hit_present": False,
            "foreign_study_hit_count": 0,
            "foreign_study_hit_present": False,
            "regulatory_hit_count": 0,
            "total_hit_count": 0,
            "study_specific_grounding_met": True,
            "study_specific_grounding_gap": False,
            "grounding_source_ids_used": [],
            "foreign_source_ids_detected": [],
            "failure_flags": {
                "missing_selected_study_grounding": False,
                "foreign_study_contamination": False,
                "regulatory_only_grounding": False,
                "unsupported_claim_risk": False,
                "omitted_required_element": False,
                "overconfident_answer": False,
                "malformed_structured_output": False,
                "grounding_gap_declared": False,
            },
        }

    case_failure_flags = {
        "missing_selected_study_grounding": bool(summary["draft"].get("failure_flags", {}).get("missing_selected_study_grounding"))
        or bool(summary["qa_answers"].get("failure_flags", {}).get("missing_selected_study_grounding")),
        "foreign_study_contamination": bool(summary["draft"].get("failure_flags", {}).get("foreign_study_contamination"))
        or bool(summary["qa_answers"].get("failure_flags", {}).get("foreign_study_contamination")),
        "regulatory_only_grounding": bool(summary["draft"].get("failure_flags", {}).get("regulatory_only_grounding"))
        or bool(summary["qa_answers"].get("failure_flags", {}).get("regulatory_only_grounding")),
        "unsupported_claim_risk": bool(summary["draft"].get("failure_flags", {}).get("unsupported_claim_risk"))
        or bool(summary["qa_answers"].get("failure_flags", {}).get("unsupported_claim_risk")),
        "omitted_required_element": bool(summary["draft"].get("failure_flags", {}).get("omitted_required_element")),
        "overconfident_answer": bool(summary["qa_answers"].get("failure_flags", {}).get("overconfident_answer")),
        "malformed_structured_output": bool(summary["structured_record"].get("failure_flags", {}).get("malformed_structured_output")),
        "grounding_gap_declared": bool(summary["draft"].get("failure_flags", {}).get("grounding_gap_declared"))
        or bool(summary["qa_answers"].get("failure_flags", {}).get("grounding_gap_declared")),
    }
    summary["failure_taxonomy"] = {
        "draft": summary["draft"].get("failure_flags", {}),
        "structured_record": summary["structured_record"].get("failure_flags", {}),
        "qa_answers": summary["qa_answers"].get("failure_flags", {}),
        "case_failure_flags": case_failure_flags,
    }

    qualitative_bundle = {
        "run_id": run_id,
        "metadata": summary_metadata,
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
