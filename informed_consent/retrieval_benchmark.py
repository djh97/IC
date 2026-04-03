from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any
import json


def load_benchmark_spec(spec_path: Path) -> dict[str, Any]:
    payload = json.loads(spec_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Benchmark spec must be a JSON object.")
    return payload


def normalize_modes(value: Any) -> list[str]:
    if isinstance(value, str):
        candidates = [value]
    elif isinstance(value, list):
        candidates = value
    else:
        candidates = []

    normalized: list[str] = []
    for item in candidates:
        mode = str(item).strip().lower()
        if mode in {"lexical", "dense", "hybrid"} and mode not in normalized:
            normalized.append(mode)
    return normalized


def normalize_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    normalized: list[str] = []
    for item in value:
        candidate = str(item).strip()
        if candidate and candidate not in normalized:
            normalized.append(candidate)
    return normalized


def find_first_match_rank(values: list[str], expected_values: list[str]) -> int | None:
    if not expected_values:
        return None
    expected = set(expected_values)
    for index, value in enumerate(values, start=1):
        if value in expected:
            return index
    return None


def hit_at_k(values: list[str], expected_values: list[str], *, k: int) -> bool | None:
    if not expected_values:
        return None
    expected = set(expected_values)
    return any(value in expected for value in values[:k])


def reciprocal_rank(rank: int | None) -> float | None:
    if rank is None or rank <= 0:
        return None
    return round(1.0 / rank, 4)


def to_bool_average(values: list[bool | None]) -> float | None:
    usable = [value for value in values if isinstance(value, bool)]
    if not usable:
        return None
    return round(sum(1 for value in usable if value) / len(usable), 4)


def to_numeric_average(values: list[float | int | None]) -> float | None:
    usable = [float(value) for value in values if isinstance(value, (int, float))]
    if not usable:
        return None
    return round(sum(usable) / len(usable), 4)


def score_retrieval_case(
    *,
    benchmark_id: str,
    query_id: str,
    query: str,
    retrieval_mode: str,
    hits: list[dict[str, Any]],
    expected_source_ids: list[str] | None = None,
    expected_source_groups: list[str] | None = None,
    top_k: int,
    source_group_filters: list[str] | None = None,
    source_id_filters: list[str] | None = None,
    notes: str = "",
) -> dict[str, Any]:
    expected_source_ids = normalize_string_list(expected_source_ids)
    expected_source_groups = normalize_string_list(expected_source_groups)
    source_group_filters = normalize_string_list(source_group_filters)
    source_id_filters = normalize_string_list(source_id_filters)

    hit_source_ids = [str(hit.get("source_id", "")).strip() for hit in hits]
    hit_source_groups = [
        str((hit.get("metadata") or {}).get("source_group", "")).strip()
        for hit in hits
    ]

    first_source_id_rank = find_first_match_rank(hit_source_ids, expected_source_ids)
    first_source_group_rank = find_first_match_rank(hit_source_groups, expected_source_groups)

    top_hit = hits[0] if hits else {}
    top_hit_metadata = top_hit.get("metadata") or {}
    result = {
        "benchmark_id": benchmark_id,
        "query_id": query_id,
        "query": query,
        "retrieval_mode": retrieval_mode,
        "top_k": top_k,
        "returned_hit_count": len(hits),
        "expected_source_id_count": len(expected_source_ids),
        "expected_source_group_count": len(expected_source_groups),
        "source_group_filters": " | ".join(source_group_filters),
        "source_id_filters": " | ".join(source_id_filters),
        "top_hit_source_id": str(top_hit.get("source_id", "")).strip(),
        "top_hit_source_group": str(top_hit_metadata.get("source_group", "")).strip(),
        "top_hit_citation_label": str(top_hit.get("citation_label", "")).strip(),
        "top_hit_score": top_hit.get("score"),
        "unique_source_count_in_hits": len({value for value in hit_source_ids if value}),
        "unique_source_group_count_in_hits": len({value for value in hit_source_groups if value}),
        "source_id_first_relevant_rank": first_source_id_rank,
        "source_group_first_relevant_rank": first_source_group_rank,
        "source_id_mrr": reciprocal_rank(first_source_id_rank),
        "source_group_mrr": reciprocal_rank(first_source_group_rank),
        "source_id_hit_at_1": hit_at_k(hit_source_ids, expected_source_ids, k=1),
        "source_id_hit_at_3": hit_at_k(hit_source_ids, expected_source_ids, k=3),
        "source_id_hit_at_k": hit_at_k(hit_source_ids, expected_source_ids, k=top_k),
        "source_group_hit_at_1": hit_at_k(hit_source_groups, expected_source_groups, k=1),
        "source_group_hit_at_3": hit_at_k(hit_source_groups, expected_source_groups, k=3),
        "source_group_hit_at_k": hit_at_k(hit_source_groups, expected_source_groups, k=top_k),
        "hit_source_ids": " | ".join(hit_source_ids),
        "hit_source_groups": " | ".join(hit_source_groups),
        "notes": notes,
    }
    return result


def aggregate_retrieval_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    mode_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        mode_rows[str(row.get("retrieval_mode", "unknown"))].append(row)

    aggregate: dict[str, Any] = {
        "query_count": len(rows),
        "modes": {},
    }
    for mode, mode_specific_rows in sorted(mode_rows.items()):
        aggregate["modes"][mode] = {
            "query_count": len(mode_specific_rows),
            "average_returned_hit_count": to_numeric_average([row.get("returned_hit_count") for row in mode_specific_rows]),
            "average_source_id_mrr": to_numeric_average([row.get("source_id_mrr") for row in mode_specific_rows]),
            "average_source_group_mrr": to_numeric_average([row.get("source_group_mrr") for row in mode_specific_rows]),
            "source_id_hit_rate_at_1": to_bool_average([row.get("source_id_hit_at_1") for row in mode_specific_rows]),
            "source_id_hit_rate_at_3": to_bool_average([row.get("source_id_hit_at_3") for row in mode_specific_rows]),
            "source_id_hit_rate_at_k": to_bool_average([row.get("source_id_hit_at_k") for row in mode_specific_rows]),
            "source_group_hit_rate_at_1": to_bool_average([row.get("source_group_hit_at_1") for row in mode_specific_rows]),
            "source_group_hit_rate_at_3": to_bool_average([row.get("source_group_hit_at_3") for row in mode_specific_rows]),
            "source_group_hit_rate_at_k": to_bool_average([row.get("source_group_hit_at_k") for row in mode_specific_rows]),
        }
    return aggregate
