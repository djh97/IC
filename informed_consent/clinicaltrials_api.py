from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, UTC
from hashlib import sha256
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import json
import re


CLINICALTRIALS_API_ROOT = "https://clinicaltrials.gov/api/v2"
CLINICALTRIALS_STUDY_ROOT = "https://clinicaltrials.gov/study"
DEFAULT_GROUP_ID = "trial_materials"
DEFAULT_SUBDIRECTORY = "clinicaltrials_gov_api"


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def compute_sha256_bytes(payload: bytes) -> str:
    digest = sha256()
    digest.update(payload)
    return digest.hexdigest()


def normalize_nct_id(nct_id: str) -> str:
    cleaned = str(nct_id or "").strip().upper()
    if not cleaned.startswith("NCT"):
        raise ValueError(f"Expected an NCT identifier, got: {nct_id!r}")
    return cleaned


def build_source_id(nct_id: str) -> str:
    return normalize_nct_id(nct_id).lower()


def query_terms(query_term: str) -> list[str]:
    return [token for token in re.findall(r"[A-Za-z0-9]+", str(query_term).lower()) if token]


def clinicaltrials_api_headers() -> dict[str, str]:
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.9",
    }


def request_json(
    path: str,
    *,
    params: dict[str, Any] | None = None,
    timeout_seconds: int = 60,
) -> tuple[dict[str, Any], str | None]:
    base_url = f"{CLINICALTRIALS_API_ROOT}{path}"
    query = urlencode({key: value for key, value in (params or {}).items() if value is not None})
    url = f"{base_url}?{query}" if query else base_url
    request = Request(url, headers=clinicaltrials_api_headers())
    with urlopen(request, timeout=timeout_seconds) as response:
        payload = json.loads(response.read().decode("utf-8"))
        return payload, response.headers.get("Content-Type")


def fetch_study_record(nct_id: str, *, timeout_seconds: int = 60) -> dict[str, Any]:
    normalized_nct_id = normalize_nct_id(nct_id)
    payload, _ = request_json(f"/studies/{normalized_nct_id}", timeout_seconds=timeout_seconds)
    return payload


def study_matches_query_terms(study: dict[str, Any], query_term: str) -> bool:
    tokens = query_terms(query_term)
    if not tokens:
        return True

    protocol_section = study.get("protocolSection", {})
    identification = protocol_section.get("identificationModule", {})
    description = protocol_section.get("descriptionModule", {})
    conditions = protocol_section.get("conditionsModule", {})

    candidate_text = " ".join(
        [
            str(identification.get("briefTitle") or ""),
            str(identification.get("officialTitle") or ""),
            str(description.get("briefSummary") or ""),
            " ".join(str(item) for item in conditions.get("conditions", []) if str(item).strip()),
            " ".join(str(item) for item in conditions.get("keywords", []) if str(item).strip()),
        ]
    ).lower()
    return all(token in candidate_text for token in tokens)


def search_study_records(
    *,
    query_term: str,
    max_studies: int = 10,
    page_size: int = 10,
    timeout_seconds: int = 60,
) -> list[dict[str, Any]]:
    if not query_term.strip():
        return []

    studies: list[dict[str, Any]] = []
    next_page_token: str | None = None
    seen_nct_ids: set[str] = set()

    while len(studies) < max_studies:
        params: dict[str, Any] = {
            "query.term": query_term,
            "pageSize": min(page_size, max_studies - len(studies)),
        }
        if next_page_token:
            params["pageToken"] = next_page_token

        payload, _ = request_json("/studies", params=params, timeout_seconds=timeout_seconds)
        for study in payload.get("studies", []):
            if not isinstance(study, dict):
                continue
            if not study_matches_query_terms(study, query_term):
                continue
            identification = (
                study.get("protocolSection", {})
                .get("identificationModule", {})
            )
            nct_id = identification.get("nctId")
            if not isinstance(nct_id, str):
                continue
            normalized = normalize_nct_id(nct_id)
            if normalized in seen_nct_ids:
                continue
            seen_nct_ids.add(normalized)
            studies.append(study)
            if len(studies) >= max_studies:
                break

        next_page_token = payload.get("nextPageToken")
        if not next_page_token:
            break

    return studies


def extract_study_metadata(study: dict[str, Any]) -> dict[str, Any]:
    protocol_section = study.get("protocolSection", {})
    identification = protocol_section.get("identificationModule", {})
    status_module = protocol_section.get("statusModule", {})
    design_module = protocol_section.get("designModule", {})

    nct_id = normalize_nct_id(str(identification.get("nctId")))
    brief_title = str(identification.get("briefTitle") or "").strip()
    official_title = str(identification.get("officialTitle") or "").strip()
    title = brief_title or official_title or nct_id

    phases = design_module.get("phases")
    if isinstance(phases, list):
        phases = [str(item) for item in phases if str(item).strip()]
    else:
        phases = []

    return {
        "nct_id": nct_id,
        "title": title,
        "brief_title": brief_title,
        "official_title": official_title,
        "overall_status": str(status_module.get("overallStatus") or "").strip(),
        "study_type": str(design_module.get("studyType") or "").strip(),
        "phases": phases,
        "has_results": bool(study.get("hasResults")),
    }


def build_manifest_item(
    study: dict[str, Any],
    *,
    saved_path: Path,
    query_term: str | None = None,
) -> dict[str, Any]:
    metadata = extract_study_metadata(study)
    payload = json.dumps(study, indent=2, ensure_ascii=True).encode("utf-8")
    return {
        "group_id": DEFAULT_GROUP_ID,
        "source_id": build_source_id(metadata["nct_id"]),
        "source_type": "study_record",
        "authority": "ClinicalTrials.gov",
        "url": f"{CLINICALTRIALS_STUDY_ROOT}/{metadata['nct_id']}",
        "api_url": f"{CLINICALTRIALS_API_ROOT}/studies/{metadata['nct_id']}",
        "original_status": "planned",
        "downloaded_at": utc_now_iso(),
        "download_status": "downloaded",
        "content_type": "application/json",
        "saved_path": str(saved_path.resolve()),
        "byte_size": len(payload),
        "sha256": compute_sha256_bytes(payload),
        "title": metadata["title"],
        "nct_id": metadata["nct_id"],
        "overall_status": metadata["overall_status"],
        "study_type": metadata["study_type"],
        "phases": metadata["phases"],
        "has_results": metadata["has_results"],
        "record_format": "clinicaltrials_gov_api_v2",
        "query_term": query_term,
    }


def merge_manifest_items(
    manifest_path: Path,
    new_items: list[dict[str, Any]],
    *,
    output_root: Path,
    fetch_context: dict[str, Any],
) -> dict[str, Any]:
    existing_items: list[dict[str, Any]] = []
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("items"), list):
            existing_items = [item for item in payload["items"] if isinstance(item, dict)]

    merged_items_by_source_id: dict[str, dict[str, Any]] = {}
    for item in existing_items:
        source_id = str(item.get("source_id", "")).strip()
        if source_id:
            merged_items_by_source_id[source_id] = item
    for item in new_items:
        source_id = str(item.get("source_id", "")).strip()
        if source_id:
            merged_items_by_source_id[source_id] = item

    merged_items = list(merged_items_by_source_id.values())
    manifest = {
        "downloaded_at": utc_now_iso(),
        "output_root": str(output_root.resolve()),
        "item_count": len(merged_items),
        "downloaded_count": sum(1 for item in merged_items if item.get("download_status") == "downloaded"),
        "failed_count": sum(1 for item in merged_items if item.get("download_status") == "failed"),
        "last_requested_item_count": int(fetch_context.get("requested_item_count", 0)),
        "last_completed_item_count": len(new_items),
        "last_fetch_context": fetch_context,
        "items": merged_items,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def fetch_and_save_studies(
    *,
    output_root: Path,
    nct_ids: list[str] | None = None,
    query_term: str | None = None,
    max_studies: int = 10,
    page_size: int = 10,
    timeout_seconds: int = 60,
) -> dict[str, Any]:
    if not nct_ids and not query_term:
        raise ValueError("Provide at least one --nct-id or a --query-term to fetch ClinicalTrials.gov studies.")

    manifests_dir = output_root / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifests_dir / "download_manifest.json"

    target_dir = output_root / DEFAULT_GROUP_ID / DEFAULT_SUBDIRECTORY
    target_dir.mkdir(parents=True, exist_ok=True)

    fetched_studies: list[dict[str, Any]] = []
    fetch_errors: list[dict[str, Any]] = []
    seen_nct_ids: set[str] = set()

    for nct_id in nct_ids or []:
        normalized_nct_id = normalize_nct_id(nct_id)
        if normalized_nct_id in seen_nct_ids:
            continue
        try:
            fetched_studies.append(fetch_study_record(normalized_nct_id, timeout_seconds=timeout_seconds))
            seen_nct_ids.add(normalized_nct_id)
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
            fetch_errors.append(
                {
                    "source_id": build_source_id(normalized_nct_id),
                    "nct_id": normalized_nct_id,
                    "downloaded_at": utc_now_iso(),
                    "download_status": "failed",
                    "error": str(exc),
                }
            )

    if query_term:
        for study in search_study_records(
            query_term=query_term,
            max_studies=max_studies,
            page_size=page_size,
            timeout_seconds=timeout_seconds,
        ):
            metadata = extract_study_metadata(study)
            if metadata["nct_id"] in seen_nct_ids:
                continue
            seen_nct_ids.add(metadata["nct_id"])
            fetched_studies.append(study)

    saved_items: list[dict[str, Any]] = []
    for study in fetched_studies:
        metadata = extract_study_metadata(study)
        filename = f"{metadata['nct_id']}.json"
        target_path = target_dir / filename
        target_path.write_text(json.dumps(study, indent=2, ensure_ascii=True), encoding="utf-8")
        saved_items.append(
            build_manifest_item(
                study,
                saved_path=target_path,
                query_term=query_term,
            )
        )

    merged_manifest = merge_manifest_items(
        manifest_path,
        saved_items + fetch_errors,
        output_root=output_root,
        fetch_context={
            "source": "clinicaltrials_gov_api_v2",
            "query_term": query_term,
            "nct_ids": [normalize_nct_id(item) for item in (nct_ids or [])],
            "requested_item_count": len(nct_ids or []) + (max_studies if query_term else 0),
            "returned_study_count": len(saved_items),
            "failed_fetch_count": len(fetch_errors),
            "page_size": page_size,
            "max_studies": max_studies,
        },
    )

    return {
        "source": "clinicaltrials_gov_api_v2",
        "output_root": str(output_root.resolve()),
        "saved_study_count": len(saved_items),
        "failed_fetch_count": len(fetch_errors),
        "saved_source_ids": [item["source_id"] for item in saved_items],
        "failed_items": fetch_errors,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_item_count": merged_manifest["item_count"],
    }
