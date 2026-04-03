from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, UTC
from hashlib import sha256
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import json
import mimetypes
import re


EXTENSION_BY_CONTENT_TYPE = {
    "application/pdf": ".pdf",
    "text/html": ".html",
    "application/json": ".json",
    "text/plain": ".txt",
    "text/markdown": ".md",
    "application/xml": ".xml",
    "text/xml": ".xml",
}


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def slugify(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", value.lower())
    return value.strip("_") or "item"


def compute_sha256_bytes(payload: bytes) -> str:
    digest = sha256()
    digest.update(payload)
    return digest.hexdigest()


def infer_extension(url: str, content_type: str | None) -> str:
    parsed = urlparse(url)
    suffix = Path(parsed.path).suffix.lower()
    if suffix:
        return suffix
    if content_type:
        normalized = content_type.split(";")[0].strip().lower()
        if normalized in EXTENSION_BY_CONTENT_TYPE:
            return EXTENSION_BY_CONTENT_TYPE[normalized]
        guessed = mimetypes.guess_extension(normalized)
        if guessed:
            return guessed
    return ".bin"


@dataclass(slots=True)
class PublicSourcePlanItem:
    group_id: str
    source_id: str
    source_type: str
    authority: str
    url: str
    status: str


def build_download_plan(
    registry: dict[str, Any],
    *,
    group_ids: set[str] | None = None,
    source_ids: set[str] | None = None,
) -> list[PublicSourcePlanItem]:
    items: list[PublicSourcePlanItem] = []
    for group in registry.get("source_groups", []):
        group_id = str(group.get("group_id", "")).strip()
        if not group_id:
            continue
        if group_ids and group_id not in group_ids:
            continue
        for item in group.get("items", []):
            source_id = str(item.get("source_id", "")).strip()
            if not source_id:
                continue
            if source_ids and source_id not in source_ids:
                continue
            items.append(
                PublicSourcePlanItem(
                    group_id=group_id,
                    source_id=source_id,
                    source_type=str(item.get("type", "unknown")),
                    authority=str(item.get("authority", "")),
                    url=str(item.get("url", "")),
                    status=str(item.get("status", "planned")),
                )
            )
    return items


def download_plan_items(
    plan: list[PublicSourcePlanItem],
    *,
    output_root: Path,
    registry_snapshot: dict[str, Any] | None = None,
    timeout_seconds: int = 60,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    manifests_dir = output_root / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifests_dir / "download_manifest.json"

    if registry_snapshot is not None:
        (manifests_dir / "registry_snapshot.json").write_text(
            json.dumps(registry_snapshot, indent=2),
            encoding="utf-8",
        )

    existing_items: list[dict[str, Any]] = []
    if manifest_path.exists():
        existing_payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        if isinstance(existing_payload, dict) and isinstance(existing_payload.get("items"), list):
            existing_items = [item for item in existing_payload["items"] if isinstance(item, dict)]

    download_results: list[dict[str, Any]] = []
    for item in plan:
        target_dir = output_root / item.group_id
        target_dir.mkdir(parents=True, exist_ok=True)

        request = Request(
            item.url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
                "Accept": "*/*",
                "Accept-Language": "en-US,en;q=0.9",
            },
        )
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                payload = response.read()
                content_type = response.headers.get("Content-Type")
                extension = infer_extension(item.url, content_type)
                filename = f"{slugify(item.source_id)}{extension}"
                target_path = target_dir / filename
                target_path.write_bytes(payload)

                result = {
                    "group_id": item.group_id,
                    "source_id": item.source_id,
                    "source_type": item.source_type,
                    "authority": item.authority,
                    "url": item.url,
                    "original_status": item.status,
                    "downloaded_at": utc_now_iso(),
                    "download_status": "downloaded",
                    "content_type": content_type,
                    "saved_path": str(target_path.resolve()),
                    "byte_size": len(payload),
                    "sha256": compute_sha256_bytes(payload),
                }
                download_results.append(result)
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            download_results.append(
                {
                    "group_id": item.group_id,
                    "source_id": item.source_id,
                    "source_type": item.source_type,
                    "authority": item.authority,
                    "url": item.url,
                    "original_status": item.status,
                    "downloaded_at": utc_now_iso(),
                    "download_status": "failed",
                    "error": str(exc),
                }
            )

    merged_items_by_source_id: dict[str, dict[str, Any]] = {}
    for item in existing_items:
        source_id = str(item.get("source_id", "")).strip()
        if source_id:
            merged_items_by_source_id[source_id] = item
    for item in download_results:
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
        "last_requested_item_count": len(plan),
        "last_completed_item_count": len(download_results),
        "items": merged_items,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return manifest
