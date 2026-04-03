from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, UTC
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4
import csv
import json

from .config import AppConfig
from .schemas import get_schema_bundle
from .types import AgentHandoff, EvaluationRecord, PipelineStageRecord, RunManifest


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def serialize_value(value: Any) -> Any:
    if is_dataclass(value):
        return {key: serialize_value(item) for key, item in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): serialize_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize_value(item) for item in value]
    return value


def compute_file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ArtifactStore:
    def __init__(self, config: AppConfig):
        self.config = config
        self.root = config.paths.artifact_root
        self.runs_dir = self.root / "runs"
        self.batches_dir = self.root / "batches"
        self.evaluations_dir = self.root / "evaluations"
        self.tables_dir = self.root / "tables"
        self.figures_dir = self.root / "figures"
        self.schemas_dir = self.root / "schemas"
        self._ensure_base_dirs()

    def _ensure_base_dirs(self) -> None:
        for path in (
            self.root,
            self.runs_dir,
            self.batches_dir,
            self.evaluations_dir,
            self.tables_dir,
            self.figures_dir,
            self.schemas_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def create_run(
        self,
        study_id: str,
        site_id: str,
        purpose: str,
        tags: list[str] | None = None,
        notes: str = "",
    ) -> RunManifest:
        created_at = utc_now_iso()
        run_id = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
        run_dir = self.runs_dir / run_id
        inputs_dir = run_dir / "inputs"
        outputs_dir = run_dir / "outputs"
        logs_dir = run_dir / "logs"
        figures_dir = run_dir / "figures"

        for path in (run_dir, inputs_dir, outputs_dir, logs_dir, figures_dir):
            path.mkdir(parents=True, exist_ok=True)

        manifest = RunManifest(
            run_id=run_id,
            created_at=created_at,
            study_id=study_id,
            site_id=site_id,
            purpose=purpose,
            model_config=serialize_value(self.config.models),
            retrieval_config=serialize_value(self.config.retrieval),
            artifact_paths={
                "run_dir": str(run_dir),
                "inputs_dir": str(inputs_dir),
                "outputs_dir": str(outputs_dir),
                "logs_dir": str(logs_dir),
                "figures_dir": str(figures_dir),
                "evaluation_log": str(self.evaluations_dir / f"{run_id}.jsonl"),
            },
            tags=tags or [],
            notes=notes,
        )
        self.write_json(run_dir / "manifest.json", manifest)
        self.export_schema_bundle(run_id=run_id)
        return manifest

    def run_path(self, run_id: str, *parts: str) -> Path:
        return self.runs_dir / run_id / Path(*parts)

    def write_json(self, path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(serialize_value(payload), indent=2), encoding="utf-8")

    def write_text(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def append_jsonl(self, path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(serialize_value(payload)))
            handle.write("\n")

    def write_jsonl(self, path: Path, rows: Iterable[Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(serialize_value(row)))
                handle.write("\n")

    def record_stage(self, run_id: str, record: PipelineStageRecord) -> None:
        self.append_jsonl(self.run_path(run_id, "logs", "stage_log.jsonl"), record)

    def record_agent_handoff(
        self,
        run_id: str,
        handoff: AgentHandoff,
        *,
        filename: str | None = None,
    ) -> Path:
        handoff_dir = self.run_path(run_id, "outputs", "agent_handoffs")
        handoff_dir.mkdir(parents=True, exist_ok=True)
        target = handoff_dir / (filename or f"{handoff.handoff_id}.json")
        self.write_json(target, handoff)
        self.append_jsonl(handoff_dir / "handoff_index.jsonl", handoff)
        return target

    def record_evaluation(self, record: EvaluationRecord) -> None:
        self.append_jsonl(self.evaluations_dir / f"{record.run_id}.jsonl", record)

    def write_table_csv(self, name: str, rows: Iterable[dict[str, Any]]) -> Path:
        rows = list(rows)
        path = self.tables_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not rows:
            path.write_text("", encoding="utf-8")
            return path
        fieldnames = list(rows[0].keys())
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(serialize_value(row))
        return path

    def export_schema_bundle(self, run_id: str | None = None) -> None:
        bundle = get_schema_bundle()
        for filename, payload in bundle.items():
            if filename == "schema_version":
                target = self.schemas_dir / "schema_version.json"
            elif run_id:
                target = self.run_path(run_id, "logs", filename)
            else:
                target = self.schemas_dir / filename
            self.write_json(target, payload)
