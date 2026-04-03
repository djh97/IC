from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class PatientProfile:
    participant_id: str | None = None
    age: int | None = None
    language: str = "en"
    health_literacy: str = "medium"
    jurisdiction: str = "US"
    notes: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ConsentSourceDocument:
    source_id: str
    title: str
    source_type: str
    path: str
    sha256: str
    byte_size: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SourceTextUnit:
    unit_id: str
    source_id: str
    text: str
    char_count: int
    token_count_estimate: int
    citation_label: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ChunkRecord:
    chunk_id: str
    source_id: str
    text: str
    char_count: int
    token_count_estimate: int
    citation_label: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievalHit:
    source_id: str
    chunk_id: str
    rank: int
    score: float
    citation_label: str
    excerpt: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ConversationTurn:
    turn_id: str
    speaker: str
    turn_type: str
    text: str
    citations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StructuredConsentRecord:
    purposes: list[str] = field(default_factory=list)
    data_types: list[str] = field(default_factory=list)
    valid_until: str | None = None
    withdrawal_policy: str | None = None
    participant_rights: list[str] = field(default_factory=list)
    consent_summary: str | None = None
    cited_markers: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PipelineStageRecord:
    stage_name: str
    status: str
    started_at: str
    ended_at: str
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    notes: str = ""


@dataclass(slots=True)
class AgentHandoff:
    handoff_id: str
    run_id: str
    from_agent: str
    to_agent: str
    purpose: str
    created_at: str
    payload: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvaluationRecord:
    run_id: str
    case_id: str
    metric_name: str
    metric_value: float | int | str
    metric_group: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RunManifest:
    run_id: str
    created_at: str
    study_id: str
    site_id: str
    purpose: str
    model_config: dict[str, Any]
    retrieval_config: dict[str, Any]
    artifact_paths: dict[str, str]
    tags: list[str] = field(default_factory=list)
    notes: str = ""
