from __future__ import annotations


SCHEMA_VERSION = "0.1.0"


RUN_MANIFEST_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "RunManifest",
    "type": "object",
    "required": [
        "run_id",
        "created_at",
        "study_id",
        "site_id",
        "purpose",
        "model_config",
        "retrieval_config",
        "artifact_paths",
    ],
    "properties": {
        "run_id": {"type": "string"},
        "created_at": {"type": "string"},
        "study_id": {"type": "string"},
        "site_id": {"type": "string"},
        "purpose": {"type": "string"},
        "model_config": {"type": "object"},
        "retrieval_config": {"type": "object"},
        "artifact_paths": {"type": "object"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "notes": {"type": "string"},
    },
}


STAGE_RECORD_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "PipelineStageRecord",
    "type": "object",
    "required": ["stage_name", "status", "started_at", "ended_at"],
    "properties": {
        "stage_name": {"type": "string"},
        "status": {"type": "string"},
        "started_at": {"type": "string"},
        "ended_at": {"type": "string"},
        "inputs": {"type": "object"},
        "outputs": {"type": "object"},
        "notes": {"type": "string"},
    },
}


SOURCE_TEXT_UNIT_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "SourceTextUnit",
    "type": "object",
    "required": [
        "unit_id",
        "source_id",
        "text",
        "char_count",
        "token_count_estimate",
        "citation_label",
    ],
    "properties": {
        "unit_id": {"type": "string"},
        "source_id": {"type": "string"},
        "text": {"type": "string"},
        "char_count": {"type": "integer"},
        "token_count_estimate": {"type": "integer"},
        "citation_label": {"type": "string"},
        "metadata": {"type": "object"},
    },
}


CHUNK_RECORD_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "ChunkRecord",
    "type": "object",
    "required": [
        "chunk_id",
        "source_id",
        "text",
        "char_count",
        "token_count_estimate",
        "citation_label",
    ],
    "properties": {
        "chunk_id": {"type": "string"},
        "source_id": {"type": "string"},
        "text": {"type": "string"},
        "char_count": {"type": "integer"},
        "token_count_estimate": {"type": "integer"},
        "citation_label": {"type": "string"},
        "metadata": {"type": "object"},
    },
}


STRUCTURED_CONSENT_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "StructuredConsentRecord",
    "type": "object",
    "properties": {
        "purposes": {"type": "array", "items": {"type": "string"}},
        "data_types": {"type": "array", "items": {"type": "string"}},
        "valid_until": {"type": ["string", "null"]},
        "withdrawal_policy": {"type": ["string", "null"]},
        "study_purpose_summary": {"type": ["string", "null"]},
        "study_procedures_summary": {"type": ["string", "null"]},
        "risks_summary": {"type": ["string", "null"]},
        "benefits_summary": {"type": ["string", "null"]},
        "alternatives_summary": {"type": ["string", "null"]},
        "question_rights_summary": {"type": ["string", "null"]},
        "voluntary_participation_statement": {"type": ["string", "null"]},
        "withdrawal_rights_summary": {"type": ["string", "null"]},
        "participant_rights": {"type": "array", "items": {"type": "string"}},
        "consent_summary": {"type": ["string", "null"]},
        "cited_markers": {"type": "array", "items": {"type": "string"}},
        "metadata": {"type": "object"},
    },
}


EVALUATION_RECORD_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "EvaluationRecord",
    "type": "object",
    "required": ["run_id", "case_id", "metric_name", "metric_value", "metric_group"],
    "properties": {
        "run_id": {"type": "string"},
        "case_id": {"type": "string"},
        "metric_name": {"type": "string"},
        "metric_value": {},
        "metric_group": {"type": "string"},
        "metadata": {"type": "object"},
    },
}


def get_schema_bundle() -> dict[str, dict]:
    return {
        "schema_version": {"version": SCHEMA_VERSION},
        "run_manifest.schema.json": RUN_MANIFEST_SCHEMA,
        "stage_record.schema.json": STAGE_RECORD_SCHEMA,
        "source_text_unit.schema.json": SOURCE_TEXT_UNIT_SCHEMA,
        "chunk_record.schema.json": CHUNK_RECORD_SCHEMA,
        "structured_consent.schema.json": STRUCTURED_CONSENT_SCHEMA,
        "evaluation_record.schema.json": EVALUATION_RECORD_SCHEMA,
    }
