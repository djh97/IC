from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from datetime import datetime, UTC
from pathlib import Path
from typing import Any
import csv
import json
import os
import re
import shutil
import subprocess
from hashlib import sha1
from uuid import uuid4

from .agents import (
    AgentRuntime,
    ConsentFormalizationAgent,
    ConversationalAgent,
    OrchestratorAgent,
    PersonalizationAgent,
    RAGAgent,
)
from .artifacts import ArtifactStore, compute_file_sha256, utc_now_iso
from .clinicaltrials_api import fetch_and_save_studies
from .config import AppConfig, build_default_config
from .corpus import build_chunk_records, load_chunk_records, load_source_text_units, retrieve_lexical_hits, slugify_source_id
from .evaluation import evaluate_run_outputs
from .hybrid_retrieval import build_dense_embeddings, dense_retrieve, load_dense_index, reciprocal_rank_fusion, save_dense_index
from .prompt_loader import PromptLoader
from .public_sources import download_plan_items
from .retrieval_benchmark import aggregate_retrieval_results, load_benchmark_spec, normalize_modes, normalize_string_list, score_retrieval_case
from .source_registry import SourceRegistry
from .types import ConsentSourceDocument, PatientProfile, PipelineStageRecord, RunManifest


REFERENCE_CHECKLIST = [
    {
        "element_id": "voluntary_participation",
        "description": "The consent explanation should make clear that joining the study is voluntary and a choice.",
        "preferred_source_groups": ["regulatory_guidance", "posted_consent_forms"],
        "queries": ["voluntary participation your choice", "participation is voluntary"],
    },
    {
        "element_id": "study_procedures",
        "description": "The consent explanation should describe what the participant will do in the study.",
        "preferred_source_groups": ["trial_materials", "posted_consent_forms", "regulatory_guidance"],
        "queries": ["study procedures study visits what will happen", "what you will do in this study"],
    },
    {
        "element_id": "risks",
        "description": "The consent explanation should describe foreseeable risks or discomforts.",
        "preferred_source_groups": ["regulatory_guidance", "posted_consent_forms", "trial_materials"],
        "queries": ["risks discomforts side effects", "study risks"],
    },
    {
        "element_id": "benefits",
        "description": "The consent explanation should describe possible benefits or note when benefits are uncertain.",
        "preferred_source_groups": ["regulatory_guidance", "posted_consent_forms", "trial_materials"],
        "queries": ["possible benefits no guaranteed benefit", "study benefits"],
    },
    {
        "element_id": "alternatives",
        "description": "The consent explanation should mention other options or alternatives to joining the study.",
        "preferred_source_groups": ["regulatory_guidance", "posted_consent_forms"],
        "queries": ["alternatives other options", "other choices besides joining"],
    },
    {
        "element_id": "questions",
        "description": "The consent explanation should say who to contact or that questions can be asked.",
        "preferred_source_groups": ["regulatory_guidance", "posted_consent_forms", "trial_materials"],
        "queries": ["contact questions rights", "who to contact if you have questions"],
    },
    {
        "element_id": "withdrawal_rights",
        "description": "The consent explanation should say the participant can stop later and describe penalty or loss-of-benefits language if relevant.",
        "preferred_source_groups": ["regulatory_guidance", "posted_consent_forms"],
        "queries": ["withdraw stop participating without penalty", "stop taking part"],
    },
]


class ConsentPipeline:
    def __init__(self, config: AppConfig | None = None):
        self.config = config or build_default_config()
        self.artifacts = ArtifactStore(self.config)
        self.prompts = PromptLoader(self.config)
        self.source_registry = SourceRegistry(self.config)
        self.agent_runtime = AgentRuntime(self)
        self.rag_agent = RAGAgent(self.agent_runtime)
        self.personalization_agent = PersonalizationAgent(self.agent_runtime)
        self.conversational_agent = ConversationalAgent(self.agent_runtime)
        self.formalization_agent = ConsentFormalizationAgent(self.agent_runtime)
        self.orchestrator_agent = OrchestratorAgent(
            self.agent_runtime,
            rag_agent=self.rag_agent,
            personalization_agent=self.personalization_agent,
            conversational_agent=self.conversational_agent,
            formalization_agent=self.formalization_agent,
        )
        self._git_commit_hash: str | None | bool = None

    def get_git_commit_hash(self) -> str | None:
        if self._git_commit_hash is False:
            return None
        if isinstance(self._git_commit_hash, str):
            return self._git_commit_hash
        try:
            completed = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.config.paths.project_root,
                capture_output=True,
                text=True,
                check=True,
            )
        except (OSError, subprocess.CalledProcessError):
            self._git_commit_hash = False
            return None
        commit_hash = completed.stdout.strip()
        self._git_commit_hash = commit_hash or False
        return commit_hash or None

    def build_runtime_metadata(
        self,
        *,
        config_path: str | None = None,
        base_run_id: str | None = None,
        prompt_identifiers: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        index_summary_path = (
            self.artifacts.run_path(base_run_id, "outputs", "hybrid_index", "hybrid_index_summary.json")
            if base_run_id
            else None
        )
        corpus_summary_path = (
            self.artifacts.run_path(base_run_id, "outputs", "corpus_summary.json")
            if base_run_id
            else None
        )
        random_seed = os.getenv("IC_RANDOM_SEED", "").strip() or os.getenv("PYTHONHASHSEED", "").strip() or None
        return {
            "model_id": self.config.models.generator_model,
            "embedding_model_id": self.config.models.embedding_model,
            "retrieval_default_mode": self.config.retrieval.retrieval_mode,
            "retrieval_default_top_k": self.config.retrieval.top_k,
            "corpus_version": base_run_id,
            "index_version": base_run_id if index_summary_path and index_summary_path.exists() else None,
            "corpus_summary_path": str(corpus_summary_path) if corpus_summary_path and corpus_summary_path.exists() else None,
            "index_summary_path": str(index_summary_path) if index_summary_path and index_summary_path.exists() else None,
            "config_path": config_path,
            "git_commit_hash": self.get_git_commit_hash(),
            "random_seed": random_seed,
            "prompt_identifiers": prompt_identifiers or {},
        }

    def initialize_run(
        self,
        purpose: str,
        tags: list[str] | None = None,
        notes: str = "",
        study_id: str | None = None,
        site_id: str | None = None,
    ) -> RunManifest:
        manifest = self.artifacts.create_run(
            study_id=study_id or self.config.study_id,
            site_id=site_id or self.config.site_id,
            purpose=purpose,
            tags=tags,
            notes=notes,
        )
        runtime_metadata = self.build_runtime_metadata()
        context_metadata = {
            "purpose": purpose,
            "study_id": study_id or self.config.study_id,
            "site_id": site_id or self.config.site_id,
        }
        self.artifacts.update_run_manifest(
            manifest.run_id,
            {
                "runtime_metadata": runtime_metadata,
                "context_metadata": context_metadata,
            },
        )
        return manifest

    def bootstrap_run(
        self,
        purpose: str,
        source_dir: Path | None = None,
        template_path: Path | None = None,
        patient_profile_path: Path | None = None,
        tags: list[str] | None = None,
        notes: str = "",
        study_id: str | None = None,
        site_id: str | None = None,
    ) -> RunManifest:
        manifest = self.initialize_run(
            purpose=purpose,
            tags=tags,
            notes=notes,
            study_id=study_id,
            site_id=site_id,
        )
        run_id = manifest.run_id
        started_at = utc_now_iso()

        source_documents: list[ConsentSourceDocument] = []
        if source_dir is not None and source_dir.exists():
            source_documents = self.inventory_source_directory(source_dir)
            self.artifacts.write_json(
                self.artifacts.run_path(run_id, "inputs", "source_inventory.json"),
                source_documents,
            )

        if template_path is not None and template_path.exists():
            self.artifacts.write_text(
                self.artifacts.run_path(run_id, "inputs", "base_template.txt"),
                template_path.read_text(encoding="utf-8"),
            )

        if patient_profile_path is not None and patient_profile_path.exists():
            profile = self.load_patient_profile(patient_profile_path)
            self.artifacts.write_json(
                self.artifacts.run_path(run_id, "inputs", "patient_profile.json"),
                profile,
            )

        self.artifacts.record_stage(
            run_id,
            PipelineStageRecord(
                stage_name="bootstrap",
                status="completed",
                started_at=started_at,
                ended_at=utc_now_iso(),
                inputs={
                    "source_dir": str(source_dir) if source_dir else None,
                    "template_path": str(template_path) if template_path else None,
                    "patient_profile_path": str(patient_profile_path) if patient_profile_path else None,
                },
                outputs={
                    "source_count": len(source_documents),
                },
                notes="Initialized a reproducible local run scaffold for later retrieval, generation, and evaluation stages.",
            ),
        )

        return manifest

    def prepare_corpus(
        self,
        purpose: str,
        source_dir: Path,
        template_path: Path | None = None,
        patient_profile_path: Path | None = None,
        tags: list[str] | None = None,
        notes: str = "",
        study_id: str | None = None,
        site_id: str | None = None,
    ) -> RunManifest:
        if not source_dir.exists():
            raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

        manifest = self.bootstrap_run(
            purpose=purpose,
            source_dir=source_dir,
            template_path=template_path,
            patient_profile_path=patient_profile_path,
            tags=tags,
            notes=notes,
            study_id=study_id,
            site_id=site_id,
        )
        run_id = manifest.run_id
        started_at = utc_now_iso()

        source_documents = self.inventory_source_directory(source_dir)
        text_units = []
        skipped_sources: list[dict[str, Any]] = []

        for source_doc in source_documents:
            try:
                extracted_units = load_source_text_units(source_doc)
            except Exception as exc:
                skipped_sources.append(
                    {
                        "source_id": source_doc.source_id,
                        "title": source_doc.title,
                        "reason": str(exc),
                    }
                )
                continue
            if not extracted_units:
                skipped_sources.append(
                    {
                        "source_id": source_doc.source_id,
                        "title": source_doc.title,
                        "reason": "No extractable text produced.",
                    }
                )
                continue
            text_units.extend(extracted_units)

        chunk_records = build_chunk_records(
            text_units=text_units,
            chunk_size=self.config.retrieval.chunk_size,
            chunk_overlap=self.config.retrieval.chunk_overlap,
        )

        outputs_dir = self.artifacts.run_path(run_id, "outputs")
        self.artifacts.write_jsonl(outputs_dir / "source_text_units.jsonl", text_units)
        self.artifacts.write_jsonl(outputs_dir / "chunk_records.jsonl", chunk_records)

        source_group_counts = Counter(str(doc.metadata.get("source_group", "unknown")).strip() or "unknown" for doc in source_documents)
        source_type_counts = Counter(doc.source_type for doc in source_documents)
        authority_counts = Counter(str(doc.metadata.get("authority", "unknown")).strip() or "unknown" for doc in source_documents)
        text_unit_count_by_source_id = Counter(unit.source_id for unit in text_units)
        chunk_count_by_source_id = Counter(chunk.source_id for chunk in chunk_records)
        source_statistics = {
            "source_group_counts": dict(sorted(source_group_counts.items())),
            "source_type_counts": dict(sorted(source_type_counts.items())),
            "authority_counts": dict(sorted(authority_counts.items())),
            "text_unit_count_by_source_id": dict(sorted(text_unit_count_by_source_id.items())),
            "chunk_count_by_source_id": dict(sorted(chunk_count_by_source_id.items())),
        }
        self.artifacts.write_json(outputs_dir / "source_statistics.json", source_statistics)

        corpus_summary = {
            "source_document_count": len(source_documents),
            "extractable_source_count": len({unit.source_id for unit in text_units}),
            "text_unit_count": len(text_units),
            "chunk_count": len(chunk_records),
            "skipped_source_count": len(skipped_sources),
            "chunk_size": self.config.retrieval.chunk_size,
            "chunk_overlap": self.config.retrieval.chunk_overlap,
            "supported_source_types": sorted({doc.source_type for doc in source_documents}),
            "source_group_count": len(source_group_counts),
            "authority_count": len(authority_counts),
        }
        self.artifacts.write_json(outputs_dir / "corpus_summary.json", corpus_summary)
        self.artifacts.write_json(outputs_dir / "skipped_sources.json", skipped_sources)

        self.artifacts.record_stage(
            run_id,
            PipelineStageRecord(
                stage_name="prepare_corpus",
                status="completed",
                started_at=started_at,
                ended_at=utc_now_iso(),
                inputs={
                    "source_dir": str(source_dir),
                    "chunk_size": self.config.retrieval.chunk_size,
                    "chunk_overlap": self.config.retrieval.chunk_overlap,
                },
                outputs=corpus_summary,
                notes=(
                    "Loaded local source documents, extracted text units with citations, "
                    "and generated retrieval-ready chunks for later grounding."
                ),
            ),
        )

        return manifest

    def build_hybrid_index(self, run_id: str) -> dict[str, Any]:
        started_at = utc_now_iso()
        chunk_path = self.artifacts.run_path(run_id, "outputs", "chunk_records.jsonl")
        chunks = load_chunk_records(chunk_path)
        if not chunks:
            raise FileNotFoundError(
                "No chunk records were found for this run. Prepare the corpus first before building the hybrid index."
            )

        embeddings = build_dense_embeddings(chunks, model_name=self.config.models.embedding_model)
        index_dir = self.artifacts.run_path(run_id, "outputs", "hybrid_index")
        paths = save_dense_index(
            index_dir,
            embeddings=embeddings,
            chunks=chunks,
            model_name=self.config.models.embedding_model,
        )

        summary = {
            "retrieval_mode": "hybrid",
            "embedding_model": self.config.models.embedding_model,
            "chunk_count": len(chunks),
            "embedding_dimension": int(embeddings.shape[1]) if len(embeddings.shape) == 2 else 0,
            **paths,
        }
        self.artifacts.write_json(index_dir / "hybrid_index_summary.json", summary)
        self.artifacts.record_stage(
            run_id,
            PipelineStageRecord(
                stage_name="build_hybrid_index",
                status="completed",
                started_at=started_at,
                ended_at=utc_now_iso(),
                outputs=summary,
                notes=(
                    "Built a local dense embedding index on CPU and saved the hybrid retrieval artifacts for reuse."
                ),
            ),
        )
        return summary

    def query_prepared_corpus(
        self,
        run_id: str,
        query: str,
        top_k: int | None = None,
        retrieval_mode: str | None = None,
        source_group_filters: list[str] | None = None,
        source_id_filters: list[str] | None = None,
        filter_logic: str | None = None,
    ) -> list[dict[str, Any]]:
        retrieval = self.rag_agent.retrieve_evidence(
            run_id=run_id,
            query=query,
            top_k=top_k,
            retrieval_mode=retrieval_mode,
            source_group_filters=source_group_filters,
            source_id_filters=source_id_filters,
            filter_logic=filter_logic,
            purpose="corpus_query",
        )
        return retrieval["retrieval_hits"]

    def retrieve_prepared_corpus(
        self,
        run_id: str,
        query: str,
        top_k: int | None = None,
        retrieval_mode: str | None = None,
        source_group_filters: list[str] | None = None,
        source_id_filters: list[str] | None = None,
        filter_logic: str | None = None,
    ) -> dict[str, Any]:
        chunk_path = self.artifacts.run_path(run_id, "outputs", "chunk_records.jsonl")
        chunks = load_chunk_records(chunk_path)
        normalized_group_filters = {item.strip() for item in (source_group_filters or []) if item and item.strip()}
        normalized_source_id_filters = {item.strip() for item in (source_id_filters or []) if item and item.strip()}
        normalized_filter_logic = self.normalize_filter_logic(filter_logic)
        selected_indices = [
            index
            for index, chunk in enumerate(chunks)
            if self.chunk_matches_retrieval_filters(
                chunk,
                source_group_filters=normalized_group_filters,
                source_id_filters=normalized_source_id_filters,
                filter_logic=normalized_filter_logic,
            )
        ]
        filtered_chunks = [chunks[index] for index in selected_indices]
        requested_top_k = top_k or self.config.retrieval.top_k
        lexical_hits = retrieve_lexical_hits(filtered_chunks, query=query, top_k=requested_top_k)

        requested_mode = str(retrieval_mode or self.config.retrieval.retrieval_mode).strip().lower()
        if requested_mode not in {"lexical", "dense", "hybrid"}:
            requested_mode = self.config.retrieval.retrieval_mode
        index_dir = self.artifacts.run_path(run_id, "outputs", "hybrid_index")
        dense_available = (index_dir / "dense_embeddings.npy").exists() and (index_dir / "dense_index_metadata.json").exists()

        if not filtered_chunks:
            return {
                "mode_used": "lexical",
                "dense_available": dense_available,
                "lexical_hits": [],
                "dense_hits": [],
                "hits": [],
                "source_group_filters": sorted(normalized_group_filters),
                "source_id_filters": sorted(normalized_source_id_filters),
                "filter_logic_used": normalized_filter_logic,
                "filtered_chunk_count": 0,
            }

        if requested_mode == "lexical" or not dense_available:
            return {
                "mode_used": "lexical",
                "dense_available": dense_available,
                "lexical_hits": [asdict(hit) for hit in lexical_hits],
                "dense_hits": [],
                "hits": lexical_hits,
                "source_group_filters": sorted(normalized_group_filters),
                "source_id_filters": sorted(normalized_source_id_filters),
                "filter_logic_used": normalized_filter_logic,
                "filtered_chunk_count": len(filtered_chunks),
            }

        embeddings, metadata = load_dense_index(index_dir)
        filtered_embeddings = embeddings[selected_indices]
        dense_hits = dense_retrieve(
            query,
            chunks=filtered_chunks,
            embeddings=filtered_embeddings,
            model_name=str(metadata["model_name"]),
            top_k=requested_top_k,
        )

        if requested_mode == "dense":
            return {
                "mode_used": "dense",
                "dense_available": True,
                "lexical_hits": [asdict(hit) for hit in lexical_hits],
                "dense_hits": [asdict(hit) for hit in dense_hits],
                "hits": dense_hits,
                "source_group_filters": sorted(normalized_group_filters),
                "source_id_filters": sorted(normalized_source_id_filters),
                "filter_logic_used": normalized_filter_logic,
                "filtered_chunk_count": len(filtered_chunks),
            }

        fused_hits = reciprocal_rank_fusion(
            [lexical_hits, dense_hits],
            top_k=requested_top_k,
            rrf_k=self.config.retrieval.rrf_k,
        )
        return {
            "mode_used": "hybrid",
            "dense_available": True,
            "lexical_hits": [asdict(hit) for hit in lexical_hits],
            "dense_hits": [asdict(hit) for hit in dense_hits],
            "hits": fused_hits,
            "source_group_filters": sorted(normalized_group_filters),
            "source_id_filters": sorted(normalized_source_id_filters),
            "filter_logic_used": normalized_filter_logic,
            "filtered_chunk_count": len(filtered_chunks),
        }

    def normalize_filter_logic(self, filter_logic: str | None) -> str:
        normalized = str(filter_logic or "intersection").strip().lower()
        if normalized not in {"intersection", "union"}:
            return "intersection"
        return normalized

    def normalize_workflow_variant(self, workflow_variant: str | None) -> str:
        normalized = str(workflow_variant or "full_agentic").strip().lower()
        if normalized not in {"full_agentic", "generic_rag", "vanilla_llm"}:
            return "full_agentic"
        return normalized

    def chunk_matches_retrieval_filters(
        self,
        chunk: ChunkRecord,
        *,
        source_group_filters: set[str],
        source_id_filters: set[str],
        filter_logic: str,
    ) -> bool:
        matches_source_id = chunk.source_id in source_id_filters if source_id_filters else False
        matches_source_group = (
            str(chunk.metadata.get("source_group", "")).strip() in source_group_filters if source_group_filters else False
        )

        if not source_group_filters and not source_id_filters:
            return True
        if not source_group_filters:
            return matches_source_id
        if not source_id_filters:
            return matches_source_group
        if filter_logic == "union":
            return matches_source_id or matches_source_group
        return matches_source_id and matches_source_group

    def draft_personalized_consent(
        self,
        run_id: str,
        patient_profile_path: Path | None = None,
        template_path: Path | None = None,
        generation_query: str | None = None,
        top_k: int | None = None,
        retrieval_mode: str | None = None,
        source_group_filters: list[str] | None = None,
        source_id_filters: list[str] | None = None,
        filter_logic: str | None = None,
        workflow_variant: str | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        return self.orchestrator_agent.personalize_consent(
            run_id=run_id,
            patient_profile_path=patient_profile_path,
            template_path=template_path,
            generation_query=generation_query,
            top_k=top_k,
            retrieval_mode=retrieval_mode,
            source_group_filters=source_group_filters,
            source_id_filters=source_id_filters,
            filter_logic=filter_logic,
            workflow_variant=self.normalize_workflow_variant(workflow_variant),
            dry_run=dry_run,
        )

    def formalize_consent(
        self,
        run_id: str,
        patient_profile_path: Path | None = None,
        draft_path: Path | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        return self.orchestrator_agent.formalize_consent(
            run_id=run_id,
            patient_profile_path=patient_profile_path,
            draft_path=draft_path,
            dry_run=dry_run,
        )

    def answer_consent_question(
        self,
        run_id: str,
        question: str,
        patient_profile_path: Path | None = None,
        top_k: int | None = None,
        retrieval_mode: str | None = None,
        source_group_filters: list[str] | None = None,
        source_id_filters: list[str] | None = None,
        filter_logic: str | None = None,
        workflow_variant: str | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        return self.orchestrator_agent.answer_question(
            run_id=run_id,
            question=question,
            patient_profile_path=patient_profile_path,
            top_k=top_k,
            retrieval_mode=retrieval_mode,
            source_group_filters=source_group_filters,
            source_id_filters=source_id_filters,
            filter_logic=filter_logic,
            workflow_variant=self.normalize_workflow_variant(workflow_variant),
            dry_run=dry_run,
        )

    def handle_user_request(
        self,
        run_id: str,
        user_input: str,
        patient_profile_path: Path | None = None,
        template_path: Path | None = None,
        draft_path: Path | None = None,
        top_k: int | None = None,
        retrieval_mode: str | None = None,
        source_group_filters: list[str] | None = None,
        source_id_filters: list[str] | None = None,
        filter_logic: str | None = None,
        workflow_variant: str | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        return self.orchestrator_agent.handle_user_request(
            run_id=run_id,
            user_input=user_input,
            patient_profile_path=patient_profile_path,
            template_path=template_path,
            draft_path=draft_path,
            top_k=top_k,
            retrieval_mode=retrieval_mode,
            source_group_filters=source_group_filters,
            source_id_filters=source_id_filters,
            filter_logic=filter_logic,
            workflow_variant=self.normalize_workflow_variant(workflow_variant),
            dry_run=dry_run,
        )

    def evaluate_run(self, run_id: str) -> dict[str, Any]:
        started_at = utc_now_iso()
        run_dir = self.artifacts.runs_dir / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

        evaluation = evaluate_run_outputs(run_id=run_id, run_dir=run_dir)
        outputs_dir = self.artifacts.run_path(run_id, "outputs")

        summary_path = outputs_dir / "evaluation_summary.json"
        qualitative_path = outputs_dir / "qualitative_case_bundle.json"
        jsonl_path = self.artifacts.evaluations_dir / f"{run_id}.jsonl"
        csv_path = self.artifacts.write_table_csv(f"{run_id}_evaluation_metrics.csv", evaluation["records"])

        self.artifacts.write_json(summary_path, evaluation["summary"])
        self.artifacts.write_json(qualitative_path, evaluation["qualitative_bundle"])
        self.artifacts.write_jsonl(jsonl_path, evaluation["records"])

        self.artifacts.record_stage(
            run_id,
            PipelineStageRecord(
                stage_name="evaluate_run",
                status="completed",
                started_at=started_at,
                ended_at=utc_now_iso(),
                outputs={
                    "evaluation_record_count": len(evaluation["records"]),
                    "summary_path": str(summary_path),
                    "qualitative_bundle_path": str(qualitative_path),
                    "evaluation_jsonl_path": str(jsonl_path),
                    "evaluation_csv_path": str(csv_path),
                },
                notes=(
                    "Computed reusable evaluation metrics and qualitative summaries from saved run artifacts "
                    "without requiring live endpoint calls."
                ),
            ),
        )

        return {
            "run_id": run_id,
            "summary_path": str(summary_path),
            "qualitative_bundle_path": str(qualitative_path),
            "evaluation_jsonl_path": str(jsonl_path),
            "evaluation_csv_path": str(csv_path),
            "summary": evaluation["summary"],
        }

    def evaluate_retrieval_benchmark(
        self,
        run_id: str,
        spec_path: Path,
        *,
        modes: list[str] | None = None,
        top_k: int | None = None,
        filter_logic: str | None = None,
    ) -> dict[str, Any]:
        started_at = utc_now_iso()
        run_dir = self.artifacts.runs_dir / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

        spec = load_benchmark_spec(spec_path)
        benchmark_id = str(spec.get("benchmark_id") or spec_path.stem).strip() or spec_path.stem
        defaults = spec.get("defaults", {})
        if not isinstance(defaults, dict):
            defaults = {}

        default_modes = normalize_modes(modes) or normalize_modes(defaults.get("modes")) or [self.config.retrieval.retrieval_mode]
        default_top_k = int(top_k or defaults.get("top_k") or self.config.retrieval.top_k)
        default_group_filters = normalize_string_list(defaults.get("source_group_filters"))
        default_source_filters = normalize_string_list(defaults.get("source_id_filters"))
        default_filter_logic = self.normalize_filter_logic(filter_logic or defaults.get("filter_logic"))

        benchmark_output_dir = self.artifacts.run_path(run_id, "outputs", "retrieval_benchmarks", benchmark_id)
        benchmark_output_dir.mkdir(parents=True, exist_ok=True)

        result_rows: list[dict[str, Any]] = []
        checks = spec.get("queries", [])
        if not isinstance(checks, list):
            raise ValueError("Retrieval benchmark spec must contain a 'queries' array.")

        for raw_check in checks:
            if not isinstance(raw_check, dict):
                continue
            query_id = str(raw_check.get("query_id", "")).strip()
            query = str(raw_check.get("query", "")).strip()
            if not query_id or not query:
                raise ValueError("Each retrieval benchmark query must define non-empty 'query_id' and 'query'.")

            query_modes = normalize_modes(raw_check.get("modes")) or default_modes
            query_top_k = int(raw_check.get("top_k") or default_top_k)
            expected_source_ids = normalize_string_list(raw_check.get("expected_source_ids"))
            expected_source_groups = normalize_string_list(raw_check.get("expected_source_groups"))
            source_group_filters = normalize_string_list(raw_check.get("source_group_filters")) or default_group_filters
            source_id_filters = normalize_string_list(raw_check.get("source_id_filters")) or default_source_filters
            filter_logic = self.normalize_filter_logic(raw_check.get("filter_logic") or default_filter_logic)
            notes = str(raw_check.get("notes", ""))

            for retrieval_mode in query_modes:
                retrieval = self.retrieve_prepared_corpus(
                    run_id=run_id,
                    query=query,
                    top_k=query_top_k,
                    retrieval_mode=retrieval_mode,
                    source_group_filters=source_group_filters,
                    source_id_filters=source_id_filters,
                    filter_logic=filter_logic,
                )
                hits_payload = [asdict(hit) for hit in retrieval["hits"]]
                hits_path = benchmark_output_dir / f"{query_id}.{retrieval_mode}.hits.json"
                self.artifacts.write_json(hits_path, hits_payload)
                row = score_retrieval_case(
                    benchmark_id=benchmark_id,
                    query_id=query_id,
                    query=query,
                    retrieval_mode=retrieval["mode_used"],
                    hits=hits_payload,
                    expected_source_ids=expected_source_ids,
                    expected_source_groups=expected_source_groups,
                    top_k=query_top_k,
                    source_group_filters=source_group_filters,
                    source_id_filters=source_id_filters,
                    notes=notes,
                )
                row["filter_logic"] = filter_logic
                row["hits_path"] = str(hits_path)
                result_rows.append(row)

        aggregate_metrics = aggregate_retrieval_results(result_rows)
        summary_payload = {
            "run_id": run_id,
            "benchmark_id": benchmark_id,
            "spec_path": str(spec_path.resolve()),
            "query_count": len(checks),
            "result_count": len(result_rows),
            "modes": default_modes,
            "top_k_default": default_top_k,
            "aggregate_metrics": aggregate_metrics,
        }
        summary_path = benchmark_output_dir / "retrieval_benchmark_summary.json"
        results_jsonl_path = benchmark_output_dir / "retrieval_benchmark_results.jsonl"
        self.artifacts.write_json(summary_path, summary_payload)
        self.artifacts.write_jsonl(results_jsonl_path, result_rows)
        csv_path = self.artifacts.write_table_csv(f"{run_id}_{benchmark_id}_retrieval_benchmark.csv", result_rows)

        self.artifacts.record_stage(
            run_id,
            PipelineStageRecord(
                stage_name="evaluate_retrieval_benchmark",
                status="completed",
                started_at=started_at,
                ended_at=utc_now_iso(),
                inputs={
                    "spec_path": str(spec_path.resolve()),
                    "modes": default_modes,
                    "top_k": default_top_k,
                    "filter_logic": default_filter_logic,
                },
                outputs={
                    "benchmark_id": benchmark_id,
                    "result_count": len(result_rows),
                    "summary_path": str(summary_path),
                    "results_jsonl_path": str(results_jsonl_path),
                    "results_csv_path": str(csv_path),
                },
                notes=(
                    "Evaluated saved corpus retrieval quality across one or more retrieval modes without requiring live generation calls."
                ),
            ),
        )

        return {
            "run_id": run_id,
            "benchmark_id": benchmark_id,
            "summary_path": str(summary_path),
            "results_jsonl_path": str(results_jsonl_path),
            "results_csv_path": str(csv_path),
            "summary": summary_payload,
        }

    def export_manual_review_bundle(self, run_id: str) -> dict[str, Any]:
        run_dir = self.artifacts.runs_dir / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

        outputs_dir = self.artifacts.run_path(run_id, "outputs")
        bundle_path = outputs_dir / "qualitative_case_bundle.json"
        if not bundle_path.exists():
            raise FileNotFoundError(
                "Qualitative case bundle not found. Run evaluate-run first so the bundle can be exported for review."
            )

        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        review_rows: list[dict[str, Any]] = []

        draft = bundle.get("personalized_consent_draft")
        if draft:
            review_rows.append(
                {
                    "run_id": run_id,
                    "item_type": "personalized_consent_draft",
                    "item_id": "draft_main",
                    "prompt_or_question": "Generate a grounded personalized consent draft.",
                    "text": draft.get("personalized_consent_text", ""),
                    "citations": ", ".join(draft.get("citation_markers_used", [])),
                }
            )

        structured = bundle.get("structured_consent_record")
        if structured:
            review_rows.append(
                {
                    "run_id": run_id,
                    "item_type": "structured_consent_record",
                    "item_id": "structured_main",
                    "prompt_or_question": "Generate a structured consent record.",
                    "text": json.dumps(structured, ensure_ascii=True),
                    "citations": ", ".join(structured.get("cited_markers", [])),
                }
            )

        for qa_item in bundle.get("qa_answers", []):
            answer = qa_item.get("answer") or {}
            review_rows.append(
                {
                    "run_id": run_id,
                    "item_type": "qa_answer",
                    "item_id": qa_item.get("question_id"),
                    "prompt_or_question": qa_item.get("question"),
                    "text": answer.get("answer_text", ""),
                    "citations": ", ".join(answer.get("citation_markers_used", [])),
                }
            )

        csv_path = self.artifacts.write_table_csv(f"{run_id}_manual_review_bundle.csv", review_rows)
        return {
            "run_id": run_id,
            "manual_review_bundle_csv": str(csv_path),
            "item_count": len(review_rows),
        }

    def _load_run_source_documents(self, run_id: str) -> list[dict[str, Any]]:
        inventory_path = self.artifacts.run_path(run_id, "inputs", "source_inventory.json")
        if not inventory_path.exists():
            return []
        payload = json.loads(inventory_path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, list) else []

    def _load_run_manifest_payload(self, run_id: str) -> dict[str, Any]:
        manifest_path = self.artifacts.run_path(run_id, "manifest.json")
        if not manifest_path.exists():
            return {}
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}

    def _extract_study_reference(self, source_doc: dict[str, Any]) -> dict[str, Any] | None:
        source_path = Path(str(source_doc.get("path", "")).strip())
        if not source_path.exists():
            return None
        try:
            payload = json.loads(source_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        protocol_section = payload.get("protocolSection", {})
        if not isinstance(protocol_section, dict):
            return None
        identification = protocol_section.get("identificationModule", {})
        status_module = protocol_section.get("statusModule", {})
        description_module = protocol_section.get("descriptionModule", {})
        conditions_module = protocol_section.get("conditionsModule", {})
        design_module = protocol_section.get("designModule", {})
        design_info = design_module.get("designInfo", {}) if isinstance(design_module.get("designInfo"), dict) else {}
        arms_module = protocol_section.get("armsInterventionsModule", {})
        eligibility_module = protocol_section.get("eligibilityModule", {})
        outcomes_module = protocol_section.get("outcomesModule", {})
        document_section = payload.get("documentSection", {})

        interventions = []
        for item in arms_module.get("interventions", []) if isinstance(arms_module, dict) else []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            interventions.append(
                {
                    "type": str(item.get("type") or "").strip(),
                    "name": name,
                    "description": str(item.get("description") or "").strip(),
                }
            )

        arms = []
        for item in arms_module.get("armGroups", []) if isinstance(arms_module, dict) else []:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label") or "").strip()
            if not label:
                continue
            arms.append(
                {
                    "label": label,
                    "description": str(item.get("description") or item.get("type") or "").strip(),
                }
            )

        primary_outcomes = []
        for item in outcomes_module.get("primaryOutcomes", []) if isinstance(outcomes_module, dict) else []:
            if not isinstance(item, dict):
                continue
            measure = str(item.get("measure") or "").strip()
            if not measure:
                continue
            primary_outcomes.append(
                {
                    "measure": measure,
                    "description": str(item.get("description") or "").strip(),
                    "time_frame": str(item.get("timeFrame") or "").strip(),
                }
            )

        large_docs = (
            document_section.get("largeDocumentModule", {}).get("largeDocs", [])
            if isinstance(document_section, dict) and isinstance(document_section.get("largeDocumentModule"), dict)
            else []
        )
        available_documents = []
        for item in large_docs if isinstance(large_docs, list) else []:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label") or "").strip()
            if not label:
                continue
            available_documents.append(
                {
                    "label": label,
                    "has_icf": bool(item.get("hasIcf")),
                    "has_protocol": bool(item.get("hasProtocol")),
                    "has_sap": bool(item.get("hasSap")),
                }
            )

        return {
            "source_id": source_doc.get("source_id"),
            "title": str(identification.get("briefTitle") or identification.get("officialTitle") or source_doc.get("title") or "").strip(),
            "nct_id": str(identification.get("nctId") or source_doc.get("metadata", {}).get("nct_id") or "").strip(),
            "official_title": str(identification.get("officialTitle") or "").strip(),
            "brief_title": str(identification.get("briefTitle") or "").strip(),
            "overall_status": str(status_module.get("overallStatus") or "").strip(),
            "study_type": str(design_module.get("studyType") or "").strip(),
            "phases": [str(item).strip() for item in design_module.get("phases", []) if str(item).strip()] if isinstance(design_module, dict) else [],
            "brief_summary": str(description_module.get("briefSummary") or "").strip(),
            "conditions": [str(item).strip() for item in conditions_module.get("conditions", []) if str(item).strip()] if isinstance(conditions_module, dict) else [],
            "keywords": [str(item).strip() for item in conditions_module.get("keywords", []) if str(item).strip()] if isinstance(conditions_module, dict) else [],
            "allocation": str(design_info.get("allocation") or "").strip(),
            "intervention_model": str(design_info.get("interventionModel") or "").strip(),
            "masking": str(design_info.get("maskingInfo", {}).get("masking") or "").strip() if isinstance(design_info.get("maskingInfo"), dict) else "",
            "primary_purpose": str(design_info.get("primaryPurpose") or "").strip(),
            "enrollment_count": design_module.get("enrollmentInfo", {}).get("count") if isinstance(design_module.get("enrollmentInfo"), dict) else None,
            "enrollment_type": str(design_module.get("enrollmentInfo", {}).get("type") or "").strip() if isinstance(design_module.get("enrollmentInfo"), dict) else "",
            "interventions": interventions,
            "arms": arms,
            "eligibility_criteria": str(eligibility_module.get("eligibilityCriteria") or "").strip() if isinstance(eligibility_module, dict) else "",
            "primary_outcomes": primary_outcomes,
            "available_documents": available_documents,
            "source_path": str(source_path),
            "source_url": source_doc.get("metadata", {}).get("url"),
        }

    def resolve_study_reference_for_source_ids(
        self,
        run_id: str,
        source_id_filters: list[str] | None,
    ) -> dict[str, Any] | None:
        wanted_ids = {str(item).strip().lower() for item in (source_id_filters or []) if str(item).strip()}
        if not wanted_ids:
            return None

        source_documents = self._load_run_source_documents(run_id)
        for doc in source_documents:
            source_id = str(doc.get("source_id", "")).strip().lower()
            nct_id = str(doc.get("metadata", {}).get("nct_id", "")).strip().lower()
            if source_id in wanted_ids or nct_id in wanted_ids:
                return self._extract_study_reference(doc)
        return None

    def build_study_query_context(
        self,
        run_id: str,
        source_id_filters: list[str] | None,
    ) -> dict[str, Any]:
        study_reference = self.resolve_study_reference_for_source_ids(run_id, source_id_filters)
        manifest_payload = self._load_run_manifest_payload(run_id)
        run_notes = str(manifest_payload.get("notes", "")).strip()

        selected_source_ids = [str(item).strip().lower() for item in (source_id_filters or []) if str(item).strip()]
        if not study_reference and not selected_source_ids:
            return {}

        intervention_names = [
            str(item.get("name", "")).strip()
            for item in (study_reference.get("interventions", []) if isinstance(study_reference, dict) else [])
            if isinstance(item, dict) and str(item.get("name", "")).strip()
        ][:3]
        conditions = [
            str(item).strip()
            for item in (study_reference.get("conditions", []) if isinstance(study_reference, dict) else [])
            if str(item).strip()
        ][:3]
        keywords = [
            str(item).strip()
            for item in (study_reference.get("keywords", []) if isinstance(study_reference, dict) else [])
            if str(item).strip()
        ][:3]

        query_terms: list[str] = []
        for value in (
            *(selected_source_ids or []),
            str(study_reference.get("source_id", "")).strip() if isinstance(study_reference, dict) else "",
            str(study_reference.get("nct_id", "")).strip() if isinstance(study_reference, dict) else "",
            str(study_reference.get("title", "")).strip() if isinstance(study_reference, dict) else "",
            str(study_reference.get("brief_title", "")).strip() if isinstance(study_reference, dict) else "",
            str(study_reference.get("official_title", "")).strip() if isinstance(study_reference, dict) else "",
            str(study_reference.get("primary_purpose", "")).strip() if isinstance(study_reference, dict) else "",
            str(study_reference.get("study_type", "")).strip() if isinstance(study_reference, dict) else "",
            str(study_reference.get("brief_summary", "")).strip()[:240] if isinstance(study_reference, dict) else "",
            *conditions,
            *keywords,
            *intervention_names,
            run_notes,
        ):
            text = str(value).strip()
            if not text:
                continue
            if text not in query_terms:
                query_terms.append(text)

        return {
            "selected_source_ids": selected_source_ids,
            "study_reference": study_reference,
            "run_notes": run_notes,
            "conditions": conditions,
            "keywords": keywords,
            "intervention_names": intervention_names,
            "query_terms": " | ".join(query_terms),
        }

    def _build_regulatory_reference_checklist(
        self,
        *,
        run_id: str,
        source_documents: list[dict[str, Any]],
    ) -> dict[str, Any]:
        chunk_path = self.artifacts.run_path(run_id, "outputs", "chunk_records.jsonl")
        chunks = load_chunk_records(chunk_path)
        lookup = {str(doc.get("source_id", "")).strip(): doc for doc in source_documents}
        checklist_rows: list[dict[str, Any]] = []

        for item in REFERENCE_CHECKLIST:
            query = " ".join(item["queries"])
            hits = retrieve_lexical_hits(chunks, query=query, top_k=12)
            filtered_hits = [
                hit
                for hit in hits
                if str(hit.metadata.get("source_group", "")).strip() in set(item["preferred_source_groups"])
            ][:3]
            evidence = []
            for hit in filtered_hits:
                source_doc = lookup.get(hit.source_id, {})
                evidence.append(
                    {
                        "source_id": hit.source_id,
                        "source_group": hit.metadata.get("source_group"),
                        "citation_label": hit.citation_label,
                        "excerpt": hit.excerpt,
                        "source_path": source_doc.get("path"),
                        "source_url": source_doc.get("metadata", {}).get("url"),
                    }
                )

            checklist_rows.append(
                {
                    "element_id": item["element_id"],
                    "description": item["description"],
                    "preferred_source_groups": item["preferred_source_groups"],
                    "queries": item["queries"],
                    "reference_evidence": evidence,
                }
            )

        return {
            "checklist": checklist_rows,
            "element_count": len(checklist_rows),
        }

    def export_evaluation_reference_pack(
        self,
        run_id: str,
        *,
        source_id: str | None = None,
    ) -> dict[str, Any]:
        run_dir = self.artifacts.runs_dir / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

        source_documents = self._load_run_source_documents(run_id)
        if not source_documents:
            raise FileNotFoundError(
                "Source inventory not found for this run. Build the corpus first so the reference pack has source metadata."
            )

        study_documents = [
            doc for doc in source_documents
            if str(doc.get("source_type", "")).strip() == "study_record"
        ]
        selected_study_doc = None
        if source_id:
            wanted = source_id.strip().lower()
            for doc in study_documents:
                current = str(doc.get("source_id", "")).strip().lower()
                nct_id = str(doc.get("metadata", {}).get("nct_id", "")).strip().lower()
                if current == wanted or nct_id == wanted:
                    selected_study_doc = doc
                    break
        elif len(study_documents) == 1:
            selected_study_doc = study_documents[0]

        study_reference = self._extract_study_reference(selected_study_doc) if selected_study_doc else None
        regulatory_reference = self._build_regulatory_reference_checklist(run_id=run_id, source_documents=source_documents)

        posted_consent_forms = []
        for doc in source_documents:
            if str(doc.get("metadata", {}).get("source_group", "")).strip() != "posted_consent_forms":
                continue
            posted_consent_forms.append(
                {
                    "source_id": doc.get("source_id"),
                    "title": doc.get("title"),
                    "source_type": doc.get("source_type"),
                    "path": doc.get("path"),
                    "authority": doc.get("metadata", {}).get("authority"),
                    "url": doc.get("metadata", {}).get("url"),
                }
            )

        rubric_path = self.config.paths.configs_root / "manual_review_rubric.json"
        rubric_payload = json.loads(rubric_path.read_text(encoding="utf-8")) if rubric_path.exists() else {}

        reference_pack = {
            "run_id": run_id,
            "created_at": utc_now_iso(),
            "source_inventory_summary": {
                "source_count": len(source_documents),
                "source_group_counts": dict(
                    sorted(
                        Counter(
                            str(doc.get("metadata", {}).get("source_group", "")).strip() or "unknown"
                            for doc in source_documents
                        ).items()
                    )
                ),
                "source_type_counts": dict(
                    sorted(
                        Counter(
                            str(doc.get("source_type", "")).strip() or "unknown"
                            for doc in source_documents
                        ).items()
                    )
                ),
            },
            "study_reference": study_reference,
            "regulatory_reference": regulatory_reference,
            "posted_consent_form_references": posted_consent_forms,
            "manual_review_reference": {
                "rubric_path": str(rubric_path),
                "rubric": rubric_payload,
            },
            "comparison_guidance": {
                "study_facts_reference": "Use the study_reference section for factual checks about the specific trial.",
                "regulatory_completeness_reference": "Use the regulatory_reference checklist to assess whether required informed-consent topics are covered.",
                "document_style_reference": "Use the posted_consent_form_references as realistic participant-facing wording and structure references.",
                "human_review_reference": "Use the manual_review_reference rubric for qualitative review of draft, QA, and structured outputs.",
            },
        }

        output_path = self.artifacts.run_path(run_id, "outputs", "evaluation_reference_pack.json")
        self.artifacts.write_json(output_path, reference_pack)
        self.artifacts.record_stage(
            run_id,
            PipelineStageRecord(
                stage_name="export_evaluation_reference_pack",
                status="completed",
                started_at=utc_now_iso(),
                ended_at=utc_now_iso(),
                inputs={"source_id": source_id},
                outputs={
                    "reference_pack_path": str(output_path),
                    "posted_consent_form_count": len(posted_consent_forms),
                    "regulatory_checklist_count": regulatory_reference["element_count"],
                    "study_reference_present": bool(study_reference),
                },
                notes=(
                    "Exported a reusable evaluation reference pack combining study facts, "
                    "regulatory checklist evidence, posted consent-form references, and the manual review rubric."
                ),
            ),
        )
        return {
            "run_id": run_id,
            "reference_pack_path": str(output_path),
            "posted_consent_form_count": len(posted_consent_forms),
            "regulatory_checklist_count": regulatory_reference["element_count"],
            "study_reference_present": bool(study_reference),
        }

    def find_download_manifest_path(self, source_dir: Path) -> Path | None:
        current = source_dir.resolve()
        source_root = self.config.paths.source_data_root.resolve()
        while True:
            candidate = current / "manifests" / "download_manifest.json"
            if candidate.exists():
                return candidate
            if current == source_root or current.parent == current:
                break
            current = current.parent
        return None

    def load_download_manifest_lookup(self, source_dir: Path) -> dict[str, dict[str, Any]]:
        manifest_path = self.find_download_manifest_path(source_dir)
        if manifest_path is None:
            return {}
        payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        lookup: dict[str, dict[str, Any]] = {}
        for item in payload.get("items", []):
            saved_path = item.get("saved_path")
            if not isinstance(saved_path, str) or not saved_path.strip():
                continue
            lookup[str(Path(saved_path).resolve())] = item
        return lookup

    def create_case_run_from_corpus(
        self,
        *,
        base_run_id: str,
        purpose: str,
        tags: list[str] | None = None,
        notes: str = "",
        study_id: str | None = None,
        site_id: str | None = None,
    ) -> RunManifest:
        base_run_dir = self.artifacts.runs_dir / base_run_id
        if not base_run_dir.exists():
            raise FileNotFoundError(f"Base run directory does not exist: {base_run_dir}")

        manifest = self.initialize_run(
            purpose=purpose,
            tags=tags,
            notes=notes,
            study_id=study_id,
            site_id=site_id,
        )
        run_id = manifest.run_id
        started_at = utc_now_iso()

        copied_paths: list[str] = []
        file_relpaths = [
            Path("inputs") / "source_inventory.json",
            Path("inputs") / "base_template.txt",
            Path("outputs") / "source_text_units.jsonl",
            Path("outputs") / "chunk_records.jsonl",
            Path("outputs") / "corpus_summary.json",
            Path("outputs") / "skipped_sources.json",
        ]
        for relpath in file_relpaths:
            source_path = base_run_dir / relpath
            if not source_path.exists():
                continue
            destination = self.artifacts.run_path(run_id, *relpath.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination)
            copied_paths.append(str(relpath))

        hybrid_index_source = base_run_dir / "outputs" / "hybrid_index"
        if hybrid_index_source.exists():
            hybrid_index_destination = self.artifacts.run_path(run_id, "outputs", "hybrid_index")
            shutil.copytree(hybrid_index_source, hybrid_index_destination, dirs_exist_ok=True)
            copied_paths.append("outputs/hybrid_index")

        self.artifacts.record_stage(
            run_id,
            PipelineStageRecord(
                stage_name="attach_prepared_corpus",
                status="completed",
                started_at=started_at,
                ended_at=utc_now_iso(),
                inputs={"base_run_id": base_run_id},
                outputs={
                    "copied_path_count": len(copied_paths),
                    "copied_paths": copied_paths,
                },
                notes=(
                    "Created an isolated case run by copying the prepared corpus artifacts and any hybrid index "
                    "from the selected base run."
                ),
            ),
        )
        return manifest

    def run_batch_experiment(
        self,
        spec_path: Path,
        *,
        dry_run: bool = False,
        base_run_id_override: str | None = None,
    ) -> dict[str, Any]:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        batch_label = str(spec.get("batch_id") or "batch_experiment").strip() or "batch_experiment"
        base_run_id = str(base_run_id_override or spec.get("base_run_id", "")).strip()
        reporting_role = str(spec.get("reporting_role", "evaluation")).strip().lower() or "evaluation"
        if not base_run_id:
            raise ValueError("Batch spec must include a non-empty base_run_id.")
        defaults = spec.get("defaults", {})
        if not isinstance(defaults, dict):
            defaults = {}

        batch_run_id = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
        batch_dir = self.artifacts.batches_dir / f"{batch_run_id}-{re.sub(r'[^a-z0-9]+', '_', batch_label.lower()).strip('_')}"
        batch_dir.mkdir(parents=True, exist_ok=True)

        resolved_spec = {
            **spec,
            "batch_run_id": batch_run_id,
            "batch_dir": str(batch_dir),
            "dry_run": dry_run,
            "spec_path": str(spec_path.resolve()),
            "base_run_id_override": base_run_id_override,
        }
        batch_runtime_metadata = self.build_runtime_metadata(
            config_path=str(spec_path.resolve()),
            base_run_id=base_run_id,
        )
        expanded_cases = self.expand_batch_case_definitions(
            spec=spec,
            spec_path=spec_path,
            defaults=defaults,
        )
        resolved_spec["expanded_case_count"] = len(expanded_cases)
        (batch_dir / "batch_spec.resolved.json").write_text(json.dumps(resolved_spec, indent=2), encoding="utf-8")

        case_records: list[dict[str, Any]] = []
        metric_rows: list[dict[str, Any]] = []

        for raw_case in expanded_cases:
            case_id = str(raw_case.get("case_id", "")).strip()

            patient_profile_file_value = raw_case.get("patient_profile_file", defaults.get("patient_profile_file"))
            if not patient_profile_file_value:
                raise ValueError(f"Batch case '{case_id}' is missing patient_profile_file.")
            patient_profile_file = self.resolve_project_relative_path(
                str(patient_profile_file_value),
                base_dir=spec_path.parent,
            )
            template_file_value = raw_case.get("template_file", defaults.get("template_file"))
            template_file = (
                self.resolve_project_relative_path(
                    str(template_file_value),
                    base_dir=spec_path.parent,
                )
                if template_file_value
                else None
            )
            questions, question_set_path = self.resolve_question_list(
                questions_value=raw_case.get("questions", defaults.get("questions", [])),
                question_set_file_value=raw_case.get("question_set_file", defaults.get("question_set_file")),
                base_dir=spec_path.parent,
            )
            patient_profile_label = patient_profile_file.stem
            question_set_label = question_set_path.stem if question_set_path else ""
            generation_query = raw_case.get("generation_query", defaults.get("generation_query"))
            top_k = raw_case.get("top_k", defaults.get("top_k"))
            retrieval_source_groups = normalize_string_list(
                raw_case.get("retrieval_source_groups", defaults.get("retrieval_source_groups", []))
            )
            retrieval_source_ids = normalize_string_list(
                raw_case.get("retrieval_source_ids", defaults.get("retrieval_source_ids", []))
            )
            retrieval_filter_logic = self.normalize_filter_logic(
                raw_case.get("retrieval_filter_logic", defaults.get("retrieval_filter_logic"))
            )
            workflow_variant = self.normalize_workflow_variant(
                raw_case.get("workflow_variant", defaults.get("workflow_variant"))
            )
            retrieval_mode = str(raw_case.get("retrieval_mode", defaults.get("retrieval_mode", self.config.retrieval.retrieval_mode))).strip().lower()
            if retrieval_mode not in {"lexical", "dense", "hybrid"}:
                retrieval_mode = self.config.retrieval.retrieval_mode
            effective_retrieval_mode = "none" if workflow_variant == "vanilla_llm" else retrieval_mode
            generate_draft = bool(raw_case.get("generate_draft", defaults.get("generate_draft", True)))
            formalize = bool(raw_case.get("formalize", defaults.get("formalize", True)))
            case_notes = str(raw_case.get("notes", defaults.get("notes", "")))
            study_source_id = self.normalize_source_id_value(
                raw_case.get("study_source_id")
                or (retrieval_source_ids[0] if retrieval_source_ids else "")
            )
            study_identity = str(raw_case.get("study_id", "")).strip() or (study_source_id.upper() if study_source_id else self.config.study_id)
            site_identity = str(raw_case.get("site_id", "")).strip() or "PUBLIC-SOURCE"

            case_manifest = self.create_case_run_from_corpus(
                base_run_id=base_run_id,
                purpose=f"batch_case:{batch_label}:{case_id}",
                tags=["batch_experiment", batch_label, case_id],
                notes=case_notes,
                study_id=study_identity,
                site_id=site_identity,
            )
            case_run_id = case_manifest.run_id
            case_context_metadata = {
                "case_id": case_id,
                "batch_run_id": batch_run_id,
                "batch_id": batch_label,
                "reporting_role": reporting_role,
                "base_run_id": base_run_id,
                "config_path": str(spec_path.resolve()),
                "study_source_id": study_source_id,
                "workflow_variant": workflow_variant,
                "patient_profile_label": patient_profile_label,
                "question_set_label": question_set_label,
                "patient_profile_file": str(patient_profile_file),
                "question_set_file": str(question_set_path) if question_set_path else None,
                "template_file": str(template_file) if template_file else None,
                "retrieval_mode": effective_retrieval_mode,
                "retrieval_top_k": top_k or self.config.retrieval.top_k,
                "retrieval_filter_logic": retrieval_filter_logic,
                "retrieval_source_groups": retrieval_source_groups,
                "retrieval_source_ids": retrieval_source_ids,
                "dry_run": dry_run,
            }
            self.artifacts.update_run_manifest(
                case_run_id,
                {
                    "runtime_metadata": batch_runtime_metadata,
                    "context_metadata": case_context_metadata,
                },
            )

            case_result: dict[str, Any] = {
                "case_id": case_id,
                "case_run_id": case_run_id,
                "study_id": study_identity,
                "site_id": site_identity,
                "study_source_id": study_source_id,
                "reporting_role": reporting_role,
                "patient_profile_file": str(patient_profile_file),
                "patient_profile_label": patient_profile_label,
                "template_file": str(template_file) if template_file else None,
                "question_set_file": str(question_set_path) if question_set_path else None,
                "question_set_label": question_set_label,
                "question_count": len(questions),
                "dry_run": dry_run,
                "status": "pending",
                "workflow_variant": workflow_variant,
                "retrieval_mode": effective_retrieval_mode,
                "retrieval_top_k": top_k or self.config.retrieval.top_k,
                "retrieval_source_groups": retrieval_source_groups,
                "retrieval_source_ids": retrieval_source_ids,
                "retrieval_filter_logic": retrieval_filter_logic,
                "model_id": batch_runtime_metadata.get("model_id"),
                "embedding_model_id": batch_runtime_metadata.get("embedding_model_id"),
                "config_path": batch_runtime_metadata.get("config_path"),
                "corpus_version": batch_runtime_metadata.get("corpus_version"),
                "index_version": batch_runtime_metadata.get("index_version"),
                "random_seed": batch_runtime_metadata.get("random_seed"),
                "git_commit_hash": batch_runtime_metadata.get("git_commit_hash"),
            }
            try:
                if generate_draft:
                    draft_payload = self.draft_personalized_consent(
                        run_id=case_run_id,
                        patient_profile_path=patient_profile_file,
                        template_path=template_file,
                        generation_query=generation_query,
                        top_k=top_k,
                        retrieval_mode=retrieval_mode,
                        source_group_filters=retrieval_source_groups,
                        source_id_filters=retrieval_source_ids,
                        filter_logic=retrieval_filter_logic,
                        workflow_variant=workflow_variant,
                        dry_run=dry_run,
                    )
                    case_result["draft_output_path"] = draft_payload.get("output_path")
                    case_result["retrieval_hits_path"] = draft_payload.get("retrieval_hits_path")

                if formalize and not dry_run:
                    formalized_payload = self.formalize_consent(
                        run_id=case_run_id,
                        patient_profile_path=patient_profile_file,
                        dry_run=False,
                    )
                    case_result["structured_record_path"] = formalized_payload.get("output_path")

                qa_results: list[dict[str, Any]] = []
                for question in questions:
                    qa_payload = self.answer_consent_question(
                        run_id=case_run_id,
                        question=question,
                        patient_profile_path=patient_profile_file,
                        top_k=top_k,
                        retrieval_mode=retrieval_mode,
                        source_group_filters=retrieval_source_groups,
                        source_id_filters=retrieval_source_ids,
                        filter_logic=retrieval_filter_logic,
                        workflow_variant=workflow_variant,
                        dry_run=dry_run,
                    )
                    qa_results.append(
                        {
                            "question_id": qa_payload["question_id"],
                            "question": question,
                            "output_path": qa_payload.get("output_path"),
                        }
                    )
                case_result["qa_results"] = qa_results

                evaluation_payload = self.evaluate_run(case_run_id)
                case_result["evaluation_summary_path"] = evaluation_payload["summary_path"]
                review_payload = self.export_manual_review_bundle(case_run_id)
                case_result["manual_review_bundle_csv"] = review_payload["manual_review_bundle_csv"]
                case_result["status"] = "completed"

                summary = evaluation_payload["summary"]
                summary_metadata = summary.get("metadata", {})
                if not isinstance(summary_metadata, dict):
                    summary_metadata = {}
                case_result["model_id"] = summary_metadata.get("model_id") or case_result.get("model_id")
                case_result["embedding_model_id"] = summary_metadata.get("embedding_model_id") or case_result.get("embedding_model_id")
                case_result["draft_system_prompt_id"] = summary_metadata.get("draft_system_prompt_id")
                case_result["draft_user_prompt_id"] = summary_metadata.get("draft_user_prompt_id")
                case_result["formalization_system_prompt_id"] = summary_metadata.get("formalization_system_prompt_id")
                case_result["formalization_user_prompt_id"] = summary_metadata.get("formalization_user_prompt_id")
                case_result["qa_system_prompt_ids"] = summary_metadata.get("qa_system_prompt_ids", [])
                case_result["qa_user_prompt_ids"] = summary_metadata.get("qa_user_prompt_ids", [])
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
                qa_avg_fkg = None
                if qa_per_question:
                    grade_values = [
                        item.get("flesch_kincaid_grade")
                        for item in qa_per_question
                        if isinstance(item.get("flesch_kincaid_grade"), (int, float))
                    ]
                    if grade_values:
                        qa_avg_fkg = round(sum(grade_values) / len(grade_values), 4)
                case_result["failure_taxonomy"] = failure_taxonomy
                metric_rows.append(
                    {
                        "batch_run_id": batch_run_id,
                        "batch_id": batch_label,
                        "base_run_id": base_run_id,
                        "case_id": case_id,
                        "case_run_id": case_run_id,
                        "study_id": study_identity,
                        "study_source_id": study_source_id,
                        "patient_profile_label": patient_profile_label,
                        "question_set_label": question_set_label,
                        "workflow_variant": workflow_variant,
                        "reporting_role": reporting_role,
                        "status": case_result["status"],
                        "config_path": summary_metadata.get("config_path") or batch_runtime_metadata.get("config_path"),
                        "git_commit_hash": summary_metadata.get("git_commit_hash") or batch_runtime_metadata.get("git_commit_hash"),
                        "model_id": summary_metadata.get("model_id") or batch_runtime_metadata.get("model_id"),
                        "embedding_model_id": summary_metadata.get("embedding_model_id") or batch_runtime_metadata.get("embedding_model_id"),
                        "corpus_version": summary_metadata.get("corpus_version") or batch_runtime_metadata.get("corpus_version"),
                        "index_version": summary_metadata.get("index_version") or batch_runtime_metadata.get("index_version"),
                        "random_seed": summary_metadata.get("random_seed") or batch_runtime_metadata.get("random_seed"),
                        "retrieval_mode": effective_retrieval_mode,
                        "retrieval_top_k": summary_metadata.get("retrieval_top_k") or (top_k or self.config.retrieval.top_k),
                        "retrieval_filter_logic": retrieval_filter_logic,
                        "retrieval_source_groups": "|".join(retrieval_source_groups),
                        "retrieval_source_ids": "|".join(retrieval_source_ids),
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
            except Exception as exc:
                case_result["status"] = "failed"
                case_result["error"] = str(exc)
                metric_rows.append(
                    {
                        "batch_run_id": batch_run_id,
                        "batch_id": batch_label,
                        "base_run_id": base_run_id,
                        "case_id": case_id,
                        "case_run_id": case_run_id,
                        "study_id": study_identity,
                        "study_source_id": study_source_id,
                        "patient_profile_label": patient_profile_label,
                        "question_set_label": question_set_label,
                        "workflow_variant": workflow_variant,
                        "reporting_role": reporting_role,
                        "status": case_result["status"],
                        "config_path": batch_runtime_metadata.get("config_path"),
                        "git_commit_hash": batch_runtime_metadata.get("git_commit_hash"),
                        "model_id": batch_runtime_metadata.get("model_id"),
                        "embedding_model_id": batch_runtime_metadata.get("embedding_model_id"),
                        "corpus_version": batch_runtime_metadata.get("corpus_version"),
                        "index_version": batch_runtime_metadata.get("index_version"),
                        "random_seed": batch_runtime_metadata.get("random_seed"),
                        "retrieval_mode": effective_retrieval_mode,
                        "retrieval_top_k": top_k or self.config.retrieval.top_k,
                        "retrieval_filter_logic": retrieval_filter_logic,
                        "retrieval_source_groups": "|".join(retrieval_source_groups),
                        "retrieval_source_ids": "|".join(retrieval_source_ids),
                        "question_count": len(questions),
                        "qa_answered_count": None,
                        "qa_answered_question_count": None,
                        "qa_abstained_count": None,
                        "qa_abstained_question_count": None,
                        "qa_clarified_count": None,
                        "qa_abstention_rate": None,
                        "qa_uncertainty_flag_count": None,
                        "qa_uncertainty_rate": None,
                        "qa_unsupported_marker_count": None,
                        "qa_unsupported_sentence_count": None,
                        "qa_citationless_sentence_count": None,
                        "qa_citationless_sentence_rate": None,
                        "qa_selected_study_hit_count": None,
                        "qa_selected_study_hit_present": None,
                        "qa_foreign_study_hit_count": None,
                        "qa_foreign_study_hit_present": None,
                        "qa_regulatory_hit_count": None,
                        "qa_total_hit_count": None,
                        "qa_study_specific_grounding_met": None,
                        "qa_study_specific_grounding_gap": None,
                        "qa_grounding_source_ids_used": None,
                        "qa_foreign_source_ids_detected": None,
                        "draft_expected_study_specific_grounding": None,
                        "draft_study_specific_grounding_met": None,
                        "draft_study_specific_grounding_gap": None,
                        "draft_selected_study_hit_count": None,
                        "draft_selected_study_hit_present": None,
                        "draft_foreign_study_hit_count": None,
                        "draft_foreign_study_hit_present": None,
                        "draft_study_specific_hit_count": None,
                        "draft_regulatory_hit_count": None,
                        "draft_total_hit_count": None,
                        "draft_has_study_specific_evidence": None,
                        "draft_study_specific_hit_ratio": None,
                        "draft_grounding_source_ids_used": None,
                        "draft_foreign_source_ids_detected": None,
                        "draft_required_element_coverage_ratio": None,
                        "draft_missing_required_element_count": None,
                        "draft_citation_marker_coverage_ratio": None,
                        "draft_sentence_citation_coverage_ratio": None,
                        "draft_citationless_sentence_count": None,
                        "draft_citationless_sentence_rate": None,
                        "draft_unsupported_marker_count": None,
                        "draft_unsupported_sentence_count": None,
                        "draft_flesch_kincaid_grade": None,
                        "draft_grounding_gap_declared": None,
                        "draft_unsupported_claim_risk": None,
                        "structured_schema_repair_applied": None,
                        "structured_required_field_presence_ratio": None,
                        "structured_malformed_output": None,
                        "qa_average_citation_marker_coverage_ratio": None,
                        "qa_average_sentence_citation_coverage_ratio": None,
                        "qa_average_flesch_kincaid_grade": None,
                        "failure_missing_selected_study_grounding": None,
                        "failure_foreign_study_contamination": None,
                        "failure_regulatory_only_grounding": None,
                        "failure_unsupported_claim_risk": None,
                        "failure_omitted_required_element": None,
                        "failure_overconfident_answer": None,
                        "failure_malformed_structured_output": None,
                        "failure_grounding_gap_declared": None,
                    }
                )
            case_records.append(case_result)

        summary_payload = {
            "batch_run_id": batch_run_id,
            "batch_id": batch_label,
            "base_run_id": base_run_id,
            "reporting_role": reporting_role,
            "config_path": str(spec_path.resolve()),
            "model_id": batch_runtime_metadata.get("model_id"),
            "embedding_model_id": batch_runtime_metadata.get("embedding_model_id"),
            "corpus_version": batch_runtime_metadata.get("corpus_version"),
            "index_version": batch_runtime_metadata.get("index_version"),
            "git_commit_hash": batch_runtime_metadata.get("git_commit_hash"),
            "random_seed": batch_runtime_metadata.get("random_seed"),
            "case_count": len(case_records),
            "completed_case_count": sum(1 for case in case_records if case.get("status") == "completed"),
            "failed_case_count": sum(1 for case in case_records if case.get("status") == "failed"),
            "dry_run": dry_run,
            "cases": case_records,
        }
        (batch_dir / "batch_summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
        with (batch_dir / "batch_cases.jsonl").open("w", encoding="utf-8") as handle:
            for record in case_records:
                handle.write(json.dumps(record))
                handle.write("\n")

        metrics_csv_path = self.artifacts.write_table_csv(f"{batch_run_id}_batch_case_metrics.csv", metric_rows)
        summary_payload["case_metrics_csv"] = str(metrics_csv_path)
        completed_rows = [row for row in metric_rows if row.get("status") == "completed"]
        aggregate_metrics: dict[str, Any] = {}
        for key in (
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
        ):
            numeric_values = [
                float(row[key])
                for row in completed_rows
                if isinstance(row.get(key), (int, float))
            ]
            aggregate_metrics[f"average_{key}"] = (
                round(sum(numeric_values) / len(numeric_values), 4)
                if numeric_values
                else None
            )
        schema_repairs = [
            row.get("structured_schema_repair_applied")
            for row in completed_rows
            if isinstance(row.get("structured_schema_repair_applied"), bool)
        ]
        aggregate_metrics["structured_schema_repair_rate"] = (
            round(sum(1 for value in schema_repairs if value) / len(schema_repairs), 4)
            if schema_repairs
            else None
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
        expected_grounding_rows = [
            row
            for row in completed_rows
            if row.get("draft_expected_study_specific_grounding") is True
        ]
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
        for failure_field in (
            "failure_missing_selected_study_grounding",
            "failure_foreign_study_contamination",
            "failure_regulatory_only_grounding",
            "failure_unsupported_claim_risk",
            "failure_omitted_required_element",
            "failure_overconfident_answer",
            "failure_malformed_structured_output",
            "failure_grounding_gap_declared",
        ):
            flags = [
                row.get(failure_field)
                for row in completed_rows
                if isinstance(row.get(failure_field), bool)
            ]
            aggregate_metrics[f"{failure_field}_rate"] = (
                round(sum(1 for value in flags if value) / len(flags), 4)
                if flags
                else None
            )
        summary_payload["workflow_variants"] = sorted(
            {
                str(case.get("workflow_variant", "")).strip()
                for case in case_records
                if str(case.get("workflow_variant", "")).strip()
            }
        )
        summary_payload["study_source_ids"] = sorted(
            {
                str(case.get("study_source_id", "")).strip()
                for case in case_records
                if str(case.get("study_source_id", "")).strip()
            }
        )
        summary_payload["study_ids"] = sorted(
            {
                str(case.get("study_id", "")).strip()
                for case in case_records
                if str(case.get("study_id", "")).strip()
            }
        )
        summary_payload["patient_profile_labels"] = sorted(
            {
                str(case.get("patient_profile_label", "")).strip()
                for case in case_records
                if str(case.get("patient_profile_label", "")).strip()
            }
        )
        summary_payload["question_set_labels"] = sorted(
            {
                str(case.get("question_set_label", "")).strip()
                for case in case_records
                if str(case.get("question_set_label", "")).strip()
            }
        )
        summary_payload["retrieval_modes"] = sorted(
            {
                str(case.get("retrieval_mode", "")).strip()
                for case in case_records
                if str(case.get("retrieval_mode", "")).strip()
            }
        )
        summary_payload["retrieval_top_k_values"] = sorted(
            {
                int(case.get("retrieval_top_k"))
                for case in case_records
                if isinstance(case.get("retrieval_top_k"), int)
            }
        )
        summary_payload["retrieval_filter_logics"] = sorted(
            {
                str(case.get("retrieval_filter_logic", "")).strip()
                for case in case_records
                if str(case.get("retrieval_filter_logic", "")).strip()
            }
        )
        summary_payload["draft_system_prompt_ids"] = sorted(
            {
                str(case.get("draft_system_prompt_id", "")).strip()
                for case in case_records
                if str(case.get("draft_system_prompt_id", "")).strip()
            }
        )
        summary_payload["draft_user_prompt_ids"] = sorted(
            {
                str(case.get("draft_user_prompt_id", "")).strip()
                for case in case_records
                if str(case.get("draft_user_prompt_id", "")).strip()
            }
        )
        summary_payload["formalization_system_prompt_ids"] = sorted(
            {
                str(case.get("formalization_system_prompt_id", "")).strip()
                for case in case_records
                if str(case.get("formalization_system_prompt_id", "")).strip()
            }
        )
        summary_payload["formalization_user_prompt_ids"] = sorted(
            {
                str(case.get("formalization_user_prompt_id", "")).strip()
                for case in case_records
                if str(case.get("formalization_user_prompt_id", "")).strip()
            }
        )
        summary_payload["qa_system_prompt_ids"] = sorted(
            {
                str(prompt_id).strip()
                for case in case_records
                for prompt_id in (case.get("qa_system_prompt_ids", []) if isinstance(case.get("qa_system_prompt_ids"), list) else [])
                if str(prompt_id).strip()
            }
        )
        summary_payload["qa_user_prompt_ids"] = sorted(
            {
                str(prompt_id).strip()
                for case in case_records
                for prompt_id in (case.get("qa_user_prompt_ids", []) if isinstance(case.get("qa_user_prompt_ids"), list) else [])
                if str(prompt_id).strip()
            }
        )

        failure_summary_rows: list[dict[str, Any]] = []
        failure_fields = [
            "failure_missing_selected_study_grounding",
            "failure_foreign_study_contamination",
            "failure_regulatory_only_grounding",
            "failure_unsupported_claim_risk",
            "failure_omitted_required_element",
            "failure_overconfident_answer",
            "failure_malformed_structured_output",
            "failure_grounding_gap_declared",
        ]
        grouped_completed_rows: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in completed_rows:
            key = (
                str(row.get("workflow_variant", "")).strip() or "unknown",
                str(row.get("question_set_label", "")).strip() or "unknown",
            )
            grouped_completed_rows.setdefault(key, []).append(row)
        for (workflow_value, question_set_value), rows_for_group in sorted(grouped_completed_rows.items()):
            case_count = len(rows_for_group)
            for failure_field in failure_fields:
                flags = [bool(row.get(failure_field)) for row in rows_for_group if isinstance(row.get(failure_field), bool)]
                failure_count = sum(1 for flag in flags if flag)
                failure_summary_rows.append(
                    {
                        "batch_run_id": batch_run_id,
                        "batch_id": batch_label,
                        "workflow_variant": workflow_value,
                        "question_set_label": question_set_value,
                        "failure_type": failure_field.removeprefix("failure_"),
                        "case_count": case_count,
                        "failure_count": failure_count,
                        "failure_rate": round(failure_count / max(case_count, 1), 4),
                    }
                )
        failure_summary_csv = self.artifacts.write_table_csv(f"{batch_run_id}_batch_failure_summary.csv", failure_summary_rows)
        summary_payload["failure_summary_csv"] = str(failure_summary_csv)
        summary_payload["aggregate_metrics"] = aggregate_metrics
        (batch_dir / "batch_summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
        return summary_payload

    def compare_batch_results(self, batch_summary_paths: list[Path], *, comparison_id: str | None = None) -> dict[str, Any]:
        if len(batch_summary_paths) < 2:
            raise ValueError("At least two batch summary files are required for comparison.")

        def parse_metric_value(value: Any) -> float | bool | None:
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return float(value)
            text = str(value or "").strip()
            if not text:
                return None
            lowered = text.lower()
            if lowered in {"true", "false"}:
                return lowered == "true"
            try:
                return float(text)
            except ValueError:
                return None

        rows: list[dict[str, Any]] = []
        case_rows: list[dict[str, Any]] = []
        for summary_path in batch_summary_paths:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            aggregate_metrics = payload.get("aggregate_metrics", {})
            if not isinstance(aggregate_metrics, dict):
                aggregate_metrics = {}
            batch_id = payload.get("batch_id")
            batch_run_id = payload.get("batch_run_id")
            reporting_role = payload.get("reporting_role")
            case_metrics_csv = payload.get("case_metrics_csv")
            rows.append(
                {
                    "batch_id": batch_id,
                    "batch_run_id": batch_run_id,
                    "reporting_role": reporting_role,
                    "case_count": payload.get("case_count"),
                    "completed_case_count": payload.get("completed_case_count"),
                    "failed_case_count": payload.get("failed_case_count"),
                    "workflow_variants": "|".join(payload.get("workflow_variants", []) or []),
                    "study_source_ids": "|".join(payload.get("study_source_ids", []) or []),
                    "study_ids": "|".join(payload.get("study_ids", []) or []),
                    "patient_profile_labels": "|".join(payload.get("patient_profile_labels", []) or []),
                    "question_set_labels": "|".join(payload.get("question_set_labels", []) or []),
                    "model_id": payload.get("model_id"),
                    "embedding_model_id": payload.get("embedding_model_id"),
                    "retrieval_modes": "|".join(payload.get("retrieval_modes", []) or []),
                    "retrieval_top_k_values": "|".join(str(value) for value in (payload.get("retrieval_top_k_values", []) or [])),
                    "retrieval_filter_logics": "|".join(payload.get("retrieval_filter_logics", []) or []),
                    "draft_system_prompt_ids": "|".join(payload.get("draft_system_prompt_ids", []) or []),
                    "draft_user_prompt_ids": "|".join(payload.get("draft_user_prompt_ids", []) or []),
                    "formalization_system_prompt_ids": "|".join(payload.get("formalization_system_prompt_ids", []) or []),
                    "formalization_user_prompt_ids": "|".join(payload.get("formalization_user_prompt_ids", []) or []),
                    "qa_system_prompt_ids": "|".join(payload.get("qa_system_prompt_ids", []) or []),
                    "qa_user_prompt_ids": "|".join(payload.get("qa_user_prompt_ids", []) or []),
                    "config_path": payload.get("config_path"),
                    "corpus_version": payload.get("corpus_version"),
                    "index_version": payload.get("index_version"),
                    "git_commit_hash": payload.get("git_commit_hash"),
                    "random_seed": payload.get("random_seed"),
                    **aggregate_metrics,
                    "batch_summary_path": str(summary_path.resolve()),
                    "case_metrics_csv": case_metrics_csv,
                }
            )
            if case_metrics_csv:
                case_metrics_path = Path(str(case_metrics_csv))
                if case_metrics_path.exists():
                    with case_metrics_path.open("r", encoding="utf-8", newline="") as handle:
                        reader = csv.DictReader(handle)
                        for case_row in reader:
                            case_rows.append(
                                {
                                    **case_row,
                                    "batch_id": batch_id,
                                    "batch_run_id": batch_run_id,
                                    "reporting_role": reporting_role,
                                }
                            )

        comparison_label = re.sub(r"[^a-z0-9]+", "_", (comparison_id or "batch_comparison").lower()).strip("_") or "batch_comparison"
        csv_path = self.artifacts.write_table_csv(f"{comparison_label}.csv", rows)
        json_path = self.artifacts.tables_dir / f"{comparison_label}.json"
        case_csv_path = self.artifacts.write_table_csv(f"{comparison_label}_case_rows.csv", case_rows)

        aggregate_metric_fields = [
            "draft_selected_study_hit_present",
            "draft_foreign_study_hit_present",
            "draft_regulatory_hit_count",
            "draft_study_specific_grounding_met",
            "draft_study_specific_grounding_gap",
            "draft_required_element_coverage_ratio",
            "draft_sentence_citation_coverage_ratio",
            "draft_citationless_sentence_rate",
            "draft_unsupported_sentence_count",
            "draft_flesch_kincaid_grade",
            "qa_answered_count",
            "qa_abstained_count",
            "qa_clarified_count",
            "qa_abstention_rate",
            "qa_uncertainty_flag_count",
            "qa_uncertainty_rate",
            "qa_citationless_sentence_rate",
            "qa_unsupported_sentence_count",
            "qa_selected_study_hit_present",
            "qa_foreign_study_hit_present",
            "qa_study_specific_grounding_met",
            "qa_study_specific_grounding_gap",
            "qa_average_sentence_citation_coverage_ratio",
            "qa_average_flesch_kincaid_grade",
            "structured_required_field_presence_ratio",
            "structured_schema_repair_applied",
            "structured_malformed_output",
            "failure_missing_selected_study_grounding",
            "failure_foreign_study_contamination",
            "failure_regulatory_only_grounding",
            "failure_unsupported_claim_risk",
            "failure_omitted_required_element",
            "failure_overconfident_answer",
            "failure_malformed_structured_output",
            "failure_grounding_gap_declared",
        ]
        failure_fields = [
            "failure_missing_selected_study_grounding",
            "failure_foreign_study_contamination",
            "failure_regulatory_only_grounding",
            "failure_unsupported_claim_risk",
            "failure_omitted_required_element",
            "failure_overconfident_answer",
            "failure_malformed_structured_output",
            "failure_grounding_gap_declared",
        ]

        def build_grouped_rows(grouping_fields: tuple[str, ...]) -> list[dict[str, Any]]:
            grouped_values: dict[tuple[str, ...], dict[str, list[float]]] = {}
            grouped_counts: Counter[tuple[str, ...]] = Counter()
            grouped_metadata: dict[tuple[str, ...], dict[str, Any]] = {}
            for row in case_rows:
                key = tuple(str(row.get(field, "")).strip() or "unknown" for field in grouping_fields)
                grouped_counts[key] += 1
                grouped_values.setdefault(key, {})
                grouped_metadata.setdefault(
                    key,
                    {
                        "model_id": str(row.get("model_id", "")).strip() or None,
                        "embedding_model_id": str(row.get("embedding_model_id", "")).strip() or None,
                        "retrieval_mode": str(row.get("retrieval_mode", "")).strip() or None,
                        "retrieval_top_k": row.get("retrieval_top_k"),
                        "retrieval_filter_logic": str(row.get("retrieval_filter_logic", "")).strip() or None,
                        "draft_system_prompt_id": str(row.get("draft_system_prompt_id", "")).strip() or None,
                        "draft_user_prompt_id": str(row.get("draft_user_prompt_id", "")).strip() or None,
                        "formalization_system_prompt_id": str(row.get("formalization_system_prompt_id", "")).strip() or None,
                        "formalization_user_prompt_id": str(row.get("formalization_user_prompt_id", "")).strip() or None,
                        "qa_system_prompt_ids": str(row.get("qa_system_prompt_ids", "")).strip() or None,
                        "qa_user_prompt_ids": str(row.get("qa_user_prompt_ids", "")).strip() or None,
                        "config_path": str(row.get("config_path", "")).strip() or None,
                        "git_commit_hash": str(row.get("git_commit_hash", "")).strip() or None,
                        "corpus_version": str(row.get("corpus_version", "")).strip() or None,
                        "index_version": str(row.get("index_version", "")).strip() or None,
                        "random_seed": str(row.get("random_seed", "")).strip() or None,
                    },
                )
                for field in aggregate_metric_fields:
                    parsed = parse_metric_value(row.get(field))
                    if parsed is None:
                        continue
                    numeric_value = 1.0 if parsed is True else 0.0 if parsed is False else float(parsed)
                    grouped_values[key].setdefault(field, []).append(numeric_value)

            grouped_rows_local: list[dict[str, Any]] = []
            for key, field_map in sorted(grouped_values.items()):
                grouped_row: dict[str, Any] = {
                    "case_count": grouped_counts[key],
                    **grouped_metadata.get(key, {}),
                }
                for index, field_name in enumerate(grouping_fields):
                    grouped_row[field_name] = key[index]
                for field in aggregate_metric_fields:
                    values = field_map.get(field, [])
                    grouped_row[f"average_{field}"] = round(sum(values) / len(values), 4) if values else None
                grouped_rows_local.append(grouped_row)
            return grouped_rows_local

        grouped_rows = build_grouped_rows(("workflow_variant", "question_set_label"))
        grouped_csv_path = self.artifacts.write_table_csv(f"{comparison_label}_grouped_by_workflow_and_question_set.csv", grouped_rows)
        workflow_rows = build_grouped_rows(("workflow_variant",))
        workflow_csv_path = self.artifacts.write_table_csv(f"{comparison_label}_grouped_by_workflow.csv", workflow_rows)

        failure_summary_rows: list[dict[str, Any]] = []
        grouped_case_sets: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in case_rows:
            key = (
                str(row.get("workflow_variant", "")).strip() or "unknown",
                str(row.get("question_set_label", "")).strip() or "unknown",
            )
            grouped_case_sets.setdefault(key, []).append(row)
        for (workflow_variant, question_set_label), grouped_case_rows in sorted(grouped_case_sets.items()):
            case_count = len(grouped_case_rows)
            for failure_field in failure_fields:
                flags = [bool(parse_metric_value(row.get(failure_field))) for row in grouped_case_rows if parse_metric_value(row.get(failure_field)) is not None]
                failure_count = sum(1 for flag in flags if flag)
                failure_summary_rows.append(
                    {
                        "workflow_variant": workflow_variant,
                        "question_set_label": question_set_label,
                        "failure_type": failure_field.removeprefix("failure_"),
                        "case_count": case_count,
                        "failure_count": failure_count,
                        "failure_rate": round(failure_count / max(case_count, 1), 4),
                    }
                )
        failure_summary_csv = self.artifacts.write_table_csv(f"{comparison_label}_failure_summary.csv", failure_summary_rows)
        self.artifacts.write_json(
            json_path,
            {
                "comparison_id": comparison_label,
                "aggregate_rows": rows,
                "case_row_count": len(case_rows),
                "grouped_row_count": len(grouped_rows),
                "workflow_row_count": len(workflow_rows),
                "failure_summary_row_count": len(failure_summary_rows),
            },
        )
        return {
            "comparison_id": comparison_label,
            "row_count": len(rows),
            "comparison_csv": str(csv_path),
            "comparison_json": str(json_path),
            "case_rows_csv": str(case_csv_path),
            "grouped_comparison_csv": str(grouped_csv_path),
            "overall_workflow_csv": str(workflow_csv_path),
            "failure_summary_csv": str(failure_summary_csv),
        }

    def plan_public_sources(
        self,
        *,
        registry_path: Path | None = None,
        group_ids: list[str] | None = None,
        source_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        registry = self.source_registry.load(registry_path)
        plan = self.source_registry.plan(
            registry_path,
            group_ids=set(group_ids or []),
            source_ids=set(source_ids or []),
        )
        return {
            "registry_path": str(self.source_registry.resolve_path(registry_path)),
            "registry_version": registry.get("version"),
            "item_count": len(plan),
            "items": [asdict(item) for item in plan],
        }

    def download_public_sources(
        self,
        *,
        registry_path: Path | None = None,
        output_dir: Path | None = None,
        group_ids: list[str] | None = None,
        source_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        registry = self.source_registry.load(registry_path)
        plan = self.source_registry.plan(
            registry_path,
            group_ids=set(group_ids or []),
            source_ids=set(source_ids or []),
        )
        resolved_output_dir = (output_dir or (self.config.paths.source_data_root / "raw" / "public")).resolve()
        return download_plan_items(
            plan,
            output_root=resolved_output_dir,
            registry_snapshot=registry,
        )

    def fetch_clinicaltrials_studies(
        self,
        *,
        output_dir: Path | None = None,
        nct_ids: list[str] | None = None,
        query_term: str | None = None,
        max_studies: int = 10,
        page_size: int = 10,
    ) -> dict[str, Any]:
        resolved_output_dir = (output_dir or (self.config.paths.source_data_root / "raw" / "public")).resolve()
        return fetch_and_save_studies(
            output_root=resolved_output_dir,
            nct_ids=nct_ids,
            query_term=query_term,
            max_studies=max_studies,
            page_size=page_size,
        )

    def inventory_source_directory(self, source_dir: Path) -> list[ConsentSourceDocument]:
        documents: list[ConsentSourceDocument] = []
        manifest_lookup = self.load_download_manifest_lookup(source_dir)
        for path in sorted(source_dir.rglob("*")):
            if not path.is_file():
                continue
            relative_path = path.relative_to(source_dir)
            lower_parts = [part.lower() for part in relative_path.parts]
            if "manifests" in lower_parts:
                continue
            if path.name.lower() in {"readme.md", "readme.txt", ".gitkeep"}:
                continue
            manifest_item = manifest_lookup.get(str(path.resolve()), {})
            source_group = relative_path.parts[0] if len(relative_path.parts) > 1 else source_dir.name
            documents.append(
                ConsentSourceDocument(
                    source_id=str(manifest_item.get("source_id", "")).strip() or slugify_source_id(relative_path),
                    title=str(manifest_item.get("title", "")).strip() or path.name,
                    source_type=str(manifest_item.get("source_type", "")).strip() or path.suffix.lstrip(".").lower() or "unknown",
                    path=str(path.resolve()),
                    sha256=compute_file_sha256(path),
                    byte_size=path.stat().st_size,
                    metadata={
                        "relative_path": str(relative_path),
                        "source_group": source_group,
                        **{
                            key: manifest_item[key]
                              for key in (
                                  "authority",
                                  "url",
                                  "api_url",
                                  "source_type",
                                  "content_type",
                                  "downloaded_at",
                                  "download_status",
                                  "group_id",
                                  "nct_id",
                                  "overall_status",
                                  "study_type",
                                  "has_results",
                                  "record_format",
                                  "query_term",
                              )
                              if key in manifest_item
                          },
                      },
                )
            )
        return documents

    def load_patient_profile(self, patient_profile_path: Path) -> PatientProfile | dict[str, Any]:
        raw = patient_profile_path.read_text(encoding="utf-8")
        payload = json.loads(raw)
        if isinstance(payload, dict):
            return PatientProfile(**payload)
        return {"raw_profile": payload}

    def resolve_patient_profile(self, run_id: str, patient_profile_path: Path | None) -> PatientProfile:
        candidate = patient_profile_path or self.artifacts.run_path(run_id, "inputs", "patient_profile.json")
        if not candidate.exists():
            raise FileNotFoundError(
                "Patient profile is required. Provide --patient-profile-file or store inputs/patient_profile.json in the run."
            )
        loaded = self.load_patient_profile(candidate)
        if isinstance(loaded, PatientProfile):
            return loaded
        return PatientProfile(notes={"raw_profile": loaded})

    def resolve_base_template_text(self, run_id: str, template_path: Path | None) -> str:
        candidate = template_path or self.artifacts.run_path(run_id, "inputs", "base_template.txt")
        if candidate.exists():
            return candidate.read_text(encoding="utf-8").strip()
        return ""

    def build_personalization_query(self, patient_profile: PatientProfile, base_template_text: str) -> str:
        template_excerpt = base_template_text[:500].replace("\n", " ").strip()
        profile_terms = [
            f"language {patient_profile.language}",
            f"health literacy {patient_profile.health_literacy}",
            f"jurisdiction {patient_profile.jurisdiction}",
            "clinical trial informed consent",
            "study purpose procedures risks benefits alternatives confidentiality voluntary participation withdrawal rights participant questions",
        ]
        if template_excerpt:
            profile_terms.append(template_excerpt)
        return " | ".join(profile_terms)

    def format_retrieval_context(
        self,
        retrieval_hits: list[dict[str, Any]],
        *,
        marker_lookup: dict[str, str] | None = None,
    ) -> str:
        blocks: list[str] = []
        for index, hit in enumerate(retrieval_hits, start=1):
            marker = marker_lookup.get(hit["chunk_id"], f"[{index}]") if marker_lookup else f"[{index}]"
            blocks.append(
                "\n".join(
                    [
                        f"{marker} {hit['citation_label']}",
                        hit["excerpt"],
                    ]
                )
            )
        return "\n\n".join(blocks)

    def classify_retrieval_hit_role(self, hit: dict[str, Any]) -> str:
        source_group = str(hit.get("metadata", {}).get("source_group", "")).strip().lower()
        if source_group == "trial_materials":
            return "study_specific"
        if source_group == "regulatory_guidance":
            return "regulatory"
        return "other"

    def build_role_separated_evidence_package(self, retrieval_hits: list[dict[str, Any]]) -> dict[str, Any]:
        grouped_hits: dict[str, list[dict[str, Any]]] = {
            "study_specific": [],
            "regulatory": [],
            "other": [],
        }
        for hit in retrieval_hits:
            grouped_hits[self.classify_retrieval_hit_role(hit)].append(hit)

        citation_map = self.build_citation_map(retrieval_hits)
        marker_lookup = {entry["chunk_id"]: entry["marker"] for entry in citation_map}
        study_context = self.format_retrieval_context(grouped_hits["study_specific"], marker_lookup=marker_lookup)
        regulatory_context = self.format_retrieval_context(grouped_hits["regulatory"], marker_lookup=marker_lookup)
        other_context = self.format_retrieval_context(grouped_hits["other"], marker_lookup=marker_lookup)

        def role_citation_entries(role_hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
            allowed_chunk_ids = {hit["chunk_id"] for hit in role_hits}
            return [entry for entry in citation_map if entry["chunk_id"] in allowed_chunk_ids]

        return {
            "study_specific_hits": grouped_hits["study_specific"],
            "regulatory_hits": grouped_hits["regulatory"],
            "other_hits": grouped_hits["other"],
            "study_specific_citation_map": role_citation_entries(grouped_hits["study_specific"]),
            "regulatory_citation_map": role_citation_entries(grouped_hits["regulatory"]),
            "other_citation_map": role_citation_entries(grouped_hits["other"]),
            "citation_map": citation_map,
            "study_specific_context": study_context,
            "regulatory_context": regulatory_context,
            "other_context": other_context,
            "combined_context": self.format_retrieval_context(retrieval_hits, marker_lookup=marker_lookup),
            "role_counts": {
                "study_specific": len(grouped_hits["study_specific"]),
                "regulatory": len(grouped_hits["regulatory"]),
                "other": len(grouped_hits["other"]),
            },
        }

    def build_readability_guidance(self, patient_profile: PatientProfile, *, task_type: str) -> str:
        literacy = patient_profile.health_literacy.lower().strip()
        if task_type == "qa":
            if literacy == "low":
                return (
                    "Aim for about a U.S. grade 6 to 7 reading level. Answer in 2 to 3 short sentences. "
                    "Prefer about 8 to 12 words per sentence. Use one idea per sentence. Use everyday words, "
                    "avoid jargon, and explain any necessary study term in plain language."
                )
            if literacy == "medium":
                return (
                    "Aim for about a U.S. grade 8 reading level. Answer in 2 to 3 concise sentences. "
                    "Prefer about 10 to 16 words per sentence. Use one idea per sentence and define unfamiliar study terms briefly."
                )
            return (
                "Keep the answer concise, direct, and understandable to a general participant audience."
            )

        if literacy == "low":
            return (
                "Aim for about a U.S. grade 6 reading level. Use 6 to 8 short sentences or bullet-like lines. "
                "Keep most sentences under 14 words when possible. Cover these topics explicitly: choice, what happens in the study, "
                "risks, possible benefits or lack of guaranteed benefit, other options, questions/contact, and stopping later. "
                "Use everyday words, avoid legalistic wording, and explain any needed study term in plain language."
            )
        if literacy == "medium":
            return (
                "Aim for about a U.S. grade 8 reading level. Use 6 to 8 concise sentences or short bullet-like lines. "
                "Keep sentences direct, cover all core consent topics explicitly, and define unfamiliar terms briefly."
            )
        return "Keep the language concise, participant-facing, and easy to follow."

    def build_question_id(self, question: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "_", question.lower()).strip("_")
        slug = slug[:40] or "question"
        suffix = sha1(question.encode("utf-8")).hexdigest()[:8]
        return f"{slug}_{suffix}"

    def normalize_questions(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    def normalize_source_id_value(self, value: Any) -> str:
        candidate = str(value or "").strip().lower()
        candidate = re.sub(r"[^a-z0-9]+", "_", candidate)
        return candidate.strip("_")

    def resolve_project_relative_path(self, value: str | Path, *, base_dir: Path | None = None) -> Path:
        candidate = Path(value)
        if candidate.is_absolute():
            return candidate
        if base_dir is not None:
            candidate_from_spec = (base_dir / candidate).resolve()
            if candidate_from_spec.exists():
                return candidate_from_spec
        return (self.config.paths.project_root / candidate).resolve()

    def resolve_question_list(
        self,
        *,
        questions_value: Any,
        question_set_file_value: Any,
        base_dir: Path,
    ) -> tuple[list[str], Path | None]:
        direct_questions = self.normalize_questions(questions_value)
        if direct_questions:
            return direct_questions, None

        if not question_set_file_value:
            return [], None

        question_set_path = self.resolve_project_relative_path(str(question_set_file_value), base_dir=base_dir)
        payload = json.loads(question_set_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return self.normalize_questions(payload), question_set_path
        if isinstance(payload, dict):
            return self.normalize_questions(payload.get("questions", [])), question_set_path
        raise ValueError(f"Unsupported question-set file structure: {question_set_path}")

    def load_study_cohort(
        self,
        *,
        cohort_file_value: Any,
        base_dir: Path,
    ) -> tuple[list[dict[str, Any]], Path | None]:
        if not cohort_file_value:
            return [], None

        cohort_path = self.resolve_project_relative_path(str(cohort_file_value), base_dir=base_dir)
        payload = json.loads(cohort_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Unsupported study cohort file structure: {cohort_path}")

        studies_payload = payload.get("studies", [])
        if not isinstance(studies_payload, list):
            raise ValueError(f"Study cohort file must define a studies list: {cohort_path}")

        studies: list[dict[str, Any]] = []
        for raw_study in studies_payload:
            if not isinstance(raw_study, dict):
                continue
            source_id = self.normalize_source_id_value(raw_study.get("source_id"))
            if not source_id:
                raise ValueError(f"Study cohort entries must define a non-empty source_id: {cohort_path}")
            studies.append(
                {
                    **raw_study,
                    "source_id": source_id,
                }
            )
        return studies, cohort_path

    def expand_batch_case_definitions(
        self,
        *,
        spec: dict[str, Any],
        spec_path: Path,
        defaults: dict[str, Any],
    ) -> list[dict[str, Any]]:
        expanded_cases: list[dict[str, Any]] = []
        seen_case_ids: set[str] = set()

        for raw_case in spec.get("cases", []):
            if not isinstance(raw_case, dict):
                continue
            case_id = str(raw_case.get("case_id", "")).strip()
            if not case_id:
                raise ValueError("Each batch case must define a non-empty case_id.")
            if case_id in seen_case_ids:
                raise ValueError(f"Duplicate batch case_id detected: {case_id}")
            seen_case_ids.add(case_id)
            expanded_cases.append(raw_case)

        case_matrix = spec.get("case_matrix")
        if not case_matrix:
            return expanded_cases
        if not isinstance(case_matrix, dict):
            raise ValueError("case_matrix must be a JSON object when provided.")

        study_entries, cohort_path = self.load_study_cohort(
            cohort_file_value=case_matrix.get("study_cohort_file"),
            base_dir=spec_path.parent,
        )
        direct_study_source_ids = normalize_string_list(case_matrix.get("study_source_ids", []))
        if not study_entries and direct_study_source_ids:
            study_entries = [{"source_id": source_id} for source_id in direct_study_source_ids]
        if not study_entries:
            raise ValueError("case_matrix requires either study_cohort_file or study_source_ids.")

        patient_profile_files = normalize_string_list(
            case_matrix.get("patient_profile_files", defaults.get("patient_profile_files", []))
        )
        question_set_files = normalize_string_list(
            case_matrix.get("question_set_files", defaults.get("question_set_files", []))
        )
        if not patient_profile_files:
            raise ValueError("case_matrix must define patient_profile_files or inherit them from defaults.")
        if not question_set_files:
            raise ValueError("case_matrix must define question_set_files or inherit them from defaults.")

        matrix_notes = str(case_matrix.get("notes", "")).strip()

        for study in study_entries:
            source_id = self.normalize_source_id_value(study.get("source_id"))
            if not source_id:
                continue
            study_notes = str(study.get("notes", "")).strip()
            study_template = study.get("template_file", case_matrix.get("template_file"))
            study_generation_query = study.get("generation_query", case_matrix.get("generation_query"))
            study_top_k = study.get("top_k", case_matrix.get("top_k"))
            study_retrieval_mode = study.get("retrieval_mode", case_matrix.get("retrieval_mode"))
            study_workflow_variant = study.get("workflow_variant", case_matrix.get("workflow_variant"))
            study_filter_logic = study.get("retrieval_filter_logic", case_matrix.get("retrieval_filter_logic"))
            study_source_groups = normalize_string_list(
                study.get("retrieval_source_groups", case_matrix.get("retrieval_source_groups", []))
            )
            explicit_source_ids = normalize_string_list(study.get("retrieval_source_ids", []))
            retrieval_source_ids = explicit_source_ids or [source_id]
            generate_draft = study.get("generate_draft", case_matrix.get("generate_draft"))
            formalize = study.get("formalize", case_matrix.get("formalize"))

            for patient_profile_file in patient_profile_files:
                profile_label = Path(patient_profile_file).stem
                for question_set_file in question_set_files:
                    question_label = Path(question_set_file).stem
                    case_id = f"{source_id}_{profile_label}_{question_label}".lower()
                    if case_id in seen_case_ids:
                        raise ValueError(f"Duplicate batch case_id detected after matrix expansion: {case_id}")
                    seen_case_ids.add(case_id)
                    combined_notes = " ".join(part for part in (matrix_notes, study_notes) if part).strip()
                    case_payload: dict[str, Any] = {
                        "case_id": case_id,
                        "study_source_id": source_id,
                        "study_id": source_id.upper(),
                        "site_id": str(study.get("site_id", case_matrix.get("site_id", "PUBLIC-SOURCE"))).strip() or "PUBLIC-SOURCE",
                        "patient_profile_file": patient_profile_file,
                        "question_set_file": question_set_file,
                        "retrieval_source_ids": retrieval_source_ids,
                        "notes": combined_notes,
                        "study_cohort_file": str(cohort_path) if cohort_path else None,
                    }
                    if study_template is not None:
                        case_payload["template_file"] = study_template
                    if study_generation_query is not None:
                        case_payload["generation_query"] = study_generation_query
                    if study_top_k is not None:
                        case_payload["top_k"] = study_top_k
                    if study_retrieval_mode is not None:
                        case_payload["retrieval_mode"] = study_retrieval_mode
                    if study_workflow_variant is not None:
                        case_payload["workflow_variant"] = study_workflow_variant
                    if study_filter_logic is not None:
                        case_payload["retrieval_filter_logic"] = study_filter_logic
                    if study_source_groups:
                        case_payload["retrieval_source_groups"] = study_source_groups
                    if generate_draft is not None:
                        case_payload["generate_draft"] = generate_draft
                    if formalize is not None:
                        case_payload["formalize"] = formalize
                    expanded_cases.append(case_payload)

        return expanded_cases

    def build_citation_map(self, retrieval_hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        citation_map: list[dict[str, Any]] = []
        for index, hit in enumerate(retrieval_hits, start=1):
            citation_map.append(
                {
                    "marker": f"[{index}]",
                    "source_id": hit["source_id"],
                    "chunk_id": hit["chunk_id"],
                    "label": hit["citation_label"],
                    "excerpt": hit["excerpt"],
                    "metadata": hit.get("metadata", {}),
                }
            )
        return citation_map

    def upsert_qa_index_entry(self, qa_index_path: Path, entry: dict[str, Any]) -> None:
        existing_entries: list[dict[str, Any]] = []
        if qa_index_path.exists():
            with qa_index_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    existing_entries.append(json.loads(line))

        updated_entries: list[dict[str, Any]] = []
        replaced = False
        for existing in existing_entries:
            if existing.get("question_id") == entry.get("question_id"):
                updated_entries.append(entry)
                replaced = True
            else:
                updated_entries.append(existing)
        if not replaced:
            updated_entries.append(entry)

        self.artifacts.write_jsonl(qa_index_path, updated_entries)

    def resolve_personalized_draft(self, run_id: str, draft_path: Path | None) -> dict[str, Any]:
        candidate = draft_path or self.artifacts.run_path(run_id, "outputs", "personalized_consent_draft.json")
        if not candidate.exists():
            raise FileNotFoundError(
                "Personalized consent draft is required. Generate it first or provide --draft-file."
            )
        payload = json.loads(candidate.read_text(encoding="utf-8"))
        if "personalized_consent_text" not in payload:
            return self.normalize_personalized_draft_response(payload)
        return payload

    def split_text_into_sentences(self, text: str) -> list[str]:
        if not text.strip():
            return []
        chunks = re.findall(r"[^.!?]+(?:[.!?]+|$)", text)
        return [chunk.strip() for chunk in chunks if chunk and chunk.strip()]

    def append_marker_to_sentence(self, sentence: str, marker: str) -> str:
        if re.search(r"\[\d+\]", sentence):
            return sentence
        trimmed = sentence.rstrip()
        punctuation = ""
        while trimmed and trimmed[-1] in ".!?":
            punctuation = trimmed[-1] + punctuation
            trimmed = trimmed[:-1].rstrip()
        punctuation = punctuation or "."
        return f"{trimmed} {marker}{punctuation}"

    def inject_inline_citations(self, text: str, markers: list[str], *, max_sentences: int | None = None) -> str:
        normalized_markers = [str(marker).strip() for marker in markers if str(marker).strip()]
        if not text.strip() or not normalized_markers:
            return text

        existing_markers = set(re.findall(r"\[\d+\]", text))
        prioritized_markers = [marker for marker in normalized_markers if marker not in existing_markers]
        if not prioritized_markers:
            prioritized_markers = normalized_markers

        sentences = self.split_text_into_sentences(text)
        if not sentences:
            return text

        candidate_indices = [
            index
            for index, sentence in enumerate(sentences)
            if not re.search(r"\[\d+\]", sentence) and len(sentence.split()) >= 4
        ]
        if not candidate_indices:
            return text

        limit = max_sentences or len(normalized_markers)
        marker_index = 0
        updated_sentences = list(sentences)
        for sentence_index in candidate_indices[:limit]:
            marker = prioritized_markers[min(marker_index, len(prioritized_markers) - 1)]
            updated_sentences[sentence_index] = self.append_marker_to_sentence(updated_sentences[sentence_index], marker)
            marker_index += 1
            if marker_index >= len(prioritized_markers):
                break

        return " ".join(updated_sentences)

    def normalize_personalized_draft_response(self, payload: dict[str, Any]) -> dict[str, Any]:
        key_information_summary = None
        for key in (
            "key_information_summary",
            "summary",
            "key_summary",
            "consent_summary",
        ):
            candidate = payload.get(key)
            if isinstance(candidate, str) and candidate.strip():
                key_information_summary = candidate.strip()
                break

        text = None
        for key in (
            "personalized_consent_text",
            "consent_text",
            "summary",
            "consent_summary",
        ):
            candidate = payload.get(key)
            if isinstance(candidate, str) and candidate.strip():
                text = candidate.strip()
                break

        if not text:
            raise RuntimeError(f"Model response did not contain recognizable consent text fields: {list(payload.keys())}")
        if not key_information_summary:
            sentences = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", text) if segment.strip()]
            key_information_summary = " ".join(sentences[:2]) if sentences else text

        citation_markers = payload.get("citation_markers_used")
        if not isinstance(citation_markers, list):
            citation_markers = sorted(set(re.findall(r"\[\d+\]", text)))
        else:
            normalized_markers: list[str] = []
            for item in citation_markers:
                marker = str(item).strip()
                if re.fullmatch(r"\d+", marker):
                    marker = f"[{marker}]"
                normalized_markers.append(marker)
            citation_markers = normalized_markers

        summary_citation_markers = payload.get("key_information_citation_markers_used")
        if not isinstance(summary_citation_markers, list):
            summary_citation_markers = sorted(set(re.findall(r"\[\d+\]", key_information_summary)))
        else:
            normalized_summary_markers: list[str] = []
            for item in summary_citation_markers:
                marker = str(item).strip()
                if re.fullmatch(r"\d+", marker):
                    marker = f"[{marker}]"
                normalized_summary_markers.append(marker)
            summary_citation_markers = normalized_summary_markers

        rationale = payload.get("personalization_rationale")
        if not isinstance(rationale, list):
            rationale = []
        else:
            rationale = [str(item) for item in rationale]

        limitations = payload.get("grounding_limitations")
        if not isinstance(limitations, list):
            limitations = []
        else:
            limitations = [str(item) for item in limitations]

        repair_notes: list[str] = []
        if "key_information_summary" not in payload:
            repair_notes.append("Recovered key_information_summary from an alternate field or derived it from the full draft.")
        if "key_information_citation_markers_used" not in payload:
            repair_notes.append("Recovered key_information_citation_markers_used from inline markers in the summary.")
        elif summary_citation_markers and not re.findall(r"\[\d+\]", key_information_summary):
            key_information_summary = self.inject_inline_citations(
                key_information_summary,
                summary_citation_markers,
                max_sentences=1,
            )
            repair_notes.append("Inserted summary citation markers inline because the model returned them only in a separate field.")
        if "personalized_consent_text" not in payload:
            repair_notes.append("Recovered personalized_consent_text from an alternate field name.")
        if "citation_markers_used" not in payload:
            repair_notes.append("Recovered citation_markers_used from inline citation markers in the draft text.")
        elif citation_markers and not re.findall(r"\[\d+\]", text):
            text = self.inject_inline_citations(
                text,
                citation_markers,
                max_sentences=max(1, min(len(citation_markers), 3)),
            )
            repair_notes.append("Inserted draft citation markers inline because the model returned them only in a separate field.")
        if "personalization_rationale" not in payload:
            repair_notes.append("Missing personalization_rationale; stored as an empty list.")
        if "grounding_limitations" not in payload:
            repair_notes.append("Missing grounding_limitations; stored as an empty list.")

        return {
            "key_information_summary": key_information_summary,
            "key_information_citation_markers_used": summary_citation_markers,
            "personalized_consent_text": text,
            "citation_markers_used": citation_markers,
            "personalization_rationale": rationale,
            "grounding_limitations": limitations,
            "schema_repair_notes": repair_notes,
            "raw_response_keys": sorted(payload.keys()),
        }

    def normalize_structured_consent_record(
        self,
        payload: dict[str, Any],
        *,
        personalized_draft: dict[str, Any],
        patient_profile: PatientProfile,
    ) -> dict[str, Any]:
        original_payload = payload
        wrapper_key = None
        for candidate_key in ("consent_record", "structured_consent_record", "record"):
            candidate_value = payload.get(candidate_key)
            if isinstance(candidate_value, dict):
                wrapper_key = candidate_key
                payload = candidate_value
                break

        draft_text = personalized_draft["personalized_consent_text"]
        key_information_summary = personalized_draft.get("key_information_summary", "")
        citation_markers = list(personalized_draft.get("citation_markers_used", [])) + list(
            personalized_draft.get("key_information_citation_markers_used", [])
        )
        sentences = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", draft_text) if segment.strip()]

        purposes = payload.get("purposes")
        if not isinstance(purposes, list):
            purposes = []
        purposes = [str(item) for item in purposes if str(item).strip()]
        if not purposes and re.search(r"clinical research study|clinical trial|research study", draft_text, re.IGNORECASE):
            purposes = ["Participation in the clinical research study"]

        study_purpose_summary = payload.get("study_purpose_summary")
        if not isinstance(study_purpose_summary, str) or not study_purpose_summary.strip():
            study_purpose_summary = next(
                (sentence for sentence in sentences if re.search(r"clinical research study|clinical trial|research study", sentence, re.IGNORECASE)),
                key_information_summary or None,
            )

        study_procedures_summary = payload.get("study_procedures_summary")
        if not isinstance(study_procedures_summary, str) or not study_procedures_summary.strip():
            study_procedures_summary = next(
                (
                    sentence
                    for sentence in sentences
                    if re.search(
                        r"study steps|study procedures|research team will explain|team will explain|what happens",
                        sentence,
                        re.IGNORECASE,
                    )
                ),
                None,
            )

        data_types = payload.get("data_types")
        if not isinstance(data_types, list):
            data_types = []
        data_types = [str(item) for item in data_types if str(item).strip()]

        risks_summary = payload.get("risks_summary")
        if not isinstance(risks_summary, str) or not risks_summary.strip():
            risks_summary = next((sentence for sentence in sentences if re.search(r"\brisks?\b", sentence, re.IGNORECASE)), None)

        benefits_summary = payload.get("benefits_summary")
        if not isinstance(benefits_summary, str) or not benefits_summary.strip():
            benefits_summary = next(
                (
                    sentence
                    for sentence in sentences
                    if re.search(
                        r"\bpossible benefits?\b|\bpotential benefits?\b|\brisks? and benefits?\b|\bbenefits? and risks?\b|\bbenefits? of (the )?(study|research)\b|\bhow (the )?study may help\b|no direct benefit",
                        sentence,
                        re.IGNORECASE,
                    )
                ),
                None,
            )

        alternatives_summary = payload.get("alternatives_summary")
        if not isinstance(alternatives_summary, str) or not alternatives_summary.strip():
            alternatives_summary = next(
                (sentence for sentence in sentences if re.search(r"\balternatives?\b|\bother options\b", sentence, re.IGNORECASE)),
                None,
            )

        valid_until = payload.get("valid_until")
        if valid_until is not None:
            valid_until = str(valid_until)

        withdrawal_policy = payload.get("withdrawal_policy")
        if not isinstance(withdrawal_policy, str) or not withdrawal_policy.strip():
            withdrawal_sentence = next(
                (
                    sentence
                    for sentence in sentences
                    if re.search(
                        r"stop participating|stop taking part|withdraw|penalty|loss of benefits|losing benefits|without being punished",
                        sentence,
                        re.IGNORECASE,
                    )
                ),
                None,
            )
            withdrawal_policy = withdrawal_sentence
        if withdrawal_policy is not None:
            withdrawal_policy = withdrawal_policy.strip()

        voluntary_participation_statement = payload.get("voluntary_participation_statement")
        if not isinstance(voluntary_participation_statement, str) or not voluntary_participation_statement.strip():
            voluntary_participation_statement = next(
                (
                    sentence
                    for sentence in sentences
                    if re.search(
                        r"your choice|voluntary|choose to join|choose whether to (join|take part)|taking part is your choice",
                        sentence,
                        re.IGNORECASE,
                    )
                ),
                None,
            )

        question_rights_summary = payload.get("question_rights_summary")
        if not isinstance(question_rights_summary, str) or not question_rights_summary.strip():
            question_rights_summary = next(
                (
                    sentence
                    for sentence in sentences
                    if re.search(r"ask questions|questions at any time|contact (the )?(study|research) team|contact the team", sentence, re.IGNORECASE)
                ),
                None,
            )

        withdrawal_rights_summary = payload.get("withdrawal_rights_summary")
        if not isinstance(withdrawal_rights_summary, str) or not withdrawal_rights_summary.strip():
            withdrawal_rights_summary = next(
                (
                    sentence
                    for sentence in sentences
                    if re.search(
                        r"stop participating|stop taking part|withdraw|penalty|loss of benefits|losing benefits|without being punished",
                        sentence,
                        re.IGNORECASE,
                    )
                ),
                withdrawal_policy,
            )

        participant_rights = payload.get("participant_rights")
        if not isinstance(participant_rights, list):
            participant_rights = []
        participant_rights = [str(item) for item in participant_rights if str(item).strip()]
        if not participant_rights:
            inferred_rights: list[str] = []
            if re.search(r"your choice|choice whether to take part|voluntary|choose to join|choose whether to (join|take part)", draft_text, re.IGNORECASE):
                inferred_rights.append("Participation is voluntary.")
            if re.search(r"ask questions|questions at any time|contact (the )?(study|research) team|contact the team", draft_text, re.IGNORECASE):
                inferred_rights.append("The participant may ask questions before deciding.")
            if re.search(r"stop participating|stop taking part|withdraw", draft_text, re.IGNORECASE):
                inferred_rights.append("The participant may stop participating at any time without penalty.")
            participant_rights = inferred_rights

        consent_summary = payload.get("consent_summary")
        if not isinstance(consent_summary, str) or not consent_summary.strip():
            consent_summary = " ".join(sentences[:2]) if sentences else None

        cited_markers = payload.get("cited_markers")
        if not isinstance(cited_markers, list):
            cited_markers = list(citation_markers)
        cited_markers = [str(item) for item in cited_markers if str(item).strip()]
        normalized_markers: list[str] = []
        for marker in cited_markers:
            cleaned = marker.strip()
            if re.fullmatch(r"\d+", cleaned):
                cleaned = f"[{cleaned}]"
            normalized_markers.append(cleaned)
        cited_markers = list(dict.fromkeys(normalized_markers or list(citation_markers)))

        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}

        repair_notes: list[str] = []
        if wrapper_key:
            repair_notes.append(f"Unwrapped structured record from top-level key '{wrapper_key}'.")
        if not payload:
            repair_notes.append("Model returned an empty object; applied heuristic fallback extraction.")
        if "purposes" not in payload:
            repair_notes.append("Missing purposes; inferred or defaulted from the personalized draft.")
        if "withdrawal_policy" not in payload:
            repair_notes.append("Missing withdrawal_policy; recovered from the draft text.")
        if "study_purpose_summary" not in payload:
            repair_notes.append("Missing study_purpose_summary; inferred from the draft text.")
        if "study_procedures_summary" not in payload:
            repair_notes.append("Missing study_procedures_summary; inferred from the draft text.")
        if "risks_summary" not in payload:
            repair_notes.append("Missing risks_summary; inferred from the draft text when possible.")
        if "benefits_summary" not in payload:
            repair_notes.append("Missing benefits_summary; inferred from the draft text when possible.")
        if "alternatives_summary" not in payload:
            repair_notes.append("Missing alternatives_summary; inferred from the draft text when possible.")
        if "question_rights_summary" not in payload:
            repair_notes.append("Missing question_rights_summary; inferred from the draft text.")
        if "voluntary_participation_statement" not in payload:
            repair_notes.append("Missing voluntary_participation_statement; inferred from the draft text.")
        if "withdrawal_rights_summary" not in payload:
            repair_notes.append("Missing withdrawal_rights_summary; inferred from the draft text.")
        if "participant_rights" not in payload:
            repair_notes.append("Missing participant_rights; inferred from the draft text.")
        if "consent_summary" not in payload:
            repair_notes.append("Missing consent_summary; generated from the draft text.")
        if "cited_markers" not in payload:
            repair_notes.append("Missing cited_markers; recovered from the personalized draft.")

        metadata = {
            "participant_id": patient_profile.participant_id,
            "jurisdiction": patient_profile.jurisdiction,
            "health_literacy": patient_profile.health_literacy,
            "schema_repair_notes": repair_notes,
            "raw_response_keys": sorted(payload.keys()),
            "outer_raw_response_keys": sorted(original_payload.keys()),
            **metadata,
        }

        return {
            "purposes": purposes,
            "data_types": data_types,
            "valid_until": valid_until,
            "withdrawal_policy": withdrawal_policy,
            "study_purpose_summary": study_purpose_summary,
            "study_procedures_summary": study_procedures_summary,
            "risks_summary": risks_summary,
            "benefits_summary": benefits_summary,
            "alternatives_summary": alternatives_summary,
            "question_rights_summary": question_rights_summary,
            "voluntary_participation_statement": voluntary_participation_statement,
            "withdrawal_rights_summary": withdrawal_rights_summary,
            "participant_rights": participant_rights,
            "consent_summary": consent_summary,
            "cited_markers": cited_markers,
            "metadata": metadata,
        }

    def normalize_qa_answer_response(
        self,
        payload: dict[str, Any],
        *,
        retrieval_hits: list[dict[str, Any]],
    ) -> dict[str, Any]:
        text = None
        for key in ("answer_text", "answer", "response", "summary"):
            candidate = payload.get(key)
            if isinstance(candidate, str) and candidate.strip():
                text = candidate.strip()
                break

        if not text:
            raise RuntimeError(f"Model response did not contain recognizable answer text fields: {list(payload.keys())}")

        citation_markers = payload.get("citation_markers_used")
        if not isinstance(citation_markers, list):
            citation_markers = sorted(set(re.findall(r"\[\d+\]", text)))
        else:
            normalized_markers: list[str] = []
            for item in citation_markers:
                marker = str(item).strip()
                if re.fullmatch(r"\d+", marker):
                    marker = f"[{marker}]"
                normalized_markers.append(marker)
            citation_markers = normalized_markers

        uncertainty_noted = payload.get("uncertainty_noted")
        if not isinstance(uncertainty_noted, bool):
            uncertainty_noted = bool(
                re.search(r"\b(insufficient|uncertain|not enough|do not know|can't tell|cannot tell)\b", text, re.IGNORECASE)
            )

        limitations = payload.get("grounding_limitations")
        if not isinstance(limitations, list):
            limitations = []
        else:
            limitations = [str(item) for item in limitations]

        available_markers = [f"[{idx}]" for idx in range(1, len(retrieval_hits) + 1)]
        unsupported_markers = sorted(set(citation_markers) - set(available_markers))

        repair_notes: list[str] = []
        if "answer_text" not in payload:
            repair_notes.append("Recovered answer_text from an alternate field name.")
        if "citation_markers_used" not in payload:
            repair_notes.append("Recovered citation_markers_used from inline markers in the answer text.")
        if "uncertainty_noted" not in payload:
            repair_notes.append("Recovered uncertainty_noted from answer wording.")
        if "grounding_limitations" not in payload:
            repair_notes.append("Missing grounding_limitations; stored as an empty list.")
        if citation_markers and not re.findall(r"\[\d+\]", text):
            text = self.inject_inline_citations(
                text,
                citation_markers,
                max_sentences=max(1, min(len(citation_markers), 2)),
            )
            repair_notes.append("Inserted answer citation markers inline because the model returned them only in a separate field.")
        if unsupported_markers:
            repair_notes.append("Detected citation markers not present in the retrieved context.")

        return {
            "answer_text": text,
            "citation_markers_used": citation_markers,
            "uncertainty_noted": uncertainty_noted,
            "grounding_limitations": limitations,
            "unsupported_citation_markers": unsupported_markers,
            "available_citation_markers": available_markers,
            "schema_repair_notes": repair_notes,
            "raw_response_keys": sorted(payload.keys()),
        }

    def log_method_note(self, run_id: str, stage_name: str, note: str, outputs: dict[str, Any] | None = None) -> None:
        timestamp = utc_now_iso()
        self.artifacts.record_stage(
            run_id,
            PipelineStageRecord(
                stage_name=stage_name,
                status="completed",
                started_at=timestamp,
                ended_at=timestamp,
                outputs=outputs or {},
                notes=note,
            ),
        )
