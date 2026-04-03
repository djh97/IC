from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .evaluation import (
    build_draft_revision_audit,
    compare_draft_revision_candidates,
    evaluate_required_elements,
    summarize_personalized_draft,
)
from .hf_client import chat_json

if TYPE_CHECKING:
    from .artifacts import ArtifactStore
    from .config import AppConfig
    from .pipeline import ConsentPipeline
    from .prompt_loader import PromptLoader
    from .types import PatientProfile, PipelineStageRecord


@dataclass(slots=True)
class PromptTools:
    loader: "PromptLoader"

    def load(self, filename: str) -> str:
        return self.loader.load(filename)

    def render(self, filename: str, values: dict[str, object]) -> str:
        return self.loader.render(filename, values)

    def path(self, filename: str) -> Path:
        return self.loader.path(filename)


@dataclass(slots=True)
class ArtifactTools:
    store: "ArtifactStore"

    def run_path(self, run_id: str, *parts: str) -> Path:
        return self.store.run_path(run_id, *parts)

    def write_json(self, path: Path, payload: Any) -> None:
        self.store.write_json(path, payload)

    def write_text(self, path: Path, text: str) -> None:
        self.store.write_text(path, text)

    def record_stage(self, run_id: str, record: "PipelineStageRecord") -> None:
        self.store.record_stage(run_id, record)


@dataclass(slots=True)
class RetrievalTools:
    pipeline: "ConsentPipeline"
    config: "AppConfig"

    def retrieve_bundle(
        self,
        *,
        run_id: str,
        query: str,
        top_k: int | None = None,
        retrieval_mode: str | None = None,
        source_group_filters: list[str] | None = None,
        source_id_filters: list[str] | None = None,
        filter_logic: str | None = None,
    ) -> dict[str, Any]:
        retrieval_bundle = self.pipeline.retrieve_prepared_corpus(
            run_id=run_id,
            query=query,
            top_k=top_k,
            retrieval_mode=retrieval_mode,
            source_group_filters=source_group_filters,
            source_id_filters=source_id_filters,
            filter_logic=filter_logic,
        )
        retrieval_hits = [asdict(hit) for hit in retrieval_bundle["hits"]]
        return {
            **retrieval_bundle,
            "retrieval_hits": retrieval_hits,
            "retrieved_context": self.pipeline.format_retrieval_context(retrieval_hits),
            "citation_map": self.pipeline.build_citation_map(retrieval_hits),
            "evidence_package": self.pipeline.build_role_separated_evidence_package(retrieval_hits),
        }

    def build_personalization_query(self, patient_profile: "PatientProfile", base_template_text: str) -> str:
        return self.pipeline.build_personalization_query(patient_profile, base_template_text)

    def build_study_query_context(self, run_id: str, source_id_filters: list[str] | None) -> dict[str, Any]:
        return self.pipeline.build_study_query_context(run_id, source_id_filters)

    def build_readability_guidance(self, patient_profile: "PatientProfile", *, task_type: str) -> str:
        return self.pipeline.build_readability_guidance(patient_profile, task_type=task_type)

    def build_question_id(self, question: str) -> str:
        return self.pipeline.build_question_id(question)

    def build_citation_map(self, retrieval_hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return self.pipeline.build_citation_map(retrieval_hits)

    def format_retrieval_context(
        self,
        retrieval_hits: list[dict[str, Any]],
        *,
        marker_lookup: dict[str, str] | None = None,
    ) -> str:
        return self.pipeline.format_retrieval_context(retrieval_hits, marker_lookup=marker_lookup)

    def build_evidence_package(self, retrieval_hits: list[dict[str, Any]]) -> dict[str, Any]:
        return self.pipeline.build_role_separated_evidence_package(retrieval_hits)

    def upsert_qa_index_entry(self, qa_index_path: Path, entry: dict[str, Any]) -> None:
        self.pipeline.upsert_qa_index_entry(qa_index_path, entry)


@dataclass(slots=True)
class GenerationTools:
    def call_json_model(
        self,
        *,
        messages: list[dict[str, str]],
        schema_name: str,
        schema: dict[str, Any],
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        return chat_json(
            messages=messages,
            schema_name=schema_name,
            schema=schema,
            temperature=temperature,
            max_tokens=max_tokens,
        )


@dataclass(slots=True)
class EvaluationTools:
    def evaluate_required_elements(self, text: str) -> dict[str, bool]:
        return evaluate_required_elements(text)

    def summarize_personalized_draft(
        self,
        draft: dict[str, Any],
        *,
        available_markers: list[str],
        health_literacy: str,
    ) -> dict[str, Any]:
        return summarize_personalized_draft(
            draft,
            available_markers=available_markers,
            health_literacy=health_literacy,
        )

    def build_draft_revision_audit(
        self,
        draft_summary: dict[str, Any],
        *,
        draft_content_plan: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return build_draft_revision_audit(
            draft_summary,
            draft_content_plan=draft_content_plan,
        )

    def compare_draft_revision_candidates(
        self,
        initial_summary: dict[str, Any],
        revised_summary: dict[str, Any],
        *,
        initial_audit: dict[str, Any] | None = None,
        revised_audit: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return compare_draft_revision_candidates(
            initial_summary,
            revised_summary,
            initial_audit=initial_audit,
            revised_audit=revised_audit,
        )


@dataclass(slots=True)
class StateTools:
    pipeline: "ConsentPipeline"

    def resolve_patient_profile(self, run_id: str, patient_profile_path: Path | None):
        return self.pipeline.resolve_patient_profile(run_id, patient_profile_path)

    def resolve_base_template_text(self, run_id: str, template_path: Path | None) -> str:
        return self.pipeline.resolve_base_template_text(run_id, template_path)

    def resolve_personalized_draft(self, run_id: str, draft_path: Path | None) -> dict[str, Any]:
        return self.pipeline.resolve_personalized_draft(run_id, draft_path)

    def normalize_personalized_draft_response(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.pipeline.normalize_personalized_draft_response(payload)

    def normalize_qa_answer_response(
        self,
        payload: dict[str, Any],
        *,
        retrieval_hits: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return self.pipeline.normalize_qa_answer_response(payload, retrieval_hits=retrieval_hits)

    def normalize_structured_consent_record(
        self,
        payload: dict[str, Any],
        *,
        personalized_draft: dict[str, Any],
        patient_profile: "PatientProfile",
    ) -> dict[str, Any]:
        return self.pipeline.normalize_structured_consent_record(
            payload,
            personalized_draft=personalized_draft,
            patient_profile=patient_profile,
        )


@dataclass(slots=True)
class AgentTools:
    prompts: PromptTools
    artifacts: ArtifactTools
    retrieval: RetrievalTools
    generation: GenerationTools
    evaluation: EvaluationTools
    state: StateTools

    @classmethod
    def from_pipeline(cls, pipeline: "ConsentPipeline") -> "AgentTools":
        return cls(
            prompts=PromptTools(pipeline.prompts),
            artifacts=ArtifactTools(pipeline.artifacts),
            retrieval=RetrievalTools(pipeline, pipeline.config),
            generation=GenerationTools(),
            evaluation=EvaluationTools(),
            state=StateTools(pipeline),
        )


@dataclass(slots=True)
class RAGAgentTools:
    artifacts: ArtifactTools
    retrieval: RetrievalTools


@dataclass(slots=True)
class PersonalizationAgentTools:
    prompts: PromptTools
    artifacts: ArtifactTools
    retrieval: RetrievalTools
    generation: GenerationTools
    state: StateTools


@dataclass(slots=True)
class ConversationalAgentTools:
    prompts: PromptTools
    artifacts: ArtifactTools
    retrieval: RetrievalTools
    generation: GenerationTools
    state: StateTools


@dataclass(slots=True)
class FormalizationAgentTools:
    prompts: PromptTools
    artifacts: ArtifactTools
    generation: GenerationTools
    state: StateTools


@dataclass(slots=True)
class OrchestratorAgentTools:
    prompts: PromptTools
    artifacts: ArtifactTools
    retrieval: RetrievalTools
    generation: GenerationTools
    evaluation: EvaluationTools
    state: StateTools


@dataclass(slots=True)
class AgentToolRegistry:
    shared: AgentTools
    orchestrator: OrchestratorAgentTools
    rag: RAGAgentTools
    personalization: PersonalizationAgentTools
    conversational: ConversationalAgentTools
    formalization: FormalizationAgentTools

    @classmethod
    def from_pipeline(cls, pipeline: "ConsentPipeline") -> "AgentToolRegistry":
        shared = AgentTools.from_pipeline(pipeline)
        return cls(
            shared=shared,
            orchestrator=OrchestratorAgentTools(
                prompts=shared.prompts,
                artifacts=shared.artifacts,
                retrieval=shared.retrieval,
                generation=shared.generation,
                evaluation=shared.evaluation,
                state=shared.state,
            ),
            rag=RAGAgentTools(
                artifacts=shared.artifacts,
                retrieval=shared.retrieval,
            ),
            personalization=PersonalizationAgentTools(
                prompts=shared.prompts,
                artifacts=shared.artifacts,
                retrieval=shared.retrieval,
                generation=shared.generation,
                state=shared.state,
            ),
            conversational=ConversationalAgentTools(
                prompts=shared.prompts,
                artifacts=shared.artifacts,
                retrieval=shared.retrieval,
                generation=shared.generation,
                state=shared.state,
            ),
            formalization=FormalizationAgentTools(
                prompts=shared.prompts,
                artifacts=shared.artifacts,
                generation=shared.generation,
                state=shared.state,
            ),
        )

    def for_agent(self, agent_label: str) -> object:
        mapping = {
            "Orchestrator Agent": self.orchestrator,
            "RAG Agent": self.rag,
            "Personalization Agent": self.personalization,
            "Conversational Agent": self.conversational,
            "Consent Formalization Agent": self.formalization,
        }
        try:
            return mapping[agent_label]
        except KeyError as exc:  # pragma: no cover - defensive guard
            raise KeyError(f"No toolset registered for agent label: {agent_label}") from exc
