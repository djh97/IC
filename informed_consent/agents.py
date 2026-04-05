from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
import json
import re
import shutil
from dataclasses import field
from uuid import uuid4

from .agent_tools import AgentToolRegistry
from .artifacts import utc_now_iso
from .hf_client import chat_json  # Backward-compatible import path for tests and local patching.
from .types import AgentHandoff, PatientProfile, PipelineStageRecord

if TYPE_CHECKING:
    from .pipeline import ConsentPipeline


PERSONALIZED_CONSENT_DRAFT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "key_information_summary": {"type": "string"},
        "key_information_citation_markers_used": {"type": "array", "items": {"type": "string"}},
        "personalized_consent_text": {"type": "string"},
        "citation_markers_used": {"type": "array", "items": {"type": "string"}},
        "personalization_rationale": {"type": "array", "items": {"type": "string"}},
        "grounding_limitations": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "key_information_summary",
        "key_information_citation_markers_used",
        "personalized_consent_text",
        "citation_markers_used",
        "personalization_rationale",
        "grounding_limitations",
    ],
    "additionalProperties": False,
}

STRUCTURED_CONSENT_RECORD_SCHEMA: dict[str, Any] = {
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
        "metadata": {
            "type": "object",
            "properties": {
                "participant_id": {"type": ["string", "null"]},
                "jurisdiction": {"type": "string"},
                "health_literacy": {"type": "string"},
            },
            "required": ["participant_id", "jurisdiction", "health_literacy"],
            "additionalProperties": False,
        },
    },
    "required": [
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
    ],
    "additionalProperties": False,
}

CONSENT_QUESTION_ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer_text": {"type": "string"},
        "citation_markers_used": {"type": "array", "items": {"type": "string"}},
        "uncertainty_noted": {"type": "boolean"},
        "grounding_limitations": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "answer_text",
        "citation_markers_used",
        "uncertainty_noted",
        "grounding_limitations",
    ],
    "additionalProperties": False,
}

ORCHESTRATOR_REQUEST_ROUTE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "intent": {
            "type": "string",
            "enum": ["personalize_consent", "answer_question", "formalize_consent", "clarification"],
        },
        "reason": {"type": "string"},
        "message": {"type": ["string", "null"]},
    },
    "required": ["intent", "reason", "message"],
    "additionalProperties": False,
}

ORCHESTRATOR_QUESTION_PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "question_profile": {
            "type": "string",
            "enum": ["regulatory", "study", "study_plus_regulatory", "study_context_optional"],
        },
        "retrieval_query": {"type": "string"},
        "required_source_groups": {
            "type": "array",
            "items": {"type": "string", "enum": ["regulatory_guidance", "trial_materials"]},
        },
        "preferred_source_groups": {
            "type": "array",
            "items": {"type": "string", "enum": ["regulatory_guidance", "trial_materials"]},
        },
        "reason": {"type": "string"},
    },
    "required": [
        "question_profile",
        "retrieval_query",
        "required_source_groups",
        "preferred_source_groups",
        "reason",
    ],
    "additionalProperties": False,
}

CONSENT_PLAN_ELEMENT_IDS = (
    "voluntary_participation",
    "study_procedures",
    "risks",
    "benefits",
    "alternatives",
    "questions",
    "withdrawal_rights",
)

ORCHESTRATOR_DRAFT_PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "overall_strategy": {"type": "string"},
        "elements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "element_id": {"type": "string", "enum": list(CONSENT_PLAN_ELEMENT_IDS)},
                    "status": {"type": "string", "enum": ["supported", "partially_supported", "unsupported"]},
                    "preferred_source_role": {
                        "type": "string",
                        "enum": ["study_specific", "regulatory", "both", "other", "none"],
                    },
                    "recommended_markers": {"type": "array", "items": {"type": "string"}},
                    "instruction": {"type": "string"},
                },
                "required": [
                    "element_id",
                    "status",
                    "preferred_source_role",
                    "recommended_markers",
                    "instruction",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["overall_strategy", "elements"],
    "additionalProperties": False,
}

DRAFT_PLAN_ROLE_HINTS: dict[str, str] = {
    "voluntary_participation": "regulatory",
    "study_procedures": "study_specific",
    "risks": "study_specific",
    "benefits": "study_specific",
    "alternatives": "regulatory",
    "questions": "regulatory",
    "withdrawal_rights": "regulatory",
}

DRAFT_PLAN_INSTRUCTIONS: dict[str, str] = {
    "voluntary_participation": "State clearly that participation is voluntary and that the participant can choose whether to join.",
    "study_procedures": "Explain what the participant would do in this study using concrete study-specific steps when the evidence supports them.",
    "risks": "Mention possible study risks or say that risks will be explained before joining if the evidence is general.",
    "benefits": "Describe possible benefits or clearly say that direct benefit is not guaranteed when that is what the evidence supports. Do not rely on a sentence that only says the team will explain benefits.",
    "alternatives": "Mention alternatives or other options if the regulatory evidence supports that statement.",
    "questions": "Tell the participant they can ask questions and contact the study team when they need help.",
    "withdrawal_rights": "Explain that the participant may stop later or withdraw without penalty when the evidence supports it.",
}

ELEMENT_RECOVERY_QUERY_HINTS: dict[str, str] = {
    "voluntary_participation": "voluntary participation choice decide whether to join informed consent",
    "study_procedures": "study procedures participant will do visits assessments schedule of events follow up intervention",
    "risks": "study risks side effects discomfort burden safety monitoring",
    "benefits": "possible benefits direct benefit not guaranteed benefit to others study purpose",
    "alternatives": "alternatives other options standard care not participate other treatment choices",
    "questions": "questions contact study team ask for help participant rights contact information",
    "withdrawal_rights": "withdrawal rights stop later leave study without penalty voluntary choice",
}

QUESTION_INTENT_PREFIXES = (
    "can ",
    "could ",
    "would ",
    "will ",
    "what ",
    "why ",
    "when ",
    "where ",
    "who ",
    "how ",
    "is ",
    "are ",
    "do ",
    "does ",
    "did ",
)

REGULATORY_QUESTION_KEYWORDS = {
    "withdraw",
    "withdrawal",
    "voluntary",
    "penalty",
    "stop",
    "leave",
    "choice",
    "rights",
    "contact",
    "privacy",
    "confidentiality",
    "questions",
    "cost",
    "compensation",
    "alternatives",
    "options",
}

STUDY_QUESTION_KEYWORDS = {
    "procedure",
    "visit",
    "drug",
    "medicine",
    "treatment",
    "randomized",
    "placebo",
    "arm",
    "dose",
    "intervention",
    "condition",
    "eligibility",
    "screening",
    "test",
    "heart failure",
    "dapagliflozin",
    "purpose",
    "testing",
}

GENERIC_STUDY_REFERENCE_KEYWORDS = {
    "study",
    "trial",
}

STUDY_SPECIFIC_PATTERNS = (
    re.compile(r"\bwhat (?:is|does) (?:this |the )?(?:study|trial) (?:test|testing|about)\b"),
    re.compile(r"\bwhat would i have to do\b"),
    re.compile(r"\bwhat will i have to do\b"),
    re.compile(r"\bwhat do i have to do\b"),
    re.compile(r"\bwho can join\b"),
    re.compile(r"\bwho is eligible\b"),
    re.compile(r"\bhow many visits\b"),
    re.compile(r"\bwhat drug\b"),
    re.compile(r"\bwhat medicine\b"),
    re.compile(r"\bwhat treatment\b"),
    re.compile(r"\bwhat procedures?\b"),
    re.compile(r"\bwhat tests?\b"),
)

REGULATORY_PRIORITY_PATTERNS = (
    re.compile(r"\bcan i (?:leave|stop|quit|withdraw)\b"),
    re.compile(r"\bwithout penalty\b"),
    re.compile(r"\bmy choice\b"),
    re.compile(r"\bwho can i contact\b"),
    re.compile(r"\bwhat other options\b"),
)


def slugify_agent_value(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "agent"


def contains_any_keyword(text: str, keywords: set[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def matches_any_pattern(text: str, patterns: tuple[re.Pattern[str], ...]) -> bool:
    return any(pattern.search(text) for pattern in patterns)


@dataclass(slots=True)
class AgentRuntime:
    pipeline: "ConsentPipeline"
    tool_registry: AgentToolRegistry = field(init=False)

    @property
    def config(self):
        return self.pipeline.config

    @property
    def artifacts(self):
        return self.pipeline.artifacts

    @property
    def prompts(self):
        return self.pipeline.prompts

    def __post_init__(self) -> None:
        self.tool_registry = AgentToolRegistry.from_pipeline(self.pipeline)

    def emit_handoff(
        self,
        *,
        run_id: str,
        from_agent: str,
        to_agent: str,
        purpose: str,
        payload: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        handoff = AgentHandoff(
            handoff_id=uuid4().hex[:12],
            run_id=run_id,
            from_agent=from_agent,
            to_agent=to_agent,
            purpose=purpose,
            created_at=utc_now_iso(),
            payload=payload,
            metadata=metadata or {},
        )
        filename = (
            f"{slugify_agent_value(from_agent)}_to_{slugify_agent_value(to_agent)}_"
            f"{slugify_agent_value(purpose)}_{handoff.handoff_id}.json"
        )
        path = self.artifacts.record_agent_handoff(run_id, handoff, filename=filename)
        return {
            "path": str(path),
            "handoff_id": handoff.handoff_id,
            "purpose": purpose,
            "from_agent": from_agent,
            "to_agent": to_agent,
        }


class BaseAgent:
    agent_label = "Base Agent"

    def __init__(self, runtime: AgentRuntime):
        self.runtime = runtime

    @property
    def pipeline(self) -> "ConsentPipeline":
        return self.runtime.pipeline

    @property
    def stage_prefix(self) -> str:
        return slugify_agent_value(self.agent_label)

    @property
    def tools(self) -> Any:
        return self.runtime.tool_registry.for_agent(self.agent_label)

    def record_stage(
        self,
        run_id: str,
        *,
        stage_name: str,
        status: str,
        started_at: str,
        ended_at: str,
        inputs: dict[str, Any] | None = None,
        outputs: dict[str, Any] | None = None,
        notes: str = "",
    ) -> None:
        self.tools.artifacts.record_stage(
            run_id,
            PipelineStageRecord(
                stage_name=f"{self.stage_prefix}.{stage_name}",
                status=status,
                started_at=started_at,
                ended_at=ended_at,
                inputs=inputs or {},
                outputs=outputs or {},
                notes=notes,
            ),
        )

    def emit_handoff(
        self,
        run_id: str,
        *,
        to_agent: str,
        purpose: str,
        payload: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.runtime.emit_handoff(
            run_id=run_id,
            from_agent=self.agent_label,
            to_agent=to_agent,
            purpose=purpose,
            payload=payload,
            metadata=metadata,
        )

    def build_prompt_identifiers(
        self,
        *,
        system_prompt_filename: str,
        user_prompt_filename: str,
    ) -> dict[str, str]:
        return {
            "system_prompt_id": system_prompt_filename,
            "user_prompt_id": user_prompt_filename,
            "system_prompt_path": str(self.tools.prompts.path(system_prompt_filename)),
            "user_prompt_path": str(self.tools.prompts.path(user_prompt_filename)),
        }

    def build_generation_metadata(self) -> dict[str, Any]:
        return {
            "model_id": self.runtime.config.models.generator_model,
            "embedding_model_id": self.runtime.config.models.embedding_model,
            "temperature": self.runtime.config.models.temperature,
            "max_tokens": self.runtime.config.models.max_tokens,
        }


class RAGAgent(BaseAgent):
    agent_label = "RAG Agent"

    def retrieve_evidence(
        self,
        *,
        run_id: str,
        query: str,
        top_k: int | None = None,
        retrieval_mode: str | None = None,
        source_group_filters: list[str] | None = None,
        source_id_filters: list[str] | None = None,
        filter_logic: str | None = None,
        purpose: str,
        emit_result_to: str | None = None,
    ) -> dict[str, Any]:
        started_at = utc_now_iso()
        retrieval_bundle = self.tools.retrieval.retrieve_bundle(
            run_id=run_id,
            query=query,
            top_k=top_k,
            retrieval_mode=retrieval_mode,
            source_group_filters=source_group_filters,
            source_id_filters=source_id_filters,
            filter_logic=filter_logic,
        )
        retrieval_hits = retrieval_bundle["retrieval_hits"]
        retrieved_context = retrieval_bundle["retrieved_context"]
        citation_map = retrieval_bundle["citation_map"]
        evidence_package = retrieval_bundle["evidence_package"]

        result_handoff_path = None
        if emit_result_to:
            result_handoff = self.emit_handoff(
                run_id,
                to_agent=emit_result_to,
                purpose=f"{purpose}_result",
                payload={
                    "query": query,
                    "retrieval_mode_used": retrieval_bundle["mode_used"],
                    "retrieval_hit_count": len(retrieval_hits),
                    "dense_retrieval_available": retrieval_bundle["dense_available"],
                    "source_group_filters": retrieval_bundle["source_group_filters"],
                    "source_id_filters": retrieval_bundle["source_id_filters"],
                    "filter_logic_used": retrieval_bundle["filter_logic_used"],
                    "top_citation_labels": [hit["citation_label"] for hit in retrieval_hits[:3]],
                    "role_counts": evidence_package["role_counts"],
                },
                metadata={"purpose_family": purpose},
            )
            result_handoff_path = result_handoff["path"]

        self.record_stage(
            run_id,
            stage_name="retrieve_evidence",
            status="completed",
            started_at=started_at,
            ended_at=utc_now_iso(),
            inputs={
                "query": query,
                "top_k": top_k or self.runtime.config.retrieval.top_k,
                "retrieval_mode": retrieval_mode or self.runtime.config.retrieval.retrieval_mode,
                "source_group_filters": source_group_filters or [],
                "source_id_filters": source_id_filters or [],
                "filter_logic": filter_logic,
                "purpose": purpose,
            },
            outputs={
                "retrieval_mode_used": retrieval_bundle["mode_used"],
                "retrieval_hit_count": len(retrieval_hits),
                "dense_retrieval_available": retrieval_bundle["dense_available"],
                "study_specific_hit_count": evidence_package["role_counts"]["study_specific"],
                "regulatory_hit_count": evidence_package["role_counts"]["regulatory"],
                "result_handoff_path": result_handoff_path,
            },
            notes=(
                "Retrieved grounded evidence from the regulatory and study knowledge sources "
                "and prepared citation-ready context for downstream agents."
            ),
        )

        return {
            **retrieval_bundle,
            "retrieval_hits": retrieval_hits,
            "retrieved_context": retrieved_context,
            "citation_map": citation_map,
            "evidence_package": evidence_package,
            "result_handoff_path": result_handoff_path,
        }


class PersonalizationAgent(BaseAgent):
    agent_label = "Personalization Agent"
    revision_max_tokens = 1536

    def build_targeted_revision_guidance(
        self,
        *,
        patient_profile: PatientProfile,
        recovery_targets: list[dict[str, Any]] | None,
        draft_audit: dict[str, Any],
    ) -> str:
        target_element_ids = {
            str(target.get("element_id", "")).strip()
            for target in (recovery_targets or [])
            if isinstance(target, dict)
        }
        missing_element_ids = {
            str(element_id).strip()
            for element_id in (
                list(draft_audit.get("missing_required_elements", []))
                + list(draft_audit.get("missing_planned_required_elements", []))
            )
        }
        guidance_lines: list[str] = []

        if (
            patient_profile.health_literacy == "low"
            and "benefits" in target_element_ids
            and "benefits" in missing_element_ids
        ):
            guidance_lines.extend(
                [
                    "For low-literacy benefits coverage, add one short standalone benefits sentence in the summary and the fuller draft.",
                    "Prefer simple wording such as 'There may be no direct benefit to you [x].' or 'This study may not help you directly [x].' when that is what the evidence supports.",
                    "Do not replace that benefits sentence with a vague line that only says the team will explain risks and benefits.",
                ]
            )

        if (
            patient_profile.health_literacy == "low"
            and "alternatives" in target_element_ids
            and "alternatives" in missing_element_ids
        ):
            guidance_lines.extend(
                [
                    "For low-literacy alternatives coverage, add one short standalone alternatives sentence in the summary and the fuller draft.",
                    "Prefer simple wording such as 'You may have other treatment options besides joining [x].' or 'You may have other choices besides joining [x].' when that is what the evidence supports.",
                    "Do not leave alternatives implied; state the other-options sentence directly.",
                ]
            )

        if not guidance_lines:
            return "No extra targeted revision guidance was prepared."

        return "\n".join(f"- {line}" for line in guidance_lines)

    def generate_draft(
        self,
        *,
        run_id: str,
        patient_profile: PatientProfile,
        patient_profile_path: Path | None,
        base_template_text: str,
        template_path: Path | None,
        generation_query: str,
        retrieval_artifacts: dict[str, Any],
        draft_content_plan: dict[str, Any] | None = None,
        top_k: int | None = None,
        retrieval_mode: str | None = None,
        source_group_filters: list[str] | None = None,
        source_id_filters: list[str] | None = None,
        workflow_variant: str = "full_agentic",
        dry_run: bool = False,
        emit_result_to: str | None = None,
    ) -> dict[str, Any]:
        started_at = utc_now_iso()
        retrieval_hits = retrieval_artifacts["retrieval_hits"]
        evidence_package = retrieval_artifacts["evidence_package"]
        study_specific_context = evidence_package["study_specific_context"] or "No study-specific context was available."
        regulatory_context = evidence_package["regulatory_context"] or "No regulatory context was available."
        other_context = evidence_package["other_context"] or "No additional grounding context was available."
        combined_context = evidence_package.get("combined_context") or "No grounded context was available."

        patient_profile_json = json.dumps(asdict(patient_profile), indent=2)
        readability_guidance = self.tools.retrieval.build_readability_guidance(patient_profile, task_type="draft")
        if workflow_variant == "generic_rag":
            user_prompt_filename = "personalize_consent_generic_rag_user.txt"
            system_prompt_filename = "personalize_consent_baseline_system.txt"
            user_prompt = self.tools.prompts.render(
                user_prompt_filename,
                {
                    "participant_profile_json": patient_profile_json,
                    "base_template_text": base_template_text or "No base consent template was supplied.",
                    "generation_objective": "Produce a participant-facing informed consent draft from the retrieved context.",
                    "readability_guidance": readability_guidance,
                    "combined_grounding_context": combined_context,
                },
            )
        elif workflow_variant == "vanilla_llm":
            user_prompt_filename = "personalize_consent_vanilla_user.txt"
            system_prompt_filename = "personalize_consent_baseline_system.txt"
            user_prompt = self.tools.prompts.render(
                user_prompt_filename,
                {
                    "participant_profile_json": patient_profile_json,
                    "base_template_text": base_template_text or "No base consent template was supplied.",
                    "generation_objective": "Produce a participant-facing informed consent draft using the participant profile and template only.",
                    "readability_guidance": readability_guidance,
                },
            )
        else:
            user_prompt_filename = "personalize_consent_user.txt"
            system_prompt_filename = "personalize_consent_system.txt"
            user_prompt = self.tools.prompts.render(
                user_prompt_filename,
                {
                    "participant_profile_json": patient_profile_json,
                    "base_template_text": base_template_text or "No base consent template was supplied.",
                    "generation_objective": (
                        "Produce a grounded personalized informed consent draft that is easy for this participant to understand."
                    ),
                    "readability_guidance": readability_guidance,
                    "draft_content_plan_json": json.dumps(draft_content_plan or {}, indent=2),
                    "study_specific_context": study_specific_context,
                    "regulatory_context": regulatory_context,
                    "other_grounding_context": other_context,
                },
            )
        system_prompt = self.tools.prompts.load(system_prompt_filename)
        prompt_identifiers = self.build_prompt_identifiers(
            system_prompt_filename=system_prompt_filename,
            user_prompt_filename=user_prompt_filename,
        )

        request_bundle = {
            "agent": self.agent_label,
            "run_id": run_id,
            "patient_profile": asdict(patient_profile),
            "generation_query": generation_query,
            "workflow_variant": workflow_variant,
            **self.build_generation_metadata(),
            "draft_content_plan": draft_content_plan or {},
            "top_k": top_k or self.runtime.config.retrieval.top_k,
            "retrieval_mode_used": retrieval_artifacts["mode_used"],
            "dense_retrieval_available": retrieval_artifacts["dense_available"],
            "source_group_filters": retrieval_artifacts["source_group_filters"],
            "source_id_filters": retrieval_artifacts["source_id_filters"],
            "filter_logic_used": retrieval_artifacts["filter_logic_used"],
            "retrieval_strategy_used": retrieval_artifacts.get("scoped_retrieval_strategy")
            or ("no_retrieval" if retrieval_artifacts["mode_used"] == "none" else "single_pass"),
            "filtered_chunk_count": retrieval_artifacts["filtered_chunk_count"],
            "lexical_hits": retrieval_artifacts["lexical_hits"],
            "dense_hits": retrieval_artifacts["dense_hits"],
            "retrieval_hits": retrieval_hits,
            "evidence_package": evidence_package,
            "prompt_identifiers": prompt_identifiers,
            **prompt_identifiers,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        inputs_dir = self.tools.artifacts.run_path(run_id, "inputs")
        outputs_dir = self.tools.artifacts.run_path(run_id, "outputs")
        self.tools.artifacts.write_json(inputs_dir / "patient_profile.json", patient_profile)
        if base_template_text:
            self.tools.artifacts.write_text(inputs_dir / "base_template.txt", base_template_text)
        self.tools.artifacts.write_json(outputs_dir / "personalization_retrieval_hits.json", retrieval_hits)
        self.tools.artifacts.write_json(outputs_dir / "personalization_evidence_package.json", evidence_package)
        citation_map = self.tools.retrieval.build_citation_map(retrieval_hits)
        self.tools.artifacts.write_json(outputs_dir / "personalization_citation_map.json", citation_map)
        self.tools.artifacts.write_json(outputs_dir / "personalization_request_bundle.json", request_bundle)

        response_payload: dict[str, Any] | None = None
        output_path: str | None = None
        if not dry_run:
            raw_response_payload = self.tools.generation.call_json_model(
                messages=request_bundle["messages"],
                schema_name="personalized_consent_draft",
                schema=PERSONALIZED_CONSENT_DRAFT_SCHEMA,
                temperature=self.runtime.config.models.temperature,
                max_tokens=self.runtime.config.models.max_tokens,
            )
            response_payload = self.tools.state.normalize_personalized_draft_response(raw_response_payload)
            self.tools.artifacts.write_json(outputs_dir / "personalized_consent_draft.raw.json", raw_response_payload)
            output_path = str(outputs_dir / "personalized_consent_draft.json")
            self.tools.artifacts.write_json(outputs_dir / "personalized_consent_draft.json", response_payload)

        result_handoff_path = None
        if emit_result_to:
            result_handoff = self.emit_handoff(
                run_id,
                to_agent=emit_result_to,
                purpose="personalized_consent_draft_result",
                payload={
                    "draft_generated": response_payload is not None,
                    "dry_run": dry_run,
                    "output_path": output_path,
                    "citation_count": len(response_payload.get("citation_markers_used", [])) if response_payload else 0,
                    "summary_citation_count": len(response_payload.get("key_information_citation_markers_used", [])) if response_payload else 0,
                    "normalization_applied": bool(response_payload.get("schema_repair_notes")) if response_payload else False,
                },
            )
            result_handoff_path = result_handoff["path"]

        stage_outputs = {
            "retrieval_hit_count": len(retrieval_hits),
            "generation_query": generation_query,
            "retrieval_mode_used": retrieval_artifacts["mode_used"],
            "workflow_variant": workflow_variant,
            "dry_run": dry_run,
            "draft_generated": response_payload is not None,
            "draft_output_path": output_path,
            "citation_map_path": str(outputs_dir / "personalization_citation_map.json"),
            "evidence_package_path": str(outputs_dir / "personalization_evidence_package.json"),
            "result_handoff_path": result_handoff_path,
        }
        if response_payload is not None:
            stage_outputs["normalized_citation_count"] = len(response_payload.get("citation_markers_used", []))
            stage_outputs["summary_citation_count"] = len(response_payload.get("key_information_citation_markers_used", []))
            stage_outputs["summary_present"] = bool(response_payload.get("key_information_summary"))
            stage_outputs["normalization_applied"] = bool(response_payload.get("schema_repair_notes"))

        self.record_stage(
            run_id,
            stage_name="generate_draft",
            status="completed",
            started_at=started_at,
            ended_at=utc_now_iso(),
            inputs={
                "patient_profile_path": str(patient_profile_path) if patient_profile_path else None,
                "template_path": str(template_path) if template_path else None,
                "generation_query": generation_query,
                "workflow_variant": workflow_variant,
                "top_k": top_k or self.runtime.config.retrieval.top_k,
                "retrieval_mode": retrieval_mode or self.runtime.config.retrieval.retrieval_mode,
                "source_group_filters": source_group_filters or [],
                "source_id_filters": source_id_filters or [],
                "dry_run": dry_run,
            },
            outputs=stage_outputs,
            notes=(
                "Translated retrieved evidence, the base consent template, and participant profile into "
                "a grounded personalized consent draft."
            ),
        )

        return {
            "run_id": run_id,
            "request_bundle_path": str(outputs_dir / "personalization_request_bundle.json"),
            "retrieval_hits_path": str(outputs_dir / "personalization_retrieval_hits.json"),
            "citation_map_path": str(outputs_dir / "personalization_citation_map.json"),
            "evidence_package_path": str(outputs_dir / "personalization_evidence_package.json"),
            "output_path": output_path,
            "response": response_payload,
            "dry_run": dry_run,
            "agent_handoff_path": result_handoff_path,
        }

    def revise_draft(
        self,
        *,
        run_id: str,
        patient_profile: PatientProfile,
        patient_profile_path: Path | None,
        base_template_text: str,
        template_path: Path | None,
        generation_query: str,
        retrieval_artifacts: dict[str, Any],
        current_draft: dict[str, Any],
        draft_audit: dict[str, Any],
        draft_content_plan: dict[str, Any] | None = None,
        recovery_targets: list[dict[str, Any]] | None = None,
        focused_recovery_context: str | None = None,
        top_k: int | None = None,
        retrieval_mode: str | None = None,
        source_group_filters: list[str] | None = None,
        source_id_filters: list[str] | None = None,
        dry_run: bool = False,
        emit_result_to: str | None = None,
    ) -> dict[str, Any]:
        started_at = utc_now_iso()
        retrieval_hits = retrieval_artifacts["retrieval_hits"]
        evidence_package = retrieval_artifacts["evidence_package"]
        study_specific_context = evidence_package["study_specific_context"] or "No study-specific context was available."
        regulatory_context = evidence_package["regulatory_context"] or "No regulatory context was available."
        other_context = evidence_package["other_context"] or "No additional grounding context was available."
        targeted_revision_guidance = self.build_targeted_revision_guidance(
            patient_profile=patient_profile,
            recovery_targets=recovery_targets,
            draft_audit=draft_audit,
        )

        patient_profile_json = json.dumps(asdict(patient_profile), indent=2)
        user_prompt = self.tools.prompts.render(
            "revise_consent_draft_user.txt",
            {
                "participant_profile_json": patient_profile_json,
                "base_template_text": base_template_text or "No base consent template was supplied.",
                "current_draft_json": json.dumps(current_draft, indent=2),
                "draft_audit_json": json.dumps(draft_audit, indent=2),
                "draft_content_plan_json": json.dumps(draft_content_plan or {}, indent=2),
                "recovery_targets_json": json.dumps(recovery_targets or [], indent=2),
                "focused_recovery_context": focused_recovery_context or "No targeted recovery context was prepared.",
                "targeted_revision_guidance": targeted_revision_guidance,
                "readability_guidance": self.tools.retrieval.build_readability_guidance(patient_profile, task_type="draft"),
                "study_specific_context": study_specific_context,
                "regulatory_context": regulatory_context,
                "other_grounding_context": other_context,
            },
        )
        system_prompt = self.tools.prompts.load("revise_consent_draft_system.txt")
        prompt_identifiers = self.build_prompt_identifiers(
            system_prompt_filename="revise_consent_draft_system.txt",
            user_prompt_filename="revise_consent_draft_user.txt",
        )

        request_bundle = {
            "agent": self.agent_label,
            "run_id": run_id,
            "patient_profile": asdict(patient_profile),
            "generation_query": generation_query,
            **self.build_generation_metadata(),
            "draft_audit": draft_audit,
            "draft_content_plan": draft_content_plan or {},
            "recovery_targets": recovery_targets or [],
            "focused_recovery_context": focused_recovery_context or "",
            "targeted_revision_guidance": targeted_revision_guidance,
            "current_draft": current_draft,
            "top_k": top_k or self.runtime.config.retrieval.top_k,
            "retrieval_mode_used": retrieval_artifacts["mode_used"],
            "dense_retrieval_available": retrieval_artifacts["dense_available"],
            "source_group_filters": retrieval_artifacts["source_group_filters"],
            "source_id_filters": retrieval_artifacts["source_id_filters"],
            "filter_logic_used": retrieval_artifacts["filter_logic_used"],
            "retrieval_strategy_used": retrieval_artifacts.get("scoped_retrieval_strategy")
            or ("no_retrieval" if retrieval_artifacts["mode_used"] == "none" else "single_pass"),
            "filtered_chunk_count": retrieval_artifacts["filtered_chunk_count"],
            "lexical_hits": retrieval_artifacts["lexical_hits"],
            "dense_hits": retrieval_artifacts["dense_hits"],
            "retrieval_hits": retrieval_hits,
            "evidence_package": evidence_package,
            "prompt_identifiers": prompt_identifiers,
            **prompt_identifiers,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        outputs_dir = self.tools.artifacts.run_path(run_id, "outputs")
        request_bundle_path = outputs_dir / "draft_revision_request_bundle.json"
        self.tools.artifacts.write_json(request_bundle_path, request_bundle)

        response_payload: dict[str, Any] | None = None
        output_path: str | None = None
        raw_output_path: str | None = None
        if not dry_run:
            raw_response_payload = self.tools.generation.call_json_model(
                messages=request_bundle["messages"],
                schema_name="personalized_consent_draft_revision",
                schema=PERSONALIZED_CONSENT_DRAFT_SCHEMA,
                temperature=self.runtime.config.models.temperature,
                max_tokens=min(self.runtime.config.models.max_tokens, self.revision_max_tokens),
            )
            response_payload = self.tools.state.normalize_personalized_draft_response(raw_response_payload)
            raw_output_path = str(outputs_dir / "personalized_consent_draft.revision.raw.json")
            self.tools.artifacts.write_json(Path(raw_output_path), raw_response_payload)
            output_path = str(outputs_dir / "personalized_consent_draft.revision.json")
            self.tools.artifacts.write_json(Path(output_path), response_payload)

        result_handoff_path = None
        if emit_result_to:
            result_handoff = self.emit_handoff(
                run_id,
                to_agent=emit_result_to,
                purpose="personalized_consent_draft_revision_result",
                payload={
                    "draft_revised": response_payload is not None,
                    "dry_run": dry_run,
                    "output_path": output_path,
                    "request_bundle_path": str(request_bundle_path),
                    "revision_issue_count": len(draft_audit.get("issues", [])),
                    "normalization_applied": bool(response_payload.get("schema_repair_notes")) if response_payload else False,
                },
            )
            result_handoff_path = result_handoff["path"]

        self.record_stage(
            run_id,
            stage_name="revise_draft",
            status="completed",
            started_at=started_at,
            ended_at=utc_now_iso(),
            inputs={
                "patient_profile_path": str(patient_profile_path) if patient_profile_path else None,
                "template_path": str(template_path) if template_path else None,
                "generation_query": generation_query,
                "top_k": top_k or self.runtime.config.retrieval.top_k,
                "retrieval_mode": retrieval_mode or self.runtime.config.retrieval.retrieval_mode,
                "source_group_filters": source_group_filters or [],
                "source_id_filters": source_id_filters or [],
                "revision_issue_count": len(draft_audit.get("issues", [])),
                "missing_required_elements": draft_audit.get("missing_required_elements", []),
                "dry_run": dry_run,
            },
            outputs={
                "draft_revised": response_payload is not None,
                "draft_output_path": output_path,
                "raw_output_path": raw_output_path,
                "request_bundle_path": str(request_bundle_path),
                "result_handoff_path": result_handoff_path,
            },
            notes=(
                "Revised an initial consent draft using targeted audit feedback while staying grounded in the same evidence package."
            ),
        )

        return {
            "run_id": run_id,
            "request_bundle_path": str(request_bundle_path),
            "output_path": output_path,
            "raw_output_path": raw_output_path,
            "response": response_payload,
            "dry_run": dry_run,
            "agent_handoff_path": result_handoff_path,
        }


class ConversationalAgent(BaseAgent):
    agent_label = "Conversational Agent"

    def answer_question(
        self,
        *,
        run_id: str,
        question: str,
        patient_profile: PatientProfile,
        patient_profile_path: Path | None,
        retrieval_artifacts: dict[str, Any],
        top_k: int | None = None,
        retrieval_mode: str | None = None,
        source_group_filters: list[str] | None = None,
        source_id_filters: list[str] | None = None,
        workflow_variant: str = "full_agentic",
        dry_run: bool = False,
        emit_result_to: str | None = None,
    ) -> dict[str, Any]:
        started_at = utc_now_iso()
        question = question.strip()
        if not question:
            raise ValueError("Question must not be empty.")

        retrieval_hits = retrieval_artifacts["retrieval_hits"]
        evidence_package = retrieval_artifacts["evidence_package"]
        study_specific_context = evidence_package["study_specific_context"] or "No study-specific context was available."
        regulatory_context = evidence_package["regulatory_context"] or "No regulatory context was available."
        other_context = evidence_package["other_context"] or "No additional grounding context was available."
        combined_context = evidence_package.get("combined_context") or "No grounded context was available."
        question_id = self.tools.retrieval.build_question_id(question)
        qa_dir = self.tools.artifacts.run_path(run_id, "outputs", "qa")
        qa_dir.mkdir(parents=True, exist_ok=True)

        readability_guidance = self.tools.retrieval.build_readability_guidance(patient_profile, task_type="qa")
        if workflow_variant == "generic_rag":
            user_prompt_filename = "qa_consent_generic_rag_user.txt"
            system_prompt_filename = "qa_consent_baseline_system.txt"
            user_prompt = self.tools.prompts.render(
                user_prompt_filename,
                {
                    "participant_profile_json": json.dumps(asdict(patient_profile), indent=2),
                    "participant_question": question,
                    "readability_guidance": readability_guidance,
                    "combined_grounding_context": combined_context,
                },
            )
        elif workflow_variant == "vanilla_llm":
            user_prompt_filename = "qa_consent_vanilla_user.txt"
            system_prompt_filename = "qa_consent_baseline_system.txt"
            user_prompt = self.tools.prompts.render(
                user_prompt_filename,
                {
                    "participant_profile_json": json.dumps(asdict(patient_profile), indent=2),
                    "participant_question": question,
                    "readability_guidance": readability_guidance,
                },
            )
        else:
            user_prompt_filename = "qa_consent_user.txt"
            system_prompt_filename = "qa_consent_system.txt"
            user_prompt = self.tools.prompts.render(
                user_prompt_filename,
                {
                    "participant_profile_json": json.dumps(asdict(patient_profile), indent=2),
                    "participant_question": question,
                    "readability_guidance": readability_guidance,
                    "study_specific_context": study_specific_context,
                    "regulatory_context": regulatory_context,
                    "other_grounding_context": other_context,
                },
            )
        system_prompt = self.tools.prompts.load(system_prompt_filename)
        prompt_identifiers = self.build_prompt_identifiers(
            system_prompt_filename=system_prompt_filename,
            user_prompt_filename=user_prompt_filename,
        )

        request_bundle = {
            "agent": self.agent_label,
            "run_id": run_id,
            "question_id": question_id,
            "question": question,
            "patient_profile": asdict(patient_profile),
            "workflow_variant": workflow_variant,
            **self.build_generation_metadata(),
            "top_k": top_k or self.runtime.config.retrieval.top_k,
            "retrieval_mode_used": retrieval_artifacts["mode_used"],
            "dense_retrieval_available": retrieval_artifacts["dense_available"],
            "source_group_filters": retrieval_artifacts["source_group_filters"],
            "source_id_filters": retrieval_artifacts["source_id_filters"],
            "filter_logic_used": retrieval_artifacts["filter_logic_used"],
            "retrieval_strategy_used": retrieval_artifacts.get("scoped_retrieval_strategy")
            or ("no_retrieval" if retrieval_artifacts["mode_used"] == "none" else "single_pass"),
            "filtered_chunk_count": retrieval_artifacts["filtered_chunk_count"],
            "lexical_hits": retrieval_artifacts["lexical_hits"],
            "dense_hits": retrieval_artifacts["dense_hits"],
            "retrieval_hits": retrieval_hits,
            "evidence_package": evidence_package,
            "prompt_identifiers": prompt_identifiers,
            **prompt_identifiers,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        retrieval_path = qa_dir / f"{question_id}.retrieval_hits.json"
        evidence_package_path = qa_dir / f"{question_id}.evidence_package.json"
        request_path = qa_dir / f"{question_id}.request_bundle.json"
        self.tools.artifacts.write_json(retrieval_path, retrieval_hits)
        self.tools.artifacts.write_json(evidence_package_path, evidence_package)
        self.tools.artifacts.write_json(request_path, request_bundle)

        response_payload: dict[str, Any] | None = None
        output_path: str | None = None
        if not dry_run:
            raw_response_payload = self.tools.generation.call_json_model(
                messages=request_bundle["messages"],
                schema_name="consent_question_answer",
                schema=CONSENT_QUESTION_ANSWER_SCHEMA,
                temperature=self.runtime.config.models.temperature,
                max_tokens=self.runtime.config.models.max_tokens,
            )
            response_payload = self.tools.state.normalize_qa_answer_response(
                raw_response_payload,
                retrieval_hits=retrieval_hits,
            )
            self.tools.artifacts.write_json(qa_dir / f"{question_id}.answer.raw.json", raw_response_payload)
            output_path = str(qa_dir / f"{question_id}.answer.json")
            self.tools.artifacts.write_json(qa_dir / f"{question_id}.answer.json", response_payload)

        result_handoff_path = None
        if emit_result_to:
            result_handoff = self.emit_handoff(
                run_id,
                to_agent=emit_result_to,
                purpose="consent_question_answer_result",
                payload={
                    "question_id": question_id,
                    "question": question,
                    "answer_generated": response_payload is not None,
                    "dry_run": dry_run,
                    "output_path": output_path,
                    "citation_count": len(response_payload.get("citation_markers_used", [])) if response_payload else 0,
                    "uncertainty_noted": response_payload.get("uncertainty_noted", False) if response_payload else False,
                    "normalization_applied": bool(response_payload.get("schema_repair_notes")) if response_payload else False,
                },
            )
            result_handoff_path = result_handoff["path"]

        stage_outputs = {
            "question_id": question_id,
            "retrieval_hit_count": len(retrieval_hits),
            "retrieval_mode_used": retrieval_artifacts["mode_used"],
            "workflow_variant": workflow_variant,
            "dry_run": dry_run,
            "answer_generated": response_payload is not None,
            "answer_output_path": output_path,
            "evidence_package_path": str(evidence_package_path),
            "result_handoff_path": result_handoff_path,
        }
        if response_payload is not None:
            stage_outputs["normalized_citation_count"] = len(response_payload.get("citation_markers_used", []))
            stage_outputs["normalization_applied"] = bool(response_payload.get("schema_repair_notes"))
            stage_outputs["uncertainty_noted"] = response_payload.get("uncertainty_noted", False)

        self.record_stage(
            run_id,
            stage_name="answer_question",
            status="completed",
            started_at=started_at,
            ended_at=utc_now_iso(),
            inputs={
                "question_id": question_id,
                "question": question,
                "patient_profile_path": str(patient_profile_path) if patient_profile_path else None,
                "workflow_variant": workflow_variant,
                "top_k": top_k or self.runtime.config.retrieval.top_k,
                "retrieval_mode": retrieval_mode or self.runtime.config.retrieval.retrieval_mode,
                "source_group_filters": source_group_filters or [],
                "source_id_filters": source_id_filters or [],
                "dry_run": dry_run,
            },
            outputs=stage_outputs,
            notes=(
                "Answered a participant question using grounded retrieved context and a dedicated conversational agent prompt."
            ),
        )

        index_entry = {
            "question_id": question_id,
            "question": question,
            "retrieval_hits_path": str(retrieval_path),
            "evidence_package_path": str(evidence_package_path),
            "request_bundle_path": str(request_path),
            "answer_path": output_path,
            "workflow_variant": workflow_variant,
            "status": "answered" if output_path else "prepared",
            "dry_run": dry_run,
        }
        self.tools.retrieval.upsert_qa_index_entry(qa_dir / "qa_index.jsonl", index_entry)

        return {
            "run_id": run_id,
            "question_id": question_id,
            "retrieval_hits_path": str(retrieval_path),
            "evidence_package_path": str(evidence_package_path),
            "request_bundle_path": str(request_path),
            "output_path": output_path,
            "response": response_payload,
            "dry_run": dry_run,
            "agent_handoff_path": result_handoff_path,
        }


class ConsentFormalizationAgent(BaseAgent):
    agent_label = "Consent Formalization Agent"

    def formalize_consent(
        self,
        *,
        run_id: str,
        patient_profile: PatientProfile,
        patient_profile_path: Path | None,
        personalized_draft: dict[str, Any],
        draft_path: Path | None,
        dry_run: bool = False,
        emit_result_to: str | None = None,
    ) -> dict[str, Any]:
        started_at = utc_now_iso()
        user_prompt = self.tools.prompts.render(
            "formalize_consent_user.txt",
            {
                "participant_profile_json": json.dumps(asdict(patient_profile), indent=2),
                "key_information_summary": personalized_draft.get("key_information_summary", ""),
                "personalized_consent_text": personalized_draft["personalized_consent_text"],
                "citation_markers": ", ".join(
                    sorted(
                        set(
                            list(personalized_draft.get("citation_markers_used", []))
                            + list(personalized_draft.get("key_information_citation_markers_used", []))
                        )
                    )
                )
                or "None",
            },
        )
        system_prompt = self.tools.prompts.load("formalize_consent_system.txt")
        prompt_identifiers = self.build_prompt_identifiers(
            system_prompt_filename="formalize_consent_system.txt",
            user_prompt_filename="formalize_consent_user.txt",
        )

        request_bundle = {
            "agent": self.agent_label,
            "run_id": run_id,
            "patient_profile": asdict(patient_profile),
            **self.build_generation_metadata(),
            "prompt_identifiers": prompt_identifiers,
            **prompt_identifiers,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        outputs_dir = self.tools.artifacts.run_path(run_id, "outputs")
        self.tools.artifacts.write_json(outputs_dir / "formalization_request_bundle.json", request_bundle)

        response_payload: dict[str, Any] | None = None
        output_path: str | None = None
        if not dry_run:
            try:
                raw_response_payload = self.tools.generation.call_json_model(
                    messages=request_bundle["messages"],
                    schema_name="structured_consent_record",
                    schema=STRUCTURED_CONSENT_RECORD_SCHEMA,
                    temperature=self.runtime.config.models.temperature,
                    max_tokens=self.runtime.config.models.max_tokens,
                )
                response_payload = self.tools.state.normalize_structured_consent_record(
                    raw_response_payload,
                    personalized_draft=personalized_draft,
                    patient_profile=patient_profile,
                )
            except RuntimeError as exc:
                raw_response_payload = {
                    "error": str(exc),
                    "fallback_strategy": "heuristic_normalization_from_personalized_draft",
                }
                response_payload = self.tools.state.normalize_structured_consent_record(
                    {},
                    personalized_draft=personalized_draft,
                    patient_profile=patient_profile,
                )
                response_payload["metadata"]["schema_repair_notes"].append(
                    "Formalization model response could not be parsed; used heuristic fallback extraction."
                )
                response_payload["metadata"]["model_error"] = str(exc)

            self.tools.artifacts.write_json(outputs_dir / "structured_consent_record.raw.json", raw_response_payload)
            output_path = str(outputs_dir / "structured_consent_record.json")
            self.tools.artifacts.write_json(outputs_dir / "structured_consent_record.json", response_payload)

        result_handoff_path = None
        if emit_result_to:
            result_handoff = self.emit_handoff(
                run_id,
                to_agent=emit_result_to,
                purpose="structured_consent_record_result",
                payload={
                    "structured_record_generated": response_payload is not None,
                    "dry_run": dry_run,
                    "output_path": output_path,
                    "repair_applied": bool(response_payload.get("metadata", {}).get("schema_repair_notes")) if response_payload else False,
                    "cited_marker_count": len(response_payload.get("cited_markers", [])) if response_payload else 0,
                },
            )
            result_handoff_path = result_handoff["path"]

        stage_outputs = {
            "dry_run": dry_run,
            "structured_record_generated": response_payload is not None,
            "structured_record_path": output_path,
            "result_handoff_path": result_handoff_path,
        }
        if response_payload is not None:
            stage_outputs["repair_applied"] = bool(response_payload.get("metadata", {}).get("schema_repair_notes"))

        self.record_stage(
            run_id,
            stage_name="formalize_consent",
            status="completed",
            started_at=started_at,
            ended_at=utc_now_iso(),
            inputs={
                "patient_profile_path": str(patient_profile_path) if patient_profile_path else None,
                "draft_path": str(draft_path) if draft_path else None,
                "dry_run": dry_run,
            },
            outputs=stage_outputs,
            notes=(
                "Converted the grounded personalized consent draft into a machine-readable consent record."
            ),
        )

        return {
            "run_id": run_id,
            "request_bundle_path": str(outputs_dir / "formalization_request_bundle.json"),
            "output_path": output_path,
            "response": response_payload,
            "dry_run": dry_run,
            "agent_handoff_path": result_handoff_path,
        }


class OrchestratorAgent(BaseAgent):
    agent_label = "Orchestrator Agent"

    def __init__(
        self,
        runtime: AgentRuntime,
        *,
        rag_agent: RAGAgent,
        personalization_agent: PersonalizationAgent,
        conversational_agent: ConversationalAgent,
        formalization_agent: ConsentFormalizationAgent,
    ):
        super().__init__(runtime)
        self.rag_agent = rag_agent
        self.personalization_agent = personalization_agent
        self.conversational_agent = conversational_agent
        self.formalization_agent = formalization_agent

    def llm_planning_available(self) -> bool:
        return bool(self.runtime.config.models.endpoint_url)

    def normalize_workflow_variant(self, workflow_variant: str | None) -> str:
        normalized = str(workflow_variant or "full_agentic").strip().lower()
        if normalized not in {"full_agentic", "generic_rag", "vanilla_llm"}:
            return "full_agentic"
        return normalized

    def build_empty_retrieval_artifacts(
        self,
        *,
        source_group_filters: list[str] | None = None,
        source_id_filters: list[str] | None = None,
        filter_logic: str | None = None,
    ) -> dict[str, Any]:
        retrieval_hits: list[dict[str, Any]] = []
        evidence_package = self.tools.retrieval.build_evidence_package(retrieval_hits)
        return {
            "mode_used": "none",
            "dense_available": False,
            "lexical_hits": [],
            "dense_hits": [],
            "retrieval_hits": retrieval_hits,
            "retrieved_context": "",
            "citation_map": [],
            "evidence_package": evidence_package,
            "source_group_filters": list(source_group_filters or []),
            "source_id_filters": list(source_id_filters or []),
            "filter_logic_used": filter_logic or "intersection",
            "filtered_chunk_count": 0,
        }

    def classify_user_request_fallback(
        self,
        *,
        user_input: str,
        draft_available: bool,
        template_available: bool,
    ) -> dict[str, Any]:
        normalized = user_input.strip().lower()

        if not normalized:
            return {
                "intent": "clarification",
                "reason": "empty_input",
                "message": "The request was empty, so the orchestrator could not choose an agent.",
            }

        if any(term in normalized for term in ("formalize", "structured consent", "structured record", "json record", "machine-readable")):
            if draft_available:
                return {"intent": "formalize_consent", "reason": "formalization_keyword_match", "message": None}
            return {
                "intent": "clarification",
                "reason": "formalization_requires_draft",
                "message": "Formalization needs a personalized consent draft first.",
            }

        if "generate" in normalized or "draft" in normalized or "personalize" in normalized or "rewrite" in normalized:
            if template_available:
                return {"intent": "personalize_consent", "reason": "draft_generation_keyword_match", "message": None}
            return {
                "intent": "clarification",
                "reason": "draft_requires_template",
                "message": "Draft generation needs a base consent template or a saved template in the run.",
            }

        if normalized.endswith("?") or normalized.startswith(QUESTION_INTENT_PREFIXES):
            return {"intent": "answer_question", "reason": "question_shape_detected", "message": None}

        if draft_available and any(term in normalized for term in ("explain", "clarify", "what does", "tell me")):
            return {"intent": "answer_question", "reason": "conversational_follow_up_detected", "message": None}

        return {
            "intent": "clarification",
            "reason": "ambiguous_request",
            "message": "The request was ambiguous, so the orchestrator could not safely choose the next agent.",
        }

    def classify_user_request_with_llm(
        self,
        *,
        user_input: str,
        draft_available: bool,
        template_available: bool,
    ) -> dict[str, Any]:
        user_prompt = self.tools.prompts.render(
            "orchestrator_route_user.txt",
            {
                "user_input": user_input,
                "draft_available": json.dumps(draft_available),
                "template_available": json.dumps(template_available),
            },
        )
        payload = self.tools.generation.call_json_model(
            messages=[
                {"role": "system", "content": self.tools.prompts.load("orchestrator_route_system.txt")},
                {"role": "user", "content": user_prompt},
            ],
            schema_name="orchestrator_request_route",
            schema=ORCHESTRATOR_REQUEST_ROUTE_SCHEMA,
            temperature=0.0,
            max_tokens=220,
        )
        intent = str(payload.get("intent", "")).strip()
        reason = str(payload.get("reason", "")).strip() or "llm_route"
        message = payload.get("message")
        if isinstance(message, str):
            message = message.strip() or None
        else:
            message = None
        return {"intent": intent, "reason": reason, "message": message}

    def classify_user_request(
        self,
        *,
        user_input: str,
        run_id: str,
        template_path: Path | None = None,
        draft_path: Path | None = None,
        use_llm: bool = True,
    ) -> dict[str, Any]:
        draft_available = bool(
            draft_path
            or self.tools.artifacts.run_path(run_id, "outputs", "personalized_consent_draft.json").exists()
        )
        template_available = bool(template_path or self.tools.artifacts.run_path(run_id, "inputs", "base_template.txt").exists())
        normalized = user_input.strip().lower()
        if not normalized:
            return self.classify_user_request_fallback(
                user_input=user_input,
                draft_available=draft_available,
                template_available=template_available,
            )

        route: dict[str, Any] | None = None
        if use_llm and self.llm_planning_available():
            try:
                route = self.classify_user_request_with_llm(
                    user_input=user_input,
                    draft_available=draft_available,
                    template_available=template_available,
                )
                route["planning_mode"] = "model"
            except Exception:
                route = None

        if route is None:
            route = self.classify_user_request_fallback(
                user_input=user_input,
                draft_available=draft_available,
                template_available=template_available,
            )
            route["planning_mode"] = "fallback"

        if route["intent"] == "formalize_consent" and not draft_available:
            return {
                "intent": "clarification",
                "reason": "formalization_requires_draft",
                "message": "Formalization needs a personalized consent draft first.",
                "planning_mode": route.get("planning_mode", "fallback"),
            }
        if route["intent"] == "personalize_consent" and not template_available:
            return {
                "intent": "clarification",
                "reason": "draft_requires_template",
                "message": "Draft generation needs a base consent template or a saved template in the run.",
                "planning_mode": route.get("planning_mode", "fallback"),
            }
        return route

    def plan_personalization_grounding(
        self,
        *,
        run_id: str,
        patient_profile: PatientProfile,
        base_template_text: str,
        generation_query: str | None,
        top_k: int | None,
        retrieval_mode: str | None,
        source_group_filters: list[str] | None,
        source_id_filters: list[str] | None,
        filter_logic: str | None,
    ) -> dict[str, Any]:
        planned_source_groups = list(source_group_filters or ["regulatory_guidance", "trial_materials"])
        planned_filter_logic = filter_logic or ("union" if source_group_filters or source_id_filters else "union")
        study_query_context = self.tools.retrieval.build_study_query_context(run_id, list(source_id_filters or []))
        base_query = generation_query or self.tools.retrieval.build_personalization_query(patient_profile, base_template_text)
        study_specific_query = " | ".join(
            part
            for part in (
                study_query_context.get("query_terms", ""),
                base_query,
                "study specific trial facts purpose procedures intervention visits eligibility condition treatment",
            )
            if str(part).strip()
        )
        required_source_groups: list[str] = []
        if "regulatory_guidance" in planned_source_groups:
            required_source_groups.append("regulatory_guidance")
        if "trial_materials" in planned_source_groups or bool(source_id_filters):
            required_source_groups.append("trial_materials")
        if not required_source_groups:
            required_source_groups = ["regulatory_guidance"]

        preferred_source_groups = [
            source_group
            for source_group in ("trial_materials", "regulatory_guidance")
            if source_group in planned_source_groups or source_group in required_source_groups
        ]
        return {
            "query": study_specific_query if study_query_context else base_query,
            "study_query": study_specific_query if study_query_context else base_query,
            "regulatory_query": base_query,
            "study_query_context": study_query_context,
            "top_k": top_k or self.runtime.config.retrieval.top_k,
            "retrieval_mode": retrieval_mode or self.runtime.config.retrieval.retrieval_mode,
            "source_group_filters": planned_source_groups,
            "source_id_filters": list(source_id_filters or []),
            "filter_logic": planned_filter_logic,
            "required_source_groups": required_source_groups,
            "preferred_source_groups": preferred_source_groups,
        }

    def should_split_study_scoped_retrieval(
        self,
        *,
        source_group_filters: list[str],
        source_id_filters: list[str],
    ) -> bool:
        group_set = {str(item).strip() for item in source_group_filters if str(item).strip()}
        return bool(source_id_filters) and {"trial_materials", "regulatory_guidance"}.issubset(group_set)

    def retrieve_scoped_grounding_artifacts(
        self,
        *,
        run_id: str,
        query: str,
        top_k: int,
        retrieval_mode: str,
        source_group_filters: list[str],
        source_id_filters: list[str],
        filter_logic: str,
        purpose_prefix: str,
        emit_result_to: str,
        study_query: str | None = None,
        regulatory_query: str | None = None,
    ) -> dict[str, Any]:
        split_scoped = self.should_split_study_scoped_retrieval(
            source_group_filters=source_group_filters,
            source_id_filters=source_id_filters,
        )
        if not split_scoped:
            single_artifacts = self.rag_agent.retrieve_evidence(
                run_id=run_id,
                query=query,
                top_k=top_k,
                retrieval_mode=retrieval_mode,
                source_group_filters=source_group_filters,
                source_id_filters=source_id_filters,
                filter_logic=filter_logic,
                purpose=purpose_prefix,
                emit_result_to=emit_result_to,
            )
            return {
                "retrieval_artifacts": single_artifacts,
                "scoped_retrieval": False,
                "study_hits_path": None,
                "regulatory_hits_path": None,
                "merged_evidence_package_path": None,
                "study_result_handoff_path": single_artifacts.get("result_handoff_path"),
                "regulatory_result_handoff_path": None,
            }

        study_top_k = max(1, (top_k + 1) // 2)
        regulatory_top_k = max(1, top_k - study_top_k)
        study_artifacts = self.rag_agent.retrieve_evidence(
            run_id=run_id,
            query=study_query or query,
            top_k=study_top_k,
            retrieval_mode=retrieval_mode,
            source_group_filters=["trial_materials"],
            source_id_filters=source_id_filters,
            filter_logic="intersection",
            purpose=f"{purpose_prefix}_study",
            emit_result_to=emit_result_to,
        )
        regulatory_artifacts = self.rag_agent.retrieve_evidence(
            run_id=run_id,
            query=regulatory_query or query,
            top_k=regulatory_top_k,
            retrieval_mode=retrieval_mode,
            source_group_filters=["regulatory_guidance"],
            source_id_filters=[],
            filter_logic="intersection",
            purpose=f"{purpose_prefix}_regulatory",
            emit_result_to=emit_result_to,
        )
        merged_artifacts = self.merge_retrieval_artifacts(
            primary_artifacts=study_artifacts,
            recovery_artifacts=regulatory_artifacts,
        )
        merged_artifacts["scoped_retrieval_strategy"] = "study_plus_regulatory_split"
        merged_artifacts["study_query"] = study_query or query
        merged_artifacts["regulatory_query"] = regulatory_query or query

        outputs_dir = self.tools.artifacts.run_path(run_id, "outputs")
        study_hits_path = outputs_dir / f"{purpose_prefix}_study_retrieval_hits.json"
        regulatory_hits_path = outputs_dir / f"{purpose_prefix}_regulatory_retrieval_hits.json"
        merged_evidence_package_path = outputs_dir / f"{purpose_prefix}_merged_evidence_package.json"
        self.tools.artifacts.write_json(study_hits_path, study_artifacts["retrieval_hits"])
        self.tools.artifacts.write_json(regulatory_hits_path, regulatory_artifacts["retrieval_hits"])
        self.tools.artifacts.write_json(merged_evidence_package_path, merged_artifacts["evidence_package"])
        return {
            "retrieval_artifacts": merged_artifacts,
            "scoped_retrieval": True,
            "study_hits_path": str(study_hits_path),
            "regulatory_hits_path": str(regulatory_hits_path),
            "merged_evidence_package_path": str(merged_evidence_package_path),
            "study_result_handoff_path": study_artifacts.get("result_handoff_path"),
            "regulatory_result_handoff_path": regulatory_artifacts.get("result_handoff_path"),
        }

    def build_study_specific_personalization_enrichment_plan(
        self,
        *,
        retrieval_plan: dict[str, Any],
        retrieval_artifacts: dict[str, Any],
    ) -> dict[str, Any] | None:
        source_group_filters = list(retrieval_plan.get("source_group_filters", []))
        source_id_filters = list(retrieval_plan.get("source_id_filters", []))
        if "trial_materials" not in source_group_filters and not source_id_filters:
            return None

        current_study_hits = int(retrieval_artifacts.get("evidence_package", {}).get("role_counts", {}).get("study_specific", 0))
        minimum_study_hits = 2 if source_id_filters else 1
        if current_study_hits >= minimum_study_hits:
            return None

        query_parts = [
            str(retrieval_plan.get("query", "")).strip(),
            "study purpose procedures intervention visits schedule eligibility study specific details",
        ]
        enrichment_query = " | ".join(part for part in query_parts if part)
        return {
            "query": enrichment_query,
            "top_k": max(int(retrieval_plan.get("top_k", self.runtime.config.retrieval.top_k)), 6),
            "retrieval_mode": retrieval_plan.get("retrieval_mode", self.runtime.config.retrieval.retrieval_mode),
            "source_group_filters": ["trial_materials"],
            "source_id_filters": source_id_filters,
            "filter_logic": "intersection",
            "minimum_study_specific_hits": minimum_study_hits,
            "current_study_specific_hits": current_study_hits,
        }

    def plan_draft_content_fallback(
        self,
        *,
        patient_profile: PatientProfile,
        base_template_text: str,
        retrieval_artifacts: dict[str, Any],
    ) -> dict[str, Any]:
        evidence_package = retrieval_artifacts["evidence_package"]
        role_text = {
            "study_specific": evidence_package.get("study_specific_context", ""),
            "regulatory": evidence_package.get("regulatory_context", ""),
            "other": evidence_package.get("other_context", ""),
        }
        role_markers = {
            "study_specific": [entry["marker"] for entry in evidence_package.get("study_specific_citation_map", [])],
            "regulatory": [entry["marker"] for entry in evidence_package.get("regulatory_citation_map", [])],
            "other": [entry["marker"] for entry in evidence_package.get("other_citation_map", [])],
        }
        role_support = {
            role: self.tools.evaluation.evaluate_required_elements(text)
            for role, text in role_text.items()
        }
        template_support = self.tools.evaluation.evaluate_required_elements(base_template_text)

        elements: list[dict[str, Any]] = []
        for element_id in CONSENT_PLAN_ELEMENT_IDS:
            supporting_roles = [
                role
                for role in ("study_specific", "regulatory", "other")
                if role_support.get(role, {}).get(element_id)
            ]
            preferred_role = DRAFT_PLAN_ROLE_HINTS[element_id]
            if len(supporting_roles) >= 2:
                preferred_source_role = "both" if {
                    "study_specific",
                    "regulatory",
                }.issubset(set(supporting_roles)) else supporting_roles[0]
            elif supporting_roles:
                preferred_source_role = supporting_roles[0]
            elif template_support.get(element_id):
                preferred_source_role = preferred_role
            else:
                preferred_source_role = "none"

            supported_from_evidence = bool(supporting_roles)
            supported_from_template = bool(template_support.get(element_id))
            if supported_from_evidence and supported_from_template:
                status = "supported"
            elif supported_from_evidence or supported_from_template:
                status = "partially_supported"
            else:
                status = "unsupported"

            if preferred_source_role == "both":
                recommended_markers = list(dict.fromkeys(role_markers["study_specific"][:1] + role_markers["regulatory"][:1]))
            elif preferred_source_role in role_markers:
                recommended_markers = role_markers[preferred_source_role][:2]
            elif supporting_roles:
                recommended_markers = role_markers[supporting_roles[0]][:2]
            else:
                recommended_markers = []

            elements.append(
                {
                    "element_id": element_id,
                    "status": status,
                    "preferred_source_role": preferred_source_role,
                    "recommended_markers": recommended_markers,
                    "instruction": DRAFT_PLAN_INSTRUCTIONS[element_id],
                }
            )

        low_literacy = patient_profile.health_literacy.strip().lower() == "low"
        overall_strategy = (
            "Use separate short cited sentences for study facts and participant-rights statements. "
            "Prefer low-literacy wording and do not invent unsupported study details."
            if low_literacy
            else "Use grounded study-specific and regulatory statements with clear inline citations and complete consent coverage."
        )
        return {
            "overall_strategy": overall_strategy,
            "elements": elements,
            "planning_mode": "fallback",
            "planning_reason": "heuristic_content_plan",
        }

    def plan_draft_content_with_llm(
        self,
        *,
        patient_profile: PatientProfile,
        base_template_text: str,
        retrieval_artifacts: dict[str, Any],
        generation_query: str,
    ) -> dict[str, Any]:
        evidence_package = retrieval_artifacts["evidence_package"]
        user_prompt = self.tools.prompts.render(
            "orchestrator_draft_plan_user.txt",
            {
                "participant_profile_json": json.dumps(asdict(patient_profile), indent=2),
                "base_template_text": base_template_text or "No base consent template was supplied.",
                "generation_query": generation_query,
                "study_specific_context": evidence_package.get("study_specific_context") or "No study-specific context was available.",
                "regulatory_context": evidence_package.get("regulatory_context") or "No regulatory context was available.",
                "other_grounding_context": evidence_package.get("other_context") or "No additional grounding context was available.",
            },
        )
        payload = self.tools.generation.call_json_model(
            messages=[
                {"role": "system", "content": self.tools.prompts.load("orchestrator_draft_plan_system.txt")},
                {"role": "user", "content": user_prompt},
            ],
            schema_name="orchestrator_draft_plan",
            schema=ORCHESTRATOR_DRAFT_PLAN_SCHEMA,
            temperature=0.0,
            max_tokens=700,
        )
        return {
            "overall_strategy": str(payload.get("overall_strategy", "")).strip(),
            "elements": payload.get("elements", []),
            "planning_mode": "model",
            "planning_reason": "llm_draft_content_plan",
        }

    def normalize_draft_content_plan(
        self,
        plan: dict[str, Any],
        *,
        fallback_plan: dict[str, Any],
    ) -> dict[str, Any]:
        plan_elements = plan.get("elements")
        if not isinstance(plan_elements, list):
            plan_elements = []
        fallback_elements_by_id = {
            item["element_id"]: item
            for item in fallback_plan["elements"]
        }
        normalized_elements: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for item in plan_elements:
            if not isinstance(item, dict):
                continue
            element_id = str(item.get("element_id", "")).strip()
            if element_id not in fallback_elements_by_id or element_id in seen_ids:
                continue
            seen_ids.add(element_id)
            fallback_item = fallback_elements_by_id[element_id]
            status = str(item.get("status", fallback_item["status"])).strip()
            if status not in {"supported", "partially_supported", "unsupported"}:
                status = fallback_item["status"]
            preferred_source_role = str(item.get("preferred_source_role", fallback_item["preferred_source_role"])).strip()
            if preferred_source_role not in {"study_specific", "regulatory", "both", "other", "none"}:
                preferred_source_role = fallback_item["preferred_source_role"]
            recommended_markers = item.get("recommended_markers")
            if not isinstance(recommended_markers, list):
                recommended_markers = fallback_item["recommended_markers"]
            else:
                normalized_markers: list[str] = []
                for marker in recommended_markers:
                    marker_text = str(marker).strip()
                    if re.fullmatch(r"\d+", marker_text):
                        marker_text = f"[{marker_text}]"
                    if marker_text:
                        normalized_markers.append(marker_text)
                recommended_markers = normalized_markers or fallback_item["recommended_markers"]
            instruction = str(item.get("instruction", "")).strip() or fallback_item["instruction"]
            normalized_elements.append(
                {
                    "element_id": element_id,
                    "status": status,
                    "preferred_source_role": preferred_source_role,
                    "recommended_markers": recommended_markers,
                    "instruction": instruction,
                }
            )

        for element_id in CONSENT_PLAN_ELEMENT_IDS:
            if element_id in seen_ids:
                continue
            normalized_elements.append(fallback_elements_by_id[element_id])

        return {
            "overall_strategy": str(plan.get("overall_strategy", "")).strip() or fallback_plan["overall_strategy"],
            "elements": normalized_elements,
            "planning_mode": plan.get("planning_mode", fallback_plan["planning_mode"]),
            "planning_reason": plan.get("planning_reason", fallback_plan["planning_reason"]),
        }

    def plan_draft_content(
        self,
        *,
        patient_profile: PatientProfile,
        base_template_text: str,
        retrieval_artifacts: dict[str, Any],
        generation_query: str,
        use_llm: bool = True,
    ) -> dict[str, Any]:
        fallback_plan = self.plan_draft_content_fallback(
            patient_profile=patient_profile,
            base_template_text=base_template_text,
            retrieval_artifacts=retrieval_artifacts,
        )
        if not use_llm or not self.llm_planning_available():
            return fallback_plan
        try:
            planned = self.plan_draft_content_with_llm(
                patient_profile=patient_profile,
                base_template_text=base_template_text,
                retrieval_artifacts=retrieval_artifacts,
                generation_query=generation_query,
            )
        except Exception:
            return fallback_plan
        return self.normalize_draft_content_plan(planned, fallback_plan=fallback_plan)

    def plan_element_recovery(
        self,
        *,
        draft_audit: dict[str, Any],
        draft_content_plan: dict[str, Any],
        retrieval_plan: dict[str, Any],
    ) -> dict[str, Any] | None:
        missing_required = {
            str(item).strip()
            for item in draft_audit.get("missing_required_elements", [])
            if str(item).strip()
        }
        if not missing_required:
            return None

        plan_elements = draft_content_plan.get("elements", [])
        if not isinstance(plan_elements, list):
            return None

        targets: list[dict[str, Any]] = []
        preferred_groups: list[str] = []
        study_query_hints: list[str] = []
        regulatory_query_hints: list[str] = []
        study_targets: list[dict[str, Any]] = []
        regulatory_targets: list[dict[str, Any]] = []
        for item in plan_elements:
            if not isinstance(item, dict):
                continue
            element_id = str(item.get("element_id", "")).strip()
            if element_id not in missing_required:
                continue
            status = str(item.get("status", "")).strip()
            if status not in {"supported", "partially_supported"}:
                continue
            preferred_role = str(item.get("preferred_source_role", "none")).strip() or "none"
            if preferred_role == "study_specific":
                preferred_groups.append("trial_materials")
                study_query_hints.append(ELEMENT_RECOVERY_QUERY_HINTS.get(element_id, element_id.replace("_", " ")))
            elif preferred_role == "regulatory":
                preferred_groups.append("regulatory_guidance")
                regulatory_query_hints.append(ELEMENT_RECOVERY_QUERY_HINTS.get(element_id, element_id.replace("_", " ")))
            elif preferred_role == "both":
                preferred_groups.extend(["trial_materials", "regulatory_guidance"])
                study_query_hints.append(ELEMENT_RECOVERY_QUERY_HINTS.get(element_id, element_id.replace("_", " ")))
                regulatory_query_hints.append(ELEMENT_RECOVERY_QUERY_HINTS.get(element_id, element_id.replace("_", " ")))
            targets.append(
                {
                    "element_id": element_id,
                    "status": status,
                    "preferred_source_role": preferred_role,
                    "recommended_markers": list(item.get("recommended_markers", [])),
                    "instruction": str(item.get("instruction", "")).strip(),
                }
            )
            if preferred_role in {"study_specific", "both"}:
                study_targets.append(targets[-1])
            if preferred_role in {"regulatory", "both"}:
                regulatory_targets.append(targets[-1])

        if not targets:
            return None

        deduped_preferred_groups = list(dict.fromkeys(preferred_groups))
        top_k = max(int(retrieval_plan.get("top_k", self.runtime.config.retrieval.top_k)), 6)
        retrieval_mode = retrieval_plan.get("retrieval_mode", self.runtime.config.retrieval.retrieval_mode)
        base_query = str(retrieval_plan.get("query", "")).strip()
        retrieval_passes: list[dict[str, Any]] = []

        if study_targets:
            study_query = " | ".join(
                part for part in (base_query, " | ".join(dict.fromkeys(study_query_hints))) if part
            )
            retrieval_passes.append(
                {
                    "pass_label": "study_specific",
                    "preferred_source_role": "study_specific",
                    "query": study_query,
                    "top_k": top_k,
                    "retrieval_mode": retrieval_mode,
                    "source_group_filters": ["trial_materials"],
                    "source_id_filters": list(retrieval_plan.get("source_id_filters", [])),
                    "filter_logic": "intersection",
                    "target_elements": [dict(item) for item in study_targets],
                }
            )

        if regulatory_targets:
            regulatory_query = " | ".join(
                part for part in (base_query, " | ".join(dict.fromkeys(regulatory_query_hints))) if part
            )
            retrieval_passes.append(
                {
                    "pass_label": "regulatory",
                    "preferred_source_role": "regulatory",
                    "query": regulatory_query,
                    "top_k": top_k,
                    "retrieval_mode": retrieval_mode,
                    "source_group_filters": ["regulatory_guidance"],
                    "source_id_filters": [],
                    "filter_logic": "intersection",
                    "target_elements": [dict(item) for item in regulatory_targets],
                }
            )

        if not retrieval_passes:
            return None

        primary_pass = retrieval_passes[0]
        retrieval_strategy_effective = "split_passes" if len(retrieval_passes) > 1 else f"{primary_pass['pass_label']}_single_pass"
        return {
            "target_elements": targets,
            "query": primary_pass["query"],
            "top_k": top_k,
            "retrieval_mode": retrieval_mode,
            "source_group_filters": list(primary_pass["source_group_filters"]),
            "source_id_filters": list(primary_pass["source_id_filters"]),
            "filter_logic": primary_pass["filter_logic"],
            "preferred_source_groups": deduped_preferred_groups,
            "retrieval_strategy_effective": retrieval_strategy_effective,
            "retrieval_passes": retrieval_passes,
        }

    def merge_retrieval_artifacts(
        self,
        *,
        primary_artifacts: dict[str, Any],
        recovery_artifacts: dict[str, Any],
    ) -> dict[str, Any]:
        merged_queries: list[str] = []
        for value in list(primary_artifacts.get("recovery_queries", [])) + list(recovery_artifacts.get("recovery_queries", [])):
            query = str(value or "").strip()
            if query and query not in merged_queries:
                merged_queries.append(query)
        recovery_query = str(recovery_artifacts.get("query", "")).strip()
        if recovery_query and recovery_query not in merged_queries:
            merged_queries.append(recovery_query)

        merged_hits: list[dict[str, Any]] = []
        seen_chunk_ids: set[str] = set()
        for index, hit in enumerate(
            list(primary_artifacts.get("retrieval_hits", [])) + list(recovery_artifacts.get("retrieval_hits", [])),
            start=1,
        ):
            if not isinstance(hit, dict):
                continue
            chunk_id = str(hit.get("chunk_id", "")).strip()
            if not chunk_id or chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk_id)
            merged_hit = dict(hit)
            merged_hit["rank"] = len(merged_hits) + 1
            merged_hits.append(merged_hit)

        citation_map = self.tools.retrieval.build_citation_map(merged_hits)
        evidence_package = self.tools.retrieval.build_evidence_package(merged_hits)
        return {
            **primary_artifacts,
            "retrieval_hits": merged_hits,
            "retrieved_context": self.tools.retrieval.format_retrieval_context(merged_hits),
            "citation_map": citation_map,
            "evidence_package": evidence_package,
            "recovery_applied": True,
            "recovery_hit_count": len(recovery_artifacts.get("retrieval_hits", [])),
            "recovery_queries": merged_queries,
        }

    def plan_question_grounding_fallback(
        self,
        *,
        question: str,
        top_k: int | None,
        retrieval_mode: str | None,
        source_group_filters: list[str] | None,
        source_id_filters: list[str] | None,
        filter_logic: str | None,
    ) -> dict[str, Any]:
        normalized = question.lower()
        regulatory_needed = contains_any_keyword(normalized, REGULATORY_QUESTION_KEYWORDS) or matches_any_pattern(
            normalized,
            REGULATORY_PRIORITY_PATTERNS,
        )
        study_keyword_present = contains_any_keyword(normalized, STUDY_QUESTION_KEYWORDS)
        generic_study_reference_present = contains_any_keyword(normalized, GENERIC_STUDY_REFERENCE_KEYWORDS)
        study_pattern_match = matches_any_pattern(normalized, STUDY_SPECIFIC_PATTERNS)
        study_needed = study_keyword_present or study_pattern_match

        if source_group_filters:
            planned_source_groups = list(source_group_filters)
        elif study_needed:
            planned_source_groups = ["trial_materials", "regulatory_guidance"]
        else:
            planned_source_groups = ["regulatory_guidance", "trial_materials"]

        required_source_groups: list[str] = []
        if study_needed:
            required_source_groups.append("trial_materials")
        if regulatory_needed:
            required_source_groups.append("regulatory_guidance")
        if not required_source_groups:
            required_source_groups.append("trial_materials" if generic_study_reference_present else "regulatory_guidance")

        if study_needed and regulatory_needed:
            question_profile = "study_plus_regulatory"
        elif study_needed:
            question_profile = "study"
        elif regulatory_needed:
            question_profile = "regulatory"
        elif generic_study_reference_present:
            question_profile = "study_context_optional"
        else:
            question_profile = "regulatory"

        return {
            "query": question,
            "top_k": top_k or self.runtime.config.retrieval.top_k,
            "retrieval_mode": retrieval_mode or self.runtime.config.retrieval.retrieval_mode,
            "source_group_filters": planned_source_groups,
            "source_id_filters": list(source_id_filters or []),
            "filter_logic": filter_logic or ("union" if source_id_filters or planned_source_groups else "intersection"),
            "required_source_groups": required_source_groups,
            "preferred_source_groups": (
                ["regulatory_guidance", "trial_materials"]
                if regulatory_needed and not study_needed
                else ["trial_materials", "regulatory_guidance"]
            ),
            "question_profile": question_profile,
            "study_keyword_present": study_keyword_present,
            "generic_study_reference_present": generic_study_reference_present,
            "regulatory_keyword_present": regulatory_needed,
            "planning_mode": "fallback",
            "planning_reason": "heuristic_fallback",
        }

    def plan_question_grounding_with_llm(
        self,
        *,
        question: str,
        source_group_filters: list[str] | None,
        source_id_filters: list[str] | None,
    ) -> dict[str, Any]:
        user_prompt = self.tools.prompts.render(
            "orchestrator_question_plan_user.txt",
            {
                "question": question,
                "source_group_filters_json": json.dumps(list(source_group_filters or [])),
                "source_id_filters_json": json.dumps(list(source_id_filters or [])),
            },
        )
        payload = self.tools.generation.call_json_model(
            messages=[
                {"role": "system", "content": self.tools.prompts.load("orchestrator_question_plan_system.txt")},
                {"role": "user", "content": user_prompt},
            ],
            schema_name="orchestrator_question_plan",
            schema=ORCHESTRATOR_QUESTION_PLAN_SCHEMA,
            temperature=0.0,
            max_tokens=320,
        )
        allowed_groups = {"regulatory_guidance", "trial_materials"}
        required_source_groups = [
            group for group in payload.get("required_source_groups", []) if isinstance(group, str) and group in allowed_groups
        ]
        preferred_source_groups = [
            group for group in payload.get("preferred_source_groups", []) if isinstance(group, str) and group in allowed_groups
        ]
        question_profile = str(payload.get("question_profile", "")).strip() or "regulatory"
        retrieval_query = str(payload.get("retrieval_query", "")).strip() or question
        planning_reason = str(payload.get("reason", "")).strip() or "llm_question_plan"
        return {
            "question_profile": question_profile,
            "retrieval_query": retrieval_query,
            "required_source_groups": required_source_groups,
            "preferred_source_groups": preferred_source_groups,
            "planning_reason": planning_reason,
        }

    def plan_question_grounding(
        self,
        *,
        question: str,
        top_k: int | None,
        retrieval_mode: str | None,
        source_group_filters: list[str] | None,
        source_id_filters: list[str] | None,
        filter_logic: str | None,
        use_llm: bool = True,
    ) -> dict[str, Any]:
        fallback_plan = self.plan_question_grounding_fallback(
            question=question,
            top_k=top_k,
            retrieval_mode=retrieval_mode,
            source_group_filters=source_group_filters,
            source_id_filters=source_id_filters,
            filter_logic=filter_logic,
        )
        if not use_llm or not self.llm_planning_available():
            return fallback_plan

        try:
            llm_plan = self.plan_question_grounding_with_llm(
                question=question,
                source_group_filters=source_group_filters,
                source_id_filters=source_id_filters,
            )
        except Exception:
            return fallback_plan

        preferred_source_groups = llm_plan["preferred_source_groups"] or fallback_plan["preferred_source_groups"]
        required_source_groups = llm_plan["required_source_groups"] or fallback_plan["required_source_groups"]
        if source_group_filters:
            planned_source_groups = list(source_group_filters)
        else:
            planned_source_groups = preferred_source_groups or fallback_plan["source_group_filters"]

        return {
            "query": llm_plan["retrieval_query"] or fallback_plan["query"],
            "top_k": top_k or self.runtime.config.retrieval.top_k,
            "retrieval_mode": retrieval_mode or self.runtime.config.retrieval.retrieval_mode,
            "source_group_filters": planned_source_groups,
            "source_id_filters": list(source_id_filters or []),
            "filter_logic": filter_logic or ("union" if source_id_filters or planned_source_groups else "intersection"),
            "required_source_groups": required_source_groups,
            "preferred_source_groups": preferred_source_groups,
            "question_profile": llm_plan["question_profile"] or fallback_plan["question_profile"],
            "study_keyword_present": fallback_plan["study_keyword_present"],
            "generic_study_reference_present": fallback_plan["generic_study_reference_present"],
            "regulatory_keyword_present": fallback_plan["regulatory_keyword_present"],
            "planning_mode": "model",
            "planning_reason": llm_plan["planning_reason"],
        }

    def assess_evidence_sufficiency(
        self,
        *,
        retrieval_artifacts: dict[str, Any],
        required_source_groups: list[str],
        min_hit_count: int = 1,
    ) -> dict[str, Any]:
        retrieval_hits = retrieval_artifacts.get("retrieval_hits", [])
        observed_groups = {
            str(hit.get("metadata", {}).get("source_group", "")).strip()
            for hit in retrieval_hits
            if isinstance(hit, dict)
        }
        observed_groups.discard("")
        missing_groups = [group for group in required_source_groups if group not in observed_groups]
        hit_count = len(retrieval_hits)
        return {
            "sufficient": hit_count >= min_hit_count and not missing_groups,
            "hit_count": hit_count,
            "observed_source_groups": sorted(observed_groups),
            "missing_required_source_groups": missing_groups,
        }

    def build_clarification_response(
        self,
        *,
        run_id: str,
        reason: str,
        message: str,
        missing_required_source_groups: list[str] | None = None,
        suggested_actions: list[str] | None = None,
    ) -> dict[str, Any]:
        return {
            "run_id": run_id,
            "status": "needs_clarification",
            "reason": reason,
            "message": message,
            "missing_required_source_groups": missing_required_source_groups or [],
            "suggested_actions": suggested_actions or [],
        }

    def audit_draft_quality(
        self,
        *,
        run_id: str,
        draft_payload: dict[str, Any],
        retrieval_hits: list[dict[str, Any]],
        patient_profile: PatientProfile,
        label: str,
        draft_content_plan: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        started_at = utc_now_iso()
        available_markers = [f"[{index}]" for index in range(1, len(retrieval_hits) + 1)]
        draft_summary = self.tools.evaluation.summarize_personalized_draft(
            draft_payload,
            available_markers=available_markers,
            health_literacy=patient_profile.health_literacy,
        )
        audit = self.tools.evaluation.build_draft_revision_audit(
            draft_summary,
            draft_content_plan=draft_content_plan,
        )
        audit_path = self.tools.artifacts.run_path(run_id, "outputs", f"draft_revision_audit.{label}.json")
        self.tools.artifacts.write_json(audit_path, audit)
        self.record_stage(
            run_id,
            stage_name=f"audit_draft_quality_{label}",
            status="completed",
            started_at=started_at,
            ended_at=utc_now_iso(),
            outputs={
                "audit_path": str(audit_path),
                "needs_revision": audit["needs_revision"],
                "issue_count": len(audit["issues"]),
                "quality_score": audit["quality_score"],
                "missing_required_elements": audit["missing_required_elements"],
                "missing_planned_required_elements": audit["missing_planned_required_elements"],
            },
            notes="Audited the participant-facing consent draft against coverage, citation, and readability thresholds.",
        )
        return {
            **audit,
            "path": str(audit_path),
        }

    def snapshot_initial_personalization_artifacts(self, run_id: str) -> dict[str, str]:
        outputs_dir = self.tools.artifacts.run_path(run_id, "outputs")
        snapshots: dict[str, str] = {}
        for source_name, snapshot_name in (
            ("personalized_consent_draft.json", "personalized_consent_draft.initial.json"),
            ("personalized_consent_draft.raw.json", "personalized_consent_draft.initial.raw.json"),
            ("personalization_request_bundle.json", "personalization_request_bundle.initial.json"),
        ):
            source_path = outputs_dir / source_name
            if not source_path.exists():
                continue
            snapshot_path = outputs_dir / snapshot_name
            shutil.copy2(source_path, snapshot_path)
            snapshots[source_name] = str(snapshot_path)
        return snapshots

    def write_final_draft_artifacts(
        self,
        *,
        run_id: str,
        draft_payload: dict[str, Any],
        revision_metadata: dict[str, Any],
        raw_source_path: str | None = None,
    ) -> str:
        outputs_dir = self.tools.artifacts.run_path(run_id, "outputs")
        final_payload = dict(draft_payload)
        final_payload["revision_metadata"] = revision_metadata
        final_output_path = outputs_dir / "personalized_consent_draft.json"
        self.tools.artifacts.write_json(final_output_path, final_payload)

        if raw_source_path:
            raw_source = Path(raw_source_path)
            canonical_raw_path = outputs_dir / "personalized_consent_draft.raw.json"
            if raw_source.exists() and raw_source.resolve() != canonical_raw_path.resolve():
                shutil.copy2(raw_source, canonical_raw_path)

        return str(final_output_path)

    def personalize_consent(
        self,
        *,
        run_id: str,
        patient_profile_path: Path | None = None,
        template_path: Path | None = None,
        generation_query: str | None = None,
        top_k: int | None = None,
        retrieval_mode: str | None = None,
        source_group_filters: list[str] | None = None,
        source_id_filters: list[str] | None = None,
        filter_logic: str | None = None,
        workflow_variant: str = "full_agentic",
        dry_run: bool = False,
    ) -> dict[str, Any]:
        started_at = utc_now_iso()
        workflow_variant = self.normalize_workflow_variant(workflow_variant)
        patient_profile = self.tools.state.resolve_patient_profile(run_id, patient_profile_path)
        base_template_text = self.tools.state.resolve_base_template_text(run_id, template_path)
        retrieval_plan = self.plan_personalization_grounding(
            run_id=run_id,
            patient_profile=patient_profile,
            base_template_text=base_template_text,
            generation_query=generation_query,
            top_k=top_k,
            retrieval_mode=retrieval_mode,
            source_group_filters=source_group_filters,
            source_id_filters=source_id_filters,
            filter_logic=filter_logic,
        )
        query = retrieval_plan["query"]
        orchestrator_to_rag_path: str | None = None
        study_enrichment_plan_path: str | None = None
        study_enrichment_hits_path: str | None = None
        study_enrichment_evidence_package_path: str | None = None
        study_enrichment_request_handoff_path: str | None = None
        study_enrichment_result_handoff_path: str | None = None
        scoped_retrieval_study_hits_path: str | None = None
        scoped_retrieval_regulatory_hits_path: str | None = None
        scoped_retrieval_evidence_package_path: str | None = None
        scoped_retrieval_study_result_handoff_path: str | None = None
        scoped_retrieval_regulatory_result_handoff_path: str | None = None
        scoped_retrieval_applied = False

        if workflow_variant == "vanilla_llm":
            retrieval_artifacts = self.build_empty_retrieval_artifacts(
                source_group_filters=retrieval_plan["source_group_filters"],
                source_id_filters=retrieval_plan["source_id_filters"],
                filter_logic=retrieval_plan["filter_logic"],
            )
        else:
            retrieval_strategy = (
                "study_plus_regulatory_split"
                if self.should_split_study_scoped_retrieval(
                    source_group_filters=retrieval_plan["source_group_filters"],
                    source_id_filters=retrieval_plan["source_id_filters"],
                )
                else "single_pass"
            )
            orchestrator_to_rag = self.emit_handoff(
                run_id,
                to_agent=self.rag_agent.agent_label,
                purpose="personalization_grounding_request",
                payload={
                    "query": query,
                    "top_k": retrieval_plan["top_k"],
                    "retrieval_mode": retrieval_plan["retrieval_mode"],
                    "source_group_filters": retrieval_plan["source_group_filters"],
                    "source_id_filters": retrieval_plan["source_id_filters"],
                    "filter_logic": retrieval_plan["filter_logic"],
                    "retrieval_strategy": retrieval_strategy,
                    "study_query": retrieval_plan.get("study_query"),
                    "regulatory_query": retrieval_plan.get("regulatory_query"),
                    "study_query_context": retrieval_plan.get("study_query_context"),
                    "workflow_variant": workflow_variant,
                    "participant_profile": {
                        "participant_id": patient_profile.participant_id,
                        "health_literacy": patient_profile.health_literacy,
                        "language": patient_profile.language,
                        "jurisdiction": patient_profile.jurisdiction,
                    },
                    "base_template_present": bool(base_template_text),
                },
            )
            orchestrator_to_rag_path = orchestrator_to_rag["path"]
            scoped_retrieval_bundle = self.retrieve_scoped_grounding_artifacts(
                run_id=run_id,
                query=query,
                top_k=retrieval_plan["top_k"],
                retrieval_mode=retrieval_plan["retrieval_mode"],
                source_group_filters=retrieval_plan["source_group_filters"],
                source_id_filters=retrieval_plan["source_id_filters"],
                filter_logic=retrieval_plan["filter_logic"],
                purpose_prefix="personalization_grounding",
                emit_result_to=self.agent_label,
                study_query=retrieval_plan.get("study_query"),
                regulatory_query=retrieval_plan.get("regulatory_query"),
            )
            retrieval_artifacts = scoped_retrieval_bundle["retrieval_artifacts"]
            scoped_retrieval_applied = bool(scoped_retrieval_bundle["scoped_retrieval"])
            scoped_retrieval_study_hits_path = scoped_retrieval_bundle["study_hits_path"]
            scoped_retrieval_regulatory_hits_path = scoped_retrieval_bundle["regulatory_hits_path"]
            scoped_retrieval_evidence_package_path = scoped_retrieval_bundle["merged_evidence_package_path"]
            scoped_retrieval_study_result_handoff_path = scoped_retrieval_bundle["study_result_handoff_path"]
            scoped_retrieval_regulatory_result_handoff_path = scoped_retrieval_bundle["regulatory_result_handoff_path"]

            if workflow_variant == "full_agentic":
                study_enrichment_plan = self.build_study_specific_personalization_enrichment_plan(
                    retrieval_plan=retrieval_plan,
                    retrieval_artifacts=retrieval_artifacts,
                )
                if study_enrichment_plan:
                    enrichment_started_at = utc_now_iso()
                    study_enrichment_plan_file = self.tools.artifacts.run_path(run_id, "outputs", "draft_study_enrichment_plan.json")
                    self.tools.artifacts.write_json(study_enrichment_plan_file, study_enrichment_plan)
                    study_enrichment_plan_path = str(study_enrichment_plan_file)
                    enrichment_request_handoff = self.emit_handoff(
                        run_id,
                        to_agent=self.rag_agent.agent_label,
                        purpose="draft_study_specific_enrichment_request",
                        payload=study_enrichment_plan,
                    )
                    study_enrichment_request_handoff_path = enrichment_request_handoff["path"]
                    enrichment_retrieval = self.rag_agent.retrieve_evidence(
                        run_id=run_id,
                        query=study_enrichment_plan["query"],
                        top_k=study_enrichment_plan["top_k"],
                        retrieval_mode=study_enrichment_plan["retrieval_mode"],
                        source_group_filters=study_enrichment_plan["source_group_filters"],
                        source_id_filters=study_enrichment_plan["source_id_filters"],
                        filter_logic=study_enrichment_plan["filter_logic"],
                        purpose="draft_study_specific_enrichment",
                        emit_result_to=self.agent_label,
                    )
                    study_enrichment_result_handoff_path = enrichment_retrieval.get("result_handoff_path")
                    retrieval_artifacts = self.merge_retrieval_artifacts(
                        primary_artifacts=retrieval_artifacts,
                        recovery_artifacts=enrichment_retrieval,
                    )
                    enrichment_hits_file = self.tools.artifacts.run_path(run_id, "outputs", "draft_study_enrichment_hits.json")
                    enrichment_evidence_file = self.tools.artifacts.run_path(run_id, "outputs", "draft_study_enrichment_merged_evidence_package.json")
                    self.tools.artifacts.write_json(enrichment_hits_file, enrichment_retrieval["retrieval_hits"])
                    self.tools.artifacts.write_json(enrichment_evidence_file, retrieval_artifacts["evidence_package"])
                    study_enrichment_hits_path = str(enrichment_hits_file)
                    study_enrichment_evidence_package_path = str(enrichment_evidence_file)
                    self.record_stage(
                        run_id,
                        stage_name="enrich_personalization_study_grounding",
                        status="completed",
                        started_at=enrichment_started_at,
                        ended_at=utc_now_iso(),
                        outputs={
                            "study_enrichment_plan_path": study_enrichment_plan_path,
                            "study_enrichment_hits_path": study_enrichment_hits_path,
                            "study_enrichment_evidence_package_path": study_enrichment_evidence_package_path,
                            "study_specific_hit_count_after_enrichment": retrieval_artifacts["evidence_package"]["role_counts"]["study_specific"],
                            "study_enrichment_request_handoff_path": study_enrichment_request_handoff_path,
                            "study_enrichment_result_handoff_path": study_enrichment_result_handoff_path,
                        },
                        notes="The orchestrator ran an additional trial-material retrieval pass to strengthen study-specific grounding before draft generation.",
                    )

            if workflow_variant == "full_agentic":
                sufficiency = self.assess_evidence_sufficiency(
                    retrieval_artifacts=retrieval_artifacts,
                    required_source_groups=retrieval_plan["required_source_groups"],
                )
                if not sufficiency["sufficient"]:
                    clarification = self.build_clarification_response(
                        run_id=run_id,
                        reason="insufficient_personalization_evidence",
                        message=(
                            "The orchestrator could not find enough grounded evidence to safely draft the personalized consent."
                        ),
                        missing_required_source_groups=sufficiency["missing_required_source_groups"],
                        suggested_actions=[
                            "Add or fetch the missing study or regulatory sources for this run.",
                            "Retry with broader source filters or a study-specific source id.",
                        ],
                    )
                    self.record_stage(
                        run_id,
                        stage_name="personalize_consent",
                        status="clarification_requested",
                        started_at=started_at,
                        ended_at=utc_now_iso(),
                        inputs={
                            "patient_profile_path": str(patient_profile_path) if patient_profile_path else None,
                            "template_path": str(template_path) if template_path else None,
                            "generation_query": generation_query,
                            "workflow_variant": workflow_variant,
                        },
                        outputs={
                            "clarification_reason": clarification["reason"],
                            "missing_required_source_groups": clarification["missing_required_source_groups"],
                            "observed_source_groups": sufficiency["observed_source_groups"],
                        },
                        notes="The orchestrator paused instead of drafting because the grounding evidence was incomplete.",
                    )
                    return clarification

        draft_content_plan: dict[str, Any] | None = None
        draft_content_plan_path: str | None = None
        if workflow_variant == "full_agentic":
            draft_content_plan = self.plan_draft_content(
                patient_profile=patient_profile,
                base_template_text=base_template_text,
                retrieval_artifacts=retrieval_artifacts,
                generation_query=query,
                use_llm=not dry_run,
            )
            draft_content_plan_file = self.tools.artifacts.run_path(run_id, "outputs", "draft_content_plan.json")
            self.tools.artifacts.write_json(draft_content_plan_file, draft_content_plan)
            draft_content_plan_path = str(draft_content_plan_file)
            self.record_stage(
                run_id,
                stage_name="plan_draft_content",
                status="completed",
                started_at=started_at,
                ended_at=utc_now_iso(),
                outputs={
                    "draft_content_plan_path": draft_content_plan_path,
                    "planning_mode": draft_content_plan.get("planning_mode"),
                    "planning_reason": draft_content_plan.get("planning_reason"),
                    "element_count": len(draft_content_plan.get("elements", [])),
                },
                notes="The orchestrator produced a per-element content plan to guide grounded draft generation.",
            )

        orchestrator_to_personalization = self.emit_handoff(
            run_id,
            to_agent=self.personalization_agent.agent_label,
            purpose="personalized_consent_draft_request",
            payload={
                "generation_query": query,
                "workflow_variant": workflow_variant,
                "retrieval_hit_count": len(retrieval_artifacts["retrieval_hits"]),
                "retrieval_mode_used": retrieval_artifacts["mode_used"],
                "draft_content_plan_path": draft_content_plan_path,
                "draft_content_plan_mode": draft_content_plan.get("planning_mode") if isinstance(draft_content_plan, dict) else None,
                "unsupported_elements": [
                    item["element_id"]
                    for item in (draft_content_plan.get("elements", []) if isinstance(draft_content_plan, dict) else [])
                    if isinstance(item, dict) and item.get("status") == "unsupported"
                ],
                "base_template_present": bool(base_template_text),
                "participant_profile": {
                    "participant_id": patient_profile.participant_id,
                    "health_literacy": patient_profile.health_literacy,
                    "language": patient_profile.language,
                    "jurisdiction": patient_profile.jurisdiction,
                },
            },
        )
        draft_payload = self.personalization_agent.generate_draft(
            run_id=run_id,
            patient_profile=patient_profile,
            patient_profile_path=patient_profile_path,
            base_template_text=base_template_text,
            template_path=template_path,
            generation_query=query,
            retrieval_artifacts=retrieval_artifacts,
            draft_content_plan=draft_content_plan,
            top_k=retrieval_plan["top_k"],
            retrieval_mode=retrieval_plan["retrieval_mode"],
            source_group_filters=retrieval_plan["source_group_filters"],
            source_id_filters=retrieval_plan["source_id_filters"],
            workflow_variant=workflow_variant,
            dry_run=dry_run,
            emit_result_to=self.agent_label,
        )

        revision_attempted = False
        revision_applied = False
        decision_path: str | None = None
        initial_audit_path: str | None = None
        revision_audit_path: str | None = None
        final_audit_path: str | None = None
        final_output_path = draft_payload.get("output_path")
        final_request_bundle_path = draft_payload.get("request_bundle_path")
        revision_request_bundle_path: str | None = None
        revision_output_path: str | None = None
        revision_handoff_path: str | None = None
        revision_result_handoff_path: str | None = None
        recovery_plan_path: str | None = None
        recovery_retrieval_hits_path: str | None = None
        recovery_evidence_package_path: str | None = None
        recovery_request_handoff_path: str | None = None
        recovery_result_handoff_path: str | None = None

        if workflow_variant == "full_agentic" and not dry_run and draft_payload.get("response") is not None:
            initial_response = draft_payload["response"]
            initial_audit = self.audit_draft_quality(
                run_id=run_id,
                draft_payload=initial_response,
                retrieval_hits=retrieval_artifacts["retrieval_hits"],
                patient_profile=patient_profile,
                label="initial",
                draft_content_plan=draft_content_plan,
            )
            initial_audit_path = initial_audit["path"]
            final_audit_path = initial_audit_path
            decision: dict[str, Any] = {
                "revision_attempted": False,
                "revision_applied": False,
                "initial_audit_path": initial_audit_path,
                "final_audit_path": final_audit_path,
                "decision_reason": "initial_draft_met_quality_thresholds",
                "initial_quality_score": initial_audit["quality_score"],
            }

            selected_response = initial_response
            selected_raw_path = self.tools.artifacts.run_path(run_id, "outputs", "personalized_consent_draft.raw.json")

            if initial_audit["needs_revision"]:
                revision_attempted = True
                snapshot_paths = self.snapshot_initial_personalization_artifacts(run_id)
                revision_retrieval_artifacts = retrieval_artifacts
                recovery_targets: list[dict[str, Any]] = []
                focused_recovery_context = ""
                recovery_plan = self.plan_element_recovery(
                    draft_audit=initial_audit,
                    draft_content_plan=draft_content_plan,
                    retrieval_plan=retrieval_plan,
                )
                if recovery_plan:
                    recovery_started_at = utc_now_iso()
                    recovery_targets = list(recovery_plan["target_elements"])
                    recovery_passes = list(recovery_plan.get("retrieval_passes", []))
                    if not recovery_passes:
                        recovery_passes = [
                            {
                                "pass_label": "recovery",
                                "preferred_source_role": "study_specific",
                                "query": recovery_plan["query"],
                                "top_k": recovery_plan["top_k"],
                                "retrieval_mode": recovery_plan["retrieval_mode"],
                                "source_group_filters": recovery_plan["source_group_filters"],
                                "source_id_filters": recovery_plan["source_id_filters"],
                                "filter_logic": recovery_plan["filter_logic"],
                                "target_elements": recovery_targets,
                            }
                        ]
                    recovery_plan_file = self.tools.artifacts.run_path(run_id, "outputs", "draft_element_recovery_plan.json")
                    self.tools.artifacts.write_json(recovery_plan_file, recovery_plan)
                    recovery_plan_path = str(recovery_plan_file)
                    recovery_request_handoff_paths: list[str] = []
                    recovery_result_handoff_paths: list[str] = []
                    recovery_pass_hit_paths: list[str] = []
                    recovery_retrieval: dict[str, Any] | None = None
                    for recovery_pass in recovery_passes:
                        pass_label = str(recovery_pass.get("pass_label") or "recovery").strip() or "recovery"
                        pass_targets = list(recovery_pass.get("target_elements", []))
                        recovery_request_handoff = self.emit_handoff(
                            run_id,
                            to_agent=self.rag_agent.agent_label,
                            purpose=f"draft_element_recovery_grounding_request_{pass_label}",
                            payload={
                                "query": recovery_pass["query"],
                                "target_elements": pass_targets,
                                "top_k": recovery_pass["top_k"],
                                "retrieval_mode": recovery_pass["retrieval_mode"],
                                "source_group_filters": recovery_pass["source_group_filters"],
                                "source_id_filters": recovery_pass["source_id_filters"],
                                "filter_logic": recovery_pass["filter_logic"],
                                "retrieval_strategy_effective": recovery_plan.get("retrieval_strategy_effective"),
                            },
                        )
                        recovery_request_handoff_paths.append(recovery_request_handoff["path"])
                        pass_artifacts = self.rag_agent.retrieve_evidence(
                            run_id=run_id,
                            query=recovery_pass["query"],
                            top_k=recovery_pass["top_k"],
                            retrieval_mode=recovery_pass["retrieval_mode"],
                            source_group_filters=recovery_pass["source_group_filters"],
                            source_id_filters=recovery_pass["source_id_filters"],
                            filter_logic=recovery_pass["filter_logic"],
                            purpose=f"draft_element_recovery_grounding_{pass_label}",
                            emit_result_to=self.agent_label,
                        )
                        recovery_result_handoff_paths.append(pass_artifacts.get("result_handoff_path"))
                        pass_artifacts["recovery_queries"] = [recovery_pass["query"]]
                        pass_hits_file = self.tools.artifacts.run_path(
                            run_id,
                            "outputs",
                            f"draft_element_recovery_{pass_label}_hits.json",
                        )
                        self.tools.artifacts.write_json(pass_hits_file, pass_artifacts["retrieval_hits"])
                        recovery_pass_hit_paths.append(str(pass_hits_file))
                        if recovery_retrieval is None:
                            recovery_retrieval = pass_artifacts
                        else:
                            recovery_retrieval = self.merge_retrieval_artifacts(
                                primary_artifacts=recovery_retrieval,
                                recovery_artifacts=pass_artifacts,
                            )

                    assert recovery_retrieval is not None
                    recovery_request_handoff_path = recovery_request_handoff_paths[0] if recovery_request_handoff_paths else None
                    recovery_result_handoff_path = recovery_result_handoff_paths[0] if recovery_result_handoff_paths else None
                    revision_retrieval_artifacts = self.merge_retrieval_artifacts(
                        primary_artifacts=retrieval_artifacts,
                        recovery_artifacts=recovery_retrieval,
                    )
                    merged_citation_map = revision_retrieval_artifacts["citation_map"]
                    merged_marker_lookup = {
                        entry["chunk_id"]: entry["marker"]
                        for entry in merged_citation_map
                    }
                    focused_recovery_context = self.tools.retrieval.format_retrieval_context(
                        recovery_retrieval["retrieval_hits"],
                        marker_lookup=merged_marker_lookup,
                    )
                    recovery_hits_file = self.tools.artifacts.run_path(run_id, "outputs", "draft_element_recovery_hits.json")
                    recovery_evidence_file = self.tools.artifacts.run_path(
                        run_id,
                        "outputs",
                        "draft_element_recovery_merged_evidence_package.json",
                    )
                    self.tools.artifacts.write_json(recovery_hits_file, recovery_retrieval["retrieval_hits"])
                    self.tools.artifacts.write_json(recovery_evidence_file, revision_retrieval_artifacts["evidence_package"])
                    recovery_retrieval_hits_path = str(recovery_hits_file)
                    recovery_evidence_package_path = str(recovery_evidence_file)
                    self.record_stage(
                        run_id,
                        stage_name="recover_draft_elements",
                        status="completed",
                        started_at=recovery_started_at,
                        ended_at=utc_now_iso(),
                        inputs={
                            "query": recovery_plan["query"],
                            "target_elements": [item["element_id"] for item in recovery_targets],
                            "retrieval_strategy_effective": recovery_plan.get("retrieval_strategy_effective"),
                            "retrieval_passes": [
                                {
                                    "pass_label": item.get("pass_label"),
                                    "preferred_source_role": item.get("preferred_source_role"),
                                    "source_group_filters": item.get("source_group_filters"),
                                    "source_id_filters": item.get("source_id_filters"),
                                    "filter_logic": item.get("filter_logic"),
                                    "target_element_ids": [
                                        str(target.get("element_id") or "").strip()
                                        for target in item.get("target_elements", [])
                                        if isinstance(target, dict)
                                    ],
                                }
                                for item in recovery_passes
                            ],
                        },
                        outputs={
                            "recovery_plan_path": recovery_plan_path,
                            "recovery_hit_count": len(recovery_retrieval["retrieval_hits"]),
                            "recovery_retrieval_hits_path": recovery_retrieval_hits_path,
                            "recovery_evidence_package_path": recovery_evidence_package_path,
                            "recovery_request_handoff_path": recovery_request_handoff_path,
                            "recovery_result_handoff_path": recovery_result_handoff_path,
                            "recovery_request_handoff_paths": recovery_request_handoff_paths,
                            "recovery_result_handoff_paths": recovery_result_handoff_paths,
                            "recovery_pass_hit_paths": recovery_pass_hit_paths,
                        },
                        notes="The orchestrator issued targeted scoped recovery retrieval before revision, using separate study-specific and regulatory passes when needed.",
                    )

                revision_handoff = self.emit_handoff(
                    run_id,
                    to_agent=self.personalization_agent.agent_label,
                    purpose="personalized_consent_draft_revision_request",
                    payload={
                        "issues": initial_audit["issues"],
                        "missing_required_elements": initial_audit["missing_required_elements"],
                        "revision_targets": initial_audit["revision_targets"],
                        "quality_score": initial_audit["quality_score"],
                        "initial_audit_path": initial_audit_path,
                        "recovery_plan_path": recovery_plan_path,
                        "recovery_request_handoff_path": recovery_request_handoff_path,
                        "recovery_result_handoff_path": recovery_result_handoff_path,
                        "recovery_target_elements": [item["element_id"] for item in recovery_targets],
                    },
                )
                revision_handoff_path = revision_handoff["path"]
                try:
                    revision_payload = self.personalization_agent.revise_draft(
                        run_id=run_id,
                        patient_profile=patient_profile,
                        patient_profile_path=patient_profile_path,
                        base_template_text=base_template_text,
                        template_path=template_path,
                        generation_query=query,
                        retrieval_artifacts=revision_retrieval_artifacts,
                        current_draft=initial_response,
                        draft_audit=initial_audit,
                        draft_content_plan=draft_content_plan,
                        recovery_targets=recovery_targets,
                        focused_recovery_context=focused_recovery_context,
                        top_k=retrieval_plan["top_k"],
                        retrieval_mode=retrieval_plan["retrieval_mode"],
                        source_group_filters=retrieval_plan["source_group_filters"],
                        source_id_filters=retrieval_plan["source_id_filters"],
                        dry_run=dry_run,
                        emit_result_to=self.agent_label,
                    )
                    revision_request_bundle_path = revision_payload.get("request_bundle_path")
                    revision_output_path = revision_payload.get("output_path")
                    revision_result_handoff_path = revision_payload.get("agent_handoff_path")

                    if revision_payload.get("response") is not None:
                        revised_response = revision_payload["response"]
                        revised_audit = self.audit_draft_quality(
                            run_id=run_id,
                            draft_payload=revised_response,
                            retrieval_hits=revision_retrieval_artifacts["retrieval_hits"],
                            patient_profile=patient_profile,
                            label="revision",
                            draft_content_plan=draft_content_plan,
                        )
                        revision_audit_path = revised_audit["path"]
                        comparison = self.tools.evaluation.compare_draft_revision_candidates(
                            initial_audit["metrics"],
                            revised_audit["metrics"],
                            initial_audit=initial_audit,
                            revised_audit=revised_audit,
                        )
                        if comparison["accept_revision"]:
                            revision_applied = True
                            selected_response = revised_response
                            selected_raw_path = revision_payload.get("raw_output_path") or str(selected_raw_path)
                            final_audit_path = revision_audit_path
                            final_output_path = revision_output_path
                            final_request_bundle_path = revision_request_bundle_path or final_request_bundle_path

                        decision = {
                            "revision_attempted": True,
                            "revision_applied": revision_applied,
                            "initial_audit_path": initial_audit_path,
                            "revision_audit_path": revision_audit_path,
                            "final_audit_path": final_audit_path,
                            "initial_request_bundle_path": snapshot_paths.get(
                                "personalization_request_bundle.json",
                                draft_payload.get("request_bundle_path"),
                            ),
                            "revision_request_bundle_path": revision_request_bundle_path,
                            "initial_output_path": snapshot_paths.get(
                                "personalized_consent_draft.json",
                                draft_payload.get("output_path"),
                            ),
                            "revision_output_path": revision_output_path,
                            "recovery_plan_path": recovery_plan_path,
                            "recovery_retrieval_hits_path": recovery_retrieval_hits_path,
                            "recovery_evidence_package_path": recovery_evidence_package_path,
                            **comparison,
                        }
                    else:
                        decision = {
                            "revision_attempted": True,
                            "revision_applied": False,
                            "initial_audit_path": initial_audit_path,
                            "final_audit_path": final_audit_path,
                            "recovery_plan_path": recovery_plan_path,
                            "decision_reason": "revision_attempt_did_not_return_a_draft",
                        }
                except RuntimeError as exc:
                    decision = {
                        "revision_attempted": True,
                        "revision_applied": False,
                        "initial_audit_path": initial_audit_path,
                        "final_audit_path": final_audit_path,
                        "recovery_plan_path": recovery_plan_path,
                        "decision_reason": "revision_attempt_failed_and_initial_draft_was_kept",
                        "revision_error": str(exc),
                    }

            decision_file = self.tools.artifacts.run_path(run_id, "outputs", "draft_revision_decision.json")
            self.tools.artifacts.write_json(decision_file, decision)
            decision_path = str(decision_file)
            revision_metadata = {
                "revision_attempted": revision_attempted,
                "revision_applied": revision_applied,
                "revision_accepted": revision_applied,
                "initial_audit_path": initial_audit_path,
                "revision_audit_path": revision_audit_path,
                "final_audit_path": final_audit_path,
                "decision_path": decision_path,
                "initial_request_bundle_path": draft_payload.get("request_bundle_path"),
                "revision_request_bundle_path": revision_request_bundle_path,
                "revision_output_path": revision_output_path,
                "revision_request_handoff_path": revision_handoff_path,
                "revision_result_handoff_path": revision_result_handoff_path,
                "recovery_plan_path": recovery_plan_path,
                "recovery_retrieval_hits_path": recovery_retrieval_hits_path,
                "recovery_evidence_package_path": recovery_evidence_package_path,
                "recovery_request_handoff_path": recovery_request_handoff_path,
                "recovery_result_handoff_path": recovery_result_handoff_path,
            }
            final_output_path = self.write_final_draft_artifacts(
                run_id=run_id,
                draft_payload=selected_response,
                revision_metadata=revision_metadata,
                raw_source_path=str(selected_raw_path) if selected_raw_path else None,
            )
            draft_payload["response"] = dict(selected_response)
            draft_payload["response"]["revision_metadata"] = revision_metadata
            draft_payload["output_path"] = final_output_path
        draft_payload["draft_content_plan_path"] = draft_content_plan_path
        draft_payload["draft_content_plan_mode"] = draft_content_plan.get("planning_mode") if isinstance(draft_content_plan, dict) else None
        draft_payload["decision_path"] = decision_path

        self.record_stage(
            run_id,
            stage_name="personalize_consent",
            status="completed",
            started_at=started_at,
            ended_at=utc_now_iso(),
            inputs={
                "patient_profile_path": str(patient_profile_path) if patient_profile_path else None,
                "template_path": str(template_path) if template_path else None,
                "generation_query": generation_query,
                "workflow_variant": workflow_variant,
                "top_k": retrieval_plan["top_k"],
                "retrieval_mode": retrieval_plan["retrieval_mode"],
                "source_group_filters": retrieval_plan["source_group_filters"],
                "source_id_filters": retrieval_plan["source_id_filters"],
                "dry_run": dry_run,
            },
            outputs={
                "draft_output_path": final_output_path,
                "request_bundle_path": final_request_bundle_path,
                "orchestrator_to_rag_handoff_path": orchestrator_to_rag_path,
                "scoped_retrieval_applied": scoped_retrieval_applied,
                "scoped_retrieval_study_hits_path": scoped_retrieval_study_hits_path,
                "scoped_retrieval_regulatory_hits_path": scoped_retrieval_regulatory_hits_path,
                "scoped_retrieval_evidence_package_path": scoped_retrieval_evidence_package_path,
                "scoped_retrieval_study_result_handoff_path": scoped_retrieval_study_result_handoff_path,
                "scoped_retrieval_regulatory_result_handoff_path": scoped_retrieval_regulatory_result_handoff_path,
                "orchestrator_to_personalization_handoff_path": orchestrator_to_personalization["path"],
                "personalization_result_handoff_path": draft_payload.get("agent_handoff_path"),
                "draft_content_plan_path": draft_content_plan_path,
                "draft_content_plan_mode": draft_content_plan.get("planning_mode") if isinstance(draft_content_plan, dict) else None,
                "study_enrichment_plan_path": study_enrichment_plan_path,
                "study_enrichment_hits_path": study_enrichment_hits_path,
                "study_enrichment_evidence_package_path": study_enrichment_evidence_package_path,
                "study_enrichment_request_handoff_path": study_enrichment_request_handoff_path,
                "study_enrichment_result_handoff_path": study_enrichment_result_handoff_path,
                "initial_audit_path": initial_audit_path,
                "revision_audit_path": revision_audit_path,
                "decision_path": decision_path,
                "draft_revision_attempted": revision_attempted,
                "draft_revision_applied": revision_applied,
                "revision_request_bundle_path": revision_request_bundle_path,
                "revision_output_path": revision_output_path,
                "revision_request_handoff_path": revision_handoff_path,
                "revision_result_handoff_path": revision_result_handoff_path,
                "recovery_plan_path": recovery_plan_path,
                "recovery_retrieval_hits_path": recovery_retrieval_hits_path,
                "recovery_evidence_package_path": recovery_evidence_package_path,
                "recovery_request_handoff_path": recovery_request_handoff_path,
                "recovery_result_handoff_path": recovery_result_handoff_path,
            },
            notes=(
                "Coordinated the RAG and Personalization agents to produce a grounded participant-facing consent draft."
            ),
        )
        return draft_payload

    def answer_question(
        self,
        *,
        run_id: str,
        question: str,
        patient_profile_path: Path | None = None,
        top_k: int | None = None,
        retrieval_mode: str | None = None,
        source_group_filters: list[str] | None = None,
        source_id_filters: list[str] | None = None,
        filter_logic: str | None = None,
        workflow_variant: str = "full_agentic",
        dry_run: bool = False,
    ) -> dict[str, Any]:
        started_at = utc_now_iso()
        workflow_variant = self.normalize_workflow_variant(workflow_variant)
        patient_profile = self.tools.state.resolve_patient_profile(run_id, patient_profile_path)
        question_id = self.tools.retrieval.build_question_id(question)
        retrieval_plan = self.plan_question_grounding(
            question=question,
            top_k=top_k,
            retrieval_mode=retrieval_mode,
            source_group_filters=source_group_filters,
            source_id_filters=source_id_filters,
            filter_logic=filter_logic,
            use_llm=workflow_variant == "full_agentic" and not dry_run,
        )

        orchestrator_to_rag_path: str | None = None
        scoped_retrieval_study_hits_path: str | None = None
        scoped_retrieval_regulatory_hits_path: str | None = None
        scoped_retrieval_evidence_package_path: str | None = None
        scoped_retrieval_study_result_handoff_path: str | None = None
        scoped_retrieval_regulatory_result_handoff_path: str | None = None
        scoped_retrieval_applied = False
        if workflow_variant == "vanilla_llm":
            retrieval_artifacts = self.build_empty_retrieval_artifacts(
                source_group_filters=retrieval_plan["source_group_filters"],
                source_id_filters=retrieval_plan["source_id_filters"],
                filter_logic=retrieval_plan["filter_logic"],
            )
        else:
            retrieval_strategy = (
                "study_plus_regulatory_split"
                if self.should_split_study_scoped_retrieval(
                    source_group_filters=retrieval_plan["source_group_filters"],
                    source_id_filters=retrieval_plan["source_id_filters"],
                )
                else "single_pass"
            )
            orchestrator_to_rag = self.emit_handoff(
                run_id,
                to_agent=self.rag_agent.agent_label,
                purpose="question_answer_grounding_request",
                payload={
                    "question_id": question_id,
                    "question": question,
                    "retrieval_query": retrieval_plan["query"],
                    "top_k": retrieval_plan["top_k"],
                    "retrieval_mode": retrieval_plan["retrieval_mode"],
                    "source_group_filters": retrieval_plan["source_group_filters"],
                    "source_id_filters": retrieval_plan["source_id_filters"],
                    "filter_logic": retrieval_plan["filter_logic"],
                    "retrieval_strategy": retrieval_strategy,
                    "question_profile": retrieval_plan["question_profile"],
                    "workflow_variant": workflow_variant,
                    "planning_mode": retrieval_plan.get("planning_mode"),
                    "planning_reason": retrieval_plan.get("planning_reason"),
                    "participant_profile": {
                        "participant_id": patient_profile.participant_id,
                        "health_literacy": patient_profile.health_literacy,
                        "language": patient_profile.language,
                        "jurisdiction": patient_profile.jurisdiction,
                    },
                },
            )
            orchestrator_to_rag_path = orchestrator_to_rag["path"]
            scoped_retrieval_bundle = self.retrieve_scoped_grounding_artifacts(
                run_id=run_id,
                query=retrieval_plan["query"],
                top_k=retrieval_plan["top_k"],
                retrieval_mode=retrieval_plan["retrieval_mode"],
                source_group_filters=retrieval_plan["source_group_filters"],
                source_id_filters=retrieval_plan["source_id_filters"],
                filter_logic=retrieval_plan["filter_logic"],
                purpose_prefix="question_answer_grounding",
                emit_result_to=self.agent_label,
                study_query=retrieval_plan["query"],
                regulatory_query=retrieval_plan["query"],
            )
            retrieval_artifacts = scoped_retrieval_bundle["retrieval_artifacts"]
            scoped_retrieval_applied = bool(scoped_retrieval_bundle["scoped_retrieval"])
            scoped_retrieval_study_hits_path = scoped_retrieval_bundle["study_hits_path"]
            scoped_retrieval_regulatory_hits_path = scoped_retrieval_bundle["regulatory_hits_path"]
            scoped_retrieval_evidence_package_path = scoped_retrieval_bundle["merged_evidence_package_path"]
            scoped_retrieval_study_result_handoff_path = scoped_retrieval_bundle["study_result_handoff_path"]
            scoped_retrieval_regulatory_result_handoff_path = scoped_retrieval_bundle["regulatory_result_handoff_path"]
        sufficiency = self.assess_evidence_sufficiency(
            retrieval_artifacts=retrieval_artifacts,
            required_source_groups=retrieval_plan["required_source_groups"],
        )
        if workflow_variant == "full_agentic" and not sufficiency["sufficient"]:
            clarification = self.build_clarification_response(
                run_id=run_id,
                reason="insufficient_question_grounding",
                message=(
                    "The orchestrator could not find enough grounded evidence to answer this question safely."
                ),
                missing_required_source_groups=sufficiency["missing_required_source_groups"],
                suggested_actions=[
                    "Add study-specific material if the question is about procedures, interventions, or visits.",
                    "Add regulatory guidance if the question is about rights, withdrawal, privacy, or alternatives.",
                ],
            )
            clarification["question_id"] = question_id
            clarification["question"] = question
            qa_dir = self.tools.artifacts.run_path(run_id, "outputs", "qa")
            qa_dir.mkdir(parents=True, exist_ok=True)
            retrieval_path = qa_dir / f"{question_id}.retrieval_hits.json"
            evidence_package_path = qa_dir / f"{question_id}.evidence_package.json"
            clarification_path = qa_dir / f"{question_id}.clarification.json"
            self.tools.artifacts.write_json(retrieval_path, retrieval_artifacts["retrieval_hits"])
            self.tools.artifacts.write_json(evidence_package_path, retrieval_artifacts["evidence_package"])
            self.tools.artifacts.write_json(clarification_path, clarification)
            self.tools.retrieval.upsert_qa_index_entry(
                qa_dir / "qa_index.jsonl",
                {
                    "question_id": question_id,
                    "question": question,
                    "retrieval_hits_path": str(retrieval_path),
                    "evidence_package_path": str(evidence_package_path),
                    "clarification_path": str(clarification_path),
                    "answer_path": None,
                    "workflow_variant": workflow_variant,
                    "status": "abstained",
                    "dry_run": dry_run,
                },
            )
            self.record_stage(
                run_id,
                stage_name="answer_question",
                status="clarification_requested",
                started_at=started_at,
                ended_at=utc_now_iso(),
                inputs={
                    "question_id": question_id,
                    "question": question,
                    "retrieval_query": retrieval_plan["query"],
                    "patient_profile_path": str(patient_profile_path) if patient_profile_path else None,
                    "workflow_variant": workflow_variant,
                },
                outputs={
                    "clarification_reason": clarification["reason"],
                    "missing_required_source_groups": clarification["missing_required_source_groups"],
                    "observed_source_groups": sufficiency["observed_source_groups"],
                    "question_profile": retrieval_plan["question_profile"],
                    "planning_mode": retrieval_plan.get("planning_mode"),
                    "planning_reason": retrieval_plan.get("planning_reason"),
                },
                notes="The orchestrator paused instead of answering because the grounding evidence was incomplete.",
            )
            return clarification

        orchestrator_to_conversational = self.emit_handoff(
            run_id,
            to_agent=self.conversational_agent.agent_label,
            purpose="consent_question_answer_request",
            payload={
                "question_id": question_id,
                "question": question,
                "retrieval_hit_count": len(retrieval_artifacts["retrieval_hits"]),
                "retrieval_mode_used": retrieval_artifacts["mode_used"],
                "workflow_variant": workflow_variant,
                "planning_mode": retrieval_plan.get("planning_mode"),
                "planning_reason": retrieval_plan.get("planning_reason"),
                "participant_profile": {
                    "participant_id": patient_profile.participant_id,
                    "health_literacy": patient_profile.health_literacy,
                    "language": patient_profile.language,
                    "jurisdiction": patient_profile.jurisdiction,
                },
            },
        )
        answer_payload = self.conversational_agent.answer_question(
            run_id=run_id,
            question=question,
            patient_profile=patient_profile,
            patient_profile_path=patient_profile_path,
            retrieval_artifacts=retrieval_artifacts,
            top_k=retrieval_plan["top_k"],
            retrieval_mode=retrieval_plan["retrieval_mode"],
            source_group_filters=retrieval_plan["source_group_filters"],
            source_id_filters=retrieval_plan["source_id_filters"],
            workflow_variant=workflow_variant,
            dry_run=dry_run,
            emit_result_to=self.agent_label,
        )

        self.record_stage(
            run_id,
            stage_name="answer_question",
            status="completed",
            started_at=started_at,
            ended_at=utc_now_iso(),
            inputs={
                "question_id": question_id,
                "question": question,
                "patient_profile_path": str(patient_profile_path) if patient_profile_path else None,
                "retrieval_query": retrieval_plan["query"],
                "workflow_variant": workflow_variant,
                "top_k": retrieval_plan["top_k"],
                "retrieval_mode": retrieval_plan["retrieval_mode"],
                "source_group_filters": retrieval_plan["source_group_filters"],
                "source_id_filters": retrieval_plan["source_id_filters"],
                "planning_mode": retrieval_plan.get("planning_mode"),
                "planning_reason": retrieval_plan.get("planning_reason"),
                "dry_run": dry_run,
            },
            outputs={
                "answer_output_path": answer_payload.get("output_path"),
                "request_bundle_path": answer_payload.get("request_bundle_path"),
                "orchestrator_to_rag_handoff_path": orchestrator_to_rag_path,
                "scoped_retrieval_applied": scoped_retrieval_applied,
                "scoped_retrieval_study_hits_path": scoped_retrieval_study_hits_path,
                "scoped_retrieval_regulatory_hits_path": scoped_retrieval_regulatory_hits_path,
                "scoped_retrieval_evidence_package_path": scoped_retrieval_evidence_package_path,
                "scoped_retrieval_study_result_handoff_path": scoped_retrieval_study_result_handoff_path,
                "scoped_retrieval_regulatory_result_handoff_path": scoped_retrieval_regulatory_result_handoff_path,
                "orchestrator_to_conversational_handoff_path": orchestrator_to_conversational["path"],
                "conversational_result_handoff_path": answer_payload.get("agent_handoff_path"),
            },
            notes=(
                "Coordinated the RAG and Conversational agents to answer a participant question with grounded evidence."
            ),
        )
        return answer_payload

    def formalize_consent(
        self,
        *,
        run_id: str,
        patient_profile_path: Path | None = None,
        draft_path: Path | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        started_at = utc_now_iso()
        patient_profile = self.tools.state.resolve_patient_profile(run_id, patient_profile_path)
        personalized_draft = self.tools.state.resolve_personalized_draft(run_id, draft_path)

        orchestrator_to_formalization = self.emit_handoff(
            run_id,
            to_agent=self.formalization_agent.agent_label,
            purpose="structured_consent_record_request",
            payload={
                "draft_path": str(draft_path) if draft_path else str(self.tools.artifacts.run_path(run_id, "outputs", "personalized_consent_draft.json")),
                "key_information_summary_present": bool(personalized_draft.get("key_information_summary")),
                "citation_marker_count": len(personalized_draft.get("citation_markers_used", [])),
                "participant_profile": {
                    "participant_id": patient_profile.participant_id,
                    "health_literacy": patient_profile.health_literacy,
                    "language": patient_profile.language,
                    "jurisdiction": patient_profile.jurisdiction,
                },
            },
        )
        formalized_payload = self.formalization_agent.formalize_consent(
            run_id=run_id,
            patient_profile=patient_profile,
            patient_profile_path=patient_profile_path,
            personalized_draft=personalized_draft,
            draft_path=draft_path,
            dry_run=dry_run,
            emit_result_to=self.agent_label,
        )

        self.record_stage(
            run_id,
            stage_name="formalize_consent",
            status="completed",
            started_at=started_at,
            ended_at=utc_now_iso(),
            inputs={
                "patient_profile_path": str(patient_profile_path) if patient_profile_path else None,
                "draft_path": str(draft_path) if draft_path else None,
                "dry_run": dry_run,
            },
            outputs={
                "structured_record_path": formalized_payload.get("output_path"),
                "request_bundle_path": formalized_payload.get("request_bundle_path"),
                "orchestrator_to_formalization_handoff_path": orchestrator_to_formalization["path"],
                "formalization_result_handoff_path": formalized_payload.get("agent_handoff_path"),
            },
            notes=(
                "Coordinated the Consent Formalization Agent to convert the grounded draft into a structured consent record."
            ),
        )
        return formalized_payload

    def handle_user_request(
        self,
        *,
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
        workflow_variant: str = "full_agentic",
        dry_run: bool = False,
    ) -> dict[str, Any]:
        started_at = utc_now_iso()
        workflow_variant = self.normalize_workflow_variant(workflow_variant)
        route = self.classify_user_request(
            user_input=user_input,
            run_id=run_id,
            template_path=template_path,
            draft_path=draft_path,
            use_llm=workflow_variant == "full_agentic" and not dry_run,
        )

        routing_handoff = self.emit_handoff(
            run_id,
            to_agent=self.agent_label,
            purpose="request_routing_decision",
            payload={
                "user_input": user_input,
                "intent": route["intent"],
                "reason": route["reason"],
                "planning_mode": route.get("planning_mode"),
                "workflow_variant": workflow_variant,
            },
        )

        if route["intent"] == "clarification":
            clarification = self.build_clarification_response(
                run_id=run_id,
                reason=str(route["reason"]),
                message=str(route["message"]),
                suggested_actions=[
                    "Ask a question about the consent or study.",
                    "Request draft generation from a saved base template.",
                    "Request formalization after a draft exists.",
                ],
            )
            clarification["routing_handoff_path"] = routing_handoff["path"]
            self.record_stage(
                run_id,
                stage_name="handle_user_request",
                status="clarification_requested",
                started_at=started_at,
                ended_at=utc_now_iso(),
                inputs={"user_input": user_input},
                outputs={
                        "intent": route["intent"],
                        "reason": route["reason"],
                        "planning_mode": route.get("planning_mode"),
                        "routing_handoff_path": routing_handoff["path"],
                    },
                notes="The orchestrator could not safely route the request and returned a clarification response.",
            )
            return clarification

        if route["intent"] == "personalize_consent":
            payload = self.personalize_consent(
                run_id=run_id,
                patient_profile_path=patient_profile_path,
                template_path=template_path,
                generation_query=None,
                top_k=top_k,
                retrieval_mode=retrieval_mode,
                source_group_filters=source_group_filters,
                source_id_filters=source_id_filters,
                filter_logic=filter_logic,
                workflow_variant=workflow_variant,
                dry_run=dry_run,
            )
        elif route["intent"] == "formalize_consent":
            payload = self.formalize_consent(
                run_id=run_id,
                patient_profile_path=patient_profile_path,
                draft_path=draft_path,
                dry_run=dry_run,
            )
        else:
            payload = self.answer_question(
                run_id=run_id,
                question=user_input,
                patient_profile_path=patient_profile_path,
                top_k=top_k,
                retrieval_mode=retrieval_mode,
                source_group_filters=source_group_filters,
                source_id_filters=source_id_filters,
                filter_logic=filter_logic,
                workflow_variant=workflow_variant,
                dry_run=dry_run,
            )

        if isinstance(payload, dict):
            payload["routing"] = {
                "intent": route["intent"],
                "reason": route["reason"],
                "planning_mode": route.get("planning_mode"),
                "workflow_variant": workflow_variant,
                "routing_handoff_path": routing_handoff["path"],
            }
        self.record_stage(
            run_id,
            stage_name="handle_user_request",
            status="completed",
            started_at=started_at,
            ended_at=utc_now_iso(),
            inputs={"user_input": user_input},
            outputs={
                "intent": route["intent"],
                "reason": route["reason"],
                "planning_mode": route.get("planning_mode"),
                "workflow_variant": workflow_variant,
                "routing_handoff_path": routing_handoff["path"],
            },
            notes="The orchestrator classified the user input and selected the next specialized agent path.",
        )
        return payload
