from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from informed_consent.config import AppConfig, ModelConfig, PathConfig, RetrievalConfig
from informed_consent.pipeline import ConsentPipeline


class AgenticOrchestrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.repo_root = Path(__file__).resolve().parents[1]

        source_dir = self.temp_path / "sources"
        regulatory_dir = source_dir / "regulatory_guidance"
        trial_dir = source_dir / "trial_materials"
        regulatory_dir.mkdir(parents=True, exist_ok=True)
        trial_dir.mkdir(parents=True, exist_ok=True)
        (regulatory_dir / "regulatory_summary.txt").write_text(
            (
                "Participation is voluntary. Participants may withdraw at any time without penalty. "
                "Study staff should explain study purpose, procedures, risks, benefits, and alternatives."
            ),
            encoding="utf-8",
        )
        (trial_dir / "nct03877237.txt").write_text(
            (
                "This study tests dapagliflozin in adults with heart failure. "
                "Participants attend study visits, complete assessments, and may receive study treatment or placebo."
            ),
            encoding="utf-8",
        )

        self.patient_profile_path = self.temp_path / "patient_profile.json"
        self.patient_profile_path.write_text(
            json.dumps(
                {
                    "participant_id": "P-001",
                    "language": "en",
                    "health_literacy": "low",
                    "jurisdiction": "US",
                }
            ),
            encoding="utf-8",
        )
        self.template_path = self.temp_path / "base_template.txt"
        self.template_path.write_text(
            "This study may involve study visits, research procedures, and the option to stop later.",
            encoding="utf-8",
        )

        config = AppConfig(
            models=ModelConfig(endpoint_url="https://example.test"),
            retrieval=RetrievalConfig(retrieval_mode="lexical"),
            paths=PathConfig(
                project_root=self.repo_root,
                artifact_root=self.temp_path / "artifacts",
                source_data_root=self.temp_path / "data",
                configs_root=self.repo_root / "configs",
                prompts_root=self.repo_root / "prompts",
                docs_root=self.repo_root / "docs",
                scripts_root=self.repo_root / "scripts",
            ),
        )
        self.pipeline = ConsentPipeline(config)
        manifest = self.pipeline.prepare_corpus(purpose="test_agentic_refactor", source_dir=source_dir)
        self.run_id = manifest.run_id

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def load_handoffs(self) -> list[dict[str, object]]:
        handoff_index = self.temp_path / "artifacts" / "runs" / self.run_id / "outputs" / "agent_handoffs" / "handoff_index.jsonl"
        self.assertTrue(handoff_index.exists(), "Expected the agent handoff index to be created.")
        return [json.loads(line) for line in handoff_index.read_text(encoding="utf-8").splitlines() if line.strip()]

    def test_draft_dry_run_uses_orchestrator_rag_and_personalization_agents(self) -> None:
        payload = self.pipeline.draft_personalized_consent(
            run_id=self.run_id,
            patient_profile_path=self.patient_profile_path,
            template_path=self.template_path,
            dry_run=True,
        )

        self.assertTrue(payload["dry_run"])
        self.assertIsNone(payload["response"])

        handoffs = self.load_handoffs()
        handoff_pairs = {(item["from_agent"], item["to_agent"]) for item in handoffs}
        self.assertIn(("Orchestrator Agent", "RAG Agent"), handoff_pairs)
        self.assertIn(("RAG Agent", "Orchestrator Agent"), handoff_pairs)
        self.assertIn(("Orchestrator Agent", "Personalization Agent"), handoff_pairs)
        self.assertIn(("Personalization Agent", "Orchestrator Agent"), handoff_pairs)
        self.assertTrue(Path(payload["draft_content_plan_path"]).exists())

    def test_vanilla_draft_dry_run_skips_rag_and_content_plan(self) -> None:
        payload = self.pipeline.draft_personalized_consent(
            run_id=self.run_id,
            patient_profile_path=self.patient_profile_path,
            template_path=self.template_path,
            workflow_variant="vanilla_llm",
            dry_run=True,
        )

        self.assertTrue(payload["dry_run"])
        self.assertIsNone(payload["response"])
        self.assertIsNone(payload["draft_content_plan_path"])

        handoffs = self.load_handoffs()
        handoff_pairs = {(item["from_agent"], item["to_agent"]) for item in handoffs}
        self.assertNotIn(("Orchestrator Agent", "RAG Agent"), handoff_pairs)
        self.assertIn(("Orchestrator Agent", "Personalization Agent"), handoff_pairs)

    def test_question_dry_run_uses_orchestrator_rag_and_conversational_agents(self) -> None:
        payload = self.pipeline.answer_consent_question(
            run_id=self.run_id,
            question="Can I stop later without penalty?",
            patient_profile_path=self.patient_profile_path,
            dry_run=True,
        )

        self.assertTrue(payload["dry_run"])
        self.assertIsNone(payload["response"])

        handoffs = self.load_handoffs()
        handoff_pairs = {(item["from_agent"], item["to_agent"]) for item in handoffs}
        self.assertIn(("Orchestrator Agent", "RAG Agent"), handoff_pairs)
        self.assertIn(("RAG Agent", "Orchestrator Agent"), handoff_pairs)
        self.assertIn(("Orchestrator Agent", "Conversational Agent"), handoff_pairs)
        self.assertIn(("Conversational Agent", "Orchestrator Agent"), handoff_pairs)

    def test_agents_receive_scoped_toolsets(self) -> None:
        self.assertTrue(hasattr(self.pipeline.orchestrator_agent.tools, "evaluation"))
        self.assertTrue(hasattr(self.pipeline.orchestrator_agent.tools, "generation"))
        self.assertFalse(hasattr(self.pipeline.rag_agent.tools, "generation"))
        self.assertFalse(hasattr(self.pipeline.rag_agent.tools, "state"))
        self.assertTrue(hasattr(self.pipeline.personalization_agent.tools, "generation"))
        self.assertFalse(hasattr(self.pipeline.personalization_agent.tools, "evaluation"))
        self.assertTrue(hasattr(self.pipeline.conversational_agent.tools, "generation"))
        self.assertFalse(hasattr(self.pipeline.formalization_agent.tools, "retrieval"))

    def test_plan_element_recovery_targets_supported_missing_elements(self) -> None:
        recovery_plan = self.pipeline.orchestrator_agent.plan_element_recovery(
            draft_audit={
                "missing_required_elements": ["study_procedures", "benefits", "questions"],
            },
            draft_content_plan={
                "elements": [
                    {
                        "element_id": "study_procedures",
                        "status": "supported",
                        "preferred_source_role": "study_specific",
                        "recommended_markers": ["[1]"],
                        "instruction": "Explain study procedures.",
                    },
                    {
                        "element_id": "benefits",
                        "status": "partially_supported",
                        "preferred_source_role": "study_specific",
                        "recommended_markers": ["[2]"],
                        "instruction": "Explain benefits carefully.",
                    },
                    {
                        "element_id": "questions",
                        "status": "unsupported",
                        "preferred_source_role": "regulatory",
                        "recommended_markers": [],
                        "instruction": "Explain questions/contact.",
                    },
                ]
            },
            retrieval_plan={
                "query": "base consent query",
                "top_k": 5,
                "retrieval_mode": "hybrid",
                "source_group_filters": ["regulatory_guidance", "trial_materials"],
                "source_id_filters": ["nct03877237"],
                "filter_logic": "union",
            },
        )

        self.assertIsNotNone(recovery_plan)
        assert recovery_plan is not None
        self.assertEqual(
            [item["element_id"] for item in recovery_plan["target_elements"]],
            ["study_procedures", "benefits"],
        )
        self.assertEqual(recovery_plan["source_group_filters"], ["trial_materials"])
        self.assertIn("study procedures", recovery_plan["query"])
        self.assertIn("possible benefits", recovery_plan["query"])

    def test_personalization_grounding_requires_trial_materials_when_study_scope_is_requested(self) -> None:
        plan = self.pipeline.orchestrator_agent.plan_personalization_grounding(
            run_id=self.run_id,
            patient_profile=self.pipeline.load_patient_profile(self.patient_profile_path),
            base_template_text=self.template_path.read_text(encoding="utf-8"),
            generation_query=None,
            top_k=None,
            retrieval_mode=None,
            source_group_filters=["regulatory_guidance", "trial_materials"],
            source_id_filters=["nct03877237"],
            filter_logic="union",
        )

        self.assertEqual(plan["required_source_groups"], ["regulatory_guidance", "trial_materials"])
        self.assertIn("trial_materials", plan["preferred_source_groups"])

    def test_scoped_grounding_retrieval_splits_study_and_regulatory_passes(self) -> None:
        def fake_retrieve(**kwargs):
            source_group_filters = kwargs["source_group_filters"]
            source_id_filters = kwargs["source_id_filters"]
            if source_group_filters == ["trial_materials"]:
                self.assertEqual(source_id_filters, ["nct03877237"])
                hits = [
                    {
                        "source_id": "nct03877237",
                        "chunk_id": "study-1",
                        "rank": 1,
                        "score": 0.9,
                        "citation_label": "Target Study",
                        "excerpt": "Target study procedures and intervention.",
                        "metadata": {"source_group": "trial_materials"},
                    }
                ]
            else:
                self.assertEqual(source_group_filters, ["regulatory_guidance"])
                self.assertEqual(source_id_filters, [])
                hits = [
                    {
                        "source_id": "common_rule_2018_july",
                        "chunk_id": "reg-1",
                        "rank": 1,
                        "score": 0.8,
                        "citation_label": "Common Rule",
                        "excerpt": "Participation is voluntary and withdrawal is allowed.",
                        "metadata": {"source_group": "regulatory_guidance"},
                    }
                ]
            return {
                "query": kwargs["query"],
                "mode_used": kwargs["retrieval_mode"],
                "dense_available": True,
                "lexical_hits": [],
                "dense_hits": [],
                "retrieval_hits": hits,
                "retrieved_context": "",
                "citation_map": self.pipeline.build_citation_map(hits),
                "evidence_package": self.pipeline.build_role_separated_evidence_package(hits),
                "source_group_filters": source_group_filters,
                "source_id_filters": source_id_filters,
                "filter_logic_used": kwargs["filter_logic"],
                "filtered_chunk_count": len(hits),
                "result_handoff_path": None,
            }

        with patch.object(self.pipeline.rag_agent, "retrieve_evidence", side_effect=fake_retrieve) as mock_retrieve:
            payload = self.pipeline.orchestrator_agent.retrieve_scoped_grounding_artifacts(
                run_id=self.run_id,
                query="generic query",
                top_k=6,
                retrieval_mode="hybrid",
                source_group_filters=["regulatory_guidance", "trial_materials"],
                source_id_filters=["nct03877237"],
                filter_logic="union",
                purpose_prefix="personalization_grounding",
                emit_result_to=self.pipeline.orchestrator_agent.agent_label,
                study_query="study-specific query",
                regulatory_query="regulatory query",
            )

        self.assertTrue(payload["scoped_retrieval"])
        self.assertEqual(mock_retrieve.call_count, 2)
        merged_hits = payload["retrieval_artifacts"]["retrieval_hits"]
        self.assertEqual([hit["source_id"] for hit in merged_hits], ["nct03877237", "common_rule_2018_july"])
        self.assertEqual(payload["retrieval_artifacts"]["evidence_package"]["role_counts"]["study_specific"], 1)
        self.assertEqual(payload["retrieval_artifacts"]["evidence_package"]["role_counts"]["regulatory"], 1)

    def test_merge_retrieval_artifacts_deduplicates_and_rebuilds_citations(self) -> None:
        primary = {
            "retrieval_hits": [
                {
                    "source_id": "s1",
                    "chunk_id": "c1",
                    "rank": 1,
                    "score": 0.9,
                    "citation_label": "Source 1",
                    "excerpt": "Primary excerpt one",
                    "metadata": {"source_group": "trial_materials"},
                },
                {
                    "source_id": "s2",
                    "chunk_id": "c2",
                    "rank": 2,
                    "score": 0.8,
                    "citation_label": "Source 2",
                    "excerpt": "Primary excerpt two",
                    "metadata": {"source_group": "regulatory_guidance"},
                },
            ],
            "mode_used": "hybrid",
            "dense_available": True,
            "source_group_filters": ["trial_materials", "regulatory_guidance"],
            "source_id_filters": [],
            "filter_logic_used": "union",
            "filtered_chunk_count": 2,
            "lexical_hits": [],
            "dense_hits": [],
        }
        recovery = {
            "query": "study procedures benefits",
            "retrieval_hits": [
                {
                    "source_id": "s2",
                    "chunk_id": "c2",
                    "rank": 1,
                    "score": 0.95,
                    "citation_label": "Source 2",
                    "excerpt": "Duplicate excerpt two",
                    "metadata": {"source_group": "regulatory_guidance"},
                },
                {
                    "source_id": "s3",
                    "chunk_id": "c3",
                    "rank": 2,
                    "score": 0.7,
                    "citation_label": "Source 3",
                    "excerpt": "Recovery excerpt three",
                    "metadata": {"source_group": "trial_materials"},
                },
            ],
        }

        merged = self.pipeline.orchestrator_agent.merge_retrieval_artifacts(
            primary_artifacts=primary,
            recovery_artifacts=recovery,
        )

        self.assertEqual([hit["chunk_id"] for hit in merged["retrieval_hits"]], ["c1", "c2", "c3"])
        self.assertEqual([entry["marker"] for entry in merged["citation_map"]], ["[1]", "[2]", "[3]"])
        self.assertTrue(merged["recovery_applied"])
        self.assertEqual(merged["recovery_queries"], ["study procedures benefits"])

    def test_handle_user_request_routes_draft_generation_to_personalization_path(self) -> None:
        payload = self.pipeline.handle_user_request(
            run_id=self.run_id,
            user_input="Generate a personalized consent draft for this participant.",
            patient_profile_path=self.patient_profile_path,
            template_path=self.template_path,
            dry_run=True,
        )

        self.assertEqual(payload["routing"]["intent"], "personalize_consent")
        handoffs = self.load_handoffs()
        handoff_pairs = {(item["from_agent"], item["to_agent"]) for item in handoffs}
        self.assertIn(("Orchestrator Agent", "Personalization Agent"), handoff_pairs)

    def test_handle_user_request_requests_clarification_when_study_grounding_is_missing(self) -> None:
        payload = self.pipeline.handle_user_request(
            run_id=self.run_id,
            user_input="What drug will I take in this study?",
            patient_profile_path=self.patient_profile_path,
            source_group_filters=["regulatory_guidance"],
            dry_run=True,
        )

        self.assertEqual(payload["status"], "needs_clarification")
        self.assertEqual(payload["reason"], "insufficient_question_grounding")
        self.assertIn("trial_materials", payload["missing_required_source_groups"])

    def test_rights_question_with_generic_study_reference_uses_regulatory_required_grounding(self) -> None:
        plan = self.pipeline.orchestrator_agent.plan_question_grounding(
            question="Can I leave the study later without any penalty?",
            top_k=None,
            retrieval_mode=None,
            source_group_filters=None,
            source_id_filters=None,
            filter_logic=None,
        )

        self.assertEqual(plan["question_profile"], "regulatory")
        self.assertEqual(plan["required_source_groups"], ["regulatory_guidance"])
        self.assertIn("trial_materials", plan["source_group_filters"])

    def test_study_specific_question_with_generic_study_words_still_requires_trial_grounding(self) -> None:
        plan = self.pipeline.orchestrator_agent.plan_question_grounding(
            question="What is this study testing?",
            top_k=None,
            retrieval_mode=None,
            source_group_filters=None,
            source_id_filters=None,
            filter_logic=None,
        )

        self.assertEqual(plan["question_profile"], "study")
        self.assertEqual(plan["required_source_groups"], ["trial_materials"])

    @patch("informed_consent.agent_tools.chat_json")
    def test_model_planner_can_route_rights_question_without_trial_requirement(self, mock_chat_json) -> None:
        mock_chat_json.return_value = {
            "question_profile": "regulatory",
            "retrieval_query": "withdrawal rights penalty leaving study participant choice",
            "required_source_groups": ["regulatory_guidance"],
            "preferred_source_groups": ["regulatory_guidance", "trial_materials"],
            "reason": "The question is about withdrawal rights, not concrete study procedures.",
        }

        plan = self.pipeline.orchestrator_agent.plan_question_grounding(
            question="Can I leave the study later without any penalty?",
            top_k=None,
            retrieval_mode=None,
            source_group_filters=["regulatory_guidance", "trial_materials"],
            source_id_filters=["nct03877237"],
            filter_logic="union",
            use_llm=True,
        )

        self.assertEqual(plan["planning_mode"], "model")
        self.assertEqual(plan["question_profile"], "regulatory")
        self.assertEqual(plan["required_source_groups"], ["regulatory_guidance"])
        self.assertIn("withdrawal rights", plan["query"])

    @patch("informed_consent.agent_tools.chat_json")
    def test_model_router_can_choose_question_answer_intent(self, mock_chat_json) -> None:
        mock_chat_json.return_value = {
            "intent": "answer_question",
            "reason": "participant_follow_up_question",
            "message": None,
        }

        route = self.pipeline.orchestrator_agent.classify_user_request(
            user_input="Can I leave the study later without any penalty?",
            run_id=self.run_id,
            use_llm=True,
        )

        self.assertEqual(route["intent"], "answer_question")
        self.assertEqual(route["planning_mode"], "model")

    def test_answer_question_uses_planned_retrieval_query(self) -> None:
        planned_query = "study procedures visits assessments intervention participation"
        with patch.object(
            self.pipeline.orchestrator_agent,
            "plan_question_grounding",
            return_value={
                "query": planned_query,
                "top_k": 5,
                "retrieval_mode": "lexical",
                "source_group_filters": ["trial_materials", "regulatory_guidance"],
                "source_id_filters": [],
                "filter_logic": "union",
                "required_source_groups": ["trial_materials"],
                "preferred_source_groups": ["trial_materials", "regulatory_guidance"],
                "question_profile": "study",
                "planning_mode": "model",
                "planning_reason": "test plan",
            },
        ), patch.object(
            self.pipeline.rag_agent,
            "retrieve_evidence",
            return_value={
                "retrieval_hits": [
                    {
                        "source_id": "study_source",
                        "chunk_id": "chunk-1",
                        "citation_label": "Study Source",
                        "excerpt": "Participants will attend visits and complete walk tests.",
                        "metadata": {"source_group": "trial_materials"},
                    }
                ],
                "mode_used": "lexical",
                "evidence_package": {
                    "study_specific_context": "[1] Study Source\nParticipants will attend visits and complete walk tests.",
                    "regulatory_context": "",
                    "other_context": "",
                },
            },
        ) as mock_retrieve, patch.object(
            self.pipeline.conversational_agent,
            "answer_question",
            return_value={
                "run_id": self.run_id,
                "question_id": "q1",
                "request_bundle_path": "request.json",
                "output_path": None,
                "response": None,
                "dry_run": True,
                "agent_handoff_path": None,
            },
        ):
            self.pipeline.orchestrator_agent.answer_question(
                run_id=self.run_id,
                question="What would I have to do in this study?",
                patient_profile_path=self.patient_profile_path,
                dry_run=True,
            )

        self.assertEqual(mock_retrieve.call_args.kwargs["query"], planned_query)

    def test_question_dry_run_writes_role_separated_evidence_package(self) -> None:
        payload = self.pipeline.answer_consent_question(
            run_id=self.run_id,
            question="Can I stop later without penalty?",
            patient_profile_path=self.patient_profile_path,
            dry_run=True,
        )

        self.assertTrue(payload["dry_run"])
        evidence_package_path = Path(payload["evidence_package_path"])
        self.assertTrue(evidence_package_path.exists())
        evidence_package = json.loads(evidence_package_path.read_text(encoding="utf-8"))
        self.assertIn("regulatory_context", evidence_package)
        self.assertIn("study_specific_context", evidence_package)

    @patch("informed_consent.agent_tools.chat_json")
    def test_personalization_revision_loop_applies_improved_revision(self, mock_chat_json) -> None:
        mock_chat_json.side_effect = [
            {
                "overall_strategy": "Use grounded short sentences for each consent topic.",
                "elements": [
                    {
                        "element_id": "voluntary_participation",
                        "status": "supported",
                        "preferred_source_role": "regulatory",
                        "recommended_markers": ["[1]"],
                        "instruction": "State that joining is voluntary.",
                    },
                    {
                        "element_id": "study_procedures",
                        "status": "supported",
                        "preferred_source_role": "regulatory",
                        "recommended_markers": ["[1]"],
                        "instruction": "Explain what the participant would do.",
                    },
                    {
                        "element_id": "risks",
                        "status": "supported",
                        "preferred_source_role": "regulatory",
                        "recommended_markers": ["[1]"],
                        "instruction": "Mention possible risks.",
                    },
                    {
                        "element_id": "benefits",
                        "status": "supported",
                        "preferred_source_role": "regulatory",
                        "recommended_markers": ["[1]"],
                        "instruction": "Mention benefits or lack of direct benefit.",
                    },
                    {
                        "element_id": "alternatives",
                        "status": "supported",
                        "preferred_source_role": "regulatory",
                        "recommended_markers": ["[1]"],
                        "instruction": "Mention alternatives.",
                    },
                    {
                        "element_id": "questions",
                        "status": "supported",
                        "preferred_source_role": "regulatory",
                        "recommended_markers": ["[1]"],
                        "instruction": "Mention questions and contact rights.",
                    },
                    {
                        "element_id": "withdrawal_rights",
                        "status": "supported",
                        "preferred_source_role": "regulatory",
                        "recommended_markers": ["[1]"],
                        "instruction": "Explain withdrawal rights.",
                    },
                ],
            },
            {
                "key_information_summary": "Joining is your choice.",
                "key_information_citation_markers_used": [],
                "personalized_consent_text": "Joining is your choice. You may stop later.",
                "citation_markers_used": [],
                "personalization_rationale": ["Initial plain-language draft."],
                "grounding_limitations": [],
            },
            {
                "key_information_summary": "Joining this study is your choice [1]. You may stop later without penalty [1].",
                "key_information_citation_markers_used": ["[1]"],
                "personalized_consent_text": (
                    "Joining this study is your choice [1]. "
                    "The study team will explain the study procedures and visits [1]. "
                    "Possible risks will be explained before you decide [1]. "
                    "Possible benefits are not guaranteed [1]. "
                    "You can ask questions at any time [1]. "
                    "You may stop later without penalty [1]. "
                    "Other options may be available [1]."
                ),
                "citation_markers_used": ["[1]"],
                "personalization_rationale": ["Added missing consent topics and inline citations."],
                "grounding_limitations": [],
            },
        ]

        payload = self.pipeline.draft_personalized_consent(
            run_id=self.run_id,
            patient_profile_path=self.patient_profile_path,
            template_path=self.template_path,
            dry_run=False,
        )

        revision_metadata = payload["response"]["revision_metadata"]
        self.assertTrue(revision_metadata["revision_attempted"])
        self.assertTrue(revision_metadata["revision_applied"])

        decision_path = self.temp_path / "artifacts" / "runs" / self.run_id / "outputs" / "draft_revision_decision.json"
        self.assertTrue(decision_path.exists())
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        self.assertTrue(decision["accept_revision"])

        final_draft_path = self.temp_path / "artifacts" / "runs" / self.run_id / "outputs" / "personalized_consent_draft.json"
        final_draft = json.loads(final_draft_path.read_text(encoding="utf-8"))
        self.assertTrue(final_draft["revision_metadata"]["revision_applied"])
        self.assertIn("[1]", final_draft["personalized_consent_text"])
        self.assertTrue((self.temp_path / "artifacts" / "runs" / self.run_id / "outputs" / "draft_content_plan.json").exists())


if __name__ == "__main__":
    unittest.main()
