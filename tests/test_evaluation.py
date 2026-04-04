from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from informed_consent.evaluation import (
    build_draft_revision_audit,
    compare_draft_revision_candidates,
    evaluate_run_outputs,
    sentence_citation_metrics,
    summarize_personalized_draft,
)


class SentenceCitationMetricsTests(unittest.TestCase):
    def test_empty_text_returns_zero_metrics(self) -> None:
        metrics = sentence_citation_metrics("")
        self.assertEqual(metrics["sentence_with_citation_count"], 0.0)
        self.assertEqual(metrics["sentence_without_citation_count"], 0.0)
        self.assertEqual(metrics["sentence_citation_coverage_ratio"], 0.0)

    def test_counts_cited_and_uncited_sentences(self) -> None:
        text = "You may leave the study at any time [1]. Risks may include discomfort. Ask questions anytime [2]."
        metrics = sentence_citation_metrics(text)
        self.assertEqual(metrics["sentence_with_citation_count"], 2.0)
        self.assertEqual(metrics["sentence_without_citation_count"], 1.0)
        self.assertAlmostEqual(metrics["sentence_citation_coverage_ratio"], 0.6667, places=4)


class DraftRevisionAuditTests(unittest.TestCase):
    def test_audit_flags_missing_elements_and_low_citation_density(self) -> None:
        draft_summary = summarize_personalized_draft(
            {
                "key_information_summary": "Joining is your choice.",
                "key_information_citation_markers_used": [],
                "personalized_consent_text": "Joining is your choice. You may stop later.",
                "citation_markers_used": [],
                "personalization_rationale": [],
                "grounding_limitations": [],
            },
            available_markers=["[1]", "[2]"],
            health_literacy="low",
        )

        audit = build_draft_revision_audit(draft_summary)

        self.assertTrue(audit["needs_revision"])
        self.assertIn("study_procedures", audit["missing_required_elements"])
        self.assertIn("draft_sentence_citation_coverage_below_threshold", audit["issues"])

    def test_audit_flags_missing_planned_required_elements(self) -> None:
        draft_summary = summarize_personalized_draft(
            {
                "key_information_summary": "Joining is your choice [1].",
                "key_information_citation_markers_used": ["[1]"],
                "personalized_consent_text": "Joining is your choice [1]. You may stop later without penalty [1].",
                "citation_markers_used": ["[1]"],
                "personalization_rationale": [],
                "grounding_limitations": [],
            },
            available_markers=["[1]", "[2]"],
            health_literacy="low",
        )

        audit = build_draft_revision_audit(
            draft_summary,
            draft_content_plan={
                "elements": [
                    {"element_id": "study_procedures", "status": "partially_supported"},
                    {"element_id": "benefits", "status": "partially_supported"},
                    {"element_id": "alternatives", "status": "unsupported"},
                ]
            },
        )

        self.assertIn("planned_required_elements_missing", audit["issues"])
        self.assertIn("study_procedures", audit["missing_planned_required_elements"])
        self.assertIn("benefits", audit["missing_planned_required_elements"])

    def test_comparison_accepts_meaningfully_improved_revision(self) -> None:
        initial_summary = summarize_personalized_draft(
            {
                "key_information_summary": "Joining is your choice.",
                "key_information_citation_markers_used": [],
                "personalized_consent_text": "Joining is your choice. You may stop later.",
                "citation_markers_used": [],
                "personalization_rationale": [],
                "grounding_limitations": [],
            },
            available_markers=["[1]", "[2]", "[3]"],
            health_literacy="low",
        )
        revised_summary = summarize_personalized_draft(
            {
                "key_information_summary": "Joining is your choice [1]. You may stop later without penalty [2].",
                "key_information_citation_markers_used": ["[1]", "[2]"],
                "personalized_consent_text": (
                    "Joining this study is your choice [1]. "
                    "The study team will explain the study procedures and visits [2]. "
                    "Possible risks will be explained before you decide [2]. "
                    "Possible benefits are not guaranteed [3]. "
                    "You can ask questions at any time [1]. "
                    "You may stop later without penalty [1]. "
                    "Other options may be available [3]."
                ),
                "citation_markers_used": ["[1]", "[2]", "[3]"],
                "personalization_rationale": [],
                "grounding_limitations": [],
            },
            available_markers=["[1]", "[2]", "[3]"],
            health_literacy="low",
        )

        comparison = compare_draft_revision_candidates(initial_summary, revised_summary)

        self.assertTrue(comparison["accept_revision"])
        self.assertGreater(comparison["quality_score_delta"], 0.0)

    def test_comparison_rejects_revision_that_loses_readability_target(self) -> None:
        initial_summary = summarize_personalized_draft(
            {
                "key_information_summary": "Joining is your choice [1]. You may stop later [2].",
                "key_information_citation_markers_used": ["[1]", "[2]"],
                "personalized_consent_text": (
                    "Joining is your choice [1]. "
                    "The team will explain study steps [2]. "
                    "You may stop later without penalty [2]. "
                    "You can ask questions [1]. "
                    "Other options may be available [2]."
                ),
                "citation_markers_used": ["[1]", "[2]"],
                "personalization_rationale": [],
                "grounding_limitations": [],
            },
            available_markers=["[1]", "[2]"],
            health_literacy="low",
        )
        revised_summary = summarize_personalized_draft(
            {
                "key_information_summary": (
                    "Participation is voluntary, and you may discontinue involvement without penalty or loss of benefits [1]."
                ),
                "key_information_citation_markers_used": ["[1]"],
                "personalized_consent_text": (
                    "Participation in this clinical investigation is voluntary, and you may discontinue involvement without penalty or "
                    "loss of benefits to which you are otherwise entitled [1]. "
                    "The investigative team will explicate the procedural assessments, questionnaires, and ambulatory evaluations [2]. "
                    "Alternative therapeutic options remain available and may be discussed with the study staff [2]."
                ),
                "citation_markers_used": ["[1]", "[2]"],
                "personalization_rationale": [],
                "grounding_limitations": [],
            },
            available_markers=["[1]", "[2]"],
            health_literacy="low",
        )

        comparison = compare_draft_revision_candidates(initial_summary, revised_summary)

        self.assertFalse(comparison["accept_revision"])
        self.assertIn("revision_lost_draft_readability_target", comparison["reasons"])

    def test_comparison_rejects_revision_that_loses_planned_required_elements(self) -> None:
        initial_summary = summarize_personalized_draft(
            {
                "key_information_summary": "You can choose to join [1].",
                "key_information_citation_markers_used": ["[1]"],
                "personalized_consent_text": (
                    "You can choose to join [1]. "
                    "The team will explain study steps [1]. "
                    "Possible benefits are not guaranteed [1]. "
                    "Other options may be available [1]."
                ),
                "citation_markers_used": ["[1]"],
                "personalization_rationale": [],
                "grounding_limitations": [],
            },
            available_markers=["[1]"],
            health_literacy="low",
        )
        revised_summary = summarize_personalized_draft(
            {
                "key_information_summary": "You can choose to join [1].",
                "key_information_citation_markers_used": ["[1]"],
                "personalized_consent_text": "You can choose to join [1]. You may stop later [1].",
                "citation_markers_used": ["[1]"],
                "personalization_rationale": [],
                "grounding_limitations": [],
            },
            available_markers=["[1]"],
            health_literacy="low",
        )

        comparison = compare_draft_revision_candidates(
            initial_summary,
            revised_summary,
            initial_audit={
                "missing_planned_required_elements": [],
            },
            revised_audit={
                "missing_planned_required_elements": ["study_procedures", "benefits"],
            },
        )

        self.assertFalse(comparison["accept_revision"])
        self.assertIn("revision_lost_planned_required_elements", comparison["reasons"])


class EvaluationOutputTests(unittest.TestCase):
    def test_evaluate_run_outputs_tracks_expected_study_specific_grounding(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            (run_dir / "inputs").mkdir(parents=True, exist_ok=True)
            (run_dir / "outputs").mkdir(parents=True, exist_ok=True)
            (run_dir / "inputs" / "patient_profile.json").write_text(
                json.dumps(
                    {
                        "participant_id": "P-1",
                        "health_literacy": "medium",
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "outputs" / "personalized_consent_draft.json").write_text(
                json.dumps(
                    {
                        "key_information_summary": "You can choose to join [1].",
                        "key_information_citation_markers_used": ["[1]"],
                        "personalized_consent_text": "You can choose to join [1]. You may ask questions [1].",
                        "citation_markers_used": ["[1]"],
                        "personalization_rationale": [],
                        "grounding_limitations": [],
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "outputs" / "personalization_request_bundle.json").write_text(
                json.dumps(
                    {
                        "workflow_variant": "full_agentic",
                        "source_group_filters": ["regulatory_guidance", "trial_materials"],
                        "source_id_filters": ["nct03877237"],
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "outputs" / "personalization_evidence_package.json").write_text(
                json.dumps(
                    {
                        "role_counts": {
                            "study_specific": 0,
                            "regulatory": 2,
                            "other": 0,
                        }
                    }
                ),
                encoding="utf-8",
            )

            summary = evaluate_run_outputs("run-1", run_dir)["summary"]

        self.assertTrue(summary["draft"]["expected_study_specific_grounding"])
        self.assertFalse(summary["draft"]["study_specific_grounding_met"])
        self.assertTrue(summary["draft"]["study_specific_grounding_gap"])

    def test_evaluate_run_outputs_detects_foreign_study_contamination_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            (run_dir / "inputs").mkdir(parents=True, exist_ok=True)
            (run_dir / "outputs").mkdir(parents=True, exist_ok=True)
            (run_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "study_id": "NCT03877237",
                        "site_id": "PUBLIC-SOURCE",
                        "runtime_metadata": {
                            "model_id": "Qwen/Qwen3-8B",
                            "embedding_model_id": "BAAI/bge-small-en-v1.5",
                            "config_path": "configs/experiments/demo.json",
                            "git_commit_hash": "abc123",
                            "corpus_version": "run-1",
                            "index_version": "run-1",
                            "random_seed": 17,
                        },
                        "context_metadata": {
                            "study_source_id": "nct03877237",
                            "workflow_variant": "generic_rag",
                            "patient_profile_label": "example_us_medium_literacy",
                            "question_set_label": "study_specific_basics",
                            "retrieval_mode": "hybrid",
                            "retrieval_top_k": 6,
                            "retrieval_filter_logic": "union",
                            "base_run_id": "run-1",
                            "batch_run_id": "batch-1",
                            "batch_id": "demo",
                            "reporting_role": "pilot",
                        },
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "inputs" / "patient_profile.json").write_text(
                json.dumps({"participant_id": "P-1", "health_literacy": "medium"}),
                encoding="utf-8",
            )
            (run_dir / "outputs" / "personalized_consent_draft.json").write_text(
                json.dumps(
                    {
                        "key_information_summary": "You can choose to join [1].",
                        "key_information_citation_markers_used": ["[1]"],
                        "personalized_consent_text": "You can choose to join [1]. You may ask questions [2].",
                        "citation_markers_used": ["[1]", "[2]"],
                        "personalization_rationale": [],
                        "grounding_limitations": [],
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "outputs" / "personalization_request_bundle.json").write_text(
                json.dumps(
                    {
                        "workflow_variant": "generic_rag",
                        "source_group_filters": ["regulatory_guidance", "trial_materials"],
                        "source_id_filters": ["nct03877237"],
                        "system_prompt_id": "draft_system_v1",
                        "user_prompt_id": "draft_user_v1",
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "outputs" / "personalization_retrieval_hits.json").write_text(
                json.dumps(
                    [
                        {
                            "source_id": "nct03877237",
                            "chunk_id": "study-1",
                            "citation_label": "Target Study",
                            "excerpt": "Study details.",
                            "metadata": {"source_group": "trial_materials"},
                        },
                        {
                            "source_id": "nct99999999",
                            "chunk_id": "study-2",
                            "citation_label": "Foreign Study",
                            "excerpt": "Foreign details.",
                            "metadata": {"source_group": "trial_materials"},
                        },
                        {
                            "source_id": "fda_guidance",
                            "chunk_id": "reg-1",
                            "citation_label": "FDA Guidance",
                            "excerpt": "Participation is voluntary.",
                            "metadata": {"source_group": "regulatory_guidance"},
                        },
                    ]
                ),
                encoding="utf-8",
            )

            summary = evaluate_run_outputs("run-foreign", run_dir)["summary"]

        self.assertEqual(summary["metadata"]["question_set_label"], "study_specific_basics")
        self.assertEqual(summary["metadata"]["draft_system_prompt_id"], "draft_system_v1")
        self.assertTrue(summary["draft"]["selected_study_hit_present"])
        self.assertTrue(summary["draft"]["foreign_study_hit_present"])
        self.assertEqual(summary["draft"]["regulatory_hit_count"], 1)
        self.assertEqual(summary["draft"]["foreign_source_ids_detected"], ["nct99999999"])
        self.assertTrue(summary["failure_taxonomy"]["draft"]["foreign_study_contamination"])
        self.assertTrue(summary["failure_taxonomy"]["case_failure_flags"]["foreign_study_contamination"])

    def test_evaluate_run_outputs_counts_qa_abstentions_from_index(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            qa_dir = run_dir / "outputs" / "qa"
            qa_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "inputs").mkdir(parents=True, exist_ok=True)
            (run_dir / "inputs" / "patient_profile.json").write_text(
                json.dumps({"participant_id": "P-1", "health_literacy": "low"}),
                encoding="utf-8",
            )
            (qa_dir / "qa_index.jsonl").write_text(
                json.dumps(
                    {
                        "question_id": "q1",
                        "question": "What would I have to do in this study?",
                        "answer_path": None,
                    }
                ) + "\n",
                encoding="utf-8",
            )

            summary = evaluate_run_outputs("run-qa", run_dir)["summary"]

        self.assertTrue(summary["qa_answers"]["artifact_present"])
        self.assertEqual(summary["qa_answers"]["question_count"], 1)
        self.assertEqual(summary["qa_answers"]["answered_question_count"], 0)
        self.assertEqual(summary["qa_answers"]["abstained_question_count"], 1)
        self.assertEqual(summary["qa_answers"]["abstention_rate"], 1.0)

    def test_evaluate_run_outputs_tracks_qa_failure_signals_for_unsupported_answer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            qa_dir = run_dir / "outputs" / "qa"
            qa_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "inputs").mkdir(parents=True, exist_ok=True)
            (run_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "study_id": "NCT03877237",
                        "site_id": "PUBLIC-SOURCE",
                        "context_metadata": {
                            "study_source_id": "nct03877237",
                            "workflow_variant": "generic_rag",
                            "patient_profile_label": "example_us_medium_literacy",
                            "question_set_label": "study_specific_basics",
                        },
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "inputs" / "patient_profile.json").write_text(
                json.dumps({"participant_id": "P-1", "health_literacy": "medium"}),
                encoding="utf-8",
            )
            request_bundle_path = qa_dir / "q1.request.json"
            answer_path = qa_dir / "q1.answer.json"
            retrieval_hits_path = qa_dir / "q1.hits.json"
            request_bundle_path.write_text(
                json.dumps(
                    {
                        "source_group_filters": ["regulatory_guidance", "trial_materials"],
                        "source_id_filters": ["nct03877237"],
                        "system_prompt_id": "qa_system_v1",
                        "user_prompt_id": "qa_user_v1",
                    }
                ),
                encoding="utf-8",
            )
            answer_path.write_text(
                json.dumps(
                    {
                        "answer_text": "You would have to attend several study visits.",
                        "citation_markers_used": [],
                        "uncertainty_noted": False,
                        "grounding_limitations": [],
                    }
                ),
                encoding="utf-8",
            )
            retrieval_hits_path.write_text(
                json.dumps(
                    [
                        {
                            "source_id": "fda_guidance",
                            "chunk_id": "reg-1",
                            "citation_label": "FDA Guidance",
                            "excerpt": "Participants may ask questions.",
                            "metadata": {"source_group": "regulatory_guidance"},
                        }
                    ]
                ),
                encoding="utf-8",
            )
            (qa_dir / "qa_index.jsonl").write_text(
                json.dumps(
                    {
                        "question_id": "q1",
                        "question": "What would I have to do in this study?",
                        "answer_path": str(answer_path),
                        "request_bundle_path": str(request_bundle_path),
                        "retrieval_hits_path": str(retrieval_hits_path),
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            summary = evaluate_run_outputs("run-qa-risk", run_dir)["summary"]

        self.assertEqual(summary["qa_answers"]["answered_count"], 1)
        self.assertEqual(summary["qa_answers"]["selected_study_hit_count"], 0)
        self.assertTrue(summary["qa_answers"]["study_specific_grounding_gap"])
        self.assertGreater(summary["qa_answers"]["citationless_sentence_count"], 0)
        self.assertGreater(summary["qa_answers"]["unsupported_sentence_count"], 0)
        self.assertTrue(summary["qa_answers"]["failure_flags"]["overconfident_answer"])
        self.assertTrue(summary["failure_taxonomy"]["case_failure_flags"]["missing_selected_study_grounding"])
        self.assertTrue(summary["failure_taxonomy"]["case_failure_flags"]["overconfident_answer"])


if __name__ == "__main__":
    unittest.main()
