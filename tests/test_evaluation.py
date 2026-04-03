from __future__ import annotations

import unittest

from informed_consent.evaluation import (
    build_draft_revision_audit,
    compare_draft_revision_candidates,
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


if __name__ == "__main__":
    unittest.main()
