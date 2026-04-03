from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from informed_consent.config import AppConfig, ModelConfig, PathConfig, RetrievalConfig
from informed_consent.pipeline import ConsentPipeline
from informed_consent.types import ChunkRecord


class PipelineNormalizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = ConsentPipeline()

    def test_inject_inline_citations_adds_marker_to_uncited_sentences(self) -> None:
        text = "You can choose to join this study. You can stop at any time without penalty."
        updated = self.pipeline.inject_inline_citations(text, ["[1]", "[2]"], max_sentences=2)
        self.assertIn("[1]", updated)
        self.assertIn("[2]", updated)

    def test_inject_inline_citations_preserves_existing_markers(self) -> None:
        text = "You can choose to join this study [1]. You can stop at any time without penalty."
        updated = self.pipeline.inject_inline_citations(text, ["[1]", "[2]"], max_sentences=2)
        self.assertEqual(updated.count("[1]"), 1)
        self.assertIn("[2]", updated)

    def test_normalize_filter_logic_defaults_to_intersection(self) -> None:
        self.assertEqual(self.pipeline.normalize_filter_logic(None), "intersection")
        self.assertEqual(self.pipeline.normalize_filter_logic("invalid"), "intersection")

    def test_normalize_workflow_variant_defaults_to_full_agentic(self) -> None:
        self.assertEqual(self.pipeline.normalize_workflow_variant(None), "full_agentic")
        self.assertEqual(self.pipeline.normalize_workflow_variant("GENERIC_RAG"), "generic_rag")
        self.assertEqual(self.pipeline.normalize_workflow_variant("unknown"), "full_agentic")

    def test_chunk_matches_retrieval_filters_intersection_requires_both(self) -> None:
        chunk = ChunkRecord(
            chunk_id="c1",
            source_id="nct03877237",
            text="Example text",
            char_count=12,
            token_count_estimate=3,
            citation_label="Example",
            metadata={"source_group": "trial_materials"},
        )
        self.assertFalse(
            self.pipeline.chunk_matches_retrieval_filters(
                chunk,
                source_group_filters={"regulatory_guidance"},
                source_id_filters={"nct03877237"},
                filter_logic="intersection",
            )
        )

    def test_chunk_matches_retrieval_filters_union_accepts_either(self) -> None:
        chunk = ChunkRecord(
            chunk_id="c1",
            source_id="nct03877237",
            text="Example text",
            char_count=12,
            token_count_estimate=3,
            citation_label="Example",
            metadata={"source_group": "trial_materials"},
        )
        self.assertTrue(
            self.pipeline.chunk_matches_retrieval_filters(
                chunk,
                source_group_filters={"regulatory_guidance"},
                source_id_filters={"nct03877237"},
                filter_logic="union",
            )
        )

    def test_build_role_separated_evidence_package_groups_hits_by_source_role(self) -> None:
        retrieval_hits = [
            {
                "source_id": "nct03877237",
                "chunk_id": "study_chunk",
                "citation_label": "ClinicalTrials.gov: NCT03877237",
                "excerpt": "Participants receive dapagliflozin during the trial.",
                "metadata": {"source_group": "trial_materials"},
            },
            {
                "source_id": "fda_guidance",
                "chunk_id": "reg_chunk",
                "citation_label": "FDA Guidance",
                "excerpt": "Participation is voluntary and withdrawal is allowed at any time.",
                "metadata": {"source_group": "regulatory_guidance"},
            },
            {
                "source_id": "misc_source",
                "chunk_id": "other_chunk",
                "citation_label": "Miscellaneous Source",
                "excerpt": "Additional supporting text.",
                "metadata": {"source_group": "supporting_material"},
            },
        ]

        package = self.pipeline.build_role_separated_evidence_package(retrieval_hits)

        self.assertEqual(package["role_counts"]["study_specific"], 1)
        self.assertEqual(package["role_counts"]["regulatory"], 1)
        self.assertEqual(package["role_counts"]["other"], 1)
        self.assertIn("ClinicalTrials.gov: NCT03877237", package["study_specific_context"])
        self.assertIn("FDA Guidance", package["regulatory_context"])
        self.assertIn("Miscellaneous Source", package["other_context"])
        self.assertIn("[1] ClinicalTrials.gov: NCT03877237", package["study_specific_context"])
        self.assertIn("[2] FDA Guidance", package["regulatory_context"])
        self.assertEqual(package["citation_map"][0]["marker"], "[1]")
        self.assertEqual(package["citation_map"][1]["marker"], "[2]")

    def test_normalize_qa_answer_response_injects_inline_markers_when_only_list_field_is_present(self) -> None:
        payload = self.pipeline.normalize_qa_answer_response(
            {
                "answer_text": "Yes, you can stop at any time without penalty. You can ask questions before deciding.",
                "citation_markers_used": ["[1]", "[2]"],
                "uncertainty_noted": False,
                "grounding_limitations": [],
            },
            retrieval_hits=[
                {"chunk_id": "c1"},
                {"chunk_id": "c2"},
            ],
        )

        self.assertIn("[1]", payload["answer_text"])
        self.assertIn("Inserted answer citation markers inline", " ".join(payload["schema_repair_notes"]))

    def test_load_study_cohort_reads_source_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cohort_path = root / "cohort.json"
            cohort_path.write_text(
                json.dumps(
                    {
                        "cohort_id": "demo",
                        "studies": [
                            {"source_id": "NCT03877237"},
                            {"source_id": "nct04210375"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            studies, resolved_path = self.pipeline.load_study_cohort(
                cohort_file_value=cohort_path,
                base_dir=root,
            )

        self.assertEqual([study["source_id"] for study in studies], ["nct03877237", "nct04210375"])
        self.assertEqual(resolved_path, cohort_path.resolve())

    def test_expand_batch_case_definitions_builds_case_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cohort_path = root / "cohort.json"
            cohort_path.write_text(
                json.dumps(
                    {
                        "cohort_id": "demo",
                        "studies": [
                            {"source_id": "NCT03877237", "site_id": "HF-SITE-01"},
                            {"source_id": "NCT04210375", "workflow_variant": "generic_rag"},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            spec = {
                "batch_id": "demo",
                "case_matrix": {
                    "study_cohort_file": str(cohort_path),
                    "patient_profile_files": [
                        "configs/patient_profiles/example_us_low_literacy.json",
                        "configs/patient_profiles/example_us_medium_literacy.json",
                    ],
                    "question_set_files": [
                        "configs/question_sets/consent_rights_basic.json",
                        "configs/question_sets/study_specific_basics.json",
                    ],
                },
            }
            expanded_cases = self.pipeline.expand_batch_case_definitions(
                spec=spec,
                spec_path=root / "spec.json",
                defaults={},
            )

        self.assertEqual(len(expanded_cases), 8)
        self.assertEqual(expanded_cases[0]["retrieval_source_ids"], ["nct03877237"])
        self.assertEqual(expanded_cases[0]["study_source_id"], "nct03877237")
        self.assertEqual(expanded_cases[0]["study_id"], "NCT03877237")
        self.assertEqual(expanded_cases[0]["site_id"], "HF-SITE-01")
        case_ids = {case["case_id"] for case in expanded_cases}
        self.assertIn(
            "nct03877237_example_us_low_literacy_consent_rights_basic",
            case_ids,
        )
        self.assertIn(
            "nct04210375_example_us_medium_literacy_study_specific_basics",
            case_ids,
        )
        generic_rag_cases = [case for case in expanded_cases if case["study_source_id"] == "nct04210375"]
        self.assertTrue(generic_rag_cases)
        self.assertTrue(all(case["workflow_variant"] == "generic_rag" for case in generic_rag_cases))

    def test_compare_batch_results_exports_aggregate_table(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            repo_root = Path(__file__).resolve().parents[1]
            config = AppConfig(
                models=ModelConfig(endpoint_url="https://example.test"),
                retrieval=RetrievalConfig(retrieval_mode="lexical"),
                paths=PathConfig(
                    project_root=repo_root,
                    artifact_root=root / "artifacts",
                    source_data_root=root / "data",
                    configs_root=repo_root / "configs",
                    prompts_root=repo_root / "prompts",
                    docs_root=repo_root / "docs",
                    scripts_root=repo_root / "scripts",
                ),
            )
            pipeline = ConsentPipeline(config)
            summary_a = root / "batch_a.json"
            summary_b = root / "batch_b.json"
            summary_a.write_text(
                json.dumps(
                    {
                        "batch_id": "full_system",
                        "batch_run_id": "batch-a",
                        "reporting_role": "scientific_evaluation",
                        "case_count": 10,
                        "completed_case_count": 10,
                        "failed_case_count": 0,
                        "case_metrics_csv": "a.csv",
                        "aggregate_metrics": {
                            "average_draft_required_element_coverage_ratio": 0.8,
                        },
                    }
                ),
                encoding="utf-8",
            )
            summary_b.write_text(
                json.dumps(
                    {
                        "batch_id": "generic_rag",
                        "batch_run_id": "batch-b",
                        "reporting_role": "baseline",
                        "case_count": 10,
                        "completed_case_count": 10,
                        "failed_case_count": 0,
                        "case_metrics_csv": "b.csv",
                        "aggregate_metrics": {
                            "average_draft_required_element_coverage_ratio": 0.7,
                        },
                    }
                ),
                encoding="utf-8",
            )

            payload = pipeline.compare_batch_results(
                [summary_a, summary_b],
                comparison_id="demo_compare",
            )

            self.assertEqual(payload["row_count"], 2)
            self.assertTrue(Path(payload["comparison_csv"]).exists())
            self.assertTrue(Path(payload["comparison_json"]).exists())


if __name__ == "__main__":
    unittest.main()
