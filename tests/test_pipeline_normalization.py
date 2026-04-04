from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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
            case_metrics_a = root / "a.csv"
            case_metrics_b = root / "b.csv"
            case_metrics_a.write_text(
                "\n".join(
                    [
                        (
                            "workflow_variant,question_set_label,model_id,embedding_model_id,retrieval_mode,retrieval_top_k,"
                            "retrieval_filter_logic,draft_system_prompt_id,draft_user_prompt_id,qa_system_prompt_ids,"
                            "qa_user_prompt_ids,config_path,git_commit_hash,corpus_version,index_version,random_seed,"
                            "draft_required_element_coverage_ratio,draft_study_specific_grounding_met,qa_abstention_rate,"
                            "failure_missing_selected_study_grounding"
                        ),
                        (
                            "full_agentic,study_specific_basics,Qwen/Qwen3-8B,BAAI/bge-small-en-v1.5,hybrid,6,union,"
                            "draft_system_v1,draft_user_v1,qa_system_v1,qa_user_v1,configs/experiments/full.json,"
                            "abc123,run-1,run-1,17,0.9,True,0.0,False"
                        ),
                    ]
                ),
                encoding="utf-8",
            )
            case_metrics_b.write_text(
                "\n".join(
                    [
                        (
                            "workflow_variant,question_set_label,model_id,embedding_model_id,retrieval_mode,retrieval_top_k,"
                            "retrieval_filter_logic,draft_system_prompt_id,draft_user_prompt_id,qa_system_prompt_ids,"
                            "qa_user_prompt_ids,config_path,git_commit_hash,corpus_version,index_version,random_seed,"
                            "draft_required_element_coverage_ratio,draft_study_specific_grounding_met,qa_abstention_rate,"
                            "failure_missing_selected_study_grounding"
                        ),
                        (
                            "generic_rag,study_specific_basics,Qwen/Qwen3-8B,BAAI/bge-small-en-v1.5,hybrid,6,union,"
                            "draft_system_v1,draft_user_v1,qa_system_v1,qa_user_v1,configs/experiments/generic.json,"
                            "abc123,run-1,run-1,17,0.7,False,0.2,True"
                        ),
                    ]
                ),
                encoding="utf-8",
            )
            summary_a.write_text(
                json.dumps(
                    {
                        "batch_id": "full_system",
                        "batch_run_id": "batch-a",
                        "reporting_role": "scientific_evaluation",
                        "workflow_variants": ["full_agentic"],
                        "study_source_ids": ["nct03877237"],
                        "study_ids": ["NCT03877237"],
                        "patient_profile_labels": ["example_us_medium_literacy"],
                        "question_set_labels": ["study_specific_basics"],
                        "model_id": "Qwen/Qwen3-8B",
                        "embedding_model_id": "BAAI/bge-small-en-v1.5",
                        "retrieval_modes": ["hybrid"],
                        "retrieval_top_k_values": [6],
                        "retrieval_filter_logics": ["union"],
                        "draft_system_prompt_ids": ["draft_system_v1"],
                        "draft_user_prompt_ids": ["draft_user_v1"],
                        "formalization_system_prompt_ids": ["formalization_system_v1"],
                        "formalization_user_prompt_ids": ["formalization_user_v1"],
                        "qa_system_prompt_ids": ["qa_system_v1"],
                        "qa_user_prompt_ids": ["qa_user_v1"],
                        "config_path": "configs/experiments/full.json",
                        "corpus_version": "run-1",
                        "index_version": "run-1",
                        "git_commit_hash": "abc123",
                        "random_seed": 17,
                        "case_count": 10,
                        "completed_case_count": 10,
                        "failed_case_count": 0,
                        "case_metrics_csv": str(case_metrics_a),
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
                        "workflow_variants": ["generic_rag"],
                        "study_source_ids": ["nct03877237"],
                        "study_ids": ["NCT03877237"],
                        "patient_profile_labels": ["example_us_medium_literacy"],
                        "question_set_labels": ["study_specific_basics"],
                        "model_id": "Qwen/Qwen3-8B",
                        "embedding_model_id": "BAAI/bge-small-en-v1.5",
                        "retrieval_modes": ["hybrid"],
                        "retrieval_top_k_values": [6],
                        "retrieval_filter_logics": ["union"],
                        "draft_system_prompt_ids": ["draft_system_v1"],
                        "draft_user_prompt_ids": ["draft_user_v1"],
                        "formalization_system_prompt_ids": ["formalization_system_v1"],
                        "formalization_user_prompt_ids": ["formalization_user_v1"],
                        "qa_system_prompt_ids": ["qa_system_v1"],
                        "qa_user_prompt_ids": ["qa_user_v1"],
                        "config_path": "configs/experiments/generic.json",
                        "corpus_version": "run-1",
                        "index_version": "run-1",
                        "git_commit_hash": "abc123",
                        "random_seed": 17,
                        "case_count": 10,
                        "completed_case_count": 10,
                        "failed_case_count": 0,
                        "case_metrics_csv": str(case_metrics_b),
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
            self.assertTrue(Path(payload["case_rows_csv"]).exists())
            self.assertTrue(Path(payload["grouped_comparison_csv"]).exists())
            self.assertTrue(Path(payload["overall_workflow_csv"]).exists())
            self.assertTrue(Path(payload["failure_summary_csv"]).exists())

            with Path(payload["comparison_csv"]).open("r", encoding="utf-8", newline="") as handle:
                comparison_rows = list(csv.DictReader(handle))
            self.assertEqual(comparison_rows[0]["model_id"], "Qwen/Qwen3-8B")
            self.assertEqual(comparison_rows[0]["workflow_variants"], "full_agentic")

            with Path(payload["grouped_comparison_csv"]).open("r", encoding="utf-8", newline="") as handle:
                grouped_rows = list(csv.DictReader(handle))
            self.assertEqual(grouped_rows[0]["workflow_variant"], "full_agentic")
            self.assertEqual(grouped_rows[0]["question_set_label"], "study_specific_basics")
            self.assertEqual(grouped_rows[0]["retrieval_mode"], "hybrid")
            self.assertEqual(grouped_rows[0]["draft_system_prompt_id"], "draft_system_v1")

            with Path(payload["overall_workflow_csv"]).open("r", encoding="utf-8", newline="") as handle:
                workflow_rows = list(csv.DictReader(handle))
            self.assertEqual({row["workflow_variant"] for row in workflow_rows}, {"full_agentic", "generic_rag"})

            with Path(payload["failure_summary_csv"]).open("r", encoding="utf-8", newline="") as handle:
                failure_rows = list(csv.DictReader(handle))
            matching_failure_rows = [
                row
                for row in failure_rows
                if row["workflow_variant"] == "generic_rag"
                and row["question_set_label"] == "study_specific_basics"
                and row["failure_type"] == "missing_selected_study_grounding"
            ]
            self.assertEqual(len(matching_failure_rows), 1)
            self.assertEqual(matching_failure_rows[0]["failure_count"], "1")

    def test_gonogo_specs_cover_all_three_variants_with_one_shared_case_matrix(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        spec_map = {
            "full_agentic": repo_root / "configs" / "experiments" / "pilots" / "nct03877237_gonogo_full_agentic_v1.json",
            "generic_rag": repo_root / "configs" / "experiments" / "pilots" / "nct03877237_gonogo_generic_rag_v1.json",
            "vanilla_llm": repo_root / "configs" / "experiments" / "pilots" / "nct03877237_gonogo_vanilla_llm_v1.json",
        }

        payloads = {variant: json.loads(path.read_text(encoding="utf-8")) for variant, path in spec_map.items()}

        for variant, payload in payloads.items():
            self.assertEqual(payload["defaults"]["workflow_variant"], variant)
            self.assertEqual(len(payload["cases"]), 1)
            self.assertEqual(payload["cases"][0]["study_source_id"], "nct03877237")

        patient_profiles = {payload["cases"][0]["patient_profile_file"] for payload in payloads.values()}
        question_sets = {payload["cases"][0]["question_set_file"] for payload in payloads.values()}
        self.assertEqual(len(patient_profiles), 1)
        self.assertEqual(len(question_sets), 1)

    def test_run_batch_experiment_reports_vanilla_llm_as_no_retrieval(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            repo_root = Path(__file__).resolve().parents[1]
            config = AppConfig(
                models=ModelConfig(endpoint_url="https://example.test"),
                retrieval=RetrievalConfig(retrieval_mode="hybrid"),
                paths=PathConfig(
                    project_root=repo_root,
                    artifact_root=root / "artifacts",
                    source_data_root=repo_root / "data",
                    configs_root=repo_root / "configs",
                    prompts_root=repo_root / "prompts",
                    docs_root=repo_root / "docs",
                    scripts_root=repo_root / "scripts",
                ),
            )
            pipeline = ConsentPipeline(config)
            base_run_id = "prepared-run"
            (config.paths.artifact_root / "runs" / base_run_id).mkdir(parents=True, exist_ok=True)
            spec_path = root / "vanilla_batch.json"
            spec_path.write_text(
                json.dumps(
                    {
                        "batch_id": "gonogo_vanilla_reporting",
                        "base_run_id": base_run_id,
                        "reporting_role": "pilot",
                        "defaults": {
                            "template_file": "data/raw/examples/base_consent_template.txt",
                            "top_k": 6,
                            "generate_draft": True,
                            "formalize": True,
                            "retrieval_mode": "hybrid",
                            "retrieval_filter_logic": "union",
                            "retrieval_source_groups": ["regulatory_guidance", "trial_materials"],
                            "workflow_variant": "vanilla_llm",
                        },
                        "cases": [
                            {
                                "case_id": "nct03877237_medium_study_specific_gonogo_vanilla",
                                "study_source_id": "nct03877237",
                                "study_id": "NCT03877237",
                                "site_id": "PUBLIC-SOURCE",
                                "patient_profile_file": "configs/patient_profiles/example_us_medium_literacy.json",
                                "question_set_file": "configs/question_sets/study_specific_basics.json",
                                "retrieval_source_ids": ["nct03877237"],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            summary_stub = {
                "metadata": {
                    "config_path": str(spec_path.resolve()),
                    "git_commit_hash": "abc123",
                    "model_id": "Qwen/Qwen3-8B",
                    "embedding_model_id": "BAAI/bge-small-en-v1.5",
                    "corpus_version": base_run_id,
                    "index_version": base_run_id,
                    "random_seed": 17,
                    "retrieval_top_k": 6,
                    "draft_system_prompt_id": "draft_system_v1",
                    "draft_user_prompt_id": "draft_user_v1",
                    "formalization_system_prompt_id": "formalization_system_v1",
                    "formalization_user_prompt_id": "formalization_user_v1",
                    "qa_system_prompt_ids": ["qa_system_v1"],
                    "qa_user_prompt_ids": ["qa_user_v1"],
                },
                "draft": {
                    "required_element_coverage_ratio": 0.75,
                    "citation_marker_coverage_ratio": 0.0,
                    "sentence_citation_coverage_ratio": 0.0,
                    "flesch_kincaid_grade": 6.1,
                    "expected_study_specific_grounding": False,
                    "study_specific_grounding_met": True,
                    "study_specific_grounding_gap": False,
                    "has_study_specific_evidence": False,
                    "study_specific_hit_ratio": 0.0,
                    "selected_study_hit_count": 0,
                    "selected_study_hit_present": False,
                    "foreign_study_hit_count": 0,
                    "foreign_study_hit_present": False,
                    "regulatory_hit_count": 0,
                    "total_hit_count": 0,
                    "grounding_source_ids_used": [],
                    "foreign_source_ids_detected": [],
                    "missing_required_element_count": 1,
                    "citationless_sentence_count": 2,
                    "citationless_sentence_rate": 1.0,
                    "unsupported_marker_count": 0,
                    "unsupported_sentence_count": 2,
                    "grounding_gap_declared": False,
                    "unsupported_claim_risk": True,
                    "failure_flags": {
                        "missing_selected_study_grounding": False,
                        "foreign_study_contamination": False,
                        "regulatory_only_grounding": False,
                        "unsupported_claim_risk": True,
                        "omitted_required_element": True,
                        "overconfident_answer": False,
                        "malformed_structured_output": False,
                        "grounding_gap_declared": False,
                    },
                },
                "structured_record": {
                    "required_field_presence_ratio": 1.0,
                    "schema_repair_applied": False,
                    "malformed_structured_output": False,
                    "failure_flags": {
                        "missing_selected_study_grounding": False,
                        "foreign_study_contamination": False,
                        "regulatory_only_grounding": False,
                        "unsupported_claim_risk": False,
                        "omitted_required_element": False,
                        "overconfident_answer": False,
                        "malformed_structured_output": False,
                        "grounding_gap_declared": False,
                    },
                },
                "qa_answers": {
                    "question_count": 3,
                    "answered_question_count": 3,
                    "answered_count": 3,
                    "abstained_question_count": 0,
                    "abstained_count": 0,
                    "clarified_count": 0,
                    "abstention_rate": 0.0,
                    "uncertainty_flag_count": 0,
                    "uncertainty_rate": 0.0,
                    "unsupported_marker_count": 0,
                    "unsupported_sentence_count": 0,
                    "citationless_sentence_count": 0,
                    "citationless_sentence_rate": 0.0,
                    "selected_study_hit_count": 0,
                    "selected_study_hit_present": False,
                    "foreign_study_hit_count": 0,
                    "foreign_study_hit_present": False,
                    "regulatory_hit_count": 0,
                    "total_hit_count": 0,
                    "study_specific_grounding_met": True,
                    "study_specific_grounding_gap": False,
                    "grounding_source_ids_used": [],
                    "foreign_source_ids_detected": [],
                    "average_sentence_citation_coverage_ratio": 0.0,
                    "average_flesch_kincaid_grade": 5.8,
                    "average_citation_marker_coverage_ratio": 0.0,
                    "failure_flags": {
                        "missing_selected_study_grounding": False,
                        "foreign_study_contamination": False,
                        "regulatory_only_grounding": False,
                        "unsupported_claim_risk": False,
                        "omitted_required_element": False,
                        "overconfident_answer": False,
                        "malformed_structured_output": False,
                        "grounding_gap_declared": False,
                    },
                    "per_question": [],
                },
                "failure_taxonomy": {
                    "case_failure_flags": {
                        "missing_selected_study_grounding": False,
                        "foreign_study_contamination": False,
                        "regulatory_only_grounding": False,
                        "unsupported_claim_risk": True,
                        "omitted_required_element": True,
                        "overconfident_answer": False,
                        "malformed_structured_output": False,
                        "grounding_gap_declared": False,
                    }
                },
            }

            question_counter = {"value": 0}

            def fake_answer(*args, **kwargs):
                question_counter["value"] += 1
                return {
                    "question_id": f"q{question_counter['value']}",
                    "output_path": str(root / f"answer_{question_counter['value']}.json"),
                }

            with (
                mock.patch.object(
                    pipeline,
                    "draft_personalized_consent",
                    return_value={
                        "output_path": str(root / "draft.json"),
                        "retrieval_hits_path": str(root / "draft_hits.json"),
                    },
                ),
                mock.patch.object(pipeline, "answer_consent_question", side_effect=fake_answer),
                mock.patch.object(
                    pipeline,
                    "evaluate_run",
                    return_value={
                        "summary": summary_stub,
                        "summary_path": str(root / "evaluation_summary.json"),
                    },
                ),
                mock.patch.object(
                    pipeline,
                    "export_manual_review_bundle",
                    return_value={"manual_review_bundle_csv": str(root / "manual_review.csv")},
                ),
            ):
                payload = pipeline.run_batch_experiment(spec_path, dry_run=True)

            self.assertTrue(Path(payload["case_metrics_csv"]).exists())
            self.assertTrue(Path(payload["failure_summary_csv"]).exists())
            with Path(payload["case_metrics_csv"]).open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["workflow_variant"], "vanilla_llm")
            self.assertEqual(rows[0]["retrieval_mode"], "none")
            self.assertEqual(rows[0]["question_set_label"], "study_specific_basics")
            self.assertIn("failure_unsupported_claim_risk", rows[0])
            self.assertIn("draft_selected_study_hit_present", rows[0])


if __name__ == "__main__":
    unittest.main()
