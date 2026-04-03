from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from informed_consent.config import AppConfig, ModelConfig, PathConfig, RetrievalConfig
from informed_consent.pipeline import ConsentPipeline


class EvaluationReferencePackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)
        self.repo_root = Path(__file__).resolve().parents[1]

        source_dir = self.temp_path / "public"
        (source_dir / "regulatory_guidance").mkdir(parents=True, exist_ok=True)
        (source_dir / "posted_consent_forms").mkdir(parents=True, exist_ok=True)
        (source_dir / "trial_materials" / "clinicaltrials_gov_api").mkdir(parents=True, exist_ok=True)
        (source_dir / "manifests").mkdir(parents=True, exist_ok=True)

        regulatory_path = source_dir / "regulatory_guidance" / "regulatory_summary.txt"
        regulatory_path.write_text(
            (
                "Participation is voluntary. You may withdraw without penalty. "
                "Ask questions at any time. Alternatives should be discussed. "
                "Possible benefits and risks should be explained."
            ),
            encoding="utf-8",
        )

        consent_form_path = source_dir / "posted_consent_forms" / "example_icf.txt"
        consent_form_path.write_text(
            (
                "This consent form explains study procedures, possible risks, possible benefits, "
                "other options, and who to contact with questions."
            ),
            encoding="utf-8",
        )

        study_payload = {
            "protocolSection": {
                "identificationModule": {
                    "nctId": "NCT01234567",
                    "briefTitle": "Example Trial of Consent Support",
                    "officialTitle": "Example Trial of Consent Support in Adults",
                },
                "statusModule": {"overallStatus": "RECRUITING"},
                "descriptionModule": {"briefSummary": "This is a study about consent support."},
                "conditionsModule": {"conditions": ["Heart Failure"], "keywords": ["Consent"]},
                "designModule": {
                    "studyType": "INTERVENTIONAL",
                    "phases": ["PHASE2"],
                    "designInfo": {
                        "allocation": "RANDOMIZED",
                        "interventionModel": "PARALLEL",
                        "primaryPurpose": "SUPPORTIVE_CARE",
                    },
                    "enrollmentInfo": {"count": 120, "type": "ESTIMATED"},
                },
                "armsInterventionsModule": {
                    "armGroups": [{"label": "Digital Consent", "description": "Interactive consent workflow."}],
                    "interventions": [{"type": "OTHER", "name": "Digital consent", "description": "Participant-facing eConsent support."}],
                },
                "eligibilityModule": {"eligibilityCriteria": "Adults who can provide consent."},
                "outcomesModule": {
                    "primaryOutcomes": [{"measure": "Participant understanding", "timeFrame": "2 weeks"}]
                },
            },
            "documentSection": {
                "largeDocumentModule": {
                    "largeDocs": [{"label": "Informed Consent Form", "hasIcf": True, "hasProtocol": False, "hasSap": False}]
                }
            },
            "hasResults": False,
        }
        study_path = source_dir / "trial_materials" / "clinicaltrials_gov_api" / "NCT01234567.json"
        study_path.write_text(json.dumps(study_payload), encoding="utf-8")

        manifest = {
            "items": [
                {
                    "source_id": "reg_summary",
                    "source_type": "guidance",
                    "authority": "FDA",
                    "url": "https://example.test/regulatory",
                    "saved_path": str(regulatory_path.resolve()),
                    "group_id": "regulatory_guidance",
                    "download_status": "downloaded",
                },
                {
                    "source_id": "posted_icf_example",
                    "source_type": "consent_form",
                    "authority": "ClinicalTrials.gov",
                    "url": "https://example.test/icf",
                    "saved_path": str(consent_form_path.resolve()),
                    "group_id": "posted_consent_forms",
                    "download_status": "downloaded",
                },
                {
                    "source_id": "nct01234567",
                    "source_type": "study_record",
                    "authority": "ClinicalTrials.gov",
                    "url": "https://clinicaltrials.gov/study/NCT01234567",
                    "saved_path": str(study_path.resolve()),
                    "group_id": "trial_materials",
                    "download_status": "downloaded",
                    "nct_id": "NCT01234567",
                },
            ]
        }
        (source_dir / "manifests" / "download_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

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
        manifest = self.pipeline.prepare_corpus(purpose="reference_pack_test", source_dir=source_dir)
        self.run_id = manifest.run_id

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_export_evaluation_reference_pack_includes_study_checklist_and_posted_forms(self) -> None:
        payload = self.pipeline.export_evaluation_reference_pack(self.run_id, source_id="NCT01234567")

        self.assertTrue(payload["study_reference_present"])
        self.assertEqual(payload["posted_consent_form_count"], 1)
        self.assertEqual(payload["regulatory_checklist_count"], 7)

        pack_path = Path(payload["reference_pack_path"])
        self.assertTrue(pack_path.exists())
        pack = json.loads(pack_path.read_text(encoding="utf-8"))
        self.assertEqual(pack["study_reference"]["nct_id"], "NCT01234567")
        self.assertEqual(len(pack["posted_consent_form_references"]), 1)
        self.assertEqual(len(pack["regulatory_reference"]["checklist"]), 7)


if __name__ == "__main__":
    unittest.main()
