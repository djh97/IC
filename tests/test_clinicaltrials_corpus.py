from __future__ import annotations

from pathlib import Path
import json
import tempfile
import unittest

from informed_consent.corpus import load_source_text_units
from informed_consent.types import ConsentSourceDocument


class ClinicalTrialsCorpusTests(unittest.TestCase):
    def test_clinicaltrials_json_is_normalized_into_study_sections(self) -> None:
        payload = {
            "protocolSection": {
                "identificationModule": {
                    "nctId": "NCT01234567",
                    "briefTitle": "Example Trial of Consent Support",
                    "officialTitle": "Example Trial of Consent Support in Adults",
                },
                "statusModule": {
                    "overallStatus": "RECRUITING",
                    "startDateStruct": {"date": "2025-01-01"},
                    "primaryCompletionDateStruct": {"date": "2026-06-01"},
                },
                "sponsorCollaboratorsModule": {
                    "leadSponsor": {"name": "Example University"},
                },
                "descriptionModule": {
                    "briefSummary": "This is a study about consent support.",
                    "detailedDescription": "Participants will review consent materials and ask questions.",
                },
                "conditionsModule": {
                    "conditions": ["Heart Failure"],
                    "keywords": ["Consent", "eConsent"],
                },
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
                    "armGroups": [
                        {"label": "Digital Consent", "description": "Interactive consent workflow."}
                    ],
                    "interventions": [
                        {"type": "OTHER", "name": "Digital consent", "description": "Participant-facing eConsent support."}
                    ],
                },
                "outcomesModule": {
                    "primaryOutcomes": [
                        {
                            "measure": "Participant understanding",
                            "description": "Understanding score after consent.",
                            "timeFrame": "2 weeks",
                        }
                    ]
                },
                "eligibilityModule": {
                    "eligibilityCriteria": "Inclusion Criteria: adults. Exclusion Criteria: none.",
                    "sex": "ALL",
                    "minimumAge": "18 Years",
                    "maximumAge": "80 Years",
                    "healthyVolunteers": False,
                },
                "contactsLocationsModule": {
                    "locations": [
                        {"city": "Boston", "state": "Massachusetts", "country": "United States"}
                    ]
                },
            },
            "documentSection": {
                "largeDocumentModule": {
                    "largeDocs": [
                        {"label": "Informed Consent Form", "hasIcf": True, "hasProtocol": False, "hasSap": False}
                    ]
                }
            },
            "hasResults": False,
            "derivedSection": {"miscInfoModule": {"versionHolder": "2026-04-03"}},
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "NCT01234567.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            source_doc = ConsentSourceDocument(
                source_id="nct01234567",
                title="NCT01234567.json",
                source_type="study_record",
                path=str(path),
                sha256="dummy",
                byte_size=path.stat().st_size,
                metadata={"authority": "ClinicalTrials.gov", "source_group": "trial_materials"},
            )

            units = load_source_text_units(source_doc)

        self.assertGreaterEqual(len(units), 4)
        self.assertTrue(any("NCT01234567" in unit.text for unit in units))
        self.assertTrue(any("Participant understanding" in unit.text for unit in units))
        self.assertTrue(any("Informed Consent Form" in unit.text for unit in units))
        self.assertTrue(all(unit.metadata.get("nct_id") == "NCT01234567" for unit in units))


if __name__ == "__main__":
    unittest.main()
