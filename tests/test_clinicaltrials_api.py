from __future__ import annotations

import unittest

from informed_consent.clinicaltrials_api import study_matches_query_terms


class ClinicalTrialsApiTests(unittest.TestCase):
    def test_query_term_filter_prefers_matching_studies(self) -> None:
        matching = {
            "protocolSection": {
                "identificationModule": {"briefTitle": "Heart Failure Consent Study"},
                "descriptionModule": {"briefSummary": "Participants with heart failure will be enrolled."},
                "conditionsModule": {"conditions": ["Heart Failure"], "keywords": ["Cardiology"]},
            }
        }
        non_matching = {
            "protocolSection": {
                "identificationModule": {"briefTitle": "Healthy Volunteer PK Study"},
                "descriptionModule": {"briefSummary": "Single ascending dose study in healthy adults."},
                "conditionsModule": {"conditions": ["Healthy"], "keywords": ["Pharmacokinetics"]},
            }
        }

        self.assertTrue(study_matches_query_terms(matching, "heart failure"))
        self.assertFalse(study_matches_query_terms(non_matching, "heart failure"))


if __name__ == "__main__":
    unittest.main()
