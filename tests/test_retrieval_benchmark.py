from __future__ import annotations

import unittest

from informed_consent.retrieval_benchmark import aggregate_retrieval_results, score_retrieval_case


class RetrievalBenchmarkTests(unittest.TestCase):
    def test_scores_expected_source_and_group_hits(self) -> None:
        hits = [
            {
                "source_id": "fda_informed_consent_guidance_2023",
                "citation_label": "Guide to Informed Consent",
                "score": 0.9,
                "metadata": {"source_group": "regulatory_guidance"},
            },
            {
                "source_id": "ich_e6_r3_2025",
                "citation_label": "ICH E6(R3)",
                "score": 0.8,
                "metadata": {"source_group": "regulatory_guidance"},
            },
        ]
        row = score_retrieval_case(
            benchmark_id="bench",
            query_id="withdrawal",
            query="Can I withdraw?",
            retrieval_mode="hybrid",
            hits=hits,
            expected_source_ids=["ich_e6_r3_2025"],
            expected_source_groups=["regulatory_guidance"],
            top_k=5,
        )
        self.assertEqual(row["source_id_first_relevant_rank"], 2)
        self.assertEqual(row["source_group_first_relevant_rank"], 1)
        self.assertEqual(row["source_id_hit_at_1"], False)
        self.assertEqual(row["source_id_hit_at_3"], True)
        self.assertEqual(row["source_group_hit_at_1"], True)
        self.assertEqual(row["source_id_mrr"], 0.5)
        self.assertEqual(row["source_group_mrr"], 1.0)

    def test_aggregate_groups_by_mode(self) -> None:
        rows = [
            {
                "retrieval_mode": "lexical",
                "returned_hit_count": 5,
                "source_id_mrr": 1.0,
                "source_group_mrr": 1.0,
                "source_id_hit_at_1": True,
                "source_id_hit_at_3": True,
                "source_id_hit_at_k": True,
                "source_group_hit_at_1": True,
                "source_group_hit_at_3": True,
                "source_group_hit_at_k": True,
            },
            {
                "retrieval_mode": "hybrid",
                "returned_hit_count": 5,
                "source_id_mrr": 0.5,
                "source_group_mrr": 1.0,
                "source_id_hit_at_1": False,
                "source_id_hit_at_3": True,
                "source_id_hit_at_k": True,
                "source_group_hit_at_1": True,
                "source_group_hit_at_3": True,
                "source_group_hit_at_k": True,
            },
        ]
        aggregate = aggregate_retrieval_results(rows)
        self.assertEqual(aggregate["query_count"], 2)
        self.assertEqual(aggregate["modes"]["lexical"]["source_id_hit_rate_at_1"], 1.0)
        self.assertEqual(aggregate["modes"]["hybrid"]["average_source_id_mrr"], 0.5)


if __name__ == "__main__":
    unittest.main()
