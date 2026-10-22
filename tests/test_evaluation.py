import sqlite3
import tempfile
from pathlib import Path
import unittest

from dashboard_agent.data import build_retrieval_examples, seed_database
from dashboard_agent.evaluation import build_evaluation_fixtures, overlap_audit, routing_scores, rows_equal


class EvaluationTests(unittest.TestCase):
    def test_multiset_comparison_preserves_duplicates_nulls_and_numbers(self):
        self.assertTrue(rows_equal([["b", 2.0], ["a", None]], [("a", None), ("b", 2.004)]))
        self.assertFalse(rows_equal([["a", 2], ["a", 2]], [["a", 2], ["b", 2]]))
        self.assertFalse(rows_equal([[None]], [[0]]))
        self.assertFalse(rows_equal([[float("nan")]], [[float("nan")]]))
        self.assertFalse(rows_equal([[1]], [[1.1]], abs_tol=0.01))

    def test_tolerance_matching_does_not_depend_on_greedy_order(self):
        self.assertTrue(rows_equal([[1.05], [1.0]], [[1.0], [1.1]], abs_tol=0.06))

    def test_zero_question_overlap_but_disclosed_semantic_overlap(self):
        audit = overlap_audit(build_evaluation_fixtures(), build_retrieval_examples())
        self.assertEqual(audit["exact_question_overlap_count"], 0)
        self.assertEqual(audit["id_overlap_count"], 0)
        self.assertGreater(len(audit["shared_semantic_families"]), 0)
        duplicated = [{"id": "duplicate", "question": build_retrieval_examples()[0]["question"]}]
        self.assertEqual(overlap_audit(duplicated, build_retrieval_examples())["exact_question_overlap_count"], 1)

    def test_macro_f1_includes_mistaken_error_class(self):
        scores = routing_scores(["ok", "ok", "clarify", "rejected"], ["ok", "error", "clarify", "rejected"])
        self.assertEqual(scores["labels"], ["clarify", "error", "ok", "rejected"])
        self.assertAlmostEqual(scores["value"], (1 + 0 + 2/3 + 1)/4)

    def test_every_gold_query_executes_and_known_net_total_is_independent(self):
        with tempfile.TemporaryDirectory() as temp:
            database = seed_database(Path(temp) / "demo.sqlite3")
            with sqlite3.connect(database) as db:
                for case in build_evaluation_fixtures():
                    if case["expected_status"] == "ok":
                        self.assertTrue(db.execute(case["gold_sql"]).fetchall(), case["id"])
                case = next(c for c in build_evaluation_fixtures() if c["id"] == "net_2024")
                net = db.execute(case["gold_sql"]).fetchone()[0]
                revenue = db.execute("SELECT SUM(i.quantity*i.unit_price*(1-i.discount)) FROM order_items i WHERE i.order_id IN (SELECT order_id FROM orders WHERE status='completed' AND SUBSTR(order_date,1,4)='2024')").fetchone()[0]
                refund = db.execute("SELECT SUM(amount) FROM refunds WHERE order_id IN (SELECT order_id FROM orders WHERE status='completed' AND SUBSTR(order_date,1,4)='2024')").fetchone()[0]
                self.assertAlmostEqual(net, round(revenue-refund, 2), places=2)


if __name__ == "__main__":
    unittest.main()
