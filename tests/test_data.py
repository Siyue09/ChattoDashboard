import sqlite3
import tempfile
from pathlib import Path
import unittest

from dashboard_agent.data import CATALOG, build_retrieval_examples, seed_database


class SyntheticDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = Path(self.temp.name) / "demo.sqlite3"
        seed_database(self.db)
        self.connection = sqlite3.connect(self.db)

    def tearDown(self):
        self.connection.close()
        self.temp.cleanup()

    def test_repeatable_content_and_relationships(self):
        before = list(self.connection.iterdump())
        self.connection.close()
        seed_database(self.db)
        self.connection = sqlite3.connect(self.db)
        self.assertEqual(before, list(self.connection.iterdump()))
        self.assertEqual(self.connection.execute("PRAGMA foreign_key_check").fetchall(), [])
        self.assertEqual(self.connection.execute("SELECT COUNT(*) FROM orders").fetchone()[0], 2404)

    def test_known_refund_fanout_fixture(self):
        # Order 1: 180 + 150 + 40 = 370 revenue; two refunds totaling 25.
        revenue = self.connection.execute("SELECT SUM(quantity*unit_price*(1-discount)) FROM order_items WHERE order_id=1").fetchone()[0]
        refund = self.connection.execute("SELECT SUM(amount) FROM refunds WHERE order_id=1").fetchone()[0]
        naive = self.connection.execute("SELECT SUM(i.quantity*i.unit_price*(1-i.discount))-SUM(r.amount) FROM order_items i JOIN refunds r USING(order_id) WHERE i.order_id=1").fetchone()[0]
        self.assertEqual(revenue, 370)
        self.assertEqual(refund, 25)
        self.assertEqual(revenue-refund, 345)
        self.assertNotEqual(naive, 345)

    def test_discount_status_and_snapshot_boundaries(self):
        rows = self.connection.execute("SELECT status, SUM(i.quantity*i.unit_price*(1-discount)) FROM orders o JOIN order_items i USING(order_id) WHERE o.order_id IN (1,2,3) GROUP BY status").fetchall()
        self.assertEqual(dict(rows)["completed"], 470)
        self.assertEqual(dict(rows)["cancelled"], 10000)
        active = self.connection.execute("SELECT subscription_id FROM subscriptions WHERE subscription_id<=3 AND start_date<='2024-02-29' AND (end_date IS NULL OR end_date>'2024-02-29')").fetchall()
        self.assertEqual(active, [(2,)])

    def test_refuses_to_overwrite_unrelated_database(self):
        path = Path(self.temp.name) / "unrelated.sqlite3"
        with sqlite3.connect(path) as db:
            db.execute("CREATE TABLE important_data (value TEXT)")
        with self.assertRaises(ValueError):
            seed_database(path)

    def test_synthetic_examples_are_labeled_and_catalog_grounded(self):
        examples = build_retrieval_examples()
        self.assertEqual(len(examples), 224)
        self.assertEqual(len({e["template_family"] for e in examples}), 28)
        self.assertEqual(len({e["id"] for e in examples}), len(examples))
        tables = {t["name"] for t in CATALOG["tables"]}
        metrics = {m["name"] for m in CATALOG["metrics"]}
        for example in examples:
            self.assertTrue(example["synthetic"])
            self.assertTrue(set(example["tables"]) <= tables)
            self.assertTrue(set(example["metrics"]) <= metrics)


if __name__ == "__main__":
    unittest.main()
