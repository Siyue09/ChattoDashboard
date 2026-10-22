import tempfile
from pathlib import Path
import unittest

from dashboard_agent.chart_model import recommend_chart, table_features, train_chart_model


class ChartModelTests(unittest.TestCase):
    def test_chart_axes_match_actual_column_types(self):
        self.assertEqual(recommend_chart(["revenue"], [[1000]])["type"], "metric")
        line = recommend_chart(["month", "net_revenue"], [["2025-01", 100], ["2025-02", 200], ["2025-03", 300]])
        self.assertEqual((line["type"], line["x"], line["y"]), ("line", "month", "net_revenue"))
        bar = recommend_chart(["region", "revenue"], [["West", 100], ["East", 200]])
        self.assertEqual((bar["type"], bar["x"], bar["y"]), ("bar", "region", "revenue"))

    def test_empty_result_and_all_null_measure_do_not_invent_numeric_axis(self):
        self.assertEqual(recommend_chart(["revenue"], [])["type"], "table")
        self.assertEqual(recommend_chart(["revenue"], [[None]])["type"], "table")
        _, kinds = table_features(["flag", "year", "value"], [[True, 2025, None]])
        self.assertEqual(kinds["numeric"], [])
        self.assertEqual(kinds["temporal"], ["year"])

    def test_training_is_deterministic_and_discloses_synthetic_overlap(self):
        with tempfile.TemporaryDirectory() as temp:
            first = train_chart_model(Path(temp) / "first.json")
            second = train_chart_model(Path(temp) / "second.json")
        self.assertEqual(first, second)
        self.assertEqual(first["evaluation"]["train_samples"], 480)
        self.assertEqual(first["evaluation"]["test_samples"], 120)
        self.assertGreater(first["evaluation"]["test_feature_patterns_seen_in_train"], 0)
        self.assertIn("synthetic", first["limitations"])


if __name__ == "__main__":
    unittest.main()
