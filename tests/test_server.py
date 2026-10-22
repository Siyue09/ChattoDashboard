"""Drilldown semantics and HTTP trust boundaries, without opening a network port."""

from email.message import Message
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from dashboard_agent.data import seed_database
from dashboard_agent.server import drilldown, make_handler


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.db = seed_database(Path(cls.temp.name) / "demo.sqlite3")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def request(self, path, body=None, *, method="POST", headers=None, raw_body=None):
        """Invoke the real handler with in-memory request/response streams."""
        Handler = make_handler(self.db, mode="offline", port=8765)
        handler = Handler.__new__(Handler)
        handler.path, handler.command = path, method
        handler.request_version = "HTTP/1.1"
        handler.requestline = f"{method} {path} HTTP/1.1"
        handler.wfile = io.BytesIO()
        payload = raw_body if raw_body is not None else json.dumps(body or {}).encode()
        handler.rfile = io.BytesIO(payload)
        handler.headers = Message()
        request_headers = {"Host": "127.0.0.1:8765", "Content-Type": "application/json", "Content-Length": str(len(payload))}
        request_headers.update(headers or {})
        for key, value in request_headers.items():
            handler.headers[key] = value
        getattr(handler, f"do_{method}")()
        raw_headers, response = handler.wfile.getvalue().split(b"\r\n\r\n", 1)
        status = int(raw_headers.split(b" ")[1])
        return status, json.loads(response), raw_headers.decode()

    def test_drilldown_year_and_dimension_keep_only_matching_completed_orders(self):
        result = drilldown(self.db, "region", "West", year="2025")
        self.assertTrue(result["rows"])
        positions = {name: i for i, name in enumerate(result["columns"])}
        with sqlite3.connect(self.db) as db:
            for row in result["rows"]:
                self.assertEqual(row[positions["region"]], "West")
                self.assertTrue(row[positions["order_date"]].startswith("2025-"))
                self.assertEqual(db.execute("SELECT status FROM orders WHERE order_id=?", (row[positions["order_id"]],)).fetchone()[0], "completed")
                expected = round(row[positions["quantity"]] * row[positions["unit_price"]] * (1-row[positions["discount"]]), 2)
                self.assertAlmostEqual(row[positions["revenue"]], expected)
        self.assertLessEqual(len(result["rows"]), 100)

    def test_drilldown_matches_independent_full_group_before_row_cap(self):
        result = drilldown(self.db, "month", "2024-01", year="2024")
        with sqlite3.connect(self.db) as db:
            expected = db.execute("""SELECT o.order_id,o.order_date,p.product_name,p.category,c.region,c.segment,i.quantity,i.unit_price,i.discount,ROUND(i.quantity*i.unit_price*(1-i.discount),2)
            FROM order_items i JOIN orders o ON i.order_id=o.order_id JOIN customers c ON c.customer_id=o.customer_id JOIN products p ON p.product_id=i.product_id
            WHERE o.status='completed' AND o.order_date>='2024-01-01' AND o.order_date<'2024-02-01'
            ORDER BY o.order_date,o.order_id,p.product_name LIMIT 101""").fetchall()
        self.assertEqual(result["rows"], [list(row) for row in expected[:100]])
        self.assertEqual(result["truncated"], len(expected) > 100)
        self.assertEqual(drilldown(self.db, "month", "2024-01", year="2025")["rows"], [])

    def test_drilldown_literal_injection_cannot_broaden_or_mutate_query(self):
        with sqlite3.connect(self.db) as db:
            before = list(db.iterdump())
        for value in ["West' OR 1=1 --", "West'; DROP TABLE orders; --", "O'Reilly"]:
            with self.subTest(value=value):
                self.assertEqual(drilldown(self.db, "region", value, year="2025")["rows"], [])
        with sqlite3.connect(self.db) as db:
            self.assertEqual(before, list(db.iterdump()))

    def test_drilldown_rejects_invalid_dimensions_years_and_values(self):
        for dimension in ["revenue", "region; DROP TABLE orders", "", None]:
            with self.subTest(dimension=dimension), self.assertRaises(ValueError):
                drilldown(self.db, dimension, "West")
        for year in [2025, "2025 OR 1=1", "25", "2025-01", "1999"]:
            with self.subTest(year=year), self.assertRaises(ValueError):
                drilldown(self.db, "region", "West", year=year)
        for value in [None, 123, "x"*101]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                drilldown(self.db, "region", value)

    def test_http_rejects_host_origin_content_type_and_oversized_payload(self):
        for header, value, expected in [("Host", "example.com:8765", 403), ("Origin", "https://evil.example", 403), ("Content-Type", "text/plain", 415), ("Content-Length", "8193", 413)]:
            with self.subTest(header=header):
                status, body, _ = self.request("/api/ask", {"question": "revenue"}, headers={header: value})
                self.assertEqual(status, expected)
                self.assertIn("error", body)

    def test_http_rejects_invalid_json_inputs_and_unknown_route(self):
        self.assertEqual(self.request("/api/ask", raw_body=b"{")[0], 400)
        self.assertEqual(self.request("/api/ask", raw_body=b"[]")[0], 400)
        self.assertEqual(self.request("/api/ask", {"question": ""})[0], 400)
        self.assertEqual(self.request("/api/ask", {"question": "revenue", "retrieval": "unknown"})[0], 400)
        self.assertEqual(self.request("/api/drilldown", {"dimension": "unknown", "value": "West"})[0], 400)
        self.assertEqual(self.request("/api/unknown")[0], 404)

    def test_http_returns_actual_query_and_drilldown_data_with_security_headers(self):
        status, result, headers = self.request("/api/drilldown", {"dimension": "category", "value": "Technology", "year": "2024"})
        self.assertEqual(status, 200)
        self.assertTrue(result["rows"])
        self.assertIn("Cache-Control: no-store", headers)
        self.assertIn("X-Content-Type-Options: nosniff", headers)
        self.assertIn("Referrer-Policy: no-referrer", headers)
        status, answer, _ = self.request("/api/ask", {"question": "Revenue by region in 2024"})
        self.assertEqual(status, 200)
        self.assertEqual(answer["status"], "ok")
        self.assertEqual(len(answer["rows"]), 4)


if __name__ == "__main__":
    unittest.main()
