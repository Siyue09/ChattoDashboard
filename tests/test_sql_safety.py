"""Adversarial checks of the independent SQL policy and execution boundaries."""
import sqlite3
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from dashboard_agent.data import seed_database
from dashboard_agent.sql_safety import QueryBudgetError, SQLSafetyError, execute_readonly, validate_sql


class SQLSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.db = Path(cls.temp.name) / "demo.sqlite3"
        seed_database(cls.db)
        with sqlite3.connect(cls.db) as connection:
            connection.execute("CREATE TABLE private_notes (body TEXT)")
            connection.execute("INSERT INTO private_notes VALUES ('not in the analytics catalog')")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_rejects_mutation_and_exfiltration_matrix(self):
        statements = [
            "DROP TABLE orders", "DELETE FROM orders", "UPDATE orders SET status='completed'",
            "INSERT INTO orders SELECT * FROM orders", "CREATE TABLE stolen AS SELECT * FROM orders",
            "SELECT * FROM orders; DELETE FROM orders", "SELECT 1; SELECT 2",
            "SELECT readfile('/etc/passwd')", "SELECT writefile('/tmp/demo', 'data')",
            "SELECT load_extension('/tmp/arbitrary')", "SELECT * FROM sqlite_master",
            "SELECT * FROM sqlite_schema", "ATTACH DATABASE ':memory:' AS extra",
            "PRAGMA table_info(orders)", "SELECT * FROM pragma_table_info('orders')",
            "SELECT * FROM main.orders", "SELECT * FROM private_notes",
            "WITH x AS (DELETE FROM orders RETURNING *) SELECT * FROM x",
            "WITH RECURSIVE x(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM x) SELECT * FROM x",
            "SELECT * INTO stolen FROM orders", "VACUUM INTO '/tmp/export.sqlite'",
        ]
        with sqlite3.connect(self.db) as connection:
            before = connection.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
        for sql in statements:
            with self.subTest(sql=sql), self.assertRaises(SQLSafetyError):
                execute_readonly(self.db, sql)
        with sqlite3.connect(self.db) as connection:
            self.assertEqual(before, connection.execute("SELECT COUNT(*) FROM orders").fetchone()[0])

    def test_accepts_nonrecursive_cte_and_literal_semicolons(self):
        sql = "WITH counts AS (SELECT COUNT(*) AS n FROM orders) SELECT n, ';DROP TABLE x' AS literal FROM counts"
        result = execute_readonly(self.db, sql)
        self.assertEqual(result["rows"], [[2404, ";DROP TABLE x"]])

    def test_authorizer_blocks_private_table_even_when_parser_is_bypassed(self):
        # The test deliberately removes the first defence to verify the second.
        with patch("dashboard_agent.sql_safety.validate_sql", side_effect=lambda sql, _: sql):
            with self.assertRaises(sqlite3.DatabaseError):
                execute_readonly(self.db, "SELECT body FROM private_notes")

    def test_authorizer_blocks_function_even_when_parser_is_bypassed(self):
        calls = []
        connect = sqlite3.connect

        def connection_with_function(*args, **kwargs):
            connection = connect(*args, **kwargs)
            connection.create_function("unsafe_function", 0, lambda: calls.append("called") or "secret")
            return connection

        with patch("dashboard_agent.sql_safety.sqlite3.connect", side_effect=connection_with_function):
            with patch("dashboard_agent.sql_safety.validate_sql", side_effect=lambda sql, _: sql):
                with self.assertRaises(sqlite3.DatabaseError):
                    execute_readonly(self.db, "SELECT unsafe_function()")
        self.assertEqual(calls, [])

    def test_connection_is_read_only_before_any_query(self):
        # Capture the created connection and verify that SQLite itself refuses a
        # write, even before the executor has installed its authorizer.
        connect = sqlite3.connect
        readonly = []

        def inspect_connection(*args, **kwargs):
            connection = connect(*args, **kwargs)
            try:
                connection.execute("CREATE TABLE should_not_exist (value TEXT)")
            except sqlite3.OperationalError as exc:
                readonly.append("readonly" in str(exc).lower())
            return connection

        with patch("dashboard_agent.sql_safety.sqlite3.connect", side_effect=inspect_connection):
            execute_readonly(self.db, "SELECT COUNT(*) FROM orders")
        self.assertEqual(readonly, [True])

    def test_row_limit_is_applied_outside_user_query(self):
        result = execute_readonly(self.db, "SELECT order_id FROM orders ORDER BY order_id LIMIT 2000", max_rows=7)
        self.assertEqual(result["rows"], [[number] for number in range(1, 8)])
        self.assertTrue(result["truncated"])
        exact = execute_readonly(self.db, "SELECT order_id FROM orders ORDER BY order_id LIMIT 7", max_rows=7)
        self.assertFalse(exact["truncated"])

    def test_query_instruction_budget_stops_cartesian_explosion(self):
        with self.assertRaises(QueryBudgetError):
            execute_readonly(self.db, "SELECT COUNT(*) FROM order_items a CROSS JOIN order_items b CROSS JOIN order_items c", max_vm_steps=1000)
        self.assertEqual(execute_readonly(self.db, "SELECT COUNT(*) FROM orders")["rows"], [[2404]])

    def test_wall_clock_budget_and_invalid_limits(self):
        # A controlled clock avoids a hardware-dependent sleep/timing test.
        with patch("dashboard_agent.sql_safety.time.monotonic", side_effect=[0.0, 3.0]):
            with self.assertRaises(QueryBudgetError):
                execute_readonly(self.db, "SELECT SUM(quantity) FROM order_items", timeout_seconds=2)
        for kwargs in ({"max_rows": 0}, {"max_rows": 10001}, {"timeout_seconds": 0}, {"max_vm_steps": 999}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                execute_readonly(self.db, "SELECT 1", **kwargs)

    def test_empty_and_oversized_sql(self):
        for sql in ("", "  ", "SELECT " + "1," * 16000 + "1"):
            with self.subTest(size=len(sql)), self.assertRaises(SQLSafetyError):
                validate_sql(sql)


if __name__ == "__main__":
    unittest.main()
