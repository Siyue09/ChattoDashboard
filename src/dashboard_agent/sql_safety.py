"""Defence in depth for a bounded, read-only SQLite analytics connection.

SQLGlot parsing is a structural gate, not a proof of SQL correctness. SQLite's
authorizer and read-only URI remain active even for a query which passes it.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Iterable

import sqlglot
from sqlglot import exp


class SQLSafetyError(ValueError):
    """The proposed statement violates the read-only query policy."""


class QueryBudgetError(RuntimeError):
    """Execution exceeded the time or virtual-machine instruction budget."""


DEFAULT_TABLES = frozenset({"customers", "products", "orders", "order_items", "refunds", "subscriptions"})
ALLOWED_FUNCTIONS = frozenset({
    "sum", "count", "avg", "min", "max", "round", "coalesce", "nullif",
    "abs", "lower", "upper", "length", "substr", "substring", "trim",
    "ltrim", "rtrim", "date", "datetime", "strftime", "julianday",
    "cast", "try_cast", "if", "iif", "case", "extract", "time_to_str",
    "ts_or_ds_to_date", "ts_or_ds_to_timestamp", "date_add", "date_sub", "time", "concat", "and", "or",
    "row_number", "rank", "dense_rank", "lag", "lead", "first_value", "last_value",
    "like", "glob", "ifnull", "total", "instr", "replace", "printf",
})
FORBIDDEN_NODE_KEYS = frozenset({
    "insert", "update", "delete", "drop", "create", "alter", "merge",
    "command", "pragma", "attach", "detach", "transaction", "commit",
    "rollback", "grant", "revoke", "copy", "into", "lock", "truncate",
    "analyze", "vacuum", "execute", "load_data", "use", "set",
})


def validate_sql(sql: str, allowed_tables: Iterable[str] = DEFAULT_TABLES) -> str:
    """Return normalized SQLite SQL, rejecting non-query and opaque constructs."""
    if not isinstance(sql, str) or not sql.strip():
        raise SQLSafetyError("An empty SQL statement cannot be executed.")
    if len(sql) > 30_000:
        raise SQLSafetyError("SQL exceeds the 30,000-character policy limit.")
    try:
        statements = [node for node in sqlglot.parse(sql, read="sqlite") if node is not None]
    except sqlglot.errors.ParseError as exc:
        raise SQLSafetyError(f"SQL could not be parsed: {exc}") from exc
    if len(statements) != 1:
        raise SQLSafetyError("Exactly one read-only SELECT statement is allowed.")
    tree = statements[0]
    if not isinstance(tree, (exp.Select, exp.Union, exp.Intersect, exp.Except)):
        raise SQLSafetyError("Only SELECT queries, including non-recursive CTEs, are allowed.")
    allowed = {str(table).lower() for table in allowed_tables}
    cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    for node in tree.walk():
        if node.key in FORBIDDEN_NODE_KEYS:
            raise SQLSafetyError(f"Disallowed SQL construct: {node.key}.")
        if isinstance(node, exp.With) and node.args.get("recursive"):
            raise SQLSafetyError("Recursive CTEs are disabled by the query budget policy.")
        if isinstance(node, exp.Table):
            if node.catalog or node.db:
                raise SQLSafetyError("Cross-database and schema-qualified reads are disabled.")
            if not isinstance(node.this, exp.Identifier):
                raise SQLSafetyError("Table-valued functions are disabled.")
            if node.name.lower() not in allowed | cte_names:
                raise SQLSafetyError(f"Table is outside the analytics catalog: {node.name}.")
        if isinstance(node, exp.Func):
            name = node.name if isinstance(node, exp.Anonymous) else node.sql_name()
            if name.lower() not in ALLOWED_FUNCTIONS:
                raise SQLSafetyError(f"Function is outside the analytics allowlist: {name}.")
    return tree.sql(dialect="sqlite").rstrip(";")


def execute_readonly(
    db_path: str | Path,
    sql: str,
    *,
    allowed_tables: Iterable[str] = DEFAULT_TABLES,
    max_rows: int = 500,
    timeout_seconds: float = 2.0,
    max_vm_steps: int = 2_000_000,
) -> dict:
    """Validate and execute once, with an outer result cap and engine-level budget."""
    if not 1 <= max_rows <= 10_000:
        raise ValueError("max_rows must be between 1 and 10,000.")
    if timeout_seconds <= 0 or max_vm_steps < 1_000:
        raise ValueError("Query budgets must be positive (at least 1,000 VM steps).")
    safe_sql = validate_sql(sql, allowed_tables)
    path = Path(db_path).expanduser().resolve(strict=True)
    connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=timeout_seconds)
    connection.execute("PRAGMA query_only=ON")
    connection.execute("PRAGMA trusted_schema=OFF")
    connection.enable_load_extension(False)
    allow = {str(table).lower() for table in allowed_tables}
    budget = {"steps": 0, "exceeded": False}
    deadline = time.monotonic() + timeout_seconds

    def authorize(action: int, arg1: str | None, arg2: str | None, database: str | None, trigger: str | None) -> int:
        if action == sqlite3.SQLITE_SELECT:
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_READ:
            return sqlite3.SQLITE_OK if (arg1 or "").lower() in allow and database in (None, "main") else sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_FUNCTION:
            return sqlite3.SQLITE_OK if (arg2 or "").lower() in ALLOWED_FUNCTIONS else sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_DENY

    def progress() -> int:
        budget["steps"] += 1_000
        budget["exceeded"] = budget["steps"] > max_vm_steps or time.monotonic() > deadline
        return int(budget["exceeded"])

    connection.set_authorizer(authorize)
    connection.set_progress_handler(progress, 1_000)
    try:
        cursor = connection.execute(f"SELECT * FROM ({safe_sql}) AS guarded_result LIMIT {max_rows + 1}")
        columns = [description[0] for description in cursor.description or []]
        rows = [list(row) for row in cursor.fetchall()]
        return {"sql": safe_sql, "columns": columns, "rows": rows[:max_rows], "truncated": len(rows) > max_rows}
    except sqlite3.DatabaseError as exc:
        if budget["exceeded"]:
            raise QueryBudgetError("Query exceeded the execution time or instruction budget.") from exc
        raise
    finally:
        connection.close()
