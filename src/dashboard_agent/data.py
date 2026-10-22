"""Deterministic, fictional commerce data and synthetic retrieval assets.

No records, labels, or reported performance originate from IpserLab.
The seed includes deliberate one-to-many joins and multiple refunds per order.
"""

from __future__ import annotations

import json
from pathlib import Path
import random
import sqlite3
from datetime import date, timedelta


ASSET_DIR = Path(__file__).resolve().parent / "assets"


def _columns(*specs):
    return [{"name": n, "type": t, "description": d} for n, t, d in specs]


CATALOG = {
    "schema_version": 1,
    "provenance": "Entirely fictional educational reconstruction; no employer data.",
    "date_coverage": ["2024-01-01", "2025-12-31"],
    "date_coverage_note": "Coverage refers to order dates; subsequent refund events and subscription end dates may extend beyond 2025.",
    "currency": "USD",
    "tables": [
        {"name": "customers", "description": "Fictional customer segments and US sales regions.",
         "primary_key": ["customer_id"], "keywords": ["customer", "segment", "region", "客户", "地区"],
         "columns": _columns(("customer_id", "INTEGER", "Customer identifier"), ("segment", "TEXT", "Consumer, Corporate, or Small Business"), ("region", "TEXT", "East, West, Central, or South"))},
        {"name": "products", "description": "Fictional products and merchandise categories.",
         "primary_key": ["product_id"], "keywords": ["product", "category", "产品", "品类"],
         "columns": _columns(("product_id", "INTEGER", "Product identifier"), ("category", "TEXT", "Technology, Furniture, or Office Supplies"), ("product_name", "TEXT", "Fictional product name"))},
        {"name": "orders", "description": "Orders at one row per order, including incomplete orders.",
         "primary_key": ["order_id"], "keywords": ["order", "date", "month", "year", "订单", "月份"],
         "columns": _columns(("order_id", "INTEGER", "Order identifier"), ("customer_id", "INTEGER", "Customer foreign key"), ("order_date", "TEXT", "ISO date, 2024-2025"), ("status", "TEXT", "completed, cancelled, or pending"))},
        {"name": "order_items", "description": "One or more discounted product lines per order; not order-grain.",
         "primary_key": ["order_id", "product_id"], "keywords": ["revenue", "sales", "price", "discount", "收入", "销售额"],
         "columns": _columns(("order_id", "INTEGER", "Order foreign key"), ("product_id", "INTEGER", "Product foreign key"), ("quantity", "INTEGER", "Positive units"), ("unit_price", "REAL", "USD before discount"), ("discount", "REAL", "Fraction in [0, 1]"))},
        {"name": "refunds", "description": "Refund events: an order may have several. Aggregate by order before joining lines.",
         "primary_key": ["refund_id"], "keywords": ["refund", "net", "退款", "净收入"],
         "columns": _columns(("refund_id", "INTEGER", "Refund identifier"), ("order_id", "INTEGER", "Order foreign key"), ("amount", "REAL", "Positive USD refund amount"), ("refund_date", "TEXT", "ISO event date; net revenue uses original order date"))},
        {"name": "subscriptions", "description": "Recurring plans; MRR is an as-of snapshot, never summed across snapshot months.",
         "primary_key": ["subscription_id"], "keywords": ["mrr", "subscription", "recurring", "订阅"],
         "columns": _columns(("subscription_id", "INTEGER", "Subscription identifier"), ("customer_id", "INTEGER", "Customer foreign key"), ("start_date", "TEXT", "Inclusive ISO start date"), ("end_date", "TEXT", "Exclusive ISO end date; NULL means no known end"), ("mrr", "REAL", "Monthly USD charge"))},
    ],
    "joins": [
        {"left_table": "orders", "left_column": "customer_id", "right_table": "customers", "right_column": "customer_id", "cardinality": "many-to-one"},
        {"left_table": "order_items", "left_column": "order_id", "right_table": "orders", "right_column": "order_id", "cardinality": "many-to-one"},
        {"left_table": "order_items", "left_column": "product_id", "right_table": "products", "right_column": "product_id", "cardinality": "many-to-one"},
        {"left_table": "refunds", "left_column": "order_id", "right_table": "orders", "right_column": "order_id", "cardinality": "many-to-one"},
        {"left_table": "subscriptions", "left_column": "customer_id", "right_table": "customers", "right_column": "customer_id", "cardinality": "many-to-one"},
    ],
    "metrics": [
        {"name": "revenue", "description": "Completed-order revenue after line discounts, before refunds.", "formula": "SUM(quantity * unit_price * (1 - discount)) WHERE orders.status = 'completed'", "required_tables": ["orders", "order_items"], "keywords": ["revenue", "sales", "销售额", "收入"]},
        {"name": "gross_sales", "description": "Completed-order sales before discounts and refunds.", "formula": "SUM(quantity * unit_price) WHERE orders.status = 'completed'", "required_tables": ["orders", "order_items"], "keywords": ["gross sales", "undiscounted", "原价销售额"]},
        {"name": "net_revenue", "description": "Discounted completed-order revenue minus refunds, attributed to the original order date. Aggregate lines and refunds separately to order grain. Product/category allocation of refunds is undefined.", "formula": "SUM(order_revenue - COALESCE(order_refunds, 0))", "required_tables": ["orders", "order_items", "refunds"], "keywords": ["net revenue", "after refunds", "净收入"]},
        {"name": "aov", "description": "Discounted revenue / distinct completed orders; not mean line value.", "formula": "SUM(quantity * unit_price * (1-discount)) / COUNT(DISTINCT orders.order_id)", "required_tables": ["orders", "order_items"], "keywords": ["aov", "average order value", "客单价"]},
        {"name": "orders", "description": "Number of distinct completed orders.", "formula": "COUNT(DISTINCT order_id) WHERE status = 'completed'", "required_tables": ["orders"], "keywords": ["orders", "order count", "订单数"]},
        {"name": "active_customers", "description": "Distinct customers with at least one completed order in the requested order-date window.", "formula": "COUNT(DISTINCT customer_id) WHERE status = 'completed'", "required_tables": ["orders"], "keywords": ["active customers", "unique buyers", "活跃客户"]},
        {"name": "refund_rate", "description": "Fraction of completed orders with one or more refund events; denominator is distinct completed orders. Fraction, not percent, in query results.", "formula": "COUNT(DISTINCT completed orders with refund) * 1.0 / COUNT(DISTINCT completed orders)", "required_tables": ["orders", "refunds"], "keywords": ["refund rate", "refunded orders", "退款率"]},
        {"name": "mrr", "description": "Month-end recurring revenue snapshot: start_date <= month end AND (end_date IS NULL OR end_date > month end). End dates are exclusive.", "formula": "SUM(mrr) for active subscriptions on snapshot date", "required_tables": ["subscriptions"], "keywords": ["mrr", "monthly recurring revenue", "订阅收入"]},
    ],
    "ambiguities": ["Profit requires cost data that this schema lacks.", "Revenue without a time range defaults to all available dates, disclosed in the answer.", "Net revenue by product/category requires an explicit refund allocation policy.", "Retention and churn require an agreed cohort and denominator.", "Synthetic retrieval examples are templates, not human-labeled observations."],
}


def seed_database(path: str | Path, seed: int = 42) -> str:
    """Build a repeatable SQLite fixture. Refuse to overwrite a non-demo database."""
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        with sqlite3.connect(destination) as existing:
            tables = {r[0] for r in existing.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if tables and "demo_metadata" not in tables:
                raise ValueError("Refusing to overwrite a database without demo_metadata")
    rng = random.Random(seed)
    with sqlite3.connect(destination) as conn:
        conn.execute("PRAGMA foreign_keys = OFF")
        for name in ["refunds", "order_items", "subscriptions", "orders", "products", "customers", "demo_metadata"]:
            conn.execute(f"DROP TABLE IF EXISTS {name}")
        conn.executescript("""
            CREATE TABLE demo_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, segment TEXT NOT NULL, region TEXT NOT NULL);
            CREATE TABLE products (product_id INTEGER PRIMARY KEY, category TEXT NOT NULL, product_name TEXT NOT NULL);
            CREATE TABLE orders (order_id INTEGER PRIMARY KEY, customer_id INTEGER NOT NULL REFERENCES customers(customer_id), order_date TEXT NOT NULL, status TEXT NOT NULL);
            CREATE TABLE order_items (order_id INTEGER NOT NULL REFERENCES orders(order_id), product_id INTEGER NOT NULL REFERENCES products(product_id), quantity INTEGER NOT NULL CHECK(quantity>0), unit_price REAL NOT NULL CHECK(unit_price>=0), discount REAL NOT NULL CHECK(discount BETWEEN 0 AND 1), PRIMARY KEY(order_id,product_id));
            CREATE TABLE refunds (refund_id INTEGER PRIMARY KEY, order_id INTEGER NOT NULL REFERENCES orders(order_id), amount REAL NOT NULL CHECK(amount>0), refund_date TEXT NOT NULL);
            CREATE TABLE subscriptions (subscription_id INTEGER PRIMARY KEY, customer_id INTEGER NOT NULL REFERENCES customers(customer_id), start_date TEXT NOT NULL, end_date TEXT, mrr REAL NOT NULL CHECK(mrr>=0));
            CREATE INDEX idx_orders_date ON orders(order_date);
            CREATE INDEX idx_refunds_order ON refunds(order_id);
            CREATE INDEX idx_orders_customer ON orders(customer_id);
        """)
        conn.executemany("INSERT INTO demo_metadata VALUES (?, ?)", [("provenance", "fictional educational data"), ("seed", str(seed)), ("version", "1")])
        segments, regions = ["Consumer", "Corporate", "Small Business"], ["East", "West", "Central", "South"]
        conn.executemany("INSERT INTO customers VALUES (?,?,?)", [(i, segments[(i - 1) % 3], regions[((i - 1) // 3) % 4]) for i in range(1, 401)])
        categories = ["Technology", "Furniture", "Office Supplies"]
        products = [(i, categories[(i - 1) // 8], f"{categories[(i-1)//8]} Item {((i-1)%8)+1}") for i in range(1, 25)]
        conn.executemany("INSERT INTO products VALUES (?,?,?)", products)
        base_prices = {i: ([80, 140, 8][(i - 1) // 8] + (i % 8) * [45, 36, 7][(i - 1) // 8]) for i in range(1, 25)}
        orders = [(1, 1, "2024-01-01", "completed"), (2, 2, "2024-01-01", "completed"), (3, 3, "2024-01-02", "cancelled"), (4, 4, "2025-12-31", "completed")]
        items = [(1, 1, 2, 100.0, 0.1), (1, 9, 1, 200.0, 0.25), (1, 17, 4, 10.0, 0.0), (2, 2, 1, 100.0, 0.0), (3, 3, 10, 1000.0, 0.0), (4, 24, 2, 50.0, 0.2)]
        refunds = [(1, 1, 10.0, "2024-01-10"), (2, 1, 15.0, "2024-02-01"), (3, 2, 100.0, "2024-02-01")]
        start = date(2024, 1, 1)
        for order_id in range(5, 2405):
            day = start + timedelta(days=rng.randrange(731))
            status = rng.choices(["completed", "cancelled", "pending"], weights=[82, 9, 9])[0]
            customer = rng.randrange(1, 401)
            orders.append((order_id, customer, day.isoformat(), status))
            line_count = rng.choices([1, 2, 3, 4, 5], weights=[25, 35, 25, 12, 3])[0]
            total = 0.0
            for product in rng.sample(range(1, 25), line_count):
                quantity = rng.choices([1, 2, 3, 4, 5], weights=[40, 28, 18, 9, 5])[0]
                price = round(base_prices[product] * rng.choice([0.95, 1.0, 1.05]), 2)
                discount = rng.choices([0.0, 0.05, 0.1, 0.2, 0.25], weights=[55, 15, 15, 10, 5])[0]
                items.append((order_id, product, quantity, price, discount))
                total += quantity * price * (1 - discount)
            if status == "completed" and rng.random() < 0.16:
                amount = round(total * rng.choice([0.1, 0.25, 0.5, 1.0]), 2)
                count = 2 if rng.random() < 0.3 else 1
                first = round(amount * 0.4, 2) if count == 2 else amount
                for offset, value in enumerate([first, round(amount - first, 2)][:count]):
                    refunds.append((len(refunds) + 1, order_id, value, (day + timedelta(days=rng.randint(1, 45) + offset)).isoformat()))
        conn.executemany("INSERT INTO orders VALUES (?,?,?,?)", orders)
        conn.executemany("INSERT INTO order_items VALUES (?,?,?,?,?)", items)
        conn.executemany("INSERT INTO refunds VALUES (?,?,?,?)", refunds)
        subscriptions = [(1, 1, "2024-01-01", "2024-02-29", 100.0), (2, 2, "2024-02-29", None, 50.0), (3, 3, "2025-12-31", None, 200.0)]
        for sub_id in range(4, 364):
            day = start + timedelta(days=rng.randrange(700))
            end = (day + timedelta(days=rng.randrange(30, 450))).isoformat() if rng.random() < 0.4 else None
            subscriptions.append((sub_id, rng.randrange(1, 401), day.isoformat(), end, rng.choice([19.0, 49.0, 99.0, 199.0, 499.0])))
        conn.executemany("INSERT INTO subscriptions VALUES (?,?,?,?,?)", subscriptions)
        issues = conn.execute("PRAGMA foreign_key_check").fetchall()
        if issues:
            raise AssertionError(f"Broken foreign keys: {issues}")
    return str(destination)


def build_retrieval_examples() -> list[dict]:
    """224 synthetic examples in 28 template families; not 224 independent labels."""
    examples = []
    wording = ["Show {metric} {scope}.", "Give me {metric} {scope}.", "Calculate {metric} {scope}.", "I need {metric} {scope}.", "What is {metric} {scope}?", "Report {metric} {scope}.", "Summarize {metric} {scope}.", "Display {metric} {scope}."]
    metric_labels = {"revenue": "revenue", "gross_sales": "gross sales", "net_revenue": "net revenue", "aov": "average order value", "orders": "completed order count", "active_customers": "active customers", "refund_rate": "refund rate", "mrr": "MRR"}
    base_tables = {m["name"]: m["required_tables"] for m in CATALOG["metrics"]}
    families = [(metric, dimension) for metric in ["revenue", "net_revenue", "aov", "orders", "active_customers", "refund_rate"] for dimension in ["total", "month", "region", "segment"]]
    families += [("gross_sales", "total"), ("gross_sales", "category"), ("revenue", "category"), ("mrr", "month")]
    for metric, dimension in families:
        family = f"{metric}__{dimension}"
        tables = list(base_tables[metric])
        if dimension in ["region", "segment"]:
            tables.append("customers")
        if dimension == "category":
            tables.append("products")
        for variant, template in enumerate(wording):
            period = ["in 2024", "in 2025", "across all available dates", "for January 2024", "for June 2025", "in 2024", "in 2025", "across all available dates"][variant]
            scope = period if dimension == "total" else f"by {dimension} {period}"
            examples.append({"id": f"synthetic-{family}-{variant+1:02d}", "question": template.format(metric=metric_labels[metric], scope=scope), "intent": "query", "tables": sorted(set(tables)), "metrics": [metric], "synthetic": True, "template_family": family, "provenance": "Programmatically authored template; no human label or production origin."})
    return examples


def write_assets(directory: str | Path | None = None) -> dict:
    target = Path(directory) if directory else ASSET_DIR
    target.mkdir(parents=True, exist_ok=True)
    catalog_path, examples_path = target / "catalog.json", target / "retrieval_examples.json"
    catalog_path.write_text(json.dumps(CATALOG, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    examples = build_retrieval_examples()
    examples_path.write_text(json.dumps(examples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"catalog": str(catalog_path), "examples": str(examples_path), "example_count": len(examples), "template_families": len({x["template_family"] for x in examples})}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/demo.sqlite3")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps({"database": seed_database(args.db, args.seed), **write_assets()}, indent=2))
