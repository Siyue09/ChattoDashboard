"""Loopback-only teaching UI; not a production or multi-tenant web service."""
import json
import os
from threading import Lock
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re


def drilldown(db_path, dimension, value, year=None):
    """Inspect actual order items behind a revenue group; no model involved."""
    expressions = {"month": "strftime('%Y-%m', o.order_date)", "category": "p.category",
                   "region": "c.region", "segment": "c.segment", "product": "p.product_name"}
    if dimension not in expressions or not isinstance(value, str) or len(value) > 100:
        raise ValueError("Invalid drilldown group")
    if year is not None and (not isinstance(year, str) or not re.fullmatch(r"20\d{2}", year)):
        raise ValueError("Invalid year")
    # Escape the literal only after validating the fixed dimension; the same AST
    # and SQLite authorizer gates as generated SQL apply below.
    literal = "'" + value.replace("'", "''") + "'"
    condition = expressions[dimension] + " = " + literal
    if year:
        condition += f" AND o.order_date >= '{year}-01-01' AND o.order_date < '{int(year)+1}-01-01'"
    sql = f"""SELECT o.order_id, o.order_date, p.product_name, p.category,
    c.region, c.segment, i.quantity, i.unit_price, i.discount,
    ROUND(i.quantity*i.unit_price*(1-i.discount),2) AS revenue
    FROM orders o JOIN order_items i ON i.order_id=o.order_id
    JOIN products p ON p.product_id=i.product_id
    JOIN customers c ON c.customer_id=o.customer_id
    WHERE o.status='completed' AND {condition}
    ORDER BY o.order_date, o.order_id, p.product_name"""
    from .sql_safety import execute_readonly
    return execute_readonly(db_path, sql, max_rows=100)


def make_handler(db_path, mode="offline", port=8765):
    # Keep corpus embeddings in memory, with predictable cache initialization.
    agents = {}
    agent_lock = Lock()
    class Handler(BaseHTTPRequestHandler):
        def respond(self, status, data, content_type="application/json; charset=utf-8"):
            body = data if isinstance(data, bytes) else json.dumps(data, ensure_ascii=False, default=str).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            self.wfile.write(body)

        def valid_host(self):
            return self.headers.get("Host", "") in {f"127.0.0.1:{port}", f"localhost:{port}"}

        def do_GET(self):
            if not self.valid_host():
                return self.respond(403, {"error": "Loopback host required"})
            if self.path == "/":
                page = Path(__file__).with_name("web").joinpath("index.html").read_bytes()
                return self.respond(200, page, "text/html; charset=utf-8")
            if self.path == "/api/info":
                from .data import ASSET_DIR
                catalog = json.loads((ASSET_DIR / "catalog.json").read_text())
                return self.respond(200, {"mode": mode, "embeddings": os.getenv("RETRIEVAL_EMBEDDINGS", "local"), "dataset": "Synthetic commerce / seed 42", "catalog": catalog})
            return self.respond(404, {"error": "Not found"})

        def do_POST(self):
            if not self.valid_host():
                return self.respond(403, {"error": "Loopback host required"})
            origin = self.headers.get("Origin")
            if origin and origin not in {f"http://127.0.0.1:{port}", f"http://localhost:{port}"}:
                return self.respond(403, {"error": "Cross-origin request refused"})
            if self.path not in {"/api/ask", "/api/drilldown"}:
                return self.respond(404, {"error": "Not found"})
            if self.headers.get_content_type() != "application/json":
                return self.respond(415, {"error": "Use application/json"})
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if size < 1 or size > 8192:
                    return self.respond(413, {"error": "Question payload must be 1–8192 bytes"})
                body = json.loads(self.rfile.read(size))
                if not isinstance(body, dict):
                    raise ValueError("Expected an object")
                if self.path == "/api/drilldown":
                    try:
                        result = drilldown(db_path, body.get("dimension"), body.get("value"), body.get("year"))
                    except (ValueError, TypeError):
                        raise
                    except Exception:
                        return self.respond(500, {"error": "Drilldown query could not be completed."})
                    return self.respond(200, result)
                question = body.get("question", "")
                retrieval = body.get("retrieval", "graph")
                if not isinstance(question, str) or not 1 <= len(question.strip()) <= 1200:
                    raise ValueError("Enter a question up to 1200 characters")
                if retrieval not in {"graph", "vector"}:
                    raise ValueError("Unknown retrieval mode")
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                return self.respond(400, {"error": str(exc)})
            from .engine import DashboardAgent
            try:
                with agent_lock:
                    if retrieval not in agents:
                        agents[retrieval] = DashboardAgent(db_path, mode=mode, retrieval_mode=retrieval)
                    result = agents[retrieval].run(question)
                return self.respond(200, result)
            except Exception:
                # Do not send provider details, credentials, or local file paths to a browser.
                return self.respond(500, {"error": "Run failed. Check local configuration or use offline mode."})

        def log_message(self, fmt, *args):
            pass  # Prompts are not logged by the web server.

    return Handler


def serve(db_path, port=8765, mode="offline"):
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(db_path, mode, port))
    print(f"Dashboard: http://127.0.0.1:{port}  |  {mode} mode  |  Ctrl+C to stop", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
