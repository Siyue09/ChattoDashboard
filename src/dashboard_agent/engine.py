"""A genuine LangGraph workflow for the educational chat-to-dashboard demo."""
from __future__ import annotations

from pathlib import Path
import re
import sqlite3
import time
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .providers import OpenAICompatibleProvider, ProviderError
from .retrieval import SchemaRetriever, detect_metrics
from .sql_safety import QueryBudgetError, SQLSafetyError, execute_readonly, validate_sql


class AgentState(TypedDict, total=False):
    question: str
    status: str
    intent: str
    sql: str
    columns: list[str]
    rows: list[list[Any]]
    answer: str
    chart: dict
    retrieved_tables: list[str]
    retrieved_metrics: list[str]
    retrieval: dict
    trace: list[dict]
    error: str
    attempts: int
    repair: bool
    truncated: bool
    plan: dict
    mode: str
    generation_method: str
    scope: dict


UNSAFE_REQUEST = re.compile(
    r"\b(delete|drop|truncate|insert|update|alter|attach|detach|pragma|vacuum|grant|revoke)\b"
    r"|load_extension|readfile|writefile|sqlite_master|sqlite_schema"
    r"|\bignore\b.{0,50}\binstructions?\b|\bunion\s+select\b|--|/\*"
    r"|删除|删库|修改数据库|忽略.{0,10}指令", re.I,
)

DIMENSIONS = {
    "month": "strftime('%Y-%m', o.order_date)",
    "category": "p.category",
    "region": "c.region",
    "segment": "c.segment",
    "product": "p.product_name",
}
ITEM_JOIN = "JOIN order_items i ON i.order_id = o.order_id"
NET_ITEMS = "SUM(i.quantity * i.unit_price * (1 - i.discount))"
GROSS_ITEMS = "SUM(i.quantity * i.unit_price)"


def _clarify(message: str) -> dict:
    return {"status": "clarify", "intent": "clarify", "answer": message, "sql": ""}


def _offline_plan(question: str) -> dict:
    """Recognize a documented finite template vocabulary; never pretend to be an LLM."""
    q = question.lower().strip()
    if re.search(r"\b(profit|margin|retention|churn|conversion|forecast|predict|growth|yoy|mom|median|percentile|cohort)\b|利润|留存|流失|预测|增长|转化", q):
        return _clarify("That metric or comparison is not defined in this demo's business catalog. Please specify a defined metric and its calculation.")
    metrics = detect_metrics(q)
    if "orders" not in metrics and re.search(r"\b(?:and|,)\s+(?:the\s+)?orders\b|\borders\s+(?:and|,)\s+(?:revenue|sales)\b", q):
        metrics.append("orders")
    if not metrics and re.search(r"\b(count|number|total|monthly|how many)\b.{0,30}\borders?\b|\borders?\b.{0,12}\b(month|region|segment)\b|订单数|订单量|多少.{0,8}订单", q):
        metrics = ["orders"]
    if len(metrics) != 1:
        return _clarify("Please choose one defined metric: revenue, gross sales, net revenue, orders, average order value, active customers, refund rate, or MRR.")
    metric = metrics[0]
    # Reject scopes which would otherwise be silently discarded by a template.
    scoped = q.replace("after refunds", "").replace("before refunds", "")
    if metric == "gross_sales":
        scoped = scoped.replace("before discounts", "")
    if re.search(r"\b(last|this|today|yesterday|since|before|after|between|quarter|quarterly|q[1-4]|weeks?|weekly|daily|days?|versus|vs|excluding|exclude|where|only|filter|more than|less than)\b|上个月|本月|上周|本周|季度|每天|每日|环比|同比|仅|筛选", scoped):
        return _clarify("Offline templates support all available data or one calendar year, grouped by one dimension. Please rewrite the date/filter scope explicitly, or use LLM mode.")
    if metric != "aov" and re.search(r"\b(average|mean|maximum|minimum|max|min)\b|平均|最大|最小", q):
        return _clarify("Please specify the aggregation explicitly. Offline mode defines total revenue and average order value, but does not infer other averages or extrema.")
    if re.search(r"[\"']|\b(?:in|during|for)\s+(?!(?:\d{4}\b|all\b|each\b|every\b|completed\b|the\s+year\b|calendar\s+year\b))[a-z]", q):
        return _clarify("That named reporting filter is not supported by offline templates. Please ask for all data or one calendar year, or use LLM mode.")
    years = sorted(set(re.findall(r"(?<!\d)((?:19|20)\d{2})(?!\d)", q)))
    if len(years) > 1 or re.search(r"\b\d{4}[-/]\d{1,2}|\d{1,2}月|\b(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\b", q):
        return _clarify("Offline mode supports a single calendar year or all available data; narrower or multiple date windows require LLM mode.")
    if re.search(r"\b(north|south|east|west|enterprise|consumer|small business|electronics|furniture|clothing|cancelled|canceled|pending|returned)\b", q):
        return _clarify("Offline mode does not apply named category, status, region, or customer filters. Please ask for the complete breakdown, or use LLM mode.")
    dimensions = []
    if re.search(r"\b(month|months|monthly)\b|每月|按月|月度", q):
        dimensions.append("month")
    if re.search(r"\b(category|categories)\b|品类|类别", q):
        dimensions.append("category")
    if re.search(r"\b(regions?|geography|geographic)\b|地区|区域", q):
        dimensions.append("region")
    if re.search(r"\b(segments?)\b|客户类型|客户分群", q):
        dimensions.append("segment")
    if re.search(r"\b(products?|items?)\b|产品|商品", q):
        dimensions.append("product")
    if metric == "mrr":
        # "Monthly recurring revenue" is the metric's name as well as a time phrase.
        dimensions = [dimension for dimension in dimensions if dimension != "month"]
        if dimensions:
            return _clarify("The offline MRR template supports a monthly snapshot; please remove the other grouping.")
        dimensions = ["month"]
    if len(dimensions) > 1:
        return _clarify("Offline mode supports one grouping at a time: month, category, region, segment, or product.")
    dimension = dimensions[0] if dimensions else None
    if metric == "net_revenue" and dimension in {"category", "product"}:
        return _clarify("Refunds are recorded at order level. Net revenue by category/product needs an explicit refund allocation rule before it can be calculated.")
    if metric in {"refund_rate", "aov", "orders", "active_customers"} and dimension in {"category", "product"}:
        return _clarify("This offline template does not define how multi-product orders are assigned to product/category groups. Please use month, region, or segment.")
    top_match = re.search(r"\btop\s+(\d+)\b|前\s*(\d+)", q)
    top_n = int(next(value for value in top_match.groups() if value)) if top_match else 10
    is_top = bool(re.search(r"\btop\b|\bbest-selling\b|前\s*\d+|排名", q))
    if is_top and (dimension != "product" or metric not in {"revenue", "gross_sales"}):
        return _clarify("Offline ranking supports top products by revenue or gross sales.")
    if not 1 <= top_n <= 100:
        return _clarify("Please request between 1 and 100 top products.")
    year = years[0] if years else None
    where = "o.status = 'completed'"
    if year:
        where += f" AND o.order_date >= '{year}-01-01' AND o.order_date < '{int(year) + 1}-01-01'"
    dimension_sql = DIMENSIONS.get(dimension)
    group_select = f"{dimension_sql} AS {dimension}, " if dimension_sql else ""
    group_by = f" GROUP BY {dimension_sql}" if dimension_sql else ""
    order_by = f" ORDER BY {dimension}" if dimension_sql else ""
    joins = []
    if metric in {"revenue", "gross_sales", "aov"}:
        joins.append(ITEM_JOIN)
    if dimension in {"region", "segment"}:
        joins.append("JOIN customers c ON c.customer_id = o.customer_id")
    if dimension in {"category", "product"}:
        joins.append("JOIN products p ON p.product_id = i.product_id")
    if metric == "net_revenue":
        # Aggregate each many-side independently before combining them.
        prefix = (
            "WITH order_totals AS (SELECT order_id, SUM(quantity * unit_price * (1 - discount)) AS revenue "
            "FROM order_items GROUP BY order_id), refund_totals AS (SELECT order_id, SUM(amount) AS refunded "
            "FROM refunds GROUP BY order_id) "
        )
        joins += ["JOIN order_totals ot ON ot.order_id = o.order_id", "LEFT JOIN refund_totals rt ON rt.order_id = o.order_id"]
        expression = "ROUND(COALESCE(SUM(ot.revenue - COALESCE(rt.refunded, 0)), 0), 2)"
    elif metric == "refund_rate":
        prefix = ""
        joins.append("LEFT JOIN (SELECT DISTINCT order_id FROM refunds) r ON r.order_id = o.order_id")
        expression = "ROUND(1.0 * COUNT(DISTINCT r.order_id) / NULLIF(COUNT(DISTINCT o.order_id), 0), 6)"
    else:
        prefix = ""
        expression = {
            "revenue": f"ROUND(COALESCE({NET_ITEMS}, 0), 2)",
            "gross_sales": f"ROUND(COALESCE({GROSS_ITEMS}, 0), 2)",
            "aov": f"ROUND({NET_ITEMS} / NULLIF(COUNT(DISTINCT o.order_id), 0), 2)",
            "orders": "COUNT(DISTINCT o.order_id)",
            "active_customers": "COUNT(DISTINCT o.customer_id)",
            "mrr": "0",  # handled below by the snapshot template
        }[metric]
    if metric == "mrr":
        if year:
            months = " UNION ALL ".join(f"SELECT '{year}-{month:02d}-01' AS month_start" for month in range(1, 13))
        else:
            # The demo's reporting calendar is the months present in order history.
            months = "SELECT DISTINCT date(order_date, 'start of month') AS month_start FROM orders"
        sql = (
            f"WITH months AS ({months}) SELECT strftime('%Y-%m', m.month_start) AS month, "
            "ROUND(COALESCE(SUM(s.mrr), 0), 2) AS mrr FROM months m LEFT JOIN subscriptions s "
            "ON s.start_date <= date(m.month_start, '+1 month', '-1 day') "
            "AND (s.end_date IS NULL OR s.end_date > date(m.month_start, '+1 month', '-1 day')) "
            "GROUP BY m.month_start ORDER BY m.month_start"
        )
    else:
        sql = f"{prefix}SELECT {group_select}{expression} AS {metric} FROM orders o {' '.join(joins)} WHERE {where}{group_by}{order_by}"
    if is_top:
        sql = sql.removesuffix(order_by) + f" ORDER BY {metric} DESC, {dimension} ASC LIMIT {top_n}"
    intent = f"{metric}_by_{dimension}" if dimension else f"total_{metric}"
    if is_top:
        intent = f"top_products_by_{metric}"
    return {"status": "pending", "intent": intent, "sql": sql,
            "plan": {"metric": metric, "dimension": dimension, "year": year, "top_n": top_n if is_top else None}}


class DashboardAgent:
    """Route -> retrieve -> generate -> guard -> execute -> visualize -> summarize.

    LLM mode may retry an invalid generated query once. Offline SQL uses known
    templates independently of retrieval; its accuracy is not evidence of an LLM
    or graph-retrieval quality gain.
    """

    def __init__(self, db_path: str, catalog_path: str | Path | None = None,
                 examples_path: str | Path | None = None, mode: str = "offline", retrieval_mode: str = "graph"):
        if mode not in {"offline", "llm"}:
            raise ValueError("mode must be 'offline' or 'llm'.")
        if retrieval_mode not in {"graph", "vector"}:
            raise ValueError("retrieval_mode must be 'graph' or 'vector'.")
        assets = Path(__file__).resolve().parent / "assets"
        self.db_path = str(Path(db_path).expanduser().resolve())
        self.mode = mode
        self.retrieval_mode = retrieval_mode
        self.retriever = SchemaRetriever(catalog_path or assets / "catalog.json",
                                         examples_path or assets / "retrieval_examples.json")
        self.provider = None
        self.max_rows = 500
        builder = StateGraph(AgentState)
        for name in ("route", "retrieve", "generate", "guard", "execute", "visualize", "summarize"):
            builder.add_node(name, self._timed(name, getattr(self, "_" + name)))
        builder.add_edge(START, "route")
        builder.add_conditional_edges("route", lambda state: "retrieve" if state["status"] == "pending" else "summarize")
        builder.add_edge("retrieve", "generate")
        builder.add_conditional_edges("generate", lambda state: "guard" if state["status"] == "pending" else "summarize")
        builder.add_conditional_edges("guard", self._after_guard)
        builder.add_conditional_edges("execute", self._after_execute)
        builder.add_edge("visualize", "summarize")
        builder.add_edge("summarize", END)
        self.graph = builder.compile()

    @staticmethod
    def _timed(name: str, function):
        def node(state: AgentState) -> dict:
            started = time.perf_counter()
            try:
                result = function(state)
            except Exception as exc:
                result = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
            entry = {"node": name, "duration_ms": round((time.perf_counter() - started) * 1000, 3)}
            if result.get("error"):
                entry["error"] = result["error"]
            result["trace"] = state.get("trace", []) + [entry]
            return result
        return node

    def run(self, question: str) -> dict:
        initial: AgentState = {"question": question if isinstance(question, str) else "", "status": "pending",
                              "intent": "", "sql": "", "columns": [], "rows": [], "answer": "",
                              "chart": {"type": "none", "x": None, "y": None, "title": ""},
                              "retrieved_tables": [], "retrieved_metrics": [], "trace": [], "attempts": 0,
                              "repair": False, "error": "", "mode": self.mode,
                              "scope": {},
                              "generation_method": "deterministic_templates" if self.mode == "offline" else "openai_compatible_llm"}
        result = dict(self.graph.invoke(initial, config={"recursion_limit": 20}))
        if not result.get("error"):
            result.pop("error", None)
        result.pop("plan", None)
        result.pop("repair", None)
        return result

    def _route(self, state: AgentState) -> dict:
        question = state["question"].strip()
        if not question or len(question) > 2_000:
            return _clarify("Please enter an analytics question between 1 and 2,000 characters.")
        if UNSAFE_REQUEST.search(question):
            return {"status": "rejected", "intent": "unsafe", "answer": "This demo accepts read-only analytics questions. Database changes, external reads, and instruction overrides are rejected."}
        if self.mode == "offline":
            plan = _offline_plan(question)
            # Generation is its own graph step; routing only stores the template plan.
            if plan["status"] == "pending":
                return {"status": "pending", "intent": plan["intent"], "plan": plan, "sql": ""}
            return plan
        return {"status": "pending", "intent": "analytics_query"}

    def _retrieve(self, state: AgentState) -> dict:
        retrieval = self.retriever.retrieve(state["question"], mode=self.retrieval_mode)
        return {"retrieval": retrieval, "retrieved_tables": retrieval["tables"], "retrieved_metrics": retrieval["metrics"]}

    def _generate(self, state: AgentState) -> dict:
        if state["status"] == "error" and not state.get("repair"):
            return {}
        if self.mode == "offline":
            return {"sql": state["plan"]["sql"], "attempts": 1, "status": "pending",
                    "scope": dict(state["plan"].get("plan", {}))}
        try:
            if self.provider is None:
                self.provider = OpenAICompatibleProvider()
            response = self.provider.generate_sql(state["question"], state.get("retrieval", {}).get("context", {}),
                                                   previous_sql=state.get("sql", ""), error=state.get("error", ""))
            if not response["sql"].strip():
                return {**_clarify(response.get("clarification") or "The model needs a more specific metric or reporting scope."),
                        "attempts": state["attempts"] + 1, "error": "", "repair": False}
            return {"sql": response["sql"], "attempts": state["attempts"] + 1, "status": "pending", "error": "", "repair": False}
        except ProviderError as exc:
            return {"status": "error", "error": str(exc), "repair": False}

    def _guard(self, state: AgentState) -> dict:
        try:
            sql = validate_sql(state["sql"], self.retriever.tables)
            return {"sql": sql, "status": "pending", "error": ""}
        except SQLSafetyError as exc:
            return {"status": "rejected", "error": str(exc), "repair": self.mode == "llm" and state["attempts"] < 2}

    @staticmethod
    def _after_guard(state: AgentState) -> str:
        if state.get("repair"):
            return "generate"
        return "execute" if state["status"] == "pending" else "summarize"

    def _execute(self, state: AgentState) -> dict:
        try:
            result = execute_readonly(self.db_path, state["sql"], allowed_tables=self.retriever.tables, max_rows=self.max_rows)
            return {**result, "status": "ok", "error": ""}
        except (SQLSafetyError, QueryBudgetError, sqlite3.DatabaseError) as exc:
            repair = self.mode == "llm" and state["attempts"] < 2 and not isinstance(exc, QueryBudgetError)
            return {"status": "error", "error": str(exc), "repair": repair}

    @staticmethod
    def _after_execute(state: AgentState) -> str:
        if state.get("repair"):
            return "generate"
        return "visualize" if state["status"] == "ok" else "summarize"

    def _visualize(self, state: AgentState) -> dict:
        from .chart_model import recommend_chart
        return {"chart": recommend_chart(state["columns"], state["rows"])}

    def _summarize(self, state: AgentState) -> dict:
        if state["status"] == "ok":
            if not state["rows"]:
                answer = "No rows matched this reporting scope."
            elif len(state["rows"]) == 1 and len(state["columns"]) == 1:
                value = state["rows"][0][0]
                rendered = f"{value:,.2f}" if isinstance(value, float) else str(value) if value is not None else "undefined (no qualifying orders)"
                answer = f"{state['columns'][0].replace('_', ' ').capitalize()}: {rendered}."
            else:
                answer = f"Returned {len(state['rows'])} rows with {', '.join(state['columns'])}."
            if state.get("truncated"):
                answer += f" Results are limited to the first {self.max_rows} rows."
            if self.mode == "offline":
                plan = state.get("plan", {}).get("plan", {})
                answer += f" Scope: calendar year {plan['year']}." if plan.get("year") else " Scope: all available reporting dates."
                if plan.get("metric") == "net_revenue":
                    answer += " Refunds are attributed to original order dates."
                elif plan.get("metric") == "refund_rate":
                    answer += " Refund rate is a fraction of completed orders, not a percentage value."
                elif plan.get("metric") == "mrr":
                    answer += " MRR is a month-end snapshot with exclusive subscription end dates."
            answer += " Synthetic demo data. " + ("SQL generated from explicit offline templates." if self.mode == "offline" else "SQL generated by the configured language model; review the SQL before relying on its business meaning.")
            return {"answer": answer}
        if state.get("answer"):
            return {}
        return {"answer": state.get("error") or "The request could not be completed.",
                "status": state["status"] if state["status"] in {"clarify", "rejected", "error"} else "error"}
