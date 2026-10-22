"""Pluggable vector retrieval with inspectable schema/metric graph expansion.

The default vectors are deterministic hashed bag-of-words vectors, NOT trained
semantic embeddings. RETRIEVAL_EMBEDDINGS=remote explicitly enables the configured
embedding model for catalog records, synthetic examples, and each question.
"""
from __future__ import annotations

from collections import Counter, deque
import hashlib
import json
import math
import os
from pathlib import Path
import re
import threading

from .providers import OpenAICompatibleEmbeddingProvider


METRIC_ALIASES = {
    "gross_sales": ("gross sales", "gross revenue", "毛销售", "折扣前"),
    "net_revenue": ("net revenue", "net sales", "净收入", "净营收", "扣除退款", "after refunds"),
    "revenue": ("revenue", "sales", "收入", "营收", "销售额"),
    "aov": ("aov", "average order value", "average basket", "客单价", "平均订单金额"),
    "orders": ("order count", "number of orders", "orders by", "monthly orders", "订单数", "订单量"),
    "active_customers": ("active customers", "unique customers", "distinct customers", "活跃客户", "活跃用户"),
    "refund_rate": ("refund rate", "refunded order rate", "退款率"),
    "mrr": ("mrr", "monthly recurring revenue", "subscription revenue", "订阅收入", "月度经常性收入"),
}


def detect_metrics(question: str) -> list[str]:
    question = question.lower().replace("_", " ")
    found = [name for name, aliases in METRIC_ALIASES.items() if any(alias in question for alias in aliases)]
    if any(name in found for name in ("net_revenue", "gross_sales", "mrr")) and "revenue" in found:
        found.remove("revenue")
    return found


def _tokens(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", text.lower())
    normalized = [word[:-1] if len(word) > 3 and word.endswith("s") else word for word in words]
    return normalized + [f"{a}::{b}" for a, b in zip(normalized, normalized[1:])]


def _vector(text: str, dimensions: int = 768) -> dict[int, float]:
    counts: Counter[int] = Counter()
    for token in _tokens(text):
        digest = hashlib.sha256(token.encode()).digest()
        counts[int.from_bytes(digest[:4], "big") % dimensions] += 1
    values = {index: 1 + math.log(count) for index, count in counts.items()}
    norm = math.sqrt(sum(value * value for value in values.values())) or 1
    return {index: value / norm for index, value in values.items()}


def _similarity(left: dict[int, float], right: dict[int, float]) -> float:
    return sum(value * right.get(index, 0) for index, value in left.items())


class SchemaRetriever:
    """Compare identical vector seed retrieval with and without graph expansion."""

    def __init__(self, catalog_path: str | Path, examples_path: str | Path | None = None, top_k: int = 3,
                 embeddings_mode: str | None = None):
        self.embeddings_mode = (embeddings_mode or os.getenv("RETRIEVAL_EMBEDDINGS", "local")).lower()
        if self.embeddings_mode not in {"local", "remote"}:
            raise ValueError("RETRIEVAL_EMBEDDINGS must be 'local' or 'remote'.")
        self.embedding_provider = None
        self._remote_vectors: list[tuple[float, ...]] | None = None
        self._embedding_lock = threading.Lock()
        with Path(catalog_path).open(encoding="utf-8") as handle:
            self.catalog = json.load(handle)
        self.tables = {table["name"]: table for table in self.catalog["tables"]}
        self.metrics = {metric["name"]: metric for metric in self.catalog["metrics"]}
        self.joins = self.catalog.get("joins", self.catalog.get("graph_edges", []))
        self.top_k = top_k
        self.documents = []
        for kind, items in (("table", self.tables), ("metric", self.metrics)):
            for name, record in items.items():
                # Exclude required_tables and SQL formula from the lexical vector;
                # they are structural knowledge made available through graph edges.
                surface = " ".join([name.replace("_", " "), record.get("description", ""),
                                    " ".join(record.get("keywords", []))])
                self.documents.append({"kind": kind, "name": name, "text": surface, "vector": _vector(surface)})
        self.examples = []
        if examples_path and Path(examples_path).is_file():
            with Path(examples_path).open(encoding="utf-8") as handle:
                content = json.load(handle)
            self.examples = content if isinstance(content, list) else content.get("examples", [])
        self.adjacency: dict[str, set[str]] = {name: set() for name in self.tables}
        for join in self.joins:
            left, right = join["left_table"], join["right_table"]
            self.adjacency.setdefault(left, set()).add(right)
            self.adjacency.setdefault(right, set()).add(left)

    def _path(self, start: str, finish: str) -> list[str]:
        queue = deque([(start, [start])])
        seen = {start}
        while queue:
            current, path = queue.popleft()
            if current == finish:
                return path
            for neighbor in sorted(self.adjacency.get(current, set())):
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return []

    def retrieve(self, question: str, mode: str = "graph") -> dict:
        if mode not in {"graph", "vector"}:
            raise ValueError("Retrieval mode must be 'graph' or 'vector'.")
        if self.embeddings_mode == "remote":
            # Both comparison modes share exactly these semantic seed scores.
            # Cache only this retriever's catalog/example vectors, never questions.
            with self._embedding_lock:
                if self.embedding_provider is None:
                    self.embedding_provider = OpenAICompatibleEmbeddingProvider()
                if self._remote_vectors is None:
                    texts = [document["text"] for document in self.documents] + [example["question"] for example in self.examples]
                    self._remote_vectors = self.embedding_provider.embed(texts)
                query_vector = self.embedding_provider.embed([question])[0]
            scores = [math.fsum(left * right for left, right in zip(query_vector, vector)) for vector in self._remote_vectors]
            document_scores = scores[:len(self.documents)]
            example_scores = scores[len(self.documents):]
        else:
            query = _vector(question)
            document_scores = [_similarity(query, document["vector"]) for document in self.documents]
            example_scores = [_similarity(query, _vector(example.get("question", ""))) for example in self.examples]
        score_map = {(document["kind"], document["name"]): score for document, score in zip(self.documents, document_scores)}
        ranked = sorted(self.documents, key=lambda document: (-score_map[document["kind"], document["name"]], document["name"]))
        seeds = [document for document in ranked[:self.top_k] if score_map[document["kind"], document["name"]] > 0]
        tables = {document["name"] for document in seeds if document["kind"] == "table"}
        metrics = {document["name"] for document in seeds if document["kind"] == "metric"}
        evidence = [{"reason": "semantic_vector_seed" if self.embeddings_mode == "remote" else "lexical_vector_seed",
                     "kind": doc["kind"], "name": doc["name"],
                     "score": round(score_map[doc["kind"], doc["name"]], 4)} for doc in seeds]
        if mode == "graph":
            # Entity linking has an explicit, inspectable alias dictionary.
            for metric in detect_metrics(question):
                if metric in self.metrics and metric not in metrics:
                    metrics.add(metric)
                    evidence.append({"reason": "metric_alias", "name": metric})
            for metric in sorted(metrics):
                for table in self.metrics[metric].get("required_tables", []):
                    if table in self.tables and table not in tables:
                        tables.add(table)
                        evidence.append({"reason": "metric_dependency", "from": metric, "name": table})
            initial_tables = sorted(tables)
            for index, left in enumerate(initial_tables):
                for right in initial_tables[index + 1:]:
                    for table in self._path(left, right):
                        if table not in tables:
                            tables.add(table)
                            evidence.append({"reason": "foreign_key_path", "from": left, "to": right, "name": table})
        selected_examples = [example for _, example in sorted(zip(example_scores, self.examples), key=lambda pair: -pair[0])[:2]]
        selected_joins = [join for join in self.joins if join["left_table"] in tables and join["right_table"] in tables]
        return {
            "tables": sorted(tables), "metrics": sorted(metrics), "evidence": evidence,
            "method": ("remote_semantic_embeddings" if self.embeddings_mode == "remote" else "hashed_lexical_vectors") + ("_plus_schema_graph" if mode == "graph" else ""),
            "embedding_backend": self.embeddings_mode,
            "embedding_model": self.embedding_provider.model if self.embedding_provider else "sha256-hashed-bag-of-words-768",
            "context": {"tables": [self.tables[name] for name in sorted(tables)],
                        "metrics": [self.metrics[name] for name in sorted(metrics)],
                        "joins": selected_joins, "examples": selected_examples},
        }


# Convenient alias for readers comparing retrieval implementations.
GraphRetriever = SchemaRetriever
