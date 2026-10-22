"""A tiny, inspectable chart classifier trained on synthetic table structures.

This is a nearest-centroid educational baseline, NOT the original project's LSTM.
Its synthetic holdout accuracy measures only the generator's narrow distribution.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import random
import re

from .data import ASSET_DIR


LABELS = ["metric", "line", "bar", "scatter", "table"]
FEATURES = ["log_rows", "numeric_columns", "temporal_columns", "categorical_columns", "log_columns"]


def table_features(columns, rows):
    """Infer structural types from actual non-null cells, with ISO-date support."""
    numeric, temporal, categorical = [], [], []
    for index, column in enumerate(columns):
        values = [row[index] for row in rows[:100] if len(row) > index and row[index] is not None]
        name = str(column).lower()
        temporal_name = name in {"date", "month", "year", "order_date", "snapshot_date", "week", "quarter"} or name.endswith("_date")
        iso_values = bool(values) and all(isinstance(v, str) and re.fullmatch(r"\d{4}(?:-\d{2})?(?:-\d{2})?", v) for v in values)
        if temporal_name or iso_values:
            temporal.append(column)
        elif values and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
            numeric.append(column)
        else:
            categorical.append(column)
    features = [math.log1p(len(rows)), min(len(numeric), 5), min(len(temporal), 3), min(len(categorical), 5), math.log1p(len(columns))]
    return features, {"numeric": numeric, "temporal": temporal, "categorical": categorical}


def synthetic_training_samples(seed=17):
    """600 generated feature-label pairs, 120 per class; no observed user behavior."""
    rng = random.Random(seed)
    samples = []
    for label in LABELS:
        for i in range(120):
            if label == "metric":
                count, n_num, n_time, n_cat = 1, 1, 0, 0
            elif label == "line":
                count, n_num, n_time, n_cat = rng.randint(3, 72), 1, 1, 0
            elif label == "bar":
                count, n_num, n_time, n_cat = rng.randint(2, 30), 1, 0, 1
            elif label == "scatter":
                count, n_num, n_time, n_cat = rng.randint(5, 150), 2, 0, 0
            else:
                count, n_num, n_time, n_cat = rng.randint(2, 80), 0, 0, rng.randint(1, 4)
            features = [math.log1p(count), n_num, n_time, n_cat, math.log1p(n_num+n_time+n_cat)]
            samples.append({"id": f"synthetic-chart-{label}-{i:03d}", "features": features, "label": label, "synthetic": True})
    rng.shuffle(samples)
    return samples


def _standardize(vector, means, scales):
    return [(v - mean) / scale for v, mean, scale in zip(vector, means, scales)]


def _predict(vector, model):
    scaled = _standardize(vector, model["means"], model["scales"])
    distances = {label: sum((a - b) ** 2 for a, b in zip(scaled, center)) for label, center in model["centroids"].items()}
    return min(distances, key=distances.get)


def train_chart_model(path=None, seed=17):
    """Stratified 80/20 holdout, fitting normalization and centroids on train only."""
    samples = synthetic_training_samples(seed)
    train, test = [], []
    for label in LABELS:
        group = [s for s in samples if s["label"] == label]
        train.extend(group[:96])
        test.extend(group[96:])
    means = [sum(s["features"][i] for s in train)/len(train) for i in range(len(FEATURES))]
    scales = [math.sqrt(sum((s["features"][i]-means[i])**2 for s in train)/len(train)) or 1.0 for i in range(len(FEATURES))]
    centers = {}
    for label in LABELS:
        vectors = [_standardize(s["features"], means, scales) for s in train if s["label"] == label]
        centers[label] = [sum(v[i] for v in vectors)/len(vectors) for i in range(len(FEATURES))]
    model = {"model_type": "nearest_centroid", "features": FEATURES, "means": means, "scales": scales, "centroids": centers, "seed": seed,
             "provenance": "600 generated structural feature-label pairs; no human labels, production logs, or original IpserLab model.",
             "limitations": "Holdout samples share the generator and include repeated feature patterns. Accuracy is a synthetic sanity check, not evidence of real-user chart quality or an independent-distribution benchmark."}
    confusion = {label: {pred: 0 for pred in LABELS} for label in LABELS}
    for sample in test:
        confusion[sample["label"]][_predict(sample["features"], model)] += 1
    unique_train = {tuple(s["features"]) for s in train}
    model["evaluation"] = {"train_samples": len(train), "test_samples": len(test), "test_accuracy": sum(confusion[label][label] for label in LABELS)/len(test), "confusion_matrix": confusion,
                           "split": "Stratified 96 train / 24 test per class; distinct sample IDs.", "test_feature_patterns_seen_in_train": sum(tuple(s["features"]) in unique_train for s in test), "unique_train_feature_patterns": len(unique_train)}
    target = Path(path) if path else ASSET_DIR / "chart_model.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(model, indent=2)+"\n", encoding="utf-8")
    return model


def recommend_chart(columns, rows):
    """Recommend chart axes; disclose heuristic fallback and never invent values."""
    if not columns or not rows:
        return {"type": "table", "x": None, "y": None, "title": "No matching data", "method": "empty_result_fallback"}
    vector, types = table_features(columns, rows)
    model_path = ASSET_DIR / "chart_model.json"
    if model_path.exists():
        model = json.loads(model_path.read_text(encoding="utf-8"))
        chart_type, method = _predict(vector, model), "synthetic_nearest_centroid"
    else:
        chart_type, method = "table", "heuristic_fallback_no_model"
        if len(rows) == 1 and len(columns) == 1 and types["numeric"]:
            chart_type = "metric"
        elif types["temporal"] and types["numeric"]:
            chart_type = "line"
        elif types["categorical"] and types["numeric"]:
            chart_type = "bar"
        elif len(types["numeric"]) >= 2:
            chart_type = "scatter"
    # Guardrails keep a classifier mistake from producing invalid chart axes.
    valid = {"metric": len(rows) == 1 and len(columns) == 1 and bool(types["numeric"]), "line": bool(types["temporal"] and types["numeric"]), "bar": bool(types["categorical"] and types["numeric"]), "scatter": len(types["numeric"]) >= 2, "table": True}
    if not valid.get(chart_type):
        chart_type, method = "table", method + "+axis_guardrail"
    x = y = None
    if chart_type == "line":
        x, y = types["temporal"][0], types["numeric"][0]
    elif chart_type == "bar":
        x, y = types["categorical"][0], types["numeric"][0]
    elif chart_type == "scatter":
        x, y = types["numeric"][:2]
    elif chart_type == "metric":
        y = types["numeric"][0]
    title = f"{str(y).replace('_', ' ').title()} by {str(x).replace('_', ' ').title()}" if x else (str(y).replace("_", " ").title() if y else "Query results")
    return {"type": chart_type, "x": x, "y": y, "title": title, "method": method, "training_data": "synthetic structural examples"}


if __name__ == "__main__":
    print(json.dumps(train_chart_model()["evaluation"], indent=2))
