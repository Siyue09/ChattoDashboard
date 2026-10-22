#!/usr/bin/env python3
"""Fail CI when the public, development-visible offline regressions stop passing.

This is a regression gate, not a secret benchmark or a production quality claim.
Usage: python scripts/check_evaluation.py artifacts/evaluation.json
"""

from __future__ import annotations

import json
from pathlib import Path
import sys


def check_report(report):
    failures = []
    if report.get("mode") != "offline":
        failures.append("Expected the deterministic offline regression report.")
    metrics = report.get("metrics", {})
    dataset = report.get("dataset", {})
    answerable = dataset.get("answerable_cases", 0)
    total = dataset.get("fixture_count", 0)
    if not isinstance(answerable, int) or answerable < 24:
        failures.append(f"Expected at least 24 answerable regression cases; got {answerable!r}.")
    if not isinstance(total, int) or total < 34:
        failures.append(f"Expected at least 34 total regression cases; got {total!r}.")
    for metric in ["sql_execution_success", "result_accuracy"]:
        score = metrics.get(metric, {})
        if score.get("value") != 1.0:
            failures.append(f"{metric}: expected 1.0, got {score.get('value')!r} ({score.get('numerator')}/{score.get('denominator')}).")
        if score.get("denominator") != answerable or score.get("numerator") != answerable:
            failures.append(f"{metric}: numerator and denominator must both equal all {answerable} answerable cases.")
    routing = metrics.get("response_routing_macro_f1", {})
    if routing.get("value") != 1.0:
        failures.append(f"response_routing_macro_f1: expected 1.0, got {routing.get('value')!r}.")
    if not {"ok", "clarify", "rejected"}.issubset(routing.get("labels", [])):
        failures.append("Routing evaluation must include ok, clarify, and rejected cases.")
    overlap = report.get("overlap_audit", {})
    if overlap.get("exact_question_overlap_count") != 0:
        failures.append(f"Expected zero exact question overlap; got {overlap.get('exact_question_overlap_count')!r}.")
    if overlap.get("id_overlap_count") != 0:
        failures.append(f"Expected zero example/fixture ID overlap; got {overlap.get('id_overlap_count')!r}.")
    cases = report.get("cases", [])
    if len(cases) != total:
        failures.append(f"Report contains {len(cases)} cases but declares {total}.")
    for case in cases:
        if not case.get("routing_correct"):
            failures.append(f"{case.get('id', '<unknown>')}: expected status {case.get('expected_status')}, got {case.get('actual_status')}.")
        if case.get("expected_status") == "ok" and (not case.get("sql_execution_success") or not case.get("result_correct")):
            failures.append(f"{case.get('id', '<unknown>')}: query execution or independent gold-result comparison failed.")
    return failures


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if len(arguments) != 1:
        print("Usage: python scripts/check_evaluation.py <evaluation.json>", file=sys.stderr)
        return 2
    try:
        report = json.loads(Path(arguments[0]).read_text(encoding="utf-8"))
        if not isinstance(report, dict):
            raise ValueError("Report root must be a JSON object.")
        failures = check_report(report)
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        print(f"Cannot check offline regression report: {exc}", file=sys.stderr)
        return 2
    if failures:
        print("Offline development regression gate FAILED:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print(f"Offline development regression gate passed: {report['dataset']['fixture_count']} public cases; all {report['dataset']['answerable_cases']} query results match independent gold SQL; routing F1 = 1.0; zero exact retrieval-question overlap.")
    print("These inspected development regressions are not a blind benchmark or validation of the original resume's metrics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
