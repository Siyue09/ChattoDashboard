# Evaluation of the educational reconstruction

The checked-in [case-level report](../reports/evaluation.json) was generated locally with seed 42 in offline mode. The 34 synthetic cases were visible during development and informed parser fixes. They are a regression suite, not a blind benchmark or evidence about the original internship system.

| Check | Measured result | Denominator / meaning |
| --- | --- | --- |
| SQL execution success | 24 / 24 | All answerable cases, including any mistaken clarification or error |
| SQL result accuracy | 24 / 24 | Rows compared with separately authored reference queries |
| Response routing | Macro F1 1.00 | Three status labels: ok (24), clarify (5), rejected (5) |
| Graph retrieval table precision | 62.65% | 52 relevant tables / 83 retrieved tables, summed over cases |
| Graph retrieval table recall | 91.23% | 52 retrieved required tables / 57 required tables |
| Lexical-vector table precision | 63.16% | 12 relevant tables / 19 retrieved tables |
| Lexical-vector table recall | 21.05% | 12 retrieved required tables / 57 required tables |
| Chart classifier sanity check | 116 / 120 (96.67%) | Same-generator synthetic test features after 480 training samples |

The response-routing F1 uses output status, not a production intent taxonomy. The graph comparison includes ontology aliases and schema expansion, so it is not a graph-edge-only causal ablation. SQL templates do not consume retrieval context; answer correctness cannot show retrieval's downstream value in offline mode.

There are 224 generated retrieval questions from 28 template families. Exact test-question overlap is zero; many semantic families overlap. Chart sample IDs are disjoint, but 80 of 120 test feature patterns also occur in training. Those limitations materially restrict what the scores mean.

All financial data are fictional. Refund totals are aggregated by order before joining to item totals. This avoids fanout and provides a concrete correctness test. Refunds are assigned to original order month. MRR uses month-end active subscriptions with exclusive end dates.

## Reproduce

```bash
dashboard-lab evaluate --output artifacts/evaluation.json
python scripts/check_evaluation.py artifacts/evaluation.json
python -m unittest discover -s tests -v
```

The CI gate detects regressions against these fixed development cases. It does not certify natural-language generalization, prompt-injection robustness, production readiness, human answer quality, or user time savings. Timing values vary by machine. No external LLM run or real-user study is included in this report.

## Next experiment

Freeze independently collected human questions and analyst-reviewed answers before running the model. Hold model, generation settings, context budget, alias handling, and metric definitions constant across retrieval conditions. Report execution rate, result accuracy, human rubric scores, table/column linking precision and recall, latency, retries, token cost, and confidence intervals. Keep all failures in the appropriate denominators.
