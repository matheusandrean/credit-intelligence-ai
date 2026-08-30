# Executive Reporting

`src/reporting/executive_report.py::generate_executive_report` produces a
Markdown Portfolio Executive Report with exactly the sections the project
spec calls for, each backed by a real tool/analytics call — never freehand
text:

| Section | Source |
|---|---|
| Portfolio Overview | `src.analytics.portfolio.portfolio_summary` |
| Risk Movement | `src.analytics.portfolio.monthly_kpi_trend` (MoM deltas) |
| Main Drivers | `ShapExplainer.global_importance` on a test-split sample |
| Deterioration Signals | `src.monitoring.drift.feature_drift_report` (train vs OOT test) |
| Stress Testing | `src.risk.stress_testing.run_stress_test` (severe scenario) |
| Model Health | `src.monitoring.drift.model_performance_over_time` + champion/calibration metadata |
| Points Requiring Human Attention | Threshold checks against `knowledge_base/risk_appetite.md`'s demonstrative triggers (high-risk % > 20%, any non-Stable drift, severe-scenario PD delta > 3pp) |

## How to generate it

- **Agent / chat**: ask "Summarize the portfolio for a Credit Risk
  Director" or "Gere um resumo executivo da carteira" — both Demo Mode and
  a real LLM provider route to the `generate_executive_report` tool.
- **Dashboard**: click "Generate Portfolio Executive Report" on the
  Executive Overview page.
- **Script**: `python -m src.reporting.executive_report`, which writes
  `reports/executive_report.md`.
- **API**: not exposed as a dedicated REST endpoint in this version (only
  via `/ai/chat`); adding a `GET /reports/executive` endpoint would be a
  one-line addition reusing the same tool (see `api/routers/`).

## Design note: why every section calls an existing tool

The point of this feature is explicitly to demonstrate the anti-
hallucination principle end to end: an "executive summary" is exactly the
kind of artifact an LLM could be tempted to write persuasively from
plausible-sounding invented numbers. Building it entirely by composing
already-tested tool outputs means the report is only ever as wrong as the
underlying analytics — never wrong in a new way introduced by the
reporting layer itself.
