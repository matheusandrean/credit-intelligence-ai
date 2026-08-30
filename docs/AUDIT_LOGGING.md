# Audit Logging

Every Credit Intelligence Agent interaction is logged for auditability, per
[`RESPONSIBLE_AI.md`](../RESPONSIBLE_AI.md).

## What is logged

`src/llm/audit.py::log_interaction` appends one JSON line to
`reports/audit_log.jsonl` per question:

```json
{
  "timestamp": "2026-01-15T14:32:01.123456+00:00",
  "provider": "demo",
  "question": "Compare risk bands A and D.",
  "tools_called": [
    {"name": "compare_segments", "ok": true, "error": null}
  ],
  "rag_sources": [],
  "model_version": "logistic_regression+sigmoid"
}
```

- **timestamp** — ISO 8601, UTC.
- **provider** — which LLM provider handled the question (`anthropic`,
  `openai`, `ollama`, or `demo`).
- **question** — the user's raw question text.
- **tools_called** — each tool invoked, whether it succeeded, and its
  error message if not. Never the full row-level result payload.
- **rag_sources** — `"Document > Section"` citations from any
  `retrieve_credit_policy` call in this interaction.
- **model_version** — `{champion_model}+{calibration_method}`, so a report
  generated against one model version is traceable if the champion later
  changes.

## What is never logged

- Secrets, API keys, or full provider request/response payloads.
- Full tool result data (portfolio rows, SHAP values) — only whether the
  call succeeded.
- Any information beyond what the question and tool names already reveal.

## Reading the log

`src/llm/audit.py::read_recent_audit_records(limit=50)` returns the most
recent records as parsed dicts, used by tests
(`tests/integration/test_agent.py::test_agent_logs_audit_record`) to verify
logging behavior without depending on log file layout details elsewhere in
the codebase.

## Why JSONL, not a database

A local dataset log matches the project's "everything reproducible from
this repo, nothing depends on external infrastructure" principle. A real
deployment would ship this to a centralized log/SIEM system instead of a
local file — this is a demonstration of the logging *contract* (what
fields, what's excluded), not a production logging pipeline.
