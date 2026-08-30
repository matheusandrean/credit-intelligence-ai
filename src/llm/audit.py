"""Audit logging for every Credit Intelligence Agent interaction.

Appends one JSON line per interaction to `reports/audit_log.jsonl`:
timestamp, question, tools invoked (name + ok/error, never full row-level
data), RAG sources cited, model/provider version, and analysis type. Never
logs secrets or full API responses - see RESPONSIBLE_AI.md and SECURITY.md.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from src.utils.config import PROJECT_ROOT

AUDIT_LOG_PATH = PROJECT_ROOT / "reports" / "audit_log.jsonl"


@dataclass
class ToolCallAudit:
    name: str
    ok: bool
    error: str | None = None


@dataclass
class AuditRecord:
    timestamp: str
    provider: str
    question: str
    tools_called: list[ToolCallAudit] = field(default_factory=list)
    rag_sources: list[str] = field(default_factory=list)
    model_version: str | None = None

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "provider": self.provider,
            "question": self.question,
            "tools_called": [
                {"name": t.name, "ok": t.ok, "error": t.error} for t in self.tools_called
            ],
            "rag_sources": self.rag_sources,
            "model_version": self.model_version,
        }


def log_interaction(
    provider: str,
    question: str,
    tools_called: list[ToolCallAudit],
    rag_sources: list[str],
    model_version: str | None,
    log_path: Path | None = None,
) -> None:
    # Resolved at call time (not baked in as a default) so tests can point
    # this at a temporary path via `audit.AUDIT_LOG_PATH = ...`.
    resolved_path = log_path or AUDIT_LOG_PATH
    record = AuditRecord(
        timestamp=datetime.now(UTC).isoformat(),
        provider=provider,
        question=question,
        tools_called=tools_called,
        rag_sources=rag_sources,
        model_version=model_version,
    )
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    with open(resolved_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record.to_dict(), default=str) + "\n")


def read_recent_audit_records(limit: int = 50, log_path: Path | None = None) -> list[dict]:
    resolved_path = log_path or AUDIT_LOG_PATH
    if not resolved_path.exists():
        return []
    lines = resolved_path.read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(line) for line in lines[-limit:]]
