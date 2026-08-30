"""Guardrails for the `query_portfolio` text-to-SQL tool.

The LLM must never be allowed to run destructive or unrestricted SQL. This
module enforces: single SELECT statement only, no DDL/DML keywords, no
multi-statement injection via semicolons, and a hard row-limit safety net.
"""

from __future__ import annotations

import re

FORBIDDEN_KEYWORDS: tuple[str, ...] = (
    "DROP",
    "DELETE",
    "UPDATE",
    "ALTER",
    "INSERT",
    "TRUNCATE",
    "CREATE",
    "ATTACH",
    "DETACH",
    "COPY",
    "PRAGMA",
    "GRANT",
    "REVOKE",
    "REPLACE",
    "VACUUM",
    "CALL",
    "EXPORT",
    "IMPORT",
    "INSTALL",
    "LOAD",
)

MAX_ROW_LIMIT = 500


class UnsafeSqlError(ValueError):
    pass


def _strip_sql_comments(sql: str) -> str:
    sql = re.sub(r"--.*?$", "", sql, flags=re.MULTILINE)
    sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
    return sql


def validate_select_only(sql: str) -> str:
    """Raise `UnsafeSqlError` unless `sql` is a single, safe SELECT statement.

    Returns the (possibly row-limited) statement to execute.
    """
    cleaned = _strip_sql_comments(sql).strip()
    if not cleaned:
        raise UnsafeSqlError("Empty SQL statement.")

    # Reject multiple statements (allow a single optional trailing semicolon).
    body = cleaned[:-1] if cleaned.endswith(";") else cleaned
    if ";" in body:
        raise UnsafeSqlError("Multiple SQL statements are not allowed.")

    if not re.match(r"^\s*(WITH\b.*?\bSELECT\b|SELECT)\b", body, re.IGNORECASE | re.DOTALL):
        raise UnsafeSqlError("Only SELECT (or WITH ... SELECT) statements are allowed.")

    tokens = re.findall(r"[A-Za-z_]+", body.upper())
    forbidden_hit = set(tokens) & set(FORBIDDEN_KEYWORDS)
    if forbidden_hit:
        raise UnsafeSqlError(f"Forbidden SQL keyword(s) detected: {sorted(forbidden_hit)}")

    if not re.search(r"\bLIMIT\b", body, re.IGNORECASE):
        body = f"{body}\nLIMIT {MAX_ROW_LIMIT}"

    return body
