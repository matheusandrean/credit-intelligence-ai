"""Unit tests for src.llm.tools.sql_guard."""

from __future__ import annotations

import pytest

from src.llm.tools.sql_guard import UnsafeSqlError, validate_select_only


def test_allows_simple_select() -> None:
    result = validate_select_only("SELECT * FROM portfolio")
    assert result.strip().upper().startswith("SELECT")
    assert "LIMIT" in result.upper()


def test_allows_with_cte_select() -> None:
    result = validate_select_only("WITH x AS (SELECT * FROM portfolio) SELECT * FROM x")
    assert "LIMIT" in result.upper()


def test_preserves_existing_limit() -> None:
    result = validate_select_only("SELECT * FROM portfolio LIMIT 10")
    assert result.upper().count("LIMIT") == 1


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE portfolio",
        "DELETE FROM portfolio",
        "UPDATE portfolio SET pd = 0",
        "ALTER TABLE portfolio ADD COLUMN x INT",
        "INSERT INTO portfolio VALUES (1)",
        "TRUNCATE portfolio",
        "CREATE TABLE evil (x INT)",
        "ATTACH 'x.db' AS x",
        "PRAGMA table_info(portfolio)",
    ],
)
def test_rejects_destructive_statements(sql: str) -> None:
    with pytest.raises(UnsafeSqlError):
        validate_select_only(sql)


def test_rejects_multi_statement_injection() -> None:
    with pytest.raises(UnsafeSqlError):
        validate_select_only("SELECT * FROM portfolio; DROP TABLE portfolio")


def test_rejects_non_select_start() -> None:
    with pytest.raises(UnsafeSqlError):
        validate_select_only("EXPLAIN SELECT * FROM portfolio")


def test_rejects_empty_sql() -> None:
    with pytest.raises(UnsafeSqlError):
        validate_select_only("   ")


def test_comment_hiding_drop_is_rejected() -> None:
    with pytest.raises(UnsafeSqlError):
        validate_select_only("SELECT * FROM portfolio; -- comment\nDROP TABLE portfolio")
