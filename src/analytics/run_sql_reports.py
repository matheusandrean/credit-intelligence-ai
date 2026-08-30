"""Execute every numbered SQL report in `sql/` against the real generated
parquet files and save results to `reports/sql/`.

This exists to prove the queries in `sql/` actually run against the
project's data (see PROJECT_SPEC section 42: "demonstrate SQL ability, not
just Python") rather than being untested reference text.
"""

from __future__ import annotations

import duckdb

from src.utils.config import PROJECT_ROOT
from src.utils.logging import configure_logging, get_logger

logger = get_logger(__name__)

SQL_DIR = PROJECT_ROOT / "sql"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "sql"


def run_all_reports() -> dict[str, int]:
    con = duckdb.connect(database=":memory:")
    con.execute(f"SET FILE_SEARCH_PATH='{PROJECT_ROOT.as_posix()}'")

    views_sql = (SQL_DIR / "00_views.sql").read_text(encoding="utf-8")
    for statement in [s.strip() for s in views_sql.split(";") if s.strip()]:
        con.execute(statement.replace("data/", f"{PROJECT_ROOT.as_posix()}/data/"))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    row_counts: dict[str, int] = {}

    for path in sorted(SQL_DIR.glob("*.sql")):
        if path.name.startswith("00_"):
            continue
        query = path.read_text(encoding="utf-8")
        df = con.execute(query).df()
        output_path = OUTPUT_DIR / f"{path.stem}.csv"
        df.to_csv(output_path, index=False)
        row_counts[path.name] = len(df)
        logger.info("sql_report_executed", query=path.name, n_rows=len(df))

    return row_counts


def main() -> None:
    configure_logging("INFO")
    counts = run_all_reports()
    logger.info("sql_reports_complete", reports=len(counts))


if __name__ == "__main__":
    main()
