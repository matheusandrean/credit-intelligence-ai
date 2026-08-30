-- Data mart views over the synthetic Credit Intelligence AI portfolio.
-- Run first by src/analytics/run_sql_reports.py, which registers these
-- views against the generated parquet files before executing the numbered
-- report queries in this directory.
--
-- Conceptual dimensional model (see docs/DATA_MART.md for the Mermaid ERD):
--   fact_credit_performance  -> v_portfolio   (one row per scored account)
--   fact_customer_behavior   -> v_features    (one row per scored account)
--   fact_delinquency         -> v_monthly_performance (one row per account-month)
--   dim_customer             -> customer_id + static attributes
--   dim_date                 -> observation_date / snapshot_month
--   dim_risk_band            -> risk_band A-E

CREATE OR REPLACE VIEW v_portfolio AS
    SELECT *
    FROM read_parquet('data/processed/scored_portfolio.parquet');

CREATE OR REPLACE VIEW v_features AS
    SELECT *
    FROM read_parquet('data/processed/credit_features.parquet');

CREATE OR REPLACE VIEW v_monthly_performance AS
    SELECT *
    FROM read_parquet('data/raw/monthly_performance.parquet');

CREATE OR REPLACE VIEW v_customers AS
    SELECT *
    FROM read_parquet('data/raw/customer_portfolio.parquet');
