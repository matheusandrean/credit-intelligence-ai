"""Data Quality: automated validation report."""

from __future__ import annotations

import streamlit as st

from app.theme import configure_page, demo_data_disclaimer, load_json_report

configure_page("Data Quality", icon="\U0001f9ea")
st.title("Data Quality")
demo_data_disclaimer()

report = load_json_report("data_quality_report.json")
if not report:
    st.warning("No data quality report found. Run `make validate` first.")
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Rows validated", f"{report['n_rows']:,}")
c2.metric("Schema valid", "Yes" if report["schema_valid"] else "No")
c3.metric("Overall passed", "Yes" if report["passed"] else "No")

st.markdown("---")
st.subheader("Duplicate Customer IDs")
st.metric("Duplicates found", report["duplicate_customer_ids"])

st.subheader("Protected Attributes Check")
if report["protected_attributes_found"]:
    st.error(f"Protected attributes found: {report['protected_attributes_found']}")
else:
    st.success("No protected/sensitive attributes present in the dataset.")

st.subheader("Cross-Field Consistency")
if report["consistency_failures"]:
    for f in report["consistency_failures"]:
        st.error(f)
else:
    st.success("No consistency failures detected (late-payment monotonicity, date ordering).")

st.markdown("---")
st.subheader("Null Rates by Column")
null_rates = {k: v for k, v in report["null_rates"].items() if v > 0}
if null_rates:
    st.bar_chart(null_rates)
else:
    st.info("No missing values detected.")

st.subheader("Monthly Volume Anomalies (|z| > 2.5)")
if report["monthly_volume_anomalies"]:
    st.dataframe(report["monthly_volume_anomalies"], width="stretch")
else:
    st.success("No monthly volume anomalies detected.")

st.markdown("---")
st.subheader("Default Rate by Month")
st.line_chart(report["default_rate_by_month"])
