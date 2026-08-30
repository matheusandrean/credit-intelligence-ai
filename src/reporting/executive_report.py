"""Generate the Portfolio Executive Report (spec section 60).

Every section is built exclusively from already-computed tool/analytics
output - never freehand text - so the report is reproducible and auditable
like everything else in this platform. See docs/EXECUTIVE_REPORTING.md.
"""

from __future__ import annotations

from datetime import UTC, datetime

from src.analytics.portfolio import monthly_kpi_trend, portfolio_summary
from src.llm.tools.context import get_tool_context
from src.monitoring.drift import feature_drift_report, model_performance_over_time
from src.risk.stress_testing import load_stress_scenarios, run_stress_test
from src.utils.config import PROJECT_ROOT
from src.utils.logging import get_logger

logger = get_logger(__name__)


def generate_executive_report() -> str:
    ctx = get_tool_context()
    scored, features = ctx.scored_portfolio, ctx.features

    summary = portfolio_summary(scored, features)
    trend = monthly_kpi_trend(scored, features)

    lines: list[str] = []
    lines.append("# Portfolio Executive Report")
    lines.append("")
    lines.append(f"_Generated {datetime.now(UTC).isoformat()} - synthetic demonstration data._")
    lines.append("")

    # --- Portfolio Overview ---------------------------------------------
    lines.append("## Portfolio Overview")
    lines.append("")
    lines.append(f"- Total customers: **{summary['total_customers']:,}**")
    lines.append(f"- Portfolio exposure (EAD): **${summary['portfolio_exposure']:,.0f}**")
    lines.append(f"- Average PD: **{summary['average_pd']:.2%}**")
    lines.append(
        f"- Expected Loss: **${summary['expected_loss']:,.0f}** ({summary['expected_loss_rate']:.2%} of exposure)"
    )
    lines.append(
        f"- High-risk population (bands D+E): **{summary['high_risk_population_pct']:.2%}**"
    )
    lines.append("")

    # --- Risk Movement -----------------------------------------------------
    lines.append("## Risk Movement")
    lines.append("")
    if len(trend) > 1:
        latest, prior = trend.iloc[-1], trend.iloc[-2]
        pd_delta = (latest["average_pd"] - prior["average_pd"]) * 100
        el_delta = latest["expected_loss"] - prior["expected_loss"]
        lines.append(
            f"- Average PD moved **{pd_delta:+.2f} pp** month-over-month "
            f"({prior['month']} -> {latest['month']})."
        )
        lines.append(f"- Expected Loss moved **${el_delta:+,.0f}** over the same period.")
    else:
        lines.append("- Not enough monthly history in this dataset to compute MoM movement.")
    lines.append("")

    # --- Main Drivers (global SHAP) ----------------------------------------
    lines.append("## Main Drivers")
    lines.append("")
    try:
        from src.models.train import split_xy

        categories = ctx.model_metadata["categories"]
        x_sample, _ = split_xy(features, "test", categories)
        if x_sample.empty:
            x_sample, _ = split_xy(features, "train", categories)
        sample = x_sample.sample(min(500, len(x_sample)), random_state=1)
        importance = ctx.shap_explainer.global_importance(sample).head(5)
        for row in importance.itertuples():
            lines.append(f"- `{row.feature}` (mean |SHAP| = {row.mean_abs_shap:.4f})")
    except Exception as exc:  # noqa: BLE001 - reporting must degrade gracefully
        logger.warning("executive_report_shap_unavailable", error=str(exc))
        lines.append("- SHAP driver ranking unavailable for this run.")
    lines.append("")

    # --- Deterioration Signals -----------------------------------------------
    lines.append("## Deterioration Signals")
    lines.append("")
    train = features[features["split"] == "train"]
    test = features[features["split"] == "test"]
    drift_features = [
        "debt_to_income",
        "credit_utilization",
        "late_payments_12m",
        "payment_stress_index",
        "behavioral_deterioration_index",
    ]
    drift_report = feature_drift_report(train, test, drift_features)
    flagged = drift_report[drift_report["status"] != "Stable"]
    if flagged.empty:
        lines.append("- No feature shows Monitor-level or worse drift (train vs OOT test).")
    else:
        for row in flagged.itertuples():
            lines.append(f"- `{row.feature}`: PSI={row.psi} ({row.status})")
    lines.append("")

    # --- Stress Testing ---------------------------------------------------
    lines.append("## Stress Testing")
    lines.append("")
    scenarios = load_stress_scenarios()
    severe_result = run_stress_test(scenarios["severe"], features)
    lines.append(
        f"- Severe scenario: average PD {severe_result['baseline']['average_pd']:.2%} -> "
        f"{severe_result['stressed']['average_pd']:.2%} "
        f"({severe_result['pd_delta']*100:+.2f} pp), "
        f"Expected Loss {severe_result['expected_loss_delta']:+,.0f}."
    )
    lines.append("- This is a hypothetical simulation, not a forecast (see RESPONSIBLE_AI.md).")
    lines.append("")

    # --- Model Health -----------------------------------------------------
    lines.append("## Model Health")
    lines.append("")
    perf = model_performance_over_time(scored)
    if not perf.empty:
        latest_perf = perf.iloc[-1]
        lines.append(
            f"- Latest month ({latest_perf['month']}): ROC-AUC={latest_perf['roc_auc']}, "
            f"KS={latest_perf['ks_statistic']}, predicted-PD PSI vs first month="
            f"{latest_perf['psi_vs_first_month']} ({latest_perf['psi_status']})."
        )
    lines.append(
        f"- Champion model: `{ctx.model_metadata['champion_model']}`, calibration: "
        f"`{ctx.model_metadata.get('calibration_method', 'n/a')}`."
    )
    lines.append("")

    # --- Points Requiring Human Attention -----------------------------------
    lines.append("## Points Requiring Human Attention")
    lines.append("")
    attention_points = []
    if summary["high_risk_population_pct"] > 0.20:
        attention_points.append(
            f"High-risk population ({summary['high_risk_population_pct']:.1%}) exceeds the "
            "20% monitoring threshold in risk_appetite.md."
        )
    if not flagged.empty:
        attention_points.append(
            f"{len(flagged)} feature(s) show non-Stable drift - review model applicability."
        )
    if severe_result["pd_delta"] > 0.03:
        attention_points.append(
            "Severe stress scenario increases average PD by more than 3pp - review capital/"
            "provisioning implications per risk_appetite.md."
        )
    if not attention_points:
        attention_points.append("No threshold breaches detected in this run.")
    for point in attention_points:
        lines.append(f"- {point}")
    lines.append("")

    lines.append("---")
    lines.append(
        "_This report is decision support only. No output above constitutes an automated "
        "credit decision - see RESPONSIBLE_AI.md._"
    )

    return "\n".join(lines)


def main() -> None:
    report = generate_executive_report()
    output_path = PROJECT_ROOT / "reports" / "executive_report.md"
    output_path.write_text(report, encoding="utf-8")
    logger.info("executive_report_written", path=str(output_path.relative_to(PROJECT_ROOT)))


if __name__ == "__main__":
    main()
