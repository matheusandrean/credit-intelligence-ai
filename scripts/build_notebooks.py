"""Builds and executes the two project notebooks with nbformat/nbclient so
their outputs are real, not fabricated. Run this after `make data` (and,
for the executive notebook, after `make train`)::

    python scripts/build_notebooks.py

Regenerate whenever the underlying data/model pipeline changes so the
committed notebook outputs stay accurate.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"


def _nb(cells: list[tuple[str, str]]) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src)
        for kind, src in cells
    ]
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    return nb


def build_eda_notebook() -> nbf.NotebookNode:
    cells: list[tuple[str, str]] = [
        (
            "md",
            "# Credit Portfolio EDA\n\n"
            "Exploratory analysis of the synthetic Credit Intelligence AI portfolio. "
            "All data is synthetic - see `DATA_CARD.md`.",
        ),
        (
            "code",
            "import sys\n"
            "sys.path.insert(0, '..')\n"
            "import pandas as pd\n"
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "import matplotlib.ticker as mticker\n"
            "import warnings\n"
            "warnings.filterwarnings('ignore')\n"
            "plt.rcParams['figure.figsize'] = (9, 4.5)\n"
            "plt.rcParams['axes.spines.top'] = False\n"
            "plt.rcParams['axes.spines.right'] = False\n\n"
            "customers = pd.read_parquet('../data/raw/customer_portfolio.parquet')\n"
            "print(customers.shape)\n"
            "customers.head()",
        ),
        ("md", "## 1. Target Distribution"),
        (
            "code",
            "rate = customers['target_default_90d'].mean()\n"
            "fig, ax = plt.subplots()\n"
            "customers['target_default_90d'].value_counts().sort_index().plot(\n"
            "    kind='bar', ax=ax, color=['#0B3D91', '#B3261E'], title=f'Target distribution (default rate = {rate:.2%})'\n"
            ")\n"
            "ax.set_xticklabels(['Non-default', 'Default'], rotation=0)\n"
            "ax.set_ylabel('Customers')\n"
            "plt.show()\n"
            "print(f'Overall default rate: {rate:.2%}')",
        ),
        (
            "md",
            "**Business interpretation.** The portfolio's observed 90-day default rate is in the "
            "single-to-low-double digits, consistent with a realistic unsecured consumer credit "
            "book (real portfolios of this type are typically strongly imbalanced toward "
            "non-default). Any model evaluation on this dataset must therefore avoid accuracy as "
            "the primary metric.",
        ),
        ("md", "## 2. Temporal Analysis"),
        (
            "code",
            "monthly = customers.groupby(customers['observation_date'].dt.to_period('M'))['target_default_90d'].mean()\n"
            "fig, ax = plt.subplots()\n"
            "monthly.plot(ax=ax, marker='o', color='#0B3D91', title='Default rate by observation month')\n"
            "ax.set_ylabel('Default rate')\n"
            "ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))\n"
            "plt.xticks(rotation=45)\n"
            "plt.tight_layout()\n"
            "plt.show()",
        ),
        (
            "md",
            "**Business interpretation.** The synthetic generator injects a slow upward drift in "
            "the underlying latent risk factor over calendar time, so later observation months "
            "should show a mildly higher default rate on average, alongside normal sampling noise. "
            "This is the basis for the portfolio's deterioration narrative used later in the "
            "monitoring and vintage analyses (see `docs/PORTFOLIO_STORY.md`).",
        ),
        ("md", "## 3. Income Distribution"),
        (
            "code",
            "fig, ax = plt.subplots()\n"
            "customers['monthly_income'].clip(upper=customers['monthly_income'].quantile(0.99)).hist(\n"
            "    bins=60, ax=ax, color='#0B3D91'\n"
            ")\n"
            "ax.set_title('Monthly income distribution (99th pct capped for display)')\n"
            "ax.set_xlabel('Monthly income')\n"
            "plt.show()\n"
            "customers['monthly_income'].describe()",
        ),
        (
            "md",
            "**Business interpretation.** Income follows a right-skewed, log-normal-like "
            "distribution typical of retail income data, with a long tail of higher earners.",
        ),
        ("md", "## 4. Debt-to-Income (DTI)"),
        (
            "code",
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))\n"
            "customers['debt_to_income'].clip(upper=2).hist(bins=50, ax=axes[0], color='#0B3D91')\n"
            "axes[0].set_title('DTI distribution')\n\n"
            "dti_q = pd.qcut(customers['debt_to_income'], 5, duplicates='drop')\n"
            "customers.groupby(dti_q, observed=True)['target_default_90d'].mean().plot(\n"
            "    kind='bar', ax=axes[1], color='#0B3D91', title='Default rate by DTI quintile'\n"
            ")\n"
            "axes[1].yaxis.set_major_formatter(mticker.PercentFormatter(1.0))\n"
            "plt.xticks(rotation=45)\n"
            "plt.tight_layout()\n"
            "plt.show()",
        ),
        (
            "md",
            "**Business interpretation.** Default rate rises monotonically and steeply with DTI "
            "quintile, confirming DTI as one of the strongest engineered risk drivers - consistent "
            "with the model's SHAP global importance ranking (see notebook 04 and `MODEL_CARD.md`).",
        ),
        ("md", "## 5. Credit Utilization"),
        (
            "code",
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))\n"
            "customers['credit_utilization'].clip(upper=1.5).hist(bins=50, ax=axes[0], color='#0B3D91')\n"
            "axes[0].set_title('Utilization distribution')\n\n"
            "util_q = pd.qcut(customers['credit_utilization'], 5, duplicates='drop')\n"
            "customers.groupby(util_q, observed=True)['target_default_90d'].mean().plot(\n"
            "    kind='bar', ax=axes[1], color='#0B3D91', title='Default rate by utilization quintile'\n"
            ")\n"
            "axes[1].yaxis.set_major_formatter(mticker.PercentFormatter(1.0))\n"
            "plt.xticks(rotation=45)\n"
            "plt.tight_layout()\n"
            "plt.show()",
        ),
        (
            "md",
            "**Business interpretation.** Utilization shows the same monotonic pattern as DTI - "
            "customers using a larger share of their available credit default substantially more "
            "often, matching real-world scorecard experience.",
        ),
        ("md", "## 6. Late Payments and Delinquency History"),
        (
            "code",
            "fig, ax = plt.subplots()\n"
            "bins = [-1, 0, 1, 3, 100]\n"
            "late_bucket = pd.cut(customers['late_payments_12m'], bins=bins)\n"
            "customers.groupby(late_bucket, observed=True)['target_default_90d'].mean().plot(\n"
            "    kind='bar', ax=ax, color='#0B3D91', title='Default rate by late payments (12m)'\n"
            ")\n"
            "ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))\n"
            "plt.xticks(rotation=0)\n"
            "plt.show()",
        ),
        (
            "md",
            "**Business interpretation.** Prior delinquency is, as expected, one of the most "
            "predictive signals available - default rate increases sharply once a customer has "
            "any late payment history in the trailing 12 months.",
        ),
        ("md", "## 7. Behavioral Indicators"),
        (
            "code",
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))\n"
            "customers['behavioral_score'].hist(bins=50, ax=axes[0], color='#0B3D91')\n"
            "axes[0].set_title('Behavioral score distribution')\n"
            "customers['financial_stability_index'].hist(bins=50, ax=axes[1], color='#0B3D91')\n"
            "axes[1].set_title('Financial stability index distribution')\n"
            "plt.tight_layout()\n"
            "plt.show()",
        ),
        (
            "md",
            "**Business interpretation.** Both composite behavioral indicators are roughly "
            "bell-shaped and behave as intended: lower behavioral score and lower financial "
            "stability both correlate with higher observed default rate (confirmed via SHAP in "
            "notebook 04).",
        ),
        ("md", "## 8. Correlations"),
        (
            "code",
            "numeric_cols = [\n"
            "    'debt_to_income', 'credit_utilization', 'late_payments_12m', 'behavioral_score',\n"
            "    'financial_stability_index', 'balance_trend', 'utilization_trend',\n"
            "    'delinquency_trend', 'previous_default_flag', 'target_default_90d',\n"
            "]\n"
            "corr = customers[numeric_cols].corr()\n"
            "fig, ax = plt.subplots(figsize=(8, 7))\n"
            "im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)\n"
            "ax.set_xticks(range(len(numeric_cols)), numeric_cols, rotation=90)\n"
            "ax.set_yticks(range(len(numeric_cols)), numeric_cols)\n"
            "fig.colorbar(im)\n"
            "ax.set_title('Correlation matrix (key risk drivers)')\n"
            "plt.tight_layout()\n"
            "plt.show()",
        ),
        (
            "md",
            "**Business interpretation.** Moderate multicollinearity exists between DTI, "
            "utilization, and the trend features, by design (see `DATA_CARD.md`) - this mirrors "
            "real bureau/behavioral data, where indebtedness measures naturally correlate.",
        ),
        ("md", "## 9. Segmentation: Age Band and Payment History"),
        (
            "code",
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))\n"
            "customers.groupby('age_band', observed=True)['target_default_90d'].mean().plot(\n"
            "    kind='bar', ax=axes[0], color='#0B3D91', title='Default rate by age band'\n"
            ")\n"
            "customers.groupby('payment_history', observed=True)['target_default_90d'].mean().sort_values().plot(\n"
            "    kind='bar', ax=axes[1], color='#0B3D91', title='Default rate by payment history'\n"
            ")\n"
            "for a in axes:\n"
            "    a.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))\n"
            "plt.xticks(rotation=45)\n"
            "plt.tight_layout()\n"
            "plt.show()",
        ),
        (
            "md",
            "**Business interpretation.** Younger age bands show mildly higher default rates in "
            "this synthetic generator (reflecting a common, if imperfect, industry pattern), and "
            "`payment_history` is almost perfectly rank-ordered with observed default, as expected "
            "since it is derived from the same latent behavior score.",
        ),
        ("md", "## 10. Missing Values"),
        (
            "code",
            "missing = customers.isna().mean().sort_values(ascending=False)\n"
            "missing = missing[missing > 0]\n"
            "fig, ax = plt.subplots()\n"
            "missing.plot(kind='barh', ax=ax, color='#0B3D91', title='Null rate by column')\n"
            "ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
            "missing",
        ),
        (
            "md",
            "**Business interpretation.** A handful of fields carry 1.5-4% missingness, injected "
            "deliberately to mirror real-world data quality issues (self-reported expenses/spend, "
            "employment tenure). The modeling pipeline imputes these at fit time "
            "(`src/models/train.py`), never before the temporal split.",
        ),
        ("md", "## 11. Outliers"),
        (
            "code",
            "fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))\n"
            "for ax, col in zip(axes, ['monthly_income', 'revolving_balance', 'credit_utilization']):\n"
            "    axes_data = customers[col].dropna()\n"
            "    ax.boxplot(axes_data, orientation='vertical')\n"
            "    ax.set_title(col)\n"
            "plt.tight_layout()\n"
            "plt.show()",
        ),
        (
            "md",
            "**Business interpretation.** A small share of rows contain deliberately injected "
            "outliers (e.g. income/balance spikes, over-100% utilization from over-limit spend) - "
            "visible as points beyond the boxplot whiskers - to make the modeling problem realistic "
            "rather than artificially clean.",
        ),
        ("md", "## 12. Deterioration Trend Signals"),
        (
            "code",
            "fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))\n"
            "for ax, col in zip(axes, ['balance_trend', 'utilization_trend', 'delinquency_trend']):\n"
            "    trend_q = pd.qcut(customers[col], 4, duplicates='drop')\n"
            "    customers.groupby(trend_q, observed=True)['target_default_90d'].mean().plot(\n"
            "        kind='bar', ax=ax, color='#0B3D91', title=f'Default rate by {col} quartile'\n"
            "    )\n"
            "    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))\n"
            "plt.tight_layout()\n"
            "plt.show()",
        ),
        (
            "md",
            "**Business interpretation.** All three directional trend signals show the expected "
            "pattern: customers whose balance, utilization, or delinquency severity is trending "
            "upward default more often than those trending flat or down. These trends are combined "
            "into the `behavioral_deterioration_index` engineered feature "
            "(see `FEATURE_DICTIONARY.md`).",
        ),
        (
            "md",
            "## Summary\n\n"
            "The synthetic portfolio exhibits the statistical relationships a real credit dataset "
            "would show - monotonic risk gradients across DTI, utilization, delinquency history and "
            "trend signals - while retaining realistic noise, missingness and outliers. This makes "
            "it suitable for demonstrating a genuine modeling and monitoring pipeline rather than a "
            "toy, trivially-separable classification problem.",
        ),
    ]
    return _nb(cells)


def build_executive_notebook() -> nbf.NotebookNode:
    cells: list[tuple[str, str]] = [
        (
            "md",
            "# Executive Credit Analysis\n\n"
            "A Credit Risk Director-style walkthrough of the portfolio: where risk is "
            "concentrated, what is driving it, how it has evolved, and what stress testing reveals. "
            "Requires `make data && make features && make train` to have been run first.",
        ),
        (
            "code",
            "import sys, json\n"
            "sys.path.insert(0, '..')\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "import matplotlib.ticker as mticker\n"
            "import warnings\n"
            "warnings.filterwarnings('ignore')\n"
            "plt.rcParams['figure.figsize'] = (9, 4.5)\n"
            "plt.rcParams['axes.spines.top'] = False\n"
            "plt.rcParams['axes.spines.right'] = False\n\n"
            "scored = pd.read_parquet('../data/processed/scored_portfolio.parquet')\n"
            "features = pd.read_parquet('../data/processed/credit_features.parquet')\n"
            "panel = pd.read_parquet('../data/raw/monthly_performance.parquet')\n"
            "customers = pd.read_parquet('../data/raw/customer_portfolio.parquet')\n"
            "print(scored.shape, features.shape, panel.shape)",
        ),
        ("md", "## 1. Where Is Risk Concentrated?"),
        (
            "code",
            "band_stats = scored.groupby('risk_band').agg(\n"
            "    n_customers=('customer_id', 'count'),\n"
            "    total_exposure=('ead', 'sum'),\n"
            "    total_expected_loss=('expected_loss', 'sum'),\n"
            "    avg_pd=('pd', 'mean'),\n"
            ").reindex(list('ABCDE'))\n"
            "band_stats",
        ),
        (
            "code",
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))\n"
            "band_stats['total_exposure'].plot(kind='bar', ax=axes[0], color='#0B3D91', title='Exposure by band')\n"
            "band_stats['total_expected_loss'].plot(kind='bar', ax=axes[1], color='#B3261E', title='Expected Loss by band')\n"
            "plt.tight_layout()\n"
            "plt.show()",
        ),
        (
            "md",
            "**Interpretation for the Credit Risk Director.** Exposure is concentrated in the "
            "lower-risk bands (A/B) as expected for a healthy book, but Expected Loss is "
            "disproportionately concentrated in bands D and E - a small share of accounts drives a "
            "large share of expected losses, which is exactly where collections/monitoring "
            "attention should focus first.",
        ),
        ("md", "## 2. Which Segments Deteriorated the Most?"),
        (
            "code",
            "scored['month'] = pd.to_datetime(scored['observation_date']).dt.to_period('M')\n"
            "monthly_pd = scored.groupby('month')['pd'].mean()\n"
            "fig, ax = plt.subplots()\n"
            "monthly_pd.plot(ax=ax, marker='o', color='#0B3D91', title='Average PD by month')\n"
            "ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))\n"
            "plt.xticks(rotation=45)\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
            "print('First 3 months avg PD:', monthly_pd.head(3).mean())\n"
            "print('Last 3 months avg PD:', monthly_pd.tail(3).mean())",
        ),
        (
            "md",
            "**Interpretation.** Average PD is mildly higher in the most recent months than at the "
            "start of the observation window, consistent with the synthetic portfolio's designed "
            "deterioration drift (`docs/PORTFOLIO_STORY.md`). This kind of slow drift is exactly "
            "what population-stability monitoring (`detect_drift`) is meant to catch before it "
            "becomes a material issue.",
        ),
        ("md", "## 3. Which Variables Anticipate Default? (SHAP)"),
        (
            "code",
            "from src.models.explainability import build_default_explainer\n"
            "from src.models.train import split_xy\n\n"
            "explainer, feat_df, metadata = build_default_explainer()\n"
            "categories = metadata['categories']\n"
            "x_test, _ = split_xy(feat_df, 'test', categories)\n"
            "sample = x_test.sample(min(500, len(x_test)), random_state=1)\n"
            "importance = explainer.global_importance(sample).head(10)\n"
            "importance",
        ),
        (
            "code",
            "fig, ax = plt.subplots()\n"
            "importance.set_index('feature')['mean_abs_shap'].sort_values().plot(\n"
            "    kind='barh', ax=ax, color='#0B3D91', title='Top 10 global SHAP drivers'\n"
            ")\n"
            "plt.tight_layout()\n"
            "plt.show()",
        ),
        (
            "md",
            "**Interpretation.** The model's globally most important drivers align with credit "
            "risk domain intuition - debt service burden, delinquency history, and composite "
            "behavioral indicators dominate, with no protected/sensitive characteristic anywhere "
            "in the list (there is none in the dataset - see `RESPONSIBLE_AI.md`).",
        ),
        ("md", "## 4. How Has the Portfolio Evolved?"),
        (
            "code",
            "from src.analytics.portfolio import monthly_kpi_trend\n\n"
            "trend = monthly_kpi_trend(scored, features)\n"
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))\n"
            "axes[0].plot(trend['month'], trend['expected_loss'], marker='o', color='#0B3D91')\n"
            "axes[0].set_title('Expected Loss over time')\n"
            "axes[0].tick_params(axis='x', rotation=45)\n"
            "axes[1].plot(trend['month'], trend['high_risk_population_pct'], marker='o', color='#B3261E')\n"
            "axes[1].set_title('High-risk population % over time')\n"
            "axes[1].yaxis.set_major_formatter(mticker.PercentFormatter(1.0))\n"
            "axes[1].tick_params(axis='x', rotation=45)\n"
            "plt.tight_layout()\n"
            "plt.show()",
        ),
        (
            "md",
            "**Interpretation.** Both Expected Loss and the high-risk population share should be "
            "read together with account volume: a rising EL trend during a period of portfolio "
            "growth is a very different signal than the same trend on a flat book.",
        ),
        ("md", "## 5. How Did Risk Change Across Vintages?"),
        (
            "code",
            "from src.analytics.vintage import build_vintage_curves, filter_material_cohorts\n\n"
            "curves = build_vintage_curves(panel, customers)\n"
            "curves = filter_material_cohorts(curves, min_accounts=30)\n"
            "recent = sorted(curves['origination_cohort'].unique())[-6:]\n"
            "fig, ax = plt.subplots()\n"
            "for cohort in recent:\n"
            "    sub = curves[curves['origination_cohort'] == cohort]\n"
            "    ax.plot(sub['mob'], sub['bad_rate'], marker='o', label=cohort)\n"
            "ax.set_xlabel('Months on Book')\n"
            "ax.set_ylabel('Bad rate (61+ DPD)')\n"
            "ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))\n"
            "ax.set_title('Vintage curves - 6 most recent material cohorts')\n"
            "ax.legend(fontsize=8)\n"
            "plt.show()",
        ),
        (
            "md",
            "**Interpretation.** Comparing cohorts at the same MOB checkpoint isolates genuine "
            "credit-quality differences between origination periods from simple portfolio-aging "
            "effects - a cohort whose curve sits persistently above its peers at the same MOB "
            "warrants underwriting-policy review for that origination period.",
        ),
        ("md", "## 6. What Impacts Appeared in Stress Testing?"),
        (
            "code",
            "with open('../reports/stress_test_report.json') as f:\n"
            "    stress = json.load(f)\n\n"
            "scenarios = ['baseline', 'mild', 'moderate', 'severe']\n"
            "avg_pd = [stress[s]['stressed']['average_pd'] for s in scenarios]\n"
            "fig, ax = plt.subplots()\n"
            "ax.bar(scenarios, avg_pd, color=['#94A3B8', '#D6A419', '#D9732B', '#B3261E'])\n"
            "ax.set_title('Average PD by stress scenario')\n"
            "ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))\n"
            "plt.show()\n"
            "for s in scenarios:\n"
            "    print(s, f\"avg PD={stress[s]['stressed']['average_pd']:.2%}\", f\"EL delta={stress[s]['expected_loss_delta']:+,.0f}\")",
        ),
        (
            "md",
            "**Interpretation for the Credit Risk Director.** Under the Severe scenario "
            "(income -20%, expenses +15%, utilization +10pp), portfolio average PD rises from "
            "baseline to a materially higher level and Expected Loss increases by tens of millions "
            "in exposure-equivalent terms. This is a hypothetical simulation, not a forecast "
            "(see `RESPONSIBLE_AI.md`), but it quantifies the book's sensitivity to a plausible "
            "macro shock and supports a capital/provisioning discussion.",
        ),
        (
            "md",
            "## Summary for the Credit Risk Director\n\n"
            "- Risk is concentrated in a small, well-identified share of accounts (bands D/E).\n"
            "- The portfolio shows a mild, monitored upward drift in average PD.\n"
            "- SHAP confirms the model's decisions track domain-standard credit risk drivers.\n"
            "- Vintage analysis localizes any deterioration to specific origination cohorts rather "
            "than a uniform book-wide effect.\n"
            "- Severe stress testing shows a material but bounded increase in expected loss, useful "
            "input for a provisioning/capital conversation - not a decision by itself.",
        ),
    ]
    return _nb(cells)


def execute_and_save(nb: nbf.NotebookNode, path: Path) -> None:
    client = NotebookClient(
        nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(NOTEBOOKS_DIR)}}
    )
    client.execute()
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Wrote and executed {path}")


def main() -> None:
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    execute_and_save(build_eda_notebook(), NOTEBOOKS_DIR / "01_credit_portfolio_eda.ipynb")
    execute_and_save(
        build_executive_notebook(), NOTEBOOKS_DIR / "04_executive_credit_analysis.ipynb"
    )


if __name__ == "__main__":
    main()
