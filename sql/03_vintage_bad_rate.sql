-- Vintage curves: bad rate (61+ DPD) by origination cohort and months-on-book,
-- restricted to cohorts with at least 30 observed accounts to avoid noisy
-- single-account legacy cohorts dominating the result.
WITH panel AS (
    SELECT
        mp.customer_id,
        DATE_TRUNC('month', c.origination_date) AS origination_cohort,
        mp.mob,
        mp.delinquency_bucket IN ('61-90', '90+') AS is_bad
    FROM v_monthly_performance mp
    JOIN v_customers c USING (customer_id)
),
cohort_mob AS (
    SELECT
        origination_cohort,
        mob,
        COUNT(*)                         AS n_accounts,
        SUM(is_bad::INT)                 AS n_bad
    FROM panel
    GROUP BY 1, 2
),
material_cohorts AS (
    SELECT origination_cohort
    FROM cohort_mob
    GROUP BY origination_cohort
    HAVING MAX(n_accounts) >= 30
)
SELECT
    cm.origination_cohort,
    cm.mob,
    cm.n_accounts,
    ROUND(cm.n_bad::DOUBLE / cm.n_accounts, 4) AS bad_rate
FROM cohort_mob cm
JOIN material_cohorts USING (origination_cohort)
ORDER BY 1, 2;
