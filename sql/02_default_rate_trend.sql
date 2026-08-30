-- Observed default rate and average PD by observation month (MoM trend).
SELECT
    DATE_TRUNC('month', observation_date)      AS observation_month,
    COUNT(*)                                   AS n_customers,
    ROUND(AVG(pd), 4)                          AS avg_pd,
    ROUND(AVG(target_default_90d), 4)          AS observed_default_rate
FROM v_portfolio
GROUP BY 1
ORDER BY 1;
