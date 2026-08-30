-- Deterioration watchlist per portfolio_monitoring_policy.md: customers
-- with both a positive delinquency trend and a positive utilization trend
-- above their portfolio's 75th percentile, currently in bands C-E.
WITH thresholds AS (
    SELECT
        QUANTILE_CONT(delinquency_trend, 0.75) AS delinquency_p75,
        QUANTILE_CONT(utilization_trend, 0.75) AS utilization_p75
    FROM v_features
)
SELECT
    p.customer_id,
    p.risk_band,
    p.pd,
    f.delinquency_trend,
    f.utilization_trend,
    p.expected_loss
FROM v_portfolio p
JOIN v_features f USING (customer_id)
CROSS JOIN thresholds t
WHERE f.delinquency_trend > t.delinquency_p75
  AND f.utilization_trend > t.utilization_p75
  AND p.risk_band IN ('C', 'D', 'E')
ORDER BY p.expected_loss DESC
LIMIT 100;
