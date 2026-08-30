-- Expected Loss concentration by age band and risk band.
SELECT
    f.age_band,
    p.risk_band,
    COUNT(*)                       AS n_customers,
    ROUND(SUM(p.ead), 2)           AS total_exposure,
    ROUND(SUM(p.expected_loss), 2) AS total_expected_loss
FROM v_portfolio p
JOIN v_features f USING (customer_id)
GROUP BY 1, 2
ORDER BY total_expected_loss DESC;
