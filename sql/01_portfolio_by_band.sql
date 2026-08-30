-- Portfolio composition and exposure by risk band.
SELECT
    risk_band,
    COUNT(*)                                   AS n_customers,
    ROUND(AVG(pd), 4)                          AS avg_pd,
    ROUND(SUM(ead), 2)                         AS total_exposure,
    ROUND(SUM(expected_loss), 2)               AS total_expected_loss,
    ROUND(SUM(expected_loss) / NULLIF(SUM(ead), 0), 4) AS expected_loss_rate
FROM v_portfolio
GROUP BY risk_band
ORDER BY risk_band;
