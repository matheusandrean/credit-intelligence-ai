-- Month-over-month delinquency bucket transition counts (roll rate),
-- pooled across the whole observation panel. Normalize to probabilities
-- in application code (see src/analytics/roll_rate.py for the equivalent
-- pandas implementation used by the dashboard).
WITH ordered AS (
    SELECT
        customer_id,
        mob,
        delinquency_bucket,
        LEAD(delinquency_bucket) OVER (PARTITION BY customer_id ORDER BY mob) AS next_bucket,
        LEAD(mob) OVER (PARTITION BY customer_id ORDER BY mob)               AS next_mob
    FROM v_monthly_performance
)
SELECT
    delinquency_bucket AS from_bucket,
    next_bucket         AS to_bucket,
    COUNT(*)            AS n_transitions
FROM ordered
WHERE next_mob = mob + 1
GROUP BY 1, 2
ORDER BY 1, 2;
