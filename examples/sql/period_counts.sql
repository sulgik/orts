-- One row per (period, arm): the users first exposed in that period and how many of
-- them converted within the conversion window.  This is the table OR-TS reads:
--
--     orts.allocate_from_rows(rows, seed=0)
--
-- Written for SQLite and run by tests/test_history.py.  Other warehouses differ only in
-- the three marked places; those variants are not tested here.
--
-- Parameters (named, as in sqlite3):
--     :experiment_id   the experiment to read
--     :as_of           the time the job runs, as 'YYYY-MM-DD HH:MM:SS'
--
-- Tables:
--     exposures(experiment_id, user_id, variation, exposed_at)
--     conversions(user_id, converted_at)
--
-- Choices that matter statistically:
--   * Every user is counted once, in the period of their first exposure, and a user
--     seen in two variations is dropped.  Counting a user in several periods or arms
--     would make the counts depend on how long the experiment has run.
--   * The conversion window is fixed (7 days here).  A user whose window has not closed
--     is left out, so a recent period is not under-counted.  The filter depends only on
--     when the user was exposed, never on what they did, so it removes users without
--     biasing the rates.
--   * Counts are per period, not cumulative.  Do not sum periods before this step.
WITH first_exposure AS (
    SELECT
        user_id,
        MIN(variation)   AS variation,
        MIN(exposed_at)  AS exposed_at
    FROM exposures
    WHERE experiment_id = :experiment_id
    GROUP BY user_id
    HAVING COUNT(DISTINCT variation) = 1
),
closed AS (
    SELECT *
    FROM first_exposure
    -- window end:  SQLite datetime(t, '+7 days')
    --              BigQuery TIMESTAMP_ADD(t, INTERVAL 7 DAY)
    --              Snowflake DATEADD(day, 7, t)      Postgres t + INTERVAL '7 days'
    WHERE datetime(exposed_at, '+7 days') <= :as_of
)
SELECT
    -- period:  SQLite date(t) or strftime('%Y-%m-%d %H:00', t) for hourly
    --          BigQuery DATE(t) or TIMESTAMP_TRUNC(t, HOUR)
    --          Snowflake / Postgres DATE_TRUNC('day', t)
    date(e.exposed_at)   AS period,
    e.variation          AS arm,
    COUNT(*)             AS exposures,
    SUM(CASE WHEN EXISTS (
            SELECT 1
            FROM conversions c
            WHERE c.user_id = e.user_id
              AND c.converted_at >= e.exposed_at
              AND c.converted_at <  datetime(e.exposed_at, '+7 days')
        ) THEN 1 ELSE 0 END) AS events
FROM closed e
GROUP BY 1, 2
ORDER BY 1, 2;
