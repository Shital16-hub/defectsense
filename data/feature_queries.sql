-- ═══════════════════════════════════════════════════════════════════════════
-- DefectSense Feature Queries
-- These show what the ML pipeline does in SQL for production scale
-- ═══════════════════════════════════════════════════════════════════════════


-- Query 1: Rolling 30-reading averages
-- This is what LSTM sequence builder does in Python, expressed in SQL for scale
SELECT
    machine_id,
    id,
    air_temperature,
    AVG(air_temperature) OVER (
        PARTITION BY machine_id
        ORDER BY id
        ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
    ) as rolling_avg_air_temp,
    AVG(process_temperature) OVER (
        PARTITION BY machine_id
        ORDER BY id
        ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
    ) as rolling_avg_process_temp,
    AVG(rotational_speed) OVER (
        PARTITION BY machine_id
        ORDER BY id
        ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
    ) as rolling_avg_rpm,
    AVG(torque) OVER (
        PARTITION BY machine_id
        ORDER BY id
        ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
    ) as rolling_avg_torque
FROM sensor_readings
ORDER BY machine_id, id;


-- Query 2: Failure rate by machine type
SELECT
    machine_type,
    COUNT(*) as total_readings,
    SUM(machine_failure) as total_failures,
    ROUND(
        100.0 * SUM(machine_failure) / COUNT(*),
        2
    ) as failure_rate_pct
FROM sensor_readings
GROUP BY machine_type
ORDER BY failure_rate_pct DESC;


-- Query 3: Sensor statistics — normal vs failure comparison
SELECT
    CASE WHEN machine_failure = 0
         THEN 'Normal'
         ELSE 'Failure'
    END as sample_type,
    COUNT(*) as count,
    ROUND(AVG(air_temperature)::numeric, 2)
        as avg_air_temp,
    ROUND(AVG(process_temperature)::numeric, 2)
        as avg_process_temp,
    ROUND(AVG(rotational_speed)::numeric, 2)
        as avg_rpm,
    ROUND(AVG(torque)::numeric, 2)
        as avg_torque,
    ROUND(AVG(tool_wear)::numeric, 2)
        as avg_tool_wear
FROM sensor_readings
GROUP BY machine_failure
ORDER BY machine_failure;


-- Query 4: Top 10 machines by failure count
SELECT
    machine_id,
    COUNT(*) as total_readings,
    SUM(machine_failure) as total_failures,
    ROUND(
        100.0 * SUM(machine_failure) / COUNT(*),
        2
    ) as failure_rate_pct,
    MAX(tool_wear) as max_tool_wear,
    AVG(torque) as avg_torque
FROM sensor_readings
GROUP BY machine_id
HAVING SUM(machine_failure) > 0
ORDER BY total_failures DESC
LIMIT 10;
