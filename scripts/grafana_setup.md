# Grafana + PostgreSQL Quick Setup (CVA Metrics)

This guide wires CVA's metrics table into Grafana for rich dashboards. Assumes PostgreSQL already running with the `metrics` table.

## 1) Create a read-only DB user
```sql
-- Connect as a superuser or the DB owner
CREATE USER grafana_user WITH PASSWORD 'grafana_pass';
GRANT CONNECT ON DATABASE cva_db TO grafana_user;
GRANT USAGE ON SCHEMA public TO grafana_user;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO grafana_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO grafana_user;
```

## 2) Useful views (optional but nice)
```sql
-- Average agent latency by agent (last 24h)
CREATE OR REPLACE VIEW v_agent_latency_24h AS
SELECT agent_name, AVG(value) AS avg_latency_s
FROM metrics
WHERE metric_type = 'agent_execution_time'
  AND timestamp > NOW() - INTERVAL '24 hours'
GROUP BY agent_name;

-- Tool success rate (last 24h) based on tool_execution metrics metadata.status
CREATE OR REPLACE VIEW v_tool_success_rate_24h AS
SELECT tool_name,
       AVG(CASE WHEN (metadata ->> 'status') = 'success' THEN 1 ELSE 0 END) AS success_rate
FROM metrics
WHERE metric_type = 'tool_execution'
  AND timestamp > NOW() - INTERVAL '24 hours'
GROUP BY tool_name;

-- Circuit breaker trips per hour (last 24h)
CREATE OR REPLACE VIEW v_breaker_trips_hourly AS
SELECT date_trunc('hour', timestamp) AS hour, COUNT(*) AS trips
FROM metrics
WHERE metric_type = 'circuit_breaker_trip'
  AND timestamp > NOW() - INTERVAL '24 hours'
GROUP BY hour
ORDER BY hour;
```

## 3) Add data source in Grafana
- URL: your Postgres host/port
- Database: `cva_db`
- User: `grafana_user`
- Password: `grafana_pass`
- TLS: as appropriate
- Save & Test

## 4) Starter panels/queries
1) **Agent Latency (ms)**  
   ```sql
   SELECT $__time(timestamp), agent_name, value*1000 AS latency_ms
   FROM metrics
   WHERE metric_type='agent_execution_time'
     AND $__timeFilter(timestamp)
   ```
2) **Tool Success Rate (last 24h)**  
   ```sql
   SELECT tool_name, success_rate*100 AS success_pct
   FROM v_tool_success_rate_24h
   ORDER BY success_pct DESC;
   ```
3) **Breaker Trips (hourly)**  
   ```sql
   SELECT hour AS time, trips
   FROM v_breaker_trips_hourly
   ```
4) **Agent Throughput** (calls per hour)  
   ```sql
   SELECT date_trunc('hour', timestamp) AS time, agent_name, COUNT(*) AS calls
   FROM metrics
   WHERE metric_type='agent_execution_time'
     AND $__timeFilter(timestamp)
   GROUP BY time, agent_name
   ORDER BY time
   ```
5) **Cost Tracking** (if/when you log costs)  
   ```sql
   SELECT $__time(timestamp), COALESCE((metadata->>'cost_usd')::numeric,0) AS cost_usd
   FROM metrics
   WHERE metric_type='llm_cost'
     AND $__timeFilter(timestamp)
   ```

## 5) (Optional) Add Prometheus or Loki later
If you also emit Prometheus or Loki logs, add those as data sources for CPU/memory and structured logs. But the Postgres metrics are enough for agent/tool health.

## 6) Run Grafana
If you don’t have Grafana, simplest is Docker:
```bash
docker run -d -p 3000:3000 --name grafana grafana/grafana
# Login: admin / admin (then change)
```
