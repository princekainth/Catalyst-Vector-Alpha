#!/bin/bash

echo "=== DYNAMIC AGENTS ACTIVITY REPORT ==="
echo ""

sqlite3 -header -column persistence_data/cva.db << SQL
SELECT 
    name,
    purpose,
    status,
    tasks_completed,
    datetime(created_at) as created,
    datetime(expires_at) as expires
FROM dynamic_agents 
ORDER BY created_at DESC;
SQL

echo ""
echo "=== TOOLS USED BY EACH AGENT ==="
echo ""

sqlite3 persistence_data/cva.db << SQL
SELECT 
    agent_id,
    name,
    json_extract(tools_json, '$') as tools
FROM dynamic_agents 
ORDER BY created_at DESC;
SQL

echo ""
echo "=== TASK HISTORY ==="
echo ""

sqlite3 -header -column persistence_data/cva.db << SQL
SELECT 
    agent_name,
    task_type,
    outcome,
    datetime(timestamp) as when,
    substr(details, 1, 50) as details_preview
FROM task_history 
WHERE agent_name LIKE 'agent_%'
ORDER BY timestamp DESC 
LIMIT 20;
SQL

