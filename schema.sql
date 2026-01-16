-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- System State (Global key-value store)
CREATE TABLE IF NOT EXISTS system_state (
    key TEXT PRIMARY KEY,
    value_json TEXT, -- JSON/JSONB content
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Agent State (Persistence for individual agents)
CREATE TABLE IF NOT EXISTS agent_state (
    agent_name TEXT PRIMARY KEY,
    state_json TEXT,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Task History (Execution log with semantic search)
CREATE TABLE IF NOT EXISTS task_history (
    task_id TEXT PRIMARY KEY,
    agent_name TEXT,
    task_description TEXT,
    outcome TEXT, -- 'completed', 'failed', 'skipped'
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    execution_time_seconds FLOAT,
    error_message TEXT,
    metadata_json TEXT,
    task_embedding vector(1024) -- Compatible with mxbai-embed-large dimension (usually 1024 or 768, checking model...)
);
-- Note: mxbai-embed-large is 1024 dim. 
-- Creating index for fast semantic search
CREATE INDEX IF NOT EXISTS task_history_embedding_idx ON task_history USING ivfflat (task_embedding vector_cosine_ops) WITH (lists = 100);


-- Mission History (High-level goals)
CREATE TABLE IF NOT EXISTS mission_history (
    mission_id TEXT PRIMARY KEY,
    mission_name TEXT,
    status TEXT,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    steps_total INTEGER,
    steps_completed INTEGER,
    metadata_json TEXT
);

-- Tool Usage (Analytics)
CREATE TABLE IF NOT EXISTS tool_usage (
    id SERIAL PRIMARY KEY,
    tool_name TEXT,
    success BOOLEAN,
    execution_time_seconds FLOAT,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    error_message TEXT
);

-- Metrics (Already handling in code, but good to have DDL)
CREATE TABLE IF NOT EXISTS metrics (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metric_type VARCHAR(50) NOT NULL,
    agent_name VARCHAR(100),
    tool_name VARCHAR(100),
    mission_type VARCHAR(50),
    value FLOAT,
    metadata JSONB
);

CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_metrics_type ON metrics(metric_type);
