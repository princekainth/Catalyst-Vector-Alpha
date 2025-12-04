-- CVA PostgreSQL Schema with improvements

-- Agent state with proper indexing
CREATE TABLE agent_state (
    agent_name TEXT PRIMARY KEY,
    state_json JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_agent_state_updated ON agent_state(updated_at DESC);

-- Task history with foreign key and indexing
CREATE TABLE task_history (
    id SERIAL PRIMARY KEY,
    task_id TEXT UNIQUE NOT NULL,
    agent_name TEXT REFERENCES agent_state(agent_name) ON DELETE CASCADE,
    task_description TEXT,
    outcome TEXT,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    execution_time_seconds REAL,
    error_message TEXT,
    metadata_json JSONB
);
CREATE INDEX idx_task_agent ON task_history(agent_name);
CREATE INDEX idx_task_started ON task_history(started_at DESC);
CREATE INDEX idx_task_outcome ON task_history(outcome) WHERE outcome IS NOT NULL;

-- Mission history with proper types
CREATE TABLE mission_history (
    id SERIAL PRIMARY KEY,
    mission_id TEXT UNIQUE NOT NULL,
    mission_name TEXT,
    status TEXT,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    steps_total INTEGER,
    steps_completed INTEGER,
    metadata_json JSONB
);
CREATE INDEX idx_mission_status ON mission_history(status);
CREATE INDEX idx_mission_started ON mission_history(started_at DESC);

-- System state with JSONB
CREATE TABLE system_state (
    key TEXT PRIMARY KEY,
    value_json JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Tool usage with better analytics support
CREATE TABLE tool_usage (
    id SERIAL PRIMARY KEY,
    tool_name TEXT NOT NULL,
    success BOOLEAN NOT NULL,
    execution_time_seconds REAL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    error_message TEXT
);
CREATE INDEX idx_tool_name ON tool_usage(tool_name);
CREATE INDEX idx_tool_timestamp ON tool_usage(timestamp DESC);
CREATE INDEX idx_tool_success ON tool_usage(success);

-- Dynamic agents with lifecycle tracking
CREATE TABLE dynamic_agents (
    agent_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    purpose TEXT,
    specialized_prompt TEXT,
    tools_json JSONB,
    ttl_hours REAL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ,
    parent_agent TEXT,
    status TEXT NOT NULL DEFAULT 'active',
    tasks_completed INTEGER DEFAULT 0,
    terminated_at TIMESTAMPTZ
);
CREATE INDEX idx_dynamic_status ON dynamic_agents(status);
CREATE INDEX idx_dynamic_expires ON dynamic_agents(expires_at) WHERE expires_at IS NOT NULL;
CREATE INDEX idx_dynamic_parent ON dynamic_agents(parent_agent) WHERE parent_agent IS NOT NULL;
