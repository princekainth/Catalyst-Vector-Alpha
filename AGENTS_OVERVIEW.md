# CVA Agent Architecture Overview

## Request Flow: API → Planner → Execution → Memory

### 1. Entry Point: `/api/command` (app.py)
- Receives `{"command": "user task"}`
- Creates `AGENT_PERFORM_TASK` directive → `ProtoAgent_Planner_instance_1`
- Injects into `CatalystVectorAlpha.dynamic_directive_queue`
- Returns `task_id` for tracking

### 2. Cognitive Loop (catalyst_vector_alpha.py)
`run_cognitive_loop()` continuously:
- Pulls directives from queue
- Calls `planner.perform_task(task_description, cycle_id, context)`

### 3. Planner Decision (agents.py: ProtoAgent_Planner)
`_execute_agent_specific_task()`:
- Checks AlertStore (user alerts: email, calendar)
- Checks MemeticKernel (system alerts: K8s CPU)
- **LLM reasoning** analyzes alert → decides action
- Routes to: existing agent, spawn DynamicAgent, or handle directly

### 4. Agent Spawning (agent_factory.py)
`spawn_agent(purpose, context)` - Hybrid intelligence:
- **LLM Phase**: Designs agent (name, prompt, tool selection)
- **Self-Validation**: LLM critiques its choices
- **Semantic Matching**: Vector similarity finds missed tools (threshold=0.4)
- **Safety Rules**: Hard-coded filters (e.g., Gmail ≠ K8s tools)
- Returns `DynamicAgent` with 2-5 specialized tools

### 5. Tool Execution (tool_registry.py)
`safe_call(tool_name, timeout_seconds=30, **kwargs)`:
- **Timeout protection**: ThreadPoolExecutor wrapper (prevents hangs)
- Validates: cooldowns, parameters, schema
- Logs: execution time, success/failure
- Returns: result or `[ERROR] timeout/failed`

### 6. Memory System (shared_models.py: MemeticKernel)
**Three-tier storage**:
- **Short-term**: Deque (100 items) - raw memories
- **Compression**: LLM summarizes batch → embedding → ChromaDB
- **Long-term**: Vector search `query_long_term_memory(query, top_k=3)`

**Memory types**:
- `TaskOutcome`, `AdaptiveToolUse`, `SystemAlert`, `IntentAdaptation`

### 7. Persistence (persistence_data/)
- **SQLite** (`cva.db`): Agent states, task history
- **ChromaDB** (`.chromadb/`): Vector embeddings
- **JSONL logs**: Raw activity per agent

## Key Architecture Points
- **Hybrid Intelligence**: LLM + semantic matching + safety rules
- **Fail-Safe**: 30s timeouts, LLM failures raise exceptions
- **Learning**: Compressed memories enable pattern recognition
- **Specialization**: DynamicAgents spawn on-demand with minimal tools
