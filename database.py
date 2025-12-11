# database_postgres.py - PostgreSQL persistence for CVA
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import logging
from db_postgres import get_db_connection, execute_query

logger = logging.getLogger("CatalystLogger")

class CVADatabase:
    """PostgreSQL database for CVA state persistence."""
    
    def __init__(self, db_path: str = None):
        """Initialize database (db_path ignored, kept for compatibility)."""
        # Test connection
        from db_postgres import health_check
        if not health_check():
            raise ConnectionError("Cannot connect to PostgreSQL database")
        logger.info("PostgreSQL database initialized")
        # Ensure metrics table exists
        self.create_metrics_table()
    
    # ============== AGENT STATE ==============
    def save_agent_state(self, agent_name: str, state: Dict[str, Any]) -> None:
        """Save agent state to database."""
        now = datetime.now(timezone.utc)
        state_json = json.dumps(state, default=str)
        
        execute_query('''
            INSERT INTO agent_state (agent_name, state_json, updated_at)
            VALUES (%s, %s, %s)
            ON CONFLICT (agent_name) 
            DO UPDATE SET state_json = EXCLUDED.state_json, updated_at = EXCLUDED.updated_at
        ''', (agent_name, state_json, now))
    
    def load_agent_state(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Load agent state from database."""
        rows = execute_query(
            'SELECT state_json FROM agent_state WHERE agent_name = %s',
            (agent_name,),
            fetch=True
        )
        
        if rows:
            return json.loads(rows[0]['state_json'])
        return None
    
    def load_all_agent_states(self) -> Dict[str, Dict[str, Any]]:
        """Load all agent states."""
        rows = execute_query(
            'SELECT agent_name, state_json FROM agent_state',
            fetch=True
        )
        
        return {row['agent_name']: json.loads(row['state_json']) for row in rows}
    
    def record_mission_execution(self, mission_name: str, success: bool, 
                                  duration_seconds: float = 0, error: str = None):
        """Quick method to record mission execution."""
        mission_id = f"mission_{int(datetime.now(timezone.utc).timestamp())}"
        now = datetime.now(timezone.utc)
        self.record_mission(
            mission_id=mission_id,
            mission_name=mission_name,
            status="completed" if success else "failed",
            started_at=now.isoformat(),
            completed_at=now.isoformat(),
            metadata={"duration": duration_seconds, "error": error}
        )

    # ============== TASK HISTORY ==============
    def record_task(self, task_id: str, agent_name: str, description: str,
                    outcome: str, started_at: str, completed_at: str,
                    execution_time: float, error: Optional[str] = None,
                    metadata: Optional[Dict] = None) -> None:
        """Record a task execution."""
        execute_query('''
            INSERT INTO task_history 
            (task_id, agent_name, task_description, outcome, started_at, completed_at, 
             execution_time_seconds, error_message, metadata_json)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (task_id) DO UPDATE SET
                outcome = EXCLUDED.outcome,
                completed_at = EXCLUDED.completed_at,
                execution_time_seconds = EXCLUDED.execution_time_seconds,
                error_message = EXCLUDED.error_message,
                metadata_json = EXCLUDED.metadata_json
        ''', (task_id, agent_name, description, outcome, started_at, completed_at,
              execution_time, error, json.dumps(metadata or {})))
        
        # Generate and store embedding for semantic search
        try:
            from shared_models import OllamaLLMIntegration
            import os
            llm = OllamaLLMIntegration(
                host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
                chat_model="mistral-small",
                embedding_model="mxbai-embed-large"
            )
            embedding = llm.generate_embedding(description)
            if embedding:
                execute_query('''
                    UPDATE task_history 
                    SET task_embedding = %s 
                    WHERE task_id = %s
                ''', (embedding, task_id))
        except Exception as e:
            logger.warning(f"Failed to generate task embedding: {e}")
    
    def get_recent_tasks(self, limit: int = 50, agent_name: Optional[str] = None) -> List[Dict]:
        """Get recent task history."""
        if agent_name:
            rows = execute_query('''
                SELECT * FROM task_history WHERE agent_name = %s
                ORDER BY completed_at DESC LIMIT %s
            ''', (agent_name, limit), fetch=True)
        else:
            rows = execute_query('''
                SELECT * FROM task_history ORDER BY completed_at DESC LIMIT %s
            ''', (limit,), fetch=True)
        
        return [dict(row) for row in rows]
    
    def get_task_stats(self) -> Dict[str, Any]:
        """Get task statistics."""
        rows = execute_query('''
            SELECT 
                COUNT(*) as total_tasks,
                SUM(CASE WHEN outcome = 'completed' THEN 1 ELSE 0 END) as completed,
                SUM(CASE WHEN outcome = 'failed' THEN 1 ELSE 0 END) as failed,
                SUM(CASE WHEN outcome = 'skipped' THEN 1 ELSE 0 END) as skipped,
                AVG(execution_time_seconds) as avg_execution_time
            FROM task_history
        ''', fetch=True)
        
        row = rows[0]
        total = row['total_tasks'] or 0
        completed = row['completed'] or 0
        
        return {
            "total_tasks": total,
            "completed": completed,
            "failed": row['failed'] or 0,
            "skipped": row['skipped'] or 0,
            "success_rate": round(completed / max(1, total), 4),
            "avg_execution_time_seconds": round(float(row['avg_execution_time'] or 0), 3)
        }
    
    # ============== MISSION HISTORY ==============
    def record_mission(self, mission_id: str, mission_name: str, status: str,
                       started_at: str, completed_at: Optional[str] = None,
                       steps_total: int = 0, steps_completed: int = 0,
                       metadata: Optional[Dict] = None) -> None:
        """Record a mission."""
        execute_query('''
            INSERT INTO mission_history
            (mission_id, mission_name, status, started_at, completed_at, 
             steps_total, steps_completed, metadata_json)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (mission_id) DO UPDATE SET
                status = EXCLUDED.status,
                completed_at = EXCLUDED.completed_at,
                steps_completed = EXCLUDED.steps_completed,
                metadata_json = EXCLUDED.metadata_json
        ''', (mission_id, mission_name, status, started_at, completed_at,
              steps_total, steps_completed, json.dumps(metadata or {})))
    
    def get_mission_stats(self) -> Dict[str, Any]:
        """Get mission statistics."""
        rows = execute_query('''
            SELECT 
                COUNT(*) as total_missions,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed
            FROM mission_history
        ''', fetch=True)
        
        row = rows[0]
        return {
            "total_missions": row['total_missions'] or 0,
            "completed": row['completed'] or 0,
            "failed": row['failed'] or 0
        }
    
    # ============== SYSTEM STATE ==============
    def save_system_state(self, key: str, value: Any) -> None:
        """Save a system state value."""
        now = datetime.now(timezone.utc)
        
        execute_query('''
            INSERT INTO system_state (key, value_json, updated_at)
            VALUES (%s, %s, %s)
            ON CONFLICT (key)
            DO UPDATE SET value_json = EXCLUDED.value_json, updated_at = EXCLUDED.updated_at
        ''', (key, json.dumps(value, default=str), now))
    
    def load_system_state(self, key: str, default: Any = None) -> Any:
        """Load a system state value."""
        rows = execute_query(
            'SELECT value_json FROM system_state WHERE key = %s',
            (key,),
            fetch=True
        )
        
        if rows:
            value = rows[0]['value_json']
            # Handle both JSON strings and already-parsed dictionaries
            if isinstance(value, dict):
                return value
            elif isinstance(value, str):
                return json.loads(value)
            else:
                return value
        return default
    
    # ============== TOOL USAGE ==============
    def record_tool_usage(self, tool_name: str, success: bool, 
                          execution_time: float, error: Optional[str] = None) -> None:
        """Record tool usage for analytics."""
        now = datetime.now(timezone.utc)
        
        execute_query('''
            INSERT INTO tool_usage (tool_name, success, execution_time_seconds, timestamp, error_message)
            VALUES (%s, %s, %s, %s, %s)
        ''', (tool_name, success, execution_time, now, error))
    
    def get_tool_stats(self) -> Dict[str, Any]:
        """Get tool usage statistics."""
        rows = execute_query('''
            SELECT 
                tool_name,
                COUNT(*) as calls,
                SUM(CASE WHEN success THEN 1 ELSE 0 END) as successes,
                AVG(execution_time_seconds) as avg_time
            FROM tool_usage
            GROUP BY tool_name
            ORDER BY calls DESC
        ''', fetch=True)
        
        tools = {}
        for row in rows:
            tools[row['tool_name']] = {
                "calls": row['calls'],
                "successes": row['successes'],
                "success_rate": round(row['successes'] / max(1, row['calls']), 4),
                "avg_time_seconds": round(float(row['avg_time'] or 0), 3)
            }
        
        return tools
    
    # ============== FULL STATE BACKUP ==============
    def save_full_swarm_state(self, state: Dict[str, Any]) -> None:
        """Save complete swarm state (backward compatible with JSON method)."""
        self.save_system_state("swarm_state", state)
    
    def load_full_swarm_state(self) -> Optional[Dict[str, Any]]:
        """Load complete swarm state."""
        return self.load_system_state("swarm_state")
    
    def close(self) -> None:
        """Close database connection (no-op for connection pool)."""
        pass

    def record_dynamic_agent_task(self, agent_id: str, task: Dict, result: Dict):
        """Record task execution by dynamic agent."""
        execute_query('''
            INSERT INTO task_history 
            (task_id, agent_name, task_description, outcome, started_at, completed_at, metadata_json)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        ''', (
            f"{agent_id}_{datetime.now(timezone.utc).timestamp()}",
            agent_id,
            str(task),
            "success" if result.get("success") else "failure",
            datetime.now(timezone.utc),
            datetime.now(timezone.utc),
            json.dumps({"result": result, "task": task})
        ))

    def query_similar_tasks(self, agent_name: str, task_description: str, limit: int = 5) -> List[Dict]:
        """Find semantically similar past tasks for an agent."""
        # Generate embedding for the query
        try:
            from shared_models import OllamaLLMIntegration
            import os
            llm = OllamaLLMIntegration(
                host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
                chat_model="mistral-small",
                embedding_model="mxbai-embed-large"
            )
            query_embedding = llm.generate_embedding(task_description)
            
            if not query_embedding:
                # Fallback to recent tasks if embedding fails
                return self._get_recent_tasks_fallback(agent_name, limit)
            
            # Semantic search using cosine similarity (pgvector <=>)
            rows = execute_query('''
                SELECT task_id, task_description, outcome, execution_time_seconds,
                       error_message, metadata_json, completed_at,
                       1 - (task_embedding <=> %s::vector) as similarity
                FROM task_history
                WHERE agent_name = %s AND task_embedding IS NOT NULL
                ORDER BY task_embedding <=> %s::vector
                LIMIT %s
            ''', (query_embedding, agent_name, query_embedding, limit), fetch=True)
            # Filter by similarity threshold - only return relevant memories
            SIMILARITY_THRESHOLD = 0.65  # Minimum confidence for memory relevance
            
            filtered_rows = []
            for row in rows:
                similarity = row.get('similarity', 0)
                if similarity >= SIMILARITY_THRESHOLD:
                    filtered_rows.append(dict(row))
            
            if not filtered_rows:
                logger.info(f"No sufficiently similar tasks found (threshold: {SIMILARITY_THRESHOLD})")
                return []  # Return empty if nothing is relevant enough
            
            return filtered_rows
            
        except Exception as e:
            logger.warning(f"Semantic search failed: {e}, falling back to recent tasks")
            return self._get_recent_tasks_fallback(agent_name, limit)

    def _get_recent_tasks_fallback(self, agent_name: str, limit: int) -> List[Dict]:
        """Fallback to recent tasks if semantic search fails."""
        rows = execute_query('''
            SELECT task_id, task_description, outcome, execution_time_seconds,
                   error_message, metadata_json, completed_at
            FROM task_history
            WHERE agent_name = %s
            ORDER BY completed_at DESC
            LIMIT %s
        ''', (agent_name, limit), fetch=True)
        return [dict(row) for row in rows]
    
    def get_agent_success_rate(self, agent_name: str, task_pattern: str = None) -> Dict[str, Any]:
        """Get success rate for an agent, optionally filtered by task pattern."""
        if task_pattern:
            rows = execute_query('''
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN outcome = 'completed' THEN 1 ELSE 0 END) as successful
                FROM task_history
                WHERE agent_name = %s AND task_description ILIKE %s
            ''', (agent_name, f'%{task_pattern}%'), fetch=True)
        else:
            rows = execute_query('''
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN outcome = 'completed' THEN 1 ELSE 0 END) as successful
                FROM task_history
                WHERE agent_name = %s
            ''', (agent_name,), fetch=True)
        
        row = rows[0] if rows else {"total": 0, "successful": 0}
        total = row['total'] or 0
        successful = row['successful'] or 0
        
        return {
            'total_tasks': total,
            'successful_tasks': successful,
            'success_rate': round(successful / max(1, total), 4)
        }

    def create_metrics_table(self):
        """Create table for tracking system metrics (PostgreSQL)."""
        try:
            execute_query('''
                CREATE TABLE IF NOT EXISTS metrics (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    metric_type VARCHAR(50) NOT NULL,
                    agent_name VARCHAR(100),
                    tool_name VARCHAR(100),
                    mission_type VARCHAR(50),
                    value FLOAT,
                    metadata JSONB
                )
            ''')
            # Create indexes separately
            execute_query('CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp)')
            execute_query('CREATE INDEX IF NOT EXISTS idx_metrics_type ON metrics(metric_type)')
            execute_query('CREATE INDEX IF NOT EXISTS idx_metrics_agent ON metrics(agent_name)')
            execute_query('CREATE INDEX IF NOT EXISTS idx_metrics_tool ON metrics(tool_name)')
        except Exception as e:
            logger.error(f"Failed to create metrics table: {e}")

    def record_metric(self, metric_type: str, value: float, 
                      agent_name: str = None, tool_name: str = None, 
                      mission_type: str = None, metadata: dict = None):
        """Record a single metric"""
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute('''
                    INSERT INTO metrics 
                    (metric_type, agent_name, tool_name, mission_type, value, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s)
                ''', (metric_type, agent_name, tool_name, mission_type, value, 
                      json.dumps(metadata) if metadata else None))
                conn.commit()

    def get_metrics(self, metric_type: str = None, hours: int = 24, limit: int = 100):
        """Query recent metrics"""
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=None) as cur:
                if metric_type:
                    cur.execute('''
                        SELECT timestamp, metric_type, agent_name, tool_name, 
                               mission_type, value, metadata
                        FROM metrics
                        WHERE metric_type = %s 
                          AND timestamp > NOW() - INTERVAL %s
                        ORDER BY timestamp DESC
                        LIMIT %s
                    ''', (metric_type, f'{hours} hours', limit))
                else:
                    cur.execute('''
                        SELECT timestamp, metric_type, agent_name, tool_name,
                               mission_type, value, metadata
                        FROM metrics
                        WHERE timestamp > NOW() - INTERVAL %s
                        ORDER BY timestamp DESC
                        LIMIT %s
                    ''', (f'{hours} hours', limit))
                
                return cur.fetchall()

    def create_metrics_table(self):
        """Create table for tracking system metrics (PostgreSQL)."""
        try:
            execute_query('''
                CREATE TABLE IF NOT EXISTS metrics (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                    metric_type VARCHAR(50) NOT NULL,
                    agent_name VARCHAR(100),
                    tool_name VARCHAR(100),
                    mission_type VARCHAR(50),
                    value FLOAT,
                    metadata JSONB
                )
            ''')
            # Create indexes separately
            execute_query('CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp)')
            execute_query('CREATE INDEX IF NOT EXISTS idx_metrics_type ON metrics(metric_type)')
            execute_query('CREATE INDEX IF NOT EXISTS idx_metrics_agent ON metrics(agent_name)')
            execute_query('CREATE INDEX IF NOT EXISTS idx_metrics_tool ON metrics(tool_name)')
        except Exception as e:
            logger.error(f"Failed to create metrics table: {e}")

# Global database instance
cva_db = CVADatabase()
