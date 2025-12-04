# database.py - SQLite persistence for CVA
import sqlite3
import json
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger("CatalystLogger")

class CVADatabase:
    """Thread-safe SQLite database for CVA state persistence."""
    
    def __init__(self, db_path: str = "persistence_data/cva.db"):
        self.db_path = db_path
        self._local = threading.local()
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Initialize tables
        self._init_tables()
        logger.info(f"Database initialized at {db_path}")
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
            self._configure_connection(self._local.conn)
        return self._local.conn

    # ----------------- connection helpers -----------------
    def _configure_connection(self, conn: sqlite3.Connection) -> None:
        """Apply concurrency-friendly pragmas."""
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA busy_timeout=5000;")  # 5s wait on locks
        except Exception as e:
            logger.warning(f"Failed to configure SQLite pragmas: {e}")

    def _execute_with_retry(self, cursor: sqlite3.Cursor, sql: str, params=(), retries: int = 5, backoff: float = 0.1):
        """Execute a statement with retries when the DB is locked."""
        for attempt in range(retries):
            try:
                return cursor.execute(sql, params)
            except sqlite3.OperationalError as e:
                if "locked" not in str(e).lower():
                    raise
                time.sleep(backoff * (attempt + 1))
        return cursor.execute(sql, params)

    def _commit_with_retry(self, conn: sqlite3.Connection, retries: int = 5, backoff: float = 0.1):
        """Commit with retries when the DB is locked."""
        for attempt in range(retries):
            try:
                return conn.commit()
            except sqlite3.OperationalError as e:
                if "locked" not in str(e).lower():
                    raise
                time.sleep(backoff * (attempt + 1))
        return conn.commit()
    
    def _init_tables(self) -> None:
        """Create tables if they don't exist."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Agent state table
        self._execute_with_retry(cursor, '''
            CREATE TABLE IF NOT EXISTS agent_state (
                agent_name TEXT PRIMARY KEY,
                state_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        ''')
        
        # Task history table
        self._execute_with_retry(cursor, '''
            CREATE TABLE IF NOT EXISTS task_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT UNIQUE,
                agent_name TEXT,
                task_description TEXT,
                outcome TEXT,
                started_at TEXT,
                completed_at TEXT,
                execution_time_seconds REAL,
                error_message TEXT,
                metadata_json TEXT
            )
        ''')
        
        # Mission history table
        self._execute_with_retry(cursor, '''
            CREATE TABLE IF NOT EXISTS mission_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                mission_id TEXT UNIQUE,
                mission_name TEXT,
                status TEXT,
                started_at TEXT,
                completed_at TEXT,
                steps_total INTEGER,
                steps_completed INTEGER,
                metadata_json TEXT
            )
        ''')
        
        # System state table (key-value store)
        self._execute_with_retry(cursor, '''
            CREATE TABLE IF NOT EXISTS system_state (
                key TEXT PRIMARY KEY,
                value_json TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        ''')
        
        # Tool usage stats
        self._execute_with_retry(cursor, '''
            CREATE TABLE IF NOT EXISTS tool_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tool_name TEXT,
                success INTEGER,
                execution_time_seconds REAL,
                timestamp TEXT,
                error_message TEXT
            )
        ''')
        
        self._commit_with_retry(conn)
    
    # ============== AGENT STATE ==============
    def save_agent_state(self, agent_name: str, state: Dict[str, Any]) -> None:
        """Save agent state to database."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        now = datetime.now(timezone.utc).isoformat()
        state_json = json.dumps(state, default=str)
        
        self._execute_with_retry(cursor, '''
            INSERT OR REPLACE INTO agent_state (agent_name, state_json, updated_at)
            VALUES (?, ?, ?)
        ''', (agent_name, state_json, now))
        
        self._commit_with_retry(conn)
    
    def load_agent_state(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Load agent state from database."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        self._execute_with_retry(cursor, 'SELECT state_json FROM agent_state WHERE agent_name = ?', (agent_name,))
        row = cursor.fetchone()
        
        if row:
            return json.loads(row['state_json'])
        return None
    
    def load_all_agent_states(self) -> Dict[str, Dict[str, Any]]:
        """Load all agent states."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        self._execute_with_retry(cursor, 'SELECT agent_name, state_json FROM agent_state')
        rows = cursor.fetchall()
        
        return {row['agent_name']: json.loads(row['state_json']) for row in rows}
    
    def record_mission_execution(self, mission_name: str, success: bool, 
                                  duration_seconds: float = 0, error: str = None):
        """Quick method to record mission execution."""
        mission_id = f"mission_{int(datetime.now(timezone.utc).timestamp())}"
        now = datetime.now(timezone.utc).isoformat()
        self.record_mission(
            mission_id=mission_id,
            mission_name=mission_name,
            status="completed" if success else "failed",
            started_at=now,
            completed_at=now,
            metadata={"duration": duration_seconds, "error": error}
        )

    # ============== TASK HISTORY ==============
    def record_task(self, task_id: str, agent_name: str, description: str,
                    outcome: str, started_at: str, completed_at: str,
                    execution_time: float, error: Optional[str] = None,
                    metadata: Optional[Dict] = None) -> None:
        """Record a task execution."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        self._execute_with_retry(cursor, '''
            INSERT OR REPLACE INTO task_history 
            (task_id, agent_name, task_description, outcome, started_at, completed_at, 
             execution_time_seconds, error_message, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (task_id, agent_name, description, outcome, started_at, completed_at,
              execution_time, error, json.dumps(metadata or {})))
        
        self._commit_with_retry(conn)
    
    def get_recent_tasks(self, limit: int = 50, agent_name: Optional[str] = None) -> List[Dict]:
        """Get recent task history."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if agent_name:
            self._execute_with_retry(cursor, '''
                SELECT * FROM task_history WHERE agent_name = ?
                ORDER BY completed_at DESC LIMIT ?
            ''', (agent_name, limit))
        else:
            self._execute_with_retry(cursor, '''
                SELECT * FROM task_history ORDER BY completed_at DESC LIMIT ?
            ''', (limit,))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def get_task_stats(self) -> Dict[str, Any]:
        """Get task statistics."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        self._execute_with_retry(cursor, '''
            SELECT 
                COUNT(*) as total_tasks,
                SUM(CASE WHEN outcome = 'completed' THEN 1 ELSE 0 END) as completed,
                SUM(CASE WHEN outcome = 'failed' THEN 1 ELSE 0 END) as failed,
                SUM(CASE WHEN outcome = 'skipped' THEN 1 ELSE 0 END) as skipped,
                AVG(execution_time_seconds) as avg_execution_time
            FROM task_history
        ''')
        
        row = cursor.fetchone()
        total = row['total_tasks'] or 0
        completed = row['completed'] or 0
        
        return {
            "total_tasks": total,
            "completed": completed,
            "failed": row['failed'] or 0,
            "skipped": row['skipped'] or 0,
            "success_rate": round(completed / max(1, total), 4),
            "avg_execution_time_seconds": round(row['avg_execution_time'] or 0, 3)
        }
    
    # ============== MISSION HISTORY ==============
    def record_mission(self, mission_id: str, mission_name: str, status: str,
                       started_at: str, completed_at: Optional[str] = None,
                       steps_total: int = 0, steps_completed: int = 0,
                       metadata: Optional[Dict] = None) -> None:
        """Record a mission."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        self._execute_with_retry(cursor, '''
            INSERT OR REPLACE INTO mission_history
            (mission_id, mission_name, status, started_at, completed_at, 
             steps_total, steps_completed, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (mission_id, mission_name, status, started_at, completed_at,
              steps_total, steps_completed, json.dumps(metadata or {})))
        
        self._commit_with_retry(conn)
    
    def get_mission_stats(self) -> Dict[str, Any]:
        """Get mission statistics."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        self._execute_with_retry(cursor, '''
            SELECT 
                COUNT(*) as total_missions,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed
            FROM mission_history
        ''')
        
        row = cursor.fetchone()
        return {
            "total_missions": row['total_missions'] or 0,
            "completed": row['completed'] or 0,
            "failed": row['failed'] or 0
        }
    
    # ============== SYSTEM STATE ==============
    def save_system_state(self, key: str, value: Any) -> None:
        """Save a system state value."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        now = datetime.now(timezone.utc).isoformat()
        
        self._execute_with_retry(cursor, '''
            INSERT OR REPLACE INTO system_state (key, value_json, updated_at)
            VALUES (?, ?, ?)
        ''', (key, json.dumps(value, default=str), now))
        
        self._commit_with_retry(conn)
    
    def load_system_state(self, key: str, default: Any = None) -> Any:
        """Load a system state value."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        self._execute_with_retry(cursor, 'SELECT value_json FROM system_state WHERE key = ?', (key,))
        row = cursor.fetchone()
        
        if row:
            return json.loads(row['value_json'])
        return default
    
    # ============== TOOL USAGE ==============
    def record_tool_usage(self, tool_name: str, success: bool, 
                          execution_time: float, error: Optional[str] = None) -> None:
        """Record tool usage for analytics."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        now = datetime.now(timezone.utc).isoformat()
        
        self._execute_with_retry(cursor, '''
            INSERT INTO tool_usage (tool_name, success, execution_time_seconds, timestamp, error_message)
            VALUES (?, ?, ?, ?, ?)
        ''', (tool_name, 1 if success else 0, execution_time, now, error))
        
        self._commit_with_retry(conn)
    
    def get_tool_stats(self) -> Dict[str, Any]:
        """Get tool usage statistics."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        self._execute_with_retry(cursor, '''
            SELECT 
                tool_name,
                COUNT(*) as calls,
                SUM(success) as successes,
                AVG(execution_time_seconds) as avg_time
            FROM tool_usage
            GROUP BY tool_name
            ORDER BY calls DESC
        ''')
        
        tools = {}
        for row in cursor.fetchall():
            tools[row['tool_name']] = {
                "calls": row['calls'],
                "successes": row['successes'],
                "success_rate": round(row['successes'] / max(1, row['calls']), 4),
                "avg_time_seconds": round(row['avg_time'] or 0, 3)
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
        """Close database connection."""
        if hasattr(self._local, 'conn') and self._local.conn:
            self._local.conn.close()
            self._local.conn = None



    def record_dynamic_agent_task(self, agent_id: str, task: Dict, result: Dict):
        """Record task execution by dynamic agent."""
        conn = self._get_connection()
        cursor = conn.cursor()
        self._execute_with_retry(cursor, '''
            INSERT INTO task_history 
            (task_id, agent_name, task_description, outcome, started_at, completed_at, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            f"{agent_id}_{datetime.now(timezone.utc).timestamp()}",
            agent_id,
            str(task),
            "success" if result.get("success") else "failure",
            datetime.now(timezone.utc).isoformat(),
            datetime.now(timezone.utc).isoformat(),
            json.dumps({"result": result, "task": task})
        ))
        self._commit_with_retry(conn)

# Global database instance
cva_db = CVADatabase()
