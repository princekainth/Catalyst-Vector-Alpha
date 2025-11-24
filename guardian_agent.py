# guardian_agent.py - Monitors and manages dynamic agents
import logging
from typing import Dict, List
from datetime import datetime, timezone
from agent_factory import AgentFactory
from database import CVADatabase

logger = logging.getLogger("CatalystLogger")

class GuardianAgent:
    """Monitors dynamic agents and enforces policies."""
    
    def __init__(self, factory: AgentFactory, db: CVADatabase):
        self.factory = factory
        self.db = db
        self.policies = {
            "max_agents": 50,
            "max_failed_tasks": 3,
            "max_idle_minutes": 30
        }
        logger.info("Guardian Agent initialized")
    
    def health_check(self) -> Dict:
        """Check all active agents."""
        active = self.factory.list_active()
        
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_active": len(active),
            "expired_cleaned": self.factory.cleanup_expired(),
            "policy_violations": [],
            "actions_taken": []
        }
        
        # Check max agents policy
        if len(active) > self.policies["max_agents"]:
            report["policy_violations"].append(
                f"Too many agents: {len(active)}/{self.policies['max_agents']}"
            )
        
        # Check individual agents
        for agent_info in active:
            agent_id = agent_info["agent_id"]
            agent = self.factory.get_agent(agent_id)
            
            if agent and agent.check_expiration():
                self.factory.kill_agent(agent_id)
                report["actions_taken"].append(f"Killed expired: {agent_id}")
        
        logger.info(f"Guardian health check: {report['total_active']} active, "
                   f"{report['expired_cleaned']} expired")
        
        return report
    
    def kill_agent(self, agent_id: str, reason: str) -> bool:
        """Terminate agent with reason."""
        success = self.factory.kill_agent(agent_id)
        if success:
            logger.warning(f"Guardian killed {agent_id}: {reason}")
        return success
    
    def get_metrics(self) -> Dict:
        """Get agent factory metrics."""
        conn = self.db._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                COUNT(*) as total_spawned,
                SUM(CASE WHEN status='active' THEN 1 ELSE 0 END) as active,
                SUM(CASE WHEN status='terminated' THEN 1 ELSE 0 END) as terminated,
                AVG(tasks_completed) as avg_tasks
            FROM dynamic_agents
        ''')
        
        row = cursor.fetchone()
        return {
            "total_spawned": row[0] or 0,
            "currently_active": row[1] or 0,
            "terminated": row[2] or 0,
            "avg_tasks_per_agent": round(row[3] or 0, 2)
        }
