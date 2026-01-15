
import unittest
import sys
import os
import logging
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents import ProtoAgent
from shared_models import SharedWorldModel, timestamp_now

class ConcreteTestAgent(ProtoAgent):
    def _execute_agent_specific_task(self, *args, **kwargs):
        pass

class TestHiveMind(unittest.TestCase):
    
    def setUp(self):
        # Mocks
        self.mock_sink = MagicMock()
        self.world_model = SharedWorldModel(external_log_sink=self.mock_sink)
        
        # Agent A (The Teacher)
        self.agent_a = ConcreteTestAgent(
            name="TeacherAgent",
            eidos_spec={"role": "worker"},
            message_bus=MagicMock(),
            event_monitor=MagicMock(),
            external_log_sink=self.mock_sink,
            chroma_db_path="/tmp/chroma_db",
            persistence_dir="/tmp/test_persistence",
            paused_agents_file_path="/tmp/test_paused.json",
            world_model=self.world_model
        )
        
        # Agent B (The Student)
        self.agent_b = ConcreteTestAgent(
            name="StudentAgent",
            eidos_spec={"role": "worker"},
            message_bus=MagicMock(),
            event_monitor=MagicMock(),
            external_log_sink=self.mock_sink,
            chroma_db_path="/tmp/chroma_db",
            persistence_dir="/tmp/test_persistence",
            paused_agents_file_path="/tmp/test_paused.json",
            world_model=self.world_model
        )

    def test_gossip_protocol(self):
        """Verify Agent A can broadcast success and Agent B can find it."""
        
        # 1. Agent A solves "optimize_db"
        task_desc = "Optimize database performance"
        tool_used = "run_vacuum_analyze"
        args = {"target": "all"}
        
        self.agent_a.broadcast_success(task_desc, tool_used, args)
        
        # Verify World Model has it
        self.assertEqual(len(self.world_model.knowledge_base), 1)
        insight = self.world_model.knowledge_base[0]
        self.assertEqual(insight["agent"], "TeacherAgent")
        self.assertEqual(insight["tool"], "run_vacuum_analyze")
        
        # 2. Agent B searches for "database"
        results = self.agent_b.consult_hive_mind("database optimization")
        
        # Verify Agent B found the recipe
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["tool"], "run_vacuum_analyze")
        self.assertEqual(results[0]["agent"], "TeacherAgent")
        
        print("\n[TEST] Agent B successfully learned from Agent A's experience!")

if __name__ == '__main__':
    unittest.main()
