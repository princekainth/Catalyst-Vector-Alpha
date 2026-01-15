
import unittest
import sys
import os
import logging
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents import ProtoAgent
from catalyst_vector_alpha import CatalystVectorAlpha
from agent_factory import DynamicAgent, AgentSpec
from datetime import datetime, timedelta, timezone

class ConcreteTestAgent(ProtoAgent):
    def _execute_agent_specific_task(self, *args, **kwargs):
        pass

class TestEnergyEconomics(unittest.TestCase):
    
    def setUp(self):
        # Mocks
        self.mock_bus = MagicMock()
        self.mock_monitor = MagicMock()
        self.mock_sink = MagicMock()
        
        # Helper to create agent
        self.agent = ConcreteTestAgent(
            name="TestAgent",
            eidos_spec={"role": "worker"},
            message_bus=self.mock_bus,
            event_monitor=self.mock_monitor,
            external_log_sink=self.mock_sink,
            chroma_db_path="/tmp/chroma_db",
            persistence_dir="/tmp/test_persistence",
            paused_agents_file_path="/tmp/test_paused.json",
            world_model=MagicMock()
        )

    def test_initial_energy(self):
        """Verify agent starts with max energy."""
        self.assertTrue(hasattr(self.agent, "energy"))
        self.assertEqual(self.agent.energy, 100.0)
        self.assertEqual(self.agent.max_energy, 100.0)

    def test_metabolism(self):
        """Verify tick() burns energy."""
        initial = self.agent.energy
        self.agent.tick()
        self.assertLess(self.agent.energy, initial)
        self.assertEqual(self.agent.energy, 100.0 - self.agent.metabolic_rate)

    def test_gain_energy(self):
        """Verify rewards work and cap at max."""
        self.agent.energy = 50.0
        self.agent.gain_energy(10.0)
        self.assertEqual(self.agent.energy, 60.0)
        
        # Test Cap
        self.agent.energy = 95.0
        self.agent.gain_energy(10.0)
        self.assertEqual(self.agent.energy, 100.0) # Should be capped

    def test_starvation_in_check_expiration(self):
        """Verify DynamicAgent expires when energy is 0."""
        # Create DynamicAgent
        spec = AgentSpec(
            agent_id="test_id", name="DynamicTest", purpose="Test", specialized_prompt="Prompt",
            tools=[], ttl_hours=1, created_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
            parent_agent="parent"
        )
        
        dynamic_agent = DynamicAgent(
            spec=spec, tool_registry=MagicMock(), db=MagicMock(),
            message_bus=self.mock_bus, event_monitor=self.mock_monitor, external_log_sink=self.mock_sink,
            chroma_db_path="/tmp/chroma", persistence_dir="/tmp/dir", paused_agents_file_path="/tmp/paused",
            world_model=MagicMock()
        )
        
        # Initial check
        self.assertFalse(dynamic_agent.check_expiration())
        
        # Starve it
        dynamic_agent.energy = 0
        self.assertTrue(dynamic_agent.check_expiration())
        self.assertTrue(dynamic_agent.is_expired)

if __name__ == '__main__':
    unittest.main()
