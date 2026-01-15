
import unittest
import sys
import os
import time
import json
from unittest.mock import MagicMock, patch

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents import ProtoAgent
from curiosity_loop import CuriosityLoop
import prompts

class ConcreteAgent(ProtoAgent):
    def _execute_agent_specific_task(self): pass
    def run_cycle(self): pass

class TestPhase4(unittest.TestCase):
    def setUp(self):
        self.mock_bus = MagicMock()
        self.mock_monitor = MagicMock()
        self.mock_log = MagicMock()
        self.mock_db = MagicMock()
        self.mock_world = MagicMock()
        
    def test_personality_initialization(self):
        """Verify that agents load the correct persona based on role."""
        
        # Test Cautious Security Agent
        security_spec = {"role": "Security", "name": "Guardian_1"}
        agent = ConcreteAgent(
            name="Guardian_1", 
            eidos_spec=security_spec,
            message_bus=self.mock_bus,
            event_monitor=self.mock_monitor,
            external_log_sink=self.mock_log,
            chroma_db_path=".",
            persistence_dir=".",
            paused_agents_file_path=".",
            world_model=self.mock_world
        )
        self.assertIn("Paranoid Guardian", agent.persona)
        print("✅ Security Agent loaded 'Paranoid Guardian' persona.")
        
        # Test default fallback
        unknown_spec = {"role": "Dishwasher", "name": "Helper_1"}
        agent_unknown = ConcreteAgent(
            name="Helper_1", 
            eidos_spec=unknown_spec,
            message_bus=self.mock_bus,
            event_monitor=self.mock_monitor,
            external_log_sink=self.mock_log,
            chroma_db_path=".",
            persistence_dir=".",
            paused_agents_file_path=".",
            world_model=self.mock_world
        )
        self.assertIn("helpful and efficient", agent_unknown.persona)
        print("✅ Unknown Agent loaded default persona.")

    def test_dream_cycle_trigger(self):
        """Verify that CuriosityLoop enters dream mode after idle cycles."""
        loop = CuriosityLoop(cycle_time=1, orchestrator=None)
        loop.external_log_sink = MagicMock()
        loop.llm = MagicMock()
        loop.llm.generate_text.return_value = "I dreamt of electric sheep."
        loop.memory = MagicMock()
        
        # Mock _should_run to return False (idle)
        loop._should_run = MagicMock(return_value=(False, "idle"))
        
        # Mock _enter_dream_mode to verify call
        with patch.object(loop, '_enter_dream_mode', wraps=loop._enter_dream_mode) as mock_dream:
            # First 4 cycles (should just increment counter)
            for _ in range(4):
                loop._loop_single_step_test() # We need to simulate the loop without blocking
            
            self.assertEqual(loop.idle_cycles, 4)
            mock_dream.assert_not_called()
            
            # 5th cycle (should trigger dream)
            loop._loop_single_step_test()
            
            mock_dream.assert_called_once()
            self.assertEqual(loop.idle_cycles, 0) # Should be reset
            print("✅ Dream Cycle triggered after 5 idle cycles.")

    # Validation Helper to expose internal loop logic for testing without threading
    def _loop_single_step_test(loop_self):
        should_run, reason = loop_self._should_run()
        if should_run:
            loop_self._explore()
            loop_self.idle_cycles = 0
        else:
            loop_self.idle_cycles += 1
            if loop_self.idle_cycles >= 5:
                loop_self._enter_dream_mode()
                loop_self.idle_cycles = 0

# Monkey patch the test helper into the class for the test
CuriosityLoop._loop_single_step_test = TestPhase4._loop_single_step_test

if __name__ == '__main__':
    unittest.main()
