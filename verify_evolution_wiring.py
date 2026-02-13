
import logging
import sys
import unittest
from unittest.mock import MagicMock

# Configure logging
logging.basicConfig(level=logging.INFO)

print("Setting up mocks...")
# Mock dependencies to avoid side effects
sys.modules['k8s_adapter'] = MagicMock()
sys.modules['prometheus_adapter'] = MagicMock()
sys.modules['policy_engine'] = MagicMock()
sys.modules['database'] = MagicMock()

# Mock actual imports used in tool_registry.py
sys.modules['integrations'] = MagicMock()
sys.modules['integrations.prometheus_tool'] = MagicMock()
sys.modules['integrations.k8s_actions_tool'] = MagicMock()
sys.modules['core'] = MagicMock()
sys.modules['core.ops_policy_engine'] = MagicMock()
sys.modules['config_manager'] = MagicMock()
sys.modules['tools'] = MagicMock()
sys.modules['sandbox_toolsmith'] = MagicMock()
sys.modules['tool_types'] = MagicMock()

# Mock config_manager.get_config to return a dict
sys.modules['config_manager'].get_config.return_value = {}

print("Importing tool_registry...")
# Import ToolRegistry
try:
    from tool_registry import ToolRegistry
    print("ToolRegistry imported successfully.")
except ImportError as e:
    print(f"Failed to import ToolRegistry: {e}")
    sys.exit(1)

class MockEvolutionAgent:
    def __init__(self):
        self.gaps = []
        
    def record_capability_gap(self, description, context, attempted_tool, failure_reason, source_agent):
        print(f"Captured Gap: {description}")
        self.gaps.append({
            "description": description,
            "context": context,
            "attempted_tool": attempted_tool,
            "failure_reason": failure_reason,
            "source_agent": source_agent
        })

class TestEvolutionWiring(unittest.TestCase):
    def setUp(self):
        print("Setting up test...")
        self.registry = ToolRegistry(db=None)
        # Mock _initialize_default_tools to avoid running it if it causes issues
        # But we need it to register tools? 
        # Actually simplest is to manually register a dummy tool or use one that exists
        # self.registry._initialize_default_tools = MagicMock() 
        
        self.evolution_agent = MockEvolutionAgent()
        self.registry.set_evolution_agent(self.evolution_agent)
        
    def test_wiring_circuit_breaker(self):
        tool_name = "test_broken_tool"
        
        # Simulate 3 failures to trip the breaker
        print(f"\nSimulating 3 failures for {tool_name}...")
        for i in range(3):
            self.registry._record_tool_failure(tool_name)
            
        # Check if evolution was triggered
        self.assertEqual(len(self.evolution_agent.gaps), 1, "Evolution Agent should have recorded 1 gap")
        self.assertIn("failing repeatedly", self.evolution_agent.gaps[0]["description"])
        print("SUCCESS: Evolution Agent received the gap report!")

if __name__ == '__main__':
    unittest.main()
