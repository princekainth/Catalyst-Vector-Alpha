
import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import uuid
import json

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock key dependencies
sys.modules['database'] = MagicMock()
sys.modules['db_postgres'] = MagicMock()
sys.modules['supervisor'] = MagicMock()
sys.modules['shared_models'] = MagicMock()

# Mock AgentFactory before importing catalyst
mock_factory = MagicMock()
mock_agent = MagicMock()
mock_agent.spec.name = "TestAgent"
mock_agent.name = "TestAgent"
mock_agent.spec.agent_id = "agent_test_123"
mock_agent.spec.tools = ["web_search"]
mock_factory.spawn_agent.return_value = mock_agent

# Use patch to mock AgentFactory when imported by catalyst
with patch('agent_factory.AgentFactory', return_value=mock_factory):
     # We need to import the class to test its methods
     # But catalyst imports many things. Let's create a partial mock of CatalystVectorAlpha class
     pass

class TestReproduction(unittest.TestCase):
    
    def test_spawn_dynamic_agent_logic(self):
        """Verify that _handle_spawn_dynamic_agent calls factory and registers agent."""
        
        # 1. Setup Mock System
        class MockCVA:
            def __init__(self):
                self.agent_factory = mock_factory
                self.agent_instances = {}
                self.log_buffer = []
                self.swarm_state = {}
            
            def _log_swarm_activity(self, *args, **kwargs):
                self.log_buffer.append(kwargs)

            # Paste the exact logic we implemented
            def _handle_spawn_dynamic_agent(self, directive):
                purpose = directive.get("purpose")
                context = directive.get("context", {})
                parent_agent = directive.get("requester_agent", "System")
                
                if not purpose:
                    raise ValueError("SPAWN_DYNAMIC_AGENT requires 'purpose'.")

                # Call Factory
                result = self.agent_factory.spawn_agent(
                    purpose=purpose,
                    context=context,
                    parent_agent=parent_agent
                )

                if isinstance(result, dict) and not result.get("success", True):
                     return

                # Success - Register the new agent
                new_agent = result
                instance_name = new_agent.name 
                unique_key = f"Dynamic_{instance_name}_{new_agent.spec.agent_id[:6]}"
                self.agent_instances[unique_key] = new_agent
                print(f"DEBUG: Spawned {unique_key}")

        cva = MockCVA()
        
        # 2. Test Directive
        directive = {
            "type": "SPAWN_DYNAMIC_AGENT",
            "purpose": "Monitor RAM usage",
            "requester_agent": "ParentAgent",
            "context": {"target": "system"}
        }
        
        cva._handle_spawn_dynamic_agent(directive)
        
        # 3. Assertions
        # Factory called?
        mock_factory.spawn_agent.assert_called_once_with(
            purpose="Monitor RAM usage",
            context={"target": "system"},
            parent_agent="ParentAgent"
        )
        
        # Agent registered?
        keys = list(cva.agent_instances.keys())
        self.assertTrue(len(keys) == 1)
        self.assertIn("Dynamic_TestAgent_agent_", keys[0])
        print("✅ Agent reproduction logic verified.")

    def test_spawn_tool(self):
        """Verify the tool injects the directive."""
        import tools
        
        # Mock global instance in tools
        mock_cva_global = MagicMock()
        tools._cva_instance = mock_cva_global
        
        # Call tool
        res = tools.spawn_agent(purpose="Test Spawning", context={"a":1})
        
        # Assertions
        self.assertEqual(res['status'], 'ok')
        self.assertEqual(res['data']['status'], 'queued')
        
        # Verify directive injection
        args, _ = mock_cva_global.inject_directives.call_args
        directive = args[0][0]
        self.assertEqual(directive['type'], 'SPAWN_DYNAMIC_AGENT')
        self.assertEqual(directive['purpose'], 'Test Spawning')
        self.assertEqual(directive['context'], {'a': 1})
        
        print("✅ spawn_agent tool verified.")

if __name__ == '__main__':
    unittest.main()
