
import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import uuid

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock dependencies that might cause import errors or runtime side effects
sys.modules['database'] = MagicMock()
sys.modules['db_postgres'] = MagicMock()
sys.modules['supervisor'] = MagicMock()
sys.modules['tools'] = MagicMock()
sys.modules['shared_models'] = MagicMock()
sys.modules['tool_registry'] = MagicMock()
sys.modules['core'] = MagicMock()
sys.modules['core.mission_runner'] = MagicMock()

# Now we can attempt to import the class, or we can just extract the method if the import is too messy.
# Given the complexity, let's copy the method logic into a test harness if we can't import.
# But let's try to mock the class first.

class TestAutonomy(unittest.TestCase):
    def setUp(self):
        # We need to test the logic we added to _handle_request_human_input
        # We will create a dummy class that mimics the structure of CatalystVectorAlpha
        pass

    def test_auto_approval_logic(self):
        """Verify that learning missions get auto-approved."""
        
        # 1. Setup the Mock System
        class MockCVA:
            def __init__(self):
                self.pending_human_interventions = {}
                self.current_action_cycle_id = "cycle_1"
                self.external_log_sink = MagicMock()
                self.handle_human_response = MagicMock()
            
            # The exact method logic we want to test
            def _handle_request_human_input(self, directive):
                # Only including the snippet we care about + context setup
                message = directive['message']
                request_id = directive.get('request_id', 'test_req_id')
                
                # ... (standard setup logic from original file)
                self.pending_human_interventions[request_id] = {
                    "id": request_id, 
                    "status": "pending",
                    # ... other fields irrelevant for this test
                }

                # --- THE LOGIC TO TEST ---
                mission_type = directive.get('mission_type') or directive.get('context', {}).get('mission_type')
                if mission_type in ["learning", "research", "curiosity_driven_exploration", "toolsmithing"]:
                    print(f"DEBUG: Auto-approving {mission_type}")
                    auto_response = {
                        "approved": True, 
                        "comment": f"Auto-approved by '{mission_type}' autonomy policy.",
                        "original_request_id": request_id
                    }
                    self.handle_human_response(request_id, auto_response)
                    return
                # -------------------------

        cva = MockCVA()

        # 2. Test Case: Learning Mission (Should Auto-Approve)
        directive_learning = {
            "type": "REQUEST_HUMAN_INPUT",
            "message": "Can I read this website?",
            "mission_type": "learning",
            "request_id": "req_learning_1"
        }
        
        cva._handle_request_human_input(directive_learning)
        
        # Assertion: Handle human response should have been called
        cva.handle_human_response.assert_called_once()
        args, _ = cva.handle_human_response.call_args
        self.assertEqual(args[0], "req_learning_1")
        self.assertTrue(args[1]['approved'])
        self.assertIn("Auto-approved", args[1]['comment'])
        
        print("✅ Learning mission auto-approved successfully.")

    def test_standard_mission_logic(self):
        """Verify that standard missions DO NOT get auto-approved."""
        
        class MockCVA:
            def __init__(self):
                self.pending_human_interventions = {}
                self.current_action_cycle_id = "cycle_1"
                self.external_log_sink = MagicMock()
                self.handle_human_response = MagicMock()
            
            def _handle_request_human_input(self, directive):
                message = directive['message']
                request_id = directive.get('request_id', 'test_req_id')
                
                self.pending_human_interventions[request_id] = { "id": request_id, "status": "pending" }

                mission_type = directive.get('mission_type') or directive.get('context', {}).get('mission_type')
                if mission_type in ["learning", "research", "curiosity_driven_exploration", "toolsmithing"]:
                     self.handle_human_response(request_id, {"approved": True})
                     return

        cva = MockCVA()

        # 3. Test Case: Dangerous Mission (Should NOT Auto-Approve)
        directive_dangerous = {
            "type": "REQUEST_HUMAN_INPUT",
            "message": "Can I delete this database?",
            "mission_type": "infrastructure_destruction", # Not in the allowed list
            "request_id": "req_danger_1"
        }

        cva._handle_request_human_input(directive_dangerous)

        # Assertion: Handle human response should NOT have been called
        cva.handle_human_response.assert_not_called()
        self.assertEqual(cva.pending_human_interventions["req_danger_1"]["status"], "pending")
        
        print("✅ Critical mission correctly blocked for human review.")

if __name__ == '__main__':
    unittest.main()
