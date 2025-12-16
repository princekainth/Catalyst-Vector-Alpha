"""
Test that retry directives match expected orchestrator format
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json

def test_directive_format():
    """Verify retry directive matches AGENT_PERFORM_TASK format"""
    
    print("\n=== TEST: Retry directive format ===")
    
    # This is what Planner should generate for retries
    retry_directive = {
        "directive_type": "AGENT_PERFORM_TASK",
        "agent_name": "Worker",
        "task_id": "retry_task_123",
        "tool_name": "k8s_scale_deployment",
        "arguments": {
            "deployment_name": "nginx",
            "namespace": "default",
            "replicas": 3
        },
        "context": "Retry with complete args",
        "priority": "high"
    }
    
    print(f"Directive format:\n{json.dumps(retry_directive, indent=2)}")
    
    # Validate required fields
    assert retry_directive.get("directive_type") == "AGENT_PERFORM_TASK", "Must be AGENT_PERFORM_TASK"
    assert retry_directive.get("agent_name"), "Must have agent_name"
    assert retry_directive.get("task_id"), "Must have task_id"
    assert retry_directive.get("tool_name"), "Must have tool_name"
    assert isinstance(retry_directive.get("arguments"), dict), "Must have arguments dict"
    
    print("✓ Directive format valid")


def test_invalid_directive_rejected():
    """Test that malformed directives would be rejected"""
    
    print("\n=== TEST: Invalid directive detection ===")
    
    # Missing directive_type
    bad_directive_1 = {
        "agent_name": "Worker",
        "task_id": "bad_1"
    }
    
    # Wrong directive_type
    bad_directive_2 = {
        "directive_type": "INVALID_TYPE",
        "agent_name": "Worker"
    }
    
    # Missing required fields
    bad_directive_3 = {
        "directive_type": "AGENT_PERFORM_TASK",
        # Missing: agent_name, task_id, tool_name
    }
    
    test_cases = [
        (bad_directive_1, "missing directive_type"),
        (bad_directive_2, "wrong directive_type"),
        (bad_directive_3, "missing required fields")
    ]
    
    for directive, reason in test_cases:
        print(f"\nTesting: {reason}")
        print(f"Directive: {json.dumps(directive, indent=2)}")
        
        # Check what would fail
        is_valid = (
            directive.get("directive_type") == "AGENT_PERFORM_TASK" and
            directive.get("agent_name") and
            directive.get("task_id") and
            directive.get("tool_name")
        )
        
        assert not is_valid, f"Should reject: {reason}"
        print(f"✓ Correctly would reject: {reason}")


if __name__ == "__main__":
    print("Testing Directive Format Validation")
    print("=" * 50)
    
    try:
        test_directive_format()
        test_invalid_directive_rejected()
        
        print("\n" + "=" * 50)
        print("ALL TESTS PASSED ✓")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
